import einops
import numpy as np
import torch

# fmt: off


MOD_TXT = 1
MOD_IMG = 2
MOD_REG = 9
MOD_SEP = 10


def nbytes_text():
    return 8 + 4 + 1  # token as int64, position int32, modality single byte.


def pack_text(tokens, positions="auto", out=None):
    # For convenience, we allow passing a list, as long as it's np.r_'able.
    if not isinstance(tokens, np.ndarray):
        tokens = np.r_[tuple(tokens)]

    # Then, we make sure it's int64, the only dtype that nn.Embedding accepts on CUDA.
    tokens = np.asarray(tokens, dtype=np.int64)

    if isinstance(positions, str) and positions == "auto":
        # also works for batched inputs
        positions = np.broadcast_to(np.arange(tokens.shape[-1], dtype=np.int32), tokens.shape)
    else:
        positions = np.asarray(positions, dtype=np.int32)

    nbytes = nbytes_text()
    if out is None:
        out = np.zeros((*tokens.shape, nbytes), dtype=np.uint8)
    else:
        assert out.dtype == np.uint8, "Can only pack into uint8 buffer."
        assert out.shape[-1] >= nbytes, f"Need at least {nbytes} bytes for packing."

    # View each of them as raw bytes:
    out[..., :8] = tokens[..., None].view(dtype=np.uint8)
    out[..., 8:8+4] = positions[..., None].view(dtype=np.uint8)
    out[..., -1] = MOD_TXT
    return out


def unpack_as_text(data):
    """Undo what `pack_text` did, but set non-text elements to 0."""
    assert data.ndim > 1 and data.shape[-1] >= nbytes_text(), f"Unpack wrong shape {data.shape}?"  # fmt: skip

    # Turn all non-text into zeros:
    mask = data[..., -1] == MOD_TXT
    data = data * mask[..., None]

    # This is the unpacking. Last dim after view is singleton, so drop.
    tokens = data[..., :8].contiguous().view(dtype=torch.int64)[..., 0]
    positions = data[..., 8:12].contiguous().view(dtype=torch.int32)[..., 0]

    return tokens, positions, mask


def nbytes_image(ph, pw, channels=3):
    return ph * pw * channels + 4 * 4 + 1  # pixels, ijmn, modality


def pack_image(patches, positions, out=None):
    # Assert patches dtype, seems error-prone (images are often floats)
    patches = np.asarray(patches)
    assert patches.dtype == np.uint8

    positions = np.asarray(positions, dtype=np.int32)

    *nps, ph, pw, c = patches.shape
    nbytes = nbytes_image(ph, pw, c)
    npatches = np.prod(nps)
    if out is None:
        out = np.zeros((npatches, nbytes), dtype=np.uint8)
    else:
        assert out.dtype == np.uint8, "Can only pack into uint8 buffer."
        assert out.shape[-1] >= nbytes, f"Need at least {nbytes} bytes for packing."

    n = ph * pw * c
    out[..., :n] = patches.reshape(npatches, n)  # Already uint8
    out[..., n:n+16] = positions.reshape(npatches, 4).view(dtype=np.uint8)  # 4 int32's # fmt: skip
    out[..., -1] = MOD_IMG
    return out


def nbytes_image_with_extras(ph, pw, channels=3, tiptoi=0):
    return ph * pw * channels + 4 * 4 + tiptoi * 4 + 1  # pixels, ijmn, tiptoi, modality


def pack_image_with_extras(patches, positions, out=None, add_row_sep=False, add_hw=False, tiptoi=0):
    """Pack image patches with extra data: row separators and image size tokens."""
    # Assert patches dtype, seems error-prone (images are often floats)
    patches = np.asarray(patches)
    assert patches.dtype == np.uint8
    positions = np.asarray(positions, dtype=np.int32)

    # Multiple of 4, for 2D sin/cos.
    assert tiptoi % 4 == 0

    h, w, ph, pw, c = patches.shape
    nbytes = nbytes_image_with_extras(ph, pw, c, tiptoi)

    n = h * w + h * add_row_sep + 2 * add_hw
    k = ph * pw * c

    if out is None:
        out = np.zeros((n, nbytes), dtype=np.uint8)
    else:
        assert out.shape[0] == n, f"Target buffer len {out.shape[0]} != expected len {n}"
        assert out.dtype == np.uint8, "Can only pack into uint8 buffer."
        assert out.shape[-1] >= nbytes, f"Need at least {nbytes} bytes for packing."

    out_idx = 0

    # Adding image size (in patches) as first two tokens
    if add_hw:
        out[out_idx, :8] = np.array([h], dtype=np.int64).view(dtype=np.uint8)
        out[out_idx + 1, :8] = np.array([w], dtype=np.int64).view(dtype=np.uint8)
        out[[out_idx, out_idx + 1], -1] = MOD_SEP
        out_idx += 2

    if tiptoi:
        sincos = _sincos2d(positions, tiptoi).astype(np.float32)

    patches = einops.rearrange(patches, "h w ph pw c -> h w (ph pw c)")
    for row in range(h):
        out[out_idx:out_idx + w, :k] = patches[row]  # already uint8
        out[out_idx:out_idx + w, k:k+16] = positions[row].view(dtype=np.uint8)

        if tiptoi:
            out[out_idx:out_idx + w, k+16:k+16+4*tiptoi] = sincos[row].view(dtype=np.uint8)

        out[out_idx:out_idx + w, -1] = MOD_IMG
        out_idx += w
        if add_row_sep:
            out[out_idx, :8] = np.array([row], dtype=np.int64).view(dtype=np.uint8)
            out[out_idx, -1] = MOD_SEP
            out_idx += 1

    return out


def unpack_as_image(data, ph, pw, channels=3, tiptoi=0, keep_flat=False):
    """Undo what `pack_image` did, but set non-imagepatch elements to 0."""
    assert data.ndim > 1 and data.shape[-1] >= nbytes_image(ph, pw, channels), f"Unpack wrong shape {data.shape}"  # fmt: skip

    # Turn all non-imagepatch data into zeros:
    mask = data[..., -1] == MOD_IMG
    data = data * mask[..., None]

    # This is the unpacking. Last dim after view is singleton, so drop.
    n = ph * pw * channels
    patches = data[..., :n].contiguous()
    positions = data[..., n:n+16].contiguous().view(dtype=torch.int32)

    sincos = data[..., n+16:n+16+4*tiptoi].contiguous().view(dtype=torch.float32) if tiptoi else None  # fmt: skip

    if not keep_flat:
        patches = patches.reshape(-1, ph, pw, channels)

    return patches, positions, sincos, mask


def nbytes_reg():
    return 8 + 1  # register ID as int64, modality single byte.


def pack_regs(nreg, out=None, mod_id=MOD_REG):
    # Then, we make sure it's int64, the only dtype that nn.Embedding accepts on CUDA.
    regs = np.arange(nreg, dtype=np.int64)

    nbytes = nbytes_reg()
    if out is None:
        out = np.zeros((len(regs), nbytes), dtype=np.uint8)
    else:
        assert out.dtype == np.uint8, "Can only pack into uint8 buffer."
        assert out.shape[-1] >= nbytes, f"Need at least {nbytes} bytes for packing."

    # View each of them as raw bytes:
    out[..., :8] = regs[:, None].view(dtype=np.uint8)
    out[..., -1] = mod_id
    return out


def unpack_as_reg(data, mod_id=MOD_REG):
    """Undo what `pack_text` did, but set non-text elements to 0."""
    assert data.ndim > 1 and data.shape[-1] >= nbytes_reg(), f"Unpack wrong shape {data.shape}?"  # fmt: skip

    # This is the unpacking. Last dim after view is singleton, so drop.
    regs = data[..., :8].contiguous().view(dtype=torch.int64)[..., 0]
    mask = data[..., -1] == mod_id

    return regs * mask, mask


def _sincos2d(pos, dim, temperature=1000.0):
    assert dim % 4 == 0
    d = dim // 4
    omega = temperature ** (-np.linspace(0.0, 1.0, d))
    x = pos[..., 0:1] * omega
    y = pos[..., 1:2] * omega
    return np.concatenate([np.sin(x), np.cos(x), np.sin(y), np.cos(y)], axis=-1)
