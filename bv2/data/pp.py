import numpy as np
from einops import rearrange
from PIL import Image


def sanity_check(example):
    attn_regions_keys = [k for k in example if k.startswith("attn_regions")]

    keys_with_same_size = ("tokens", "loss_weights") + tuple(attn_regions_keys)
    if len(set(len(example[k]) for k in keys_with_same_size)) != 1:
        raise ValueError(
            "Keys with unexpectedly different sizes:\n"
            + "\n".join(f"{k}: {len(example[k])}" for k in keys_with_same_size)
        )

    # fmt: off
    assert example["tokens"].dtype == np.uint8, f"{example['tokens'].dtype=}, expected int64"
    assert example["loss_weights"].dtype == np.int64, f"{example['loss_weights'].dtype=}, expected int64"
    # fmt: on
    for k in attn_regions_keys:
        assert example[k].dtype == np.int64, f"{k}: {example[k].dtype=}, expected int64"
    return example


def patchify(img, *, ph, pw):
    img = np.asarray(img)
    h, w, c = img.shape

    # Assume the image is already properly sizes as multiple from a previous step.
    assert w % pw == 0 and h % ph == 0 and c == 3, "Need proper shape; forgot pp?"

    patches = rearrange(img, "(ny ph) (nx pw) c -> ny nx ph pw c", ph=ph, pw=pw)

    # ij = y/x-index of patch ; mn = number of patches in height/width
    m, n = h // ph, w // pw
    i, j = np.mgrid[:m, :n].astype(np.int16)
    positions = np.stack([i, j, np.full((m, n), m, np.int16), np.full((m, n), n, np.int16)], axis=-1)  # fmt: skip

    return patches, positions


def unpatchify(patches, positions):
    patches = np.asarray(patches)
    pos = np.asarray(positions)

    num_patches, ph, pw, c = patches.shape

    # Channels 2 and 3 are the number of patches in height and width.
    img = np.zeros((pos[0, 2] * ph, pos[0, 3] * pw, c), dtype=patches.dtype)
    for idx in range(num_patches):
        i, j = positions[idx, 0], positions[idx, 1]
        y_start, y_end = i * ph, (i + 1) * ph
        x_start, x_end = j * pw, (j + 1) * pw
        img[y_start:y_end, x_start:x_end, :] = patches[idx]

    return img


def resize_max_patches(img, max_patches, *, ph=16, pw=16):
    orig_w, orig_h = target_w, target_h = img.size

    # First, get w/h below target pixel area if needed:
    orig_pixels = orig_w * orig_h
    max_pixels = max_patches * ph * pw
    if orig_pixels >= max_pixels:
        scale = (max_pixels / orig_pixels) ** 0.5
        target_w = int(orig_w * scale)
        target_h = int(orig_h * scale)

    # Then, round w/h down to the next multiple of pw/ph, but avoid 0 h/w.
    if target_w % pw != 0:
        target_w = max((target_w // pw) * pw, pw)
    if target_h % ph != 0:
        target_h = max((target_h // ph) * ph, ph)

    target = (target_w, target_h)
    if target == img.size:
        return img

    # Lanczos used to be called ANTIALIAS in pillow. However, it's not quite the
    # same as TF and TV's linear+antialias. Let's see if it's a bottleneck.
    return img.resize(target, Image.LANCZOS)
