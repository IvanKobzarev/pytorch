import numpy as np
import pytest
import torch
from torch.nn.attention.flex_attention import create_block_mask

import bv2.data.dpack as d
import bv2.data.pp as pp
import bv2.model as model


@pytest.mark.gpu
def test_model_simple():
    """Simple model test.

    It "simply" tests that model builds at all and
    that we can run fwd pass. There is no further checks.

    It is useful for model development.
    """
    device = torch.device("cuda")
    patch_size = 4

    m = model.SimpleTransformer(
        dim=512,
        depth=2,
        vocab=100,
        txt_unemb={"init_std": 0.01},
        img={"ph": patch_size, "pw": patch_size},
        reg={"nreg": 101},
        head_dim=64,
        kv_reduce=1,
    )
    m.init_weights(torch.Generator().manual_seed(0))
    m.to(device)

    # Training data consists of random tokens ids.
    # There are multiple examples (below is 4), packed together.
    # Optioanlly, we add a single 16x16 image with 4x4 patches
    # and "register" tokens, to test more code paths.
    ntoks = [33, 100, 40, 77]
    data, loss_weights, seqids = _prepare_data(
        device,
        ntoks=ntoks,
        add_image_and_reg=True,
        patch_size=patch_size,
    )

    def _mask_fn(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & (seqids[q_idx] == seqids[kv_idx])

    mask = {
        "attn_regions": create_block_mask(
            _mask_fn,
            Q_LEN=sum(ntoks),
            KV_LEN=sum(ntoks),
            device=data.device,
            B=None,
            H=None,
        )
    }

    x, extra = m(data, mask, loss_weights, seqids, mode="loss")


@pytest.mark.gpu
def test_model_batching():
    device = torch.device("cuda")
    patch_size = 4

    m = model.SimpleTransformer(
        dim=512,
        depth=2,
        vocab=100,
        txt_unemb={"init_std": 0.1},
        img={"ph": patch_size, "pw": patch_size},
        reg={"nreg": 101},
        head_dim=64,
        kv_reduce=1,
    )
    m.init_weights(torch.Generator().manual_seed(0))
    m.to(device)

    ntoks = [100, 150]
    data, loss_weights, seqids = _prepare_data(
        device,
        ntoks=ntoks,
        add_image_and_reg=True,
        patch_size=patch_size,
    )

    # #### Packed ###############################################
    def _mask_fn(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & (seqids[q_idx] == seqids[kv_idx])

    mask = {
        "attn_regions": create_block_mask(
            _mask_fn,
            Q_LEN=sum(ntoks),
            KV_LEN=sum(ntoks),
            device=data.device,
            B=None,
            H=None,
        )
    }

    _, res = m(data, mask, loss_weights, seqids, mode="loss")
    losses_packed = res["tok_losses"]
    losses_packed_1 = losses_packed[: ntoks[0] - 1]
    losses_packed_2 = losses_packed[ntoks[0] :]

    #### One-by-one ###########################################
    k = ntoks[0]

    mask = {
        "attn_regions": create_block_mask(
            lambda b, h, q_idx, kv_idx: q_idx >= kv_idx,
            Q_LEN=ntoks[0],
            KV_LEN=ntoks[0],
            device=data.device,
            B=None,
            H=None,
        )
    }
    _, res = m(data[:k], mask, loss_weights[:k], seqids[:k], mode="loss")
    losses_1 = res["tok_losses"]

    mask = {
        "attn_regions": create_block_mask(
            lambda b, h, q_idx, kv_idx: q_idx >= kv_idx,
            Q_LEN=ntoks[1],
            KV_LEN=ntoks[1],
            device=data.device,
            B=None,
            H=None,
        )
    }
    _, res = m(data[k:], mask, loss_weights[k:], seqids[k:], mode="loss")
    losses_2 = res["tok_losses"]

    assert torch.allclose(losses_packed_1, losses_1)
    assert torch.allclose(losses_packed_2, losses_2)

    #### Batched ###########################################

    k = max(ntoks)

    data_batched = torch.zeros(2, k, data.shape[-1], dtype=data.dtype).to(data.device)
    loss_weights_batched = torch.zeros(2, k, dtype=loss_weights.dtype).to(data.device)
    seqids_batched = -torch.ones(2, k, dtype=seqids.dtype).to(data.device)

    data_batched[0, : ntoks[0]] = data[: ntoks[0]]
    loss_weights_batched[0, : ntoks[0]] = loss_weights[: ntoks[0]]
    seqids_batched[0, : ntoks[0]] = seqids[: ntoks[0]]

    data_batched[1, : ntoks[1]] = data[ntoks[0] :]
    loss_weights_batched[1, : ntoks[1]] = loss_weights[ntoks[0] :]
    seqids_batched[1, : ntoks[1]] = seqids[ntoks[0] :]

    # fmt: off
    mask = {
        "attn_regions": create_block_mask(
            lambda b, h, q_idx, kv_idx: (q_idx >= kv_idx) & (seqids_batched[b][q_idx] >= 0),
            Q_LEN=max(ntoks),
            KV_LEN=max(ntoks),
            device=data.device,
            B=2,
            H=None,
        )
    }
    # fmt: on

    _, res = m(
        data_batched,
        mask,
        loss_weights_batched,
        seqids_batched,
        mode="loss",
    )
    losses_batched_1 = res["tok_losses"][0, ..., : ntoks[0] - 1]
    losses_batched_2 = res["tok_losses"][1, ..., : ntoks[1] - 1]

    assert torch.allclose(losses_packed_1, losses_batched_1)
    assert torch.allclose(losses_packed_2, losses_batched_2)


def _prepare_data(device, ntoks=(128, 128), add_image_and_reg=True, patch_size=4):
    nbytes = max(
        d.nbytes_text(), d.nbytes_image(patch_size, patch_size), d.nbytes_reg()
    )
    data = np.zeros((sum(ntoks), nbytes), dtype=np.uint8)

    # Fill data with text tokens first
    tokens = np.random.randint(0, 100, size=(sum(ntoks),)).astype(np.int64)

    pos = np.concatenate([np.arange(ntok) for ntok in ntoks], axis=0)
    d.pack_text(tokens, positions=pos, out=data)

    # Lets add image now. It will also conveniently add registers.
    if add_image_and_reg:
        ims, ps = patch_size * 4, 4
        k = (ims // ps) ** 2 + ims // ps + 2  # patches + row separators + size tokens
        assert k < ntoks[0], "An image should fit in the first example"
        image, positions = pp.patchify(
            np.zeros((ims, ims, 3), dtype=np.uint8), ph=ps, pw=ps
        )
        d.pack_image_with_extras(
            image, positions, out=data[:k], add_hw=True, add_row_sep=True
        )

    # Final data preparation
    data = torch.from_numpy(data).to(device)

    loss_weights = torch.ones(sum(ntoks), dtype=torch.float32).to(device)

    seqids = [np.zeros(ntok) + i for i, ntok in enumerate(ntoks)]
    seqids = np.concatenate(seqids, axis=0)
    seqids = torch.from_numpy(seqids).to(device)

    return data, loss_weights, seqids
