"""Move a BlockMask (with closure tensors and dynamic indices) to a device."""

from functools import partial

import torch


def blockmask_to_gpu(bm, device):
    """Move a BlockMask to device, handling closure tensors and _dynamo_dynamic_indices."""
    src_bm = bm
    bm = bm.to(device)
    # BlockMask.to() moves block indices but not tensors captured
    # in the mask_mod closure. Move those too so flex_attention
    # doesn't hit CPU tensors during inductor lowering.
    fn = bm.mask_mod
    if isinstance(fn, partial):
        bm.mask_mod = partial(fn.func,
            *tuple(v.to(device) if isinstance(v, torch.Tensor) else v for v in fn.args),
            **{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in fn.keywords.items()})
    elif hasattr(fn, '__closure__') and fn.__closure__:
        for cell in fn.__closure__:
            try:
                v = cell.cell_contents
                if isinstance(v, torch.Tensor):
                    cell.cell_contents = v.to(device)
            except ValueError:
                pass  # empty cell
    # .to() creates new tensors that lose _dynamo_dynamic_indices.
    # Re-apply from the originals.
    for attr in ('kv_indices', 'full_kv_indices', 'q_indices', 'full_q_indices'):
        src = getattr(src_bm, attr)
        if hasattr(src, '_dynamo_dynamic_indices'):
            dst = getattr(bm, attr)
            for dim in src._dynamo_dynamic_indices:
                torch._dynamo.mark_dynamic(dst, dim)
    return bm
