"""Move a BlockMask (with closure tensors and dynamic indices) to a device."""

from functools import partial

import torch


def blockmask_to_gpu(bm, device):
    """Move a BlockMask to device, handling closure tensors."""
    # Preserve mark_dynamic annotations: bm.to() creates new tensors that
    # lose _dynamo_dynamic_indices. Re-apply after moving.
    dynamic_marks = {}
    for attr in ("kv_indices", "full_kv_indices", "q_indices", "full_q_indices"):
        t = getattr(bm, attr)
        if hasattr(t, '_dynamo_dynamic_indices'):
            dynamic_marks[attr] = t._dynamo_dynamic_indices
    bm = bm.to(device)
    for attr, marks in dynamic_marks.items():
        getattr(bm, attr)._dynamo_dynamic_indices = marks
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
    return bm
