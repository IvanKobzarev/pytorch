"""Move a BlockMask (with closure tensors and unbacked metadata) to a device."""

from functools import partial

import torch


def blockmask_to_gpu(bm, device):
    """Move a BlockMask to device, handling closure tensors."""
    # Preserve mark_unbacked annotations: bm.to() creates new tensors that
    # lose Dynamo attrs. Re-apply after moving.
    unbacked_marks = {}
    for attr in ("kv_indices", "full_kv_indices", "q_indices", "full_q_indices"):
        t = getattr(bm, attr)
        saved = {}
        for key in (
            "_dynamo_dynamic_indices",
            "_dynamo_unbacked_indices",
            "_dynamo_strict_unbacked_indices",
            "_dynamo_unbacked_bounds",
            "_dynamo_shape_ids",
            "_dynamo_hint_overrides",
            "_specialize_on",
        ):
            if hasattr(t, key):
                saved[key] = getattr(t, key)
        if saved:
            unbacked_marks[attr] = saved
    bm = bm.to(device)
    for attr, saved in unbacked_marks.items():
        dst = getattr(bm, attr)
        for key, value in saved.items():
            setattr(dst, key, value)
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
