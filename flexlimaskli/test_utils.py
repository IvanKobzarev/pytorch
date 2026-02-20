import torch


def blockmask_to_dense(bm):
    """Convert BlockMask to dense bool tensor (CPU), works with compact indices."""
    NB = bm.kv_num_blocks.shape[-1]
    dense = torch.zeros(1, 1, NB, NB, dtype=torch.bool)
    for name in ['kv', 'full_kv']:
        nums = getattr(bm, f'{name}_num_blocks')
        idxs = getattr(bm, f'{name}_indices')
        if nums is None:
            continue
        n, idx = nums[0, 0].cpu(), idxs[0, 0].cpu()
        for qb in range(NB):
            for j in range(n[qb]):
                dense[0, 0, qb, idx[qb, j]] = True
    return dense
