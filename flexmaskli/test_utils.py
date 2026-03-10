import torch


def _mask_mod_device(bm):
    """Detect device from mask_mod closure tensors."""
    if hasattr(bm.mask_mod, 'keywords'):
        for v in bm.mask_mod.keywords.values():
            if isinstance(v, torch.Tensor):
                return v.device
    return "cpu"


def blockmask_to_dense(bm, b=None):
    """Convert BlockMask to dense bool tensor (CPU), mirroring flex_attention behavior.

    Full blocks: all True (mask_mod not called, matching flex_attention).
    Partial blocks: mask_mod evaluated per token pair.
    Absent blocks: all False.

    If b is given, only that batch element is materialized (returns (1,H,Q,KV)).
    """
    B, H, Q_LEN, KV_LEN = bm.shape
    BS_Q, BS_KV = bm.BLOCK_SIZE
    NB_Q = (Q_LEN + BS_Q - 1) // BS_Q
    device = _mask_mod_device(bm)

    b_range = range(b, b + 1) if b is not None else range(B)
    out_B = 1 if b is not None else B
    dense = torch.zeros(out_B, H, NB_Q * BS_Q, (Q_LEN + BS_KV - 1) // BS_KV * BS_KV,
                        dtype=torch.bool, device=device)

    for bi, b_actual in enumerate(b_range):
        for h in range(H):
            # Full blocks: all True
            if bm.full_kv_num_blocks is not None:
                fkv_n = bm.full_kv_num_blocks[b_actual, h]
                fkv_idx = bm.full_kv_indices[b_actual, h]
                for qb in range(NB_Q):
                    for j in range(min(fkv_n[qb].item(), fkv_idx.shape[-1])):
                        kvb = fkv_idx[qb, j].item()
                        dense[bi, h, qb*BS_Q:(qb+1)*BS_Q, kvb*BS_KV:(kvb+1)*BS_KV] = True

            # Partial blocks: evaluate mask_mod
            kv_n = bm.kv_num_blocks[b_actual, h]
            kv_idx = bm.kv_indices[b_actual, h]
            for qb in range(NB_Q):
                for j in range(min(kv_n[qb].item(), kv_idx.shape[-1])):
                    kvb = kv_idx[qb, j].item()
                    q_off = qb * BS_Q
                    kv_off = kvb * BS_KV
                    q_len = min(BS_Q, Q_LEN - q_off)
                    kv_len = min(BS_KV, KV_LEN - kv_off)
                    qi = torch.arange(q_len, device=device, dtype=torch.int32)[:, None] + q_off
                    ki = torch.arange(kv_len, device=device, dtype=torch.int32)[None, :] + kv_off
                    dense[bi, h, q_off:q_off+q_len, kv_off:kv_off+kv_len] = bm.mask_mod(b_actual, h, qi, ki)

    return dense[:, :, :Q_LEN, :KV_LEN].cpu()


def compare_block_masks(mask1, mask2, structural=True):
    """Compare two BlockMasks: structural (block indices) and/or dense (token-level).

    mask1 is treated as the reference (e.g. GPU ground truth).
    When structural=False, only checks that mask2 is a superset of mask1.
    """
    def cpu(t): return t.cpu() if t.is_cuda else t

    if structural:
        assert torch.equal(cpu(mask1.kv_num_blocks), cpu(mask2.kv_num_blocks)), \
            f"kv_num_blocks mismatch: {mask1.kv_num_blocks} vs {mask2.kv_num_blocks}"
        assert mask1.seq_lengths == mask2.seq_lengths, \
            f"seq_lengths mismatch: {mask1.seq_lengths} vs {mask2.seq_lengths}"
        assert mask1.BLOCK_SIZE == mask2.BLOCK_SIZE, \
            f"BLOCK_SIZE mismatch: {mask1.BLOCK_SIZE} vs {mask2.BLOCK_SIZE}"

        B, H, num_q_blocks, _ = mask1.kv_indices.shape
        for b in range(B):
            for h in range(H):
                for q_block in range(num_q_blocks):
                    num_valid = mask1.kv_num_blocks[b, h, q_block].item()
                    valid_indices1 = cpu(mask1.kv_indices[b, h, q_block, :num_valid])
                    valid_indices2 = cpu(mask2.kv_indices[b, h, q_block, :num_valid])
                    assert torch.equal(valid_indices1, valid_indices2), \
                        f"kv_indices mismatch at [{b},{h},{q_block}]: {valid_indices1} vs {valid_indices2}"

    dense1 = blockmask_to_dense(mask1)
    dense2 = blockmask_to_dense(mask2)
    if structural:
        assert torch.equal(dense1, dense2), \
            f"Dense masks differ! Shape: {dense1.shape}, diff locations: {(dense1 != dense2).sum().item()}"
    else:
        missing = dense1 & ~dense2
        assert not missing.any(), \
            f"CPU mask is missing blocks present in GPU ref! Missing: {missing.sum().item()}"
