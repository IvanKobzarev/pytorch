"""Test to find CPU vs GPU batchmask discrepancies for decode-like patterns.

The existing tests only compare block-level inclusion (which blocks are active),
but don't check partial vs full classification or element-level mask correctness.
This test materializes BlockMasks to element-level dense masks and compares
against the ground truth mask function.
"""

import numpy as np
import torch


def ground_truth_mask(B, ntoks, ar):
    """Compute element-level mask from the mask function definition."""
    ar_t = torch.as_tensor(ar)
    q = torch.arange(ntoks).unsqueeze(1).expand(ntoks, ntoks)
    kv = torch.arange(ntoks).unsqueeze(0).expand(ntoks, ntoks)
    masks = []
    for b in range(B):
        causal = q >= kv
        ar_q = ar_t[b, q]
        ar_kv = ar_t[b, kv]
        dense = (ar_q > 0) & (ar_kv > 0)
        same = ar_q == ar_kv
        masks.append(causal | (dense & same))
    return torch.stack(masks)


def blockmask_to_element(bm, B, ntoks, BS, ar):
    """Materialize a BlockMask to element-level dense mask (B, ntoks, ntoks).

    Full blocks → all True.
    Partial blocks → evaluate mask function element-wise.
    Missing blocks → all False.
    """
    NB = (ntoks + BS - 1) // BS
    mask = torch.zeros(B, ntoks, ntoks, dtype=torch.bool)
    ar_t = torch.as_tensor(ar)

    for b in range(B):
        # Full blocks: all elements attend
        fkv_n = bm.full_kv_num_blocks[b, 0].cpu()
        fkv_i = bm.full_kv_indices[b, 0].cpu()
        for qb in range(NB):
            for j in range(fkv_n[qb]):
                kvb = fkv_i[qb, j].item()
                qs, qe = qb * BS, min((qb + 1) * BS, ntoks)
                kvs, kve = kvb * BS, min((kvb + 1) * BS, ntoks)
                mask[b, qs:qe, kvs:kve] = True

        # Partial blocks: evaluate mask function
        kv_n = bm.kv_num_blocks[b, 0].cpu()
        kv_i = bm.kv_indices[b, 0].cpu()
        for qb in range(NB):
            for j in range(kv_n[qb]):
                kvb = kv_i[qb, j].item()
                qs, qe = qb * BS, min((qb + 1) * BS, ntoks)
                kvs, kve = kvb * BS, min((kvb + 1) * BS, ntoks)
                q_idx = torch.arange(qs, qe).unsqueeze(1).expand(qe - qs, kve - kvs)
                kv_idx = torch.arange(kvs, kve).unsqueeze(0).expand(qe - qs, kve - kvs)
                causal = q_idx >= kv_idx
                ar_q = ar_t[b, q_idx]
                ar_kv = ar_t[b, kv_idx]
                dense = (ar_q > 0) & (ar_kv > 0)
                same = ar_q == ar_kv
                mask[b, qs:qe, kvs:kve] = causal | (dense & same)

    return mask


def compare_one(fn_name, make_fn, ar, ntoks, BS, gt):
    B = ar.shape[0]
    bm = make_fn(ntoks, ar, BLOCK_SIZE=BS)
    materialized = blockmask_to_element(bm, B, ntoks, BS, ar)

    if not torch.equal(materialized, gt):
        diff = (materialized != gt)
        for b in range(B):
            if diff[b].any():
                n_diff = diff[b].sum().item()
                positions = torch.nonzero(diff[b])
                q, kv = positions[0][0].item(), positions[0][1].item()
                qb, kvb = q // BS, kv // BS
                print(f"  FAIL [{fn_name}] b={b}: {n_diff} elements differ")
                print(f"    First diff at q={q}, kv={kv} (block qb={qb}, kvb={kvb})")
                print(f"    Expected: {gt[b, q, kv].item()}, Got: {materialized[b, q, kv].item()}")
                print(f"    ar[b,q]={ar[b,q]}, ar[b,kv]={ar[b,kv]}")

                # Check block classification
                is_full = any(bm.full_kv_indices[b, 0, qb, jj] == kvb
                              for jj in range(bm.full_kv_num_blocks[b, 0, qb]))
                is_partial = any(bm.kv_indices[b, 0, qb, jj] == kvb
                                 for jj in range(bm.kv_num_blocks[b, 0, qb]))
                print(f"    Block (qb={qb},kvb={kvb}): full={is_full}, partial={is_partial}, missing={not is_full and not is_partial}")
        return False

    print(f"  OK [{fn_name}]")
    return True


def compare_masks(name, ar, ntoks, BS):
    from flexmaskli.batchmask_cpu import make_batchmask_cpu, make_batchmask_numpy

    print(f"Testing: {name}")
    gt = ground_truth_mask(ar.shape[0], ntoks, ar)
    ok = True
    ok &= compare_one("numpy", make_batchmask_numpy, ar, ntoks, BS, gt)
    ok &= compare_one("cpu/numba", make_batchmask_cpu, ar, ntoks, BS, gt)
    return ok


def test_decode_like_patterns():
    all_pass = True
    BS = 128

    # 1: contiguous dense prefix — should pass
    ntoks = 1024
    ar = np.zeros((2, ntoks), dtype=np.int64)
    ar[0, :200] = 1
    ar[1, :500] = 1
    all_pass &= compare_masks("contiguous_dense_prefix", ar, ntoks, BS)

    # 2: non-contiguous SAME value — THE BUG
    # Two separate image segments with the same attn_regions value.
    # The mask function allows bidirectional attention between them,
    # but the CPU code only finds overlap within single contiguous regions.
    ar = np.zeros((2, ntoks), dtype=np.int64)
    ar[0, 50:200] = 1   # segment A, value 1
    ar[0, 400:550] = 1  # segment B, SAME value 1
    ar[1, :100] = 1
    ar[1, 300:500] = 1  # same value 1 again
    all_pass &= compare_masks("non_contiguous_same_value", ar, ntoks, BS)

    # 3: non-contiguous same value with small block size
    BS_small = 64
    ntoks = 512
    ar = np.zeros((2, ntoks), dtype=np.int64)
    ar[0, 30:100] = 1
    ar[0, 200:300] = 1  # same value, non-contiguous
    ar[1, 50:250] = 1   # contiguous (control)
    all_pass &= compare_masks("small_bs_noncontig", ar, ntoks, BS_small)

    # 4: non-contiguous same value, block-aligned
    ntoks = 1024
    ar = np.zeros((1, ntoks), dtype=np.int64)
    ar[0, 0:256] = 1     # blocks 0,1 fully covered
    ar[0, 512:768] = 1   # blocks 4,5 fully covered, SAME value
    all_pass &= compare_masks("block_aligned_noncontig", ar, ntoks, BS)

    # 5: multiple values, some contiguous some not
    ar = np.zeros((1, ntoks), dtype=np.int64)
    ar[0, 0:128] = 1
    ar[0, 256:384] = 2   # different value — no cross-attention expected
    ar[0, 512:640] = 1   # same as first — cross-attention expected
    all_pass &= compare_masks("mixed_values_noncontig", ar, ntoks, BS)

    print()
    if all_pass:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED — non-contiguous same-value bug confirmed")
    return all_pass


if __name__ == "__main__":
    test_decode_like_patterns()
