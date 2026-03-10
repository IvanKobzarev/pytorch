"""CPU batched mask creation for single-document-per-batch-element sequences.

Three entry points:
- make_batchmask_numpy: pure numpy path
- make_batchmask_numba: numba JIT path (requires numba)
- make_batchmask_cpu: dispatcher — numba if available, else numpy
"""

from functools import partial

import numpy as np
import torch
from torch.nn.attention.flex_attention import BlockMask

try:
    import numba as nb
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


def _mask_fn_batched(b, h, q_idx, kv_idx, attn_regions):
    causal = q_idx >= kv_idx
    is_padding = (attn_regions[b, q_idx] == -1) | (attn_regions[b, kv_idx] == -1)
    dense_region = (attn_regions[b, q_idx] > 0) & (attn_regions[b, kv_idx] > 0)
    same_region = attn_regions[b, q_idx] == attn_regions[b, kv_idx]
    return (causal | (dense_region & same_region)) & ~is_padding


# ---------------------------------------------------------------------------
# Shared input preparation
# ---------------------------------------------------------------------------

def _prepare_inputs(ntoks, attn_regions_batch, BLOCK_SIZE):
    ar = attn_regions_batch.cpu().numpy() if isinstance(attn_regions_batch, torch.Tensor) else np.asarray(attn_regions_batch)
    BS = BLOCK_SIZE
    NB = (ntoks + BS - 1) // BS
    seq_len = ntoks
    # Pad to block alignment; 0 means causal-only (no dense).
    if ntoks % BS:
        pad = NB * BS - ntoks
        ar = np.pad(ar, ((0, 0), (0, pad)), constant_values=0)
    return ar, BS, NB, seq_len


def _build_mask(kv_num, kv_idx, fkv_num, fkv_idx, q_num, q_idx, fq_num, fq_idx, BS, seq_len, ar):
    to_t = torch.from_numpy
    return BlockMask(
        kv_num_blocks=to_t(kv_num)[:, None, :],
        kv_indices=to_t(kv_idx)[:, None, :, :],
        full_kv_num_blocks=to_t(fkv_num)[:, None, :],
        full_kv_indices=to_t(fkv_idx)[:, None, :, :],
        q_num_blocks=to_t(q_num)[:, None, :],
        q_indices=to_t(q_idx)[:, None, :, :],
        full_q_num_blocks=to_t(fq_num)[:, None, :],
        full_q_indices=to_t(fq_idx)[:, None, :, :],
        BLOCK_SIZE=(BS, BS),
        mask_mod=partial(_mask_fn_batched, attn_regions=to_t(ar)),
        seq_lengths=(seq_len, seq_len),
    )


# ---------------------------------------------------------------------------
# Numba JIT kernel (optional, loaded only if numba is available)
# ---------------------------------------------------------------------------

if HAS_NUMBA:
    # cache=False: numba file caching causes race conditions in multi-GPU (multi-process) setups.
    @nb.jit(nopython=True, cache=False)
    def _numba_core_batched(ar, BS, NB, max_dense):
        """Batched JIT kernel for single-document-per-batch-element."""
        B = ar.shape[0]
        ntoks = NB * BS
        kv_num = np.zeros((B, NB), dtype=np.int32)
        kv_idx = np.zeros((B, NB, NB), dtype=np.int32)
        fkv_num = np.zeros((B, NB), dtype=np.int32)
        fkv_idx = np.zeros((B, NB, NB), dtype=np.int32)

        for b in range(B):
            # --- inline dense region finding ---
            MAX_DENSE = max_dense
            dense_s = np.empty(MAX_DENSE, dtype=np.int64)
            dense_e = np.empty(MAX_DENSE, dtype=np.int64)
            dense_v = np.empty(MAX_DENSE, dtype=np.int64)
            nd = 0
            j = 0
            while j < ntoks:
                if ar[b, j] > 0:
                    v = ar[b, j]
                    k = j + 1
                    while k < ntoks and ar[b, k] == v:
                        k += 1
                    assert nd < MAX_DENSE, "too many dense regions in one batch element"
                    dense_s[nd] = j
                    dense_e[nd] = k
                    dense_v[nd] = v
                    nd += 1
                    j = k
                else:
                    j += 1

            # --- per-block negative flags for -1 handling ---
            any_neg = np.zeros(NB, dtype=nb.boolean)
            all_neg = np.zeros(NB, dtype=nb.boolean)
            for blk in range(NB):
                neg = 0
                for t in range(BS):
                    if ar[b, blk * BS + t] == -1:
                        neg += 1
                any_neg[blk] = neg > 0
                all_neg[blk] = neg == BS

            for qb in range(NB):
                if all_neg[qb]:
                    continue

                # below diagonal: full unless -1 involved
                for kvb in range(qb):
                    if all_neg[kvb]:
                        continue
                    if any_neg[qb] or any_neg[kvb]:
                        i = kv_num[b, qb]; kv_idx[b, qb, i] = kvb; kv_num[b, qb] = i + 1
                    else:
                        i = fkv_num[b, qb]; fkv_idx[b, qb, i] = kvb; fkv_num[b, qb] = i + 1

                # diagonal: full only if a dense region fully covers this block and no -1
                diag_full = False
                for d in range(nd):
                    if dense_s[d] <= qb * BS and (qb + 1) * BS <= dense_e[d]:
                        diag_full = True
                        break
                if diag_full and not any_neg[qb]:
                    i = fkv_num[b, qb]; fkv_idx[b, qb, i] = qb; fkv_num[b, qb] = i + 1
                else:
                    i = kv_num[b, qb]; kv_idx[b, qb, i] = qb; kv_num[b, qb] = i + 1

                # above diagonal: only through dense regions
                for kvb in range(qb + 1, NB):
                    found_partial = False
                    found_full = False
                    for d1 in range(nd):
                        ds1 = dense_s[d1]
                        de1 = dense_e[d1]
                        q_ov = (qb * BS < de1) and (ds1 < (qb + 1) * BS)
                        if not q_ov:
                            continue
                        q_fi = (ds1 <= qb * BS) and ((qb + 1) * BS <= de1)
                        for d2 in range(nd):
                            if dense_v[d2] != dense_v[d1]:
                                continue
                            ds2 = dense_s[d2]
                            de2 = dense_e[d2]
                            kv_ov = (kvb * BS < de2) and (ds2 < (kvb + 1) * BS)
                            if kv_ov:
                                kv_fi = (ds2 <= kvb * BS) and ((kvb + 1) * BS <= de2)
                                if q_fi and kv_fi:
                                    found_full = True
                                else:
                                    found_partial = True
                                break
                        if found_full:
                            break
                    if found_full:
                        i = fkv_num[b, qb]; fkv_idx[b, qb, i] = kvb; fkv_num[b, qb] = i + 1
                    elif found_partial:
                        i = kv_num[b, qb]; kv_idx[b, qb, i] = kvb; kv_num[b, qb] = i + 1

        # --- transpose: kv -> q ---
        q_num = np.zeros((B, NB), dtype=np.int32)
        q_idx = np.zeros((B, NB, NB), dtype=np.int32)
        fq_num = np.zeros((B, NB), dtype=np.int32)
        fq_idx = np.zeros((B, NB, NB), dtype=np.int32)
        for b in range(B):
            for qb in range(NB):
                for j in range(kv_num[b, qb]):
                    kvb = kv_idx[b, qb, j]
                    i = q_num[b, kvb]; q_idx[b, kvb, i] = qb; q_num[b, kvb] = i + 1
                for j in range(fkv_num[b, qb]):
                    kvb = fkv_idx[b, qb, j]
                    i = fq_num[b, kvb]; fq_idx[b, kvb, i] = qb; fq_num[b, kvb] = i + 1

        return kv_num, kv_idx, fkv_num, fkv_idx, q_num, q_idx, fq_num, fq_idx


# ---------------------------------------------------------------------------
# make_batchmask_numba — numba JIT path
# ---------------------------------------------------------------------------

def make_batchmask_numba(ntoks, attn_regions_batch, BLOCK_SIZE=128, max_dense=32):
    """Batched flexmask using numba JIT. Requires numba."""
    assert HAS_NUMBA, "numba is required for make_batchmask_numba"
    ar, BS, NB, seq_len = _prepare_inputs(ntoks, attn_regions_batch, BLOCK_SIZE)
    kv_num, kv_idx, fkv_num, fkv_idx, q_num, q_idx, fq_num, fq_idx = \
        _numba_core_batched(ar.astype(np.int64), BS, NB, max_dense)
    return _build_mask(kv_num, kv_idx, fkv_num, fkv_idx, q_num, q_idx, fq_num, fq_idx, BS, seq_len, ar)


# ---------------------------------------------------------------------------
# make_batchmask_numpy — pure numpy path
# ---------------------------------------------------------------------------

def make_batchmask_numpy(ntoks, attn_regions_batch, BLOCK_SIZE=128):
    """Batched flexmask using pure numpy/Python. No numba dependency."""
    ar, BS, NB, seq_len = _prepare_inputs(ntoks, attn_regions_batch, BLOCK_SIZE)
    B = ar.shape[0]
    padded_ntoks = NB * BS

    kv_num = np.zeros((B, NB), dtype=np.int32)
    kv_idx = np.zeros((B, NB, NB), dtype=np.int32)
    fkv_num = np.zeros((B, NB), dtype=np.int32)
    fkv_idx = np.zeros((B, NB, NB), dtype=np.int32)

    for b in range(B):
        # Find dense regions for this batch element
        dense = []
        j = 0
        while j < padded_ntoks:
            if ar[b, j] > 0:
                v = ar[b, j]
                k = j + 1
                while k < padded_ntoks and ar[b, k] == v:
                    k += 1
                dense.append((j, k, v))
                j = k
            else:
                j += 1

        # Per-block negative flags for -1 handling
        ar_blk = ar[b].reshape(NB, BS)
        any_neg = (ar_blk == -1).any(axis=1)
        all_neg = (ar_blk == -1).all(axis=1)

        for qb in range(NB):
            if all_neg[qb]:
                continue

            # below diagonal: full unless -1 involved
            for kvb in range(qb):
                if all_neg[kvb]:
                    continue
                if any_neg[qb] or any_neg[kvb]:
                    i = kv_num[b, qb]; kv_idx[b, qb, i] = kvb; kv_num[b, qb] = i + 1
                else:
                    i = fkv_num[b, qb]; fkv_idx[b, qb, i] = kvb; fkv_num[b, qb] = i + 1

            # diagonal: full only if a dense region fully covers this block and no -1
            diag_full = any(ds <= qb * BS and (qb + 1) * BS <= de for ds, de, _ in dense)
            if diag_full and not any_neg[qb]:
                i = fkv_num[b, qb]; fkv_idx[b, qb, i] = qb; fkv_num[b, qb] = i + 1
            else:
                i = kv_num[b, qb]; kv_idx[b, qb, i] = qb; kv_num[b, qb] = i + 1

            # above diagonal: only through dense regions
            for kvb in range(qb + 1, NB):
                is_partial = False
                is_full = False
                for ds1, de1, v1 in dense:
                    q_ov = (qb * BS < de1) and (ds1 < (qb + 1) * BS)
                    if not q_ov:
                        continue
                    q_fi = (ds1 <= qb * BS) and ((qb + 1) * BS <= de1)
                    for ds2, de2, v2 in dense:
                        if v2 != v1:
                            continue
                        kv_ov = (kvb * BS < de2) and (ds2 < (kvb + 1) * BS)
                        if kv_ov:
                            kv_fi = (ds2 <= kvb * BS) and ((kvb + 1) * BS <= de2)
                            if q_fi and kv_fi:
                                is_full = True
                            else:
                                is_partial = True
                            break
                    if is_full:
                        break
                if is_full:
                    i = fkv_num[b, qb]; fkv_idx[b, qb, i] = kvb; fkv_num[b, qb] = i + 1
                elif is_partial:
                    i = kv_num[b, qb]; kv_idx[b, qb, i] = kvb; kv_num[b, qb] = i + 1

    # --- transpose: kv -> q ---
    q_num = np.zeros((B, NB), dtype=np.int32)
    q_idx = np.zeros((B, NB, NB), dtype=np.int32)
    fq_num = np.zeros((B, NB), dtype=np.int32)
    fq_idx = np.zeros((B, NB, NB), dtype=np.int32)
    for b in range(B):
        for qb in range(NB):
            for j in range(kv_num[b, qb]):
                kvb = kv_idx[b, qb, j]
                i = q_num[b, kvb]; q_idx[b, kvb, i] = qb; q_num[b, kvb] = i + 1
            for j in range(fkv_num[b, qb]):
                kvb = fkv_idx[b, qb, j]
                i = fq_num[b, kvb]; fq_idx[b, kvb, i] = qb; fq_num[b, kvb] = i + 1

    return _build_mask(kv_num, kv_idx, fkv_num, fkv_idx, q_num, q_idx, fq_num, fq_idx, BS, seq_len, ar)


# ---------------------------------------------------------------------------
# make_batchmask_cpu — dispatcher
# ---------------------------------------------------------------------------

def make_batchmask_cpu(ntoks, attn_regions_batch, BLOCK_SIZE=128, max_dense=32):
    """Batched flexmask for single-document-per-batch-element (e.g. decoding).

    attn_regions_batch: (B, ntoks) array. Returns a BlockMask with batch dim B.
    """
    if HAS_NUMBA:
        return make_batchmask_numba(ntoks, attn_regions_batch, BLOCK_SIZE, max_dense)
    return make_batchmask_numpy(ntoks, attn_regions_batch, BLOCK_SIZE)
