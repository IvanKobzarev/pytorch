"""CPU document mask creation for packed multi-document sequences.

Three entry points:
- make_docmask_numpy: pure numpy path (vectorized dense region finding)
- make_docmask_numba: numba JIT path (requires numba)
- make_docmask_cpu: dispatcher — numba if available, else numpy
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


def _mask_fn(b, h, q_idx, kv_idx, attn_regions, document_ids):
    causal = q_idx >= kv_idx
    is_padding = (attn_regions[q_idx] == -1) | (attn_regions[kv_idx] == -1)
    dense_region = (attn_regions[q_idx] > 0) & (attn_regions[kv_idx] > 0)
    same_region = attn_regions[q_idx] == attn_regions[kv_idx]
    same_document = document_ids[q_idx] == document_ids[kv_idx]
    return (causal | (dense_region & same_region)) & same_document & ~is_padding


# ---------------------------------------------------------------------------
# Dense region finding: numpy vectorized
# ---------------------------------------------------------------------------

def _find_dense_regions_vec(ar, s, e):
    """Find contiguous runs of same attn_region > 0 within ar[s:e] using numpy."""
    chunk = ar[s:e]
    if len(chunk) == 0:
        return []
    changes = np.where(np.diff(chunk) != 0)[0] + 1
    starts = np.concatenate([[0], changes])
    ends = np.concatenate([changes, [len(chunk)]])
    vals = chunk[starts]
    mask = vals > 0
    if not np.any(mask):
        return []
    return list(zip((starts[mask] + s).tolist(), (ends[mask] + s).tolist()))


# ---------------------------------------------------------------------------
# Numba JIT kernel (optional, loaded only if numba is available)
# ---------------------------------------------------------------------------

if HAS_NUMBA:
    # cache=False: numba file caching causes race conditions in multi-GPU (multi-process) setups.
    @nb.jit(nopython=True, cache=False)
    def _numba_core(ar, di, BS, NB, seg_starts, seg_ends, max_per_row, max_dense):
        """JIT-compiled block iteration + dense region finding + transpose."""
        kv_num = np.zeros(NB, dtype=np.int32)
        kv_idx = np.zeros((NB, max_per_row), dtype=np.int32)
        fkv_num = np.zeros(NB, dtype=np.int32)
        fkv_idx = np.zeros((NB, max_per_row), dtype=np.int32)
        diag_done = np.zeros(NB, dtype=nb.boolean)

        for si in range(len(seg_starts)):
            s = seg_starts[si]
            e = seg_ends[si]
            if di[s] == -1:
                continue

            fb = s // BS
            lb = (e - 1) // BS

            # --- inline dense region finding ---
            MAX_DENSE = max_dense
            dense_s = np.empty(MAX_DENSE, dtype=np.int64)
            dense_e = np.empty(MAX_DENSE, dtype=np.int64)
            nd = 0
            j = s
            while j < e:
                if ar[j] > 0:
                    v = ar[j]
                    k = j + 1
                    while k < e and ar[k] == v:
                        k += 1
                    assert nd < MAX_DENSE, "too many dense regions in one segment"
                    dense_s[nd] = j
                    dense_e[nd] = k
                    nd += 1
                    j = k
                else:
                    j += 1

            for qb in range(fb, lb + 1):
                qp = (s <= qb * BS) and ((qb + 1) * BS <= e)

                # below diagonal
                for kvb in range(fb, qb):
                    kvp = (s <= kvb * BS) and ((kvb + 1) * BS <= e)
                    if qp and kvp:
                        i = fkv_num[qb]; fkv_idx[qb, i] = kvb; fkv_num[qb] = i + 1
                    else:
                        i = kv_num[qb]; kv_idx[qb, i] = kvb; kv_num[qb] = i + 1

                # diagonal (only once per block)
                if not diag_done[qb]:
                    diag_done[qb] = True
                    diag_full = False
                    if qp:
                        for d in range(nd):
                            if dense_s[d] <= qb * BS and (qb + 1) * BS <= dense_e[d]:
                                diag_full = True
                                break
                    if diag_full:
                        i = fkv_num[qb]; fkv_idx[qb, i] = qb; fkv_num[qb] = i + 1
                    else:
                        i = kv_num[qb]; kv_idx[qb, i] = qb; kv_num[qb] = i + 1

                # above diagonal
                for kvb in range(qb + 1, lb + 1):
                    for d in range(nd):
                        ds = dense_s[d]
                        de = dense_e[d]
                        q_ov = (qb * BS < de) and (ds < (qb + 1) * BS)
                        kv_ov = (kvb * BS < de) and (ds < (kvb + 1) * BS)
                        if q_ov and kv_ov:
                            q_fi = (ds <= qb * BS) and ((qb + 1) * BS <= de)
                            kv_fi = (ds <= kvb * BS) and ((kvb + 1) * BS <= de)
                            if q_fi and kv_fi:
                                i = fkv_num[qb]; fkv_idx[qb, i] = kvb; fkv_num[qb] = i + 1
                            else:
                                i = kv_num[qb]; kv_idx[qb, i] = kvb; kv_num[qb] = i + 1
                            break

        # --- transpose: kv -> q ---
        q_num = np.zeros(NB, dtype=np.int32)
        q_idx = np.zeros((NB, max_per_row), dtype=np.int32)
        fq_num = np.zeros(NB, dtype=np.int32)
        fq_idx = np.zeros((NB, max_per_row), dtype=np.int32)
        for qb in range(NB):
            for j in range(kv_num[qb]):
                kvb = kv_idx[qb, j]
                i = q_num[kvb]; q_idx[kvb, i] = qb; q_num[kvb] = i + 1
            for j in range(fkv_num[qb]):
                kvb = fkv_idx[qb, j]
                i = fq_num[kvb]; fq_idx[kvb, i] = qb; fq_num[kvb] = i + 1

        return kv_num, kv_idx, fkv_num, fkv_idx, q_num, q_idx, fq_num, fq_idx


# ---------------------------------------------------------------------------
# Input conversion shared by make_docmask_numpy and make_docmask_numba
# ---------------------------------------------------------------------------

def _prepare_inputs(ntoks, attn_regions, document_ids, BLOCK_SIZE):
    ar = attn_regions.cpu().numpy() if isinstance(attn_regions, torch.Tensor) else np.asarray(attn_regions)
    di = document_ids.cpu().numpy() if isinstance(document_ids, torch.Tensor) else np.asarray(document_ids)
    BS = BLOCK_SIZE
    NB = (ntoks + BS - 1) // BS
    seq_len = ntoks
    if ntoks % BS:
        pad = NB * BS - ntoks
        ar = np.pad(ar, (0, pad), constant_values=-1)
        di = np.pad(di, (0, pad), constant_values=-1)
        ntoks = NB * BS
    return ar, di, BS, NB, ntoks, seq_len


def _compute_segments_and_mpr(di, ntoks, BS, NB, max_per_row):
    changes = np.where(np.diff(di) != 0)[0] + 1
    seg_starts = np.concatenate([[0], changes])
    seg_ends = np.concatenate([changes, [ntoks]])
    seg_block_spans = (seg_ends - 1) // BS - seg_starts // BS + 1
    per_block = np.zeros(NB, dtype=np.int64)
    for si in range(len(seg_starts)):
        fb = int(seg_starts[si]) // BS
        lb = (int(seg_ends[si]) - 1) // BS
        per_block[fb:lb+1] += seg_block_spans[si]
    needed_mpr = int(per_block.max())
    if isinstance(max_per_row, int):
        assert needed_mpr <= max_per_row, f"max_per_row={max_per_row} too small, need {needed_mpr}"
        mpr = max_per_row
    else:
        mpr = needed_mpr
    return seg_starts, seg_ends, mpr


def _apply_dynamic(mask, max_per_row):
    if max_per_row == "dynamic":
        d = mask.kv_indices.ndim - 1
        torch._dynamo.mark_dynamic(mask.kv_indices, d)
        torch._dynamo.mark_dynamic(mask.full_kv_indices, d)
        torch._dynamo.mark_dynamic(mask.q_indices, d)
        torch._dynamo.mark_dynamic(mask.full_q_indices, d)


# ---------------------------------------------------------------------------
# make_docmask_numba — numba JIT path
# ---------------------------------------------------------------------------

def make_docmask_numba(ntoks, attn_regions, document_ids, BLOCK_SIZE=128, max_per_row=None, max_dense=32):
    assert HAS_NUMBA, "numba is required for make_docmask_numba"
    ar, di, BS, NB, ntoks, seq_len = _prepare_inputs(ntoks, attn_regions, document_ids, BLOCK_SIZE)
    seg_starts, seg_ends, mpr = _compute_segments_and_mpr(di, ntoks, BS, NB, max_per_row)

    kv_num, kv_idx, fkv_num, fkv_idx, q_num, q_idx, fq_num, fq_idx = \
        _numba_core(ar.astype(np.int64), di.astype(np.int64), BS, NB,
                    seg_starts.astype(np.int64), seg_ends.astype(np.int64), mpr, max_dense)

    mask_mod = partial(_mask_fn, attn_regions=torch.from_numpy(ar), document_ids=torch.from_numpy(di))
    to_t = torch.from_numpy
    mask = BlockMask(
        kv_num_blocks=to_t(kv_num)[None, None, :],
        kv_indices=to_t(kv_idx)[None, None, :, :],
        full_kv_num_blocks=to_t(fkv_num)[None, None, :],
        full_kv_indices=to_t(fkv_idx)[None, None, :, :],
        q_num_blocks=to_t(q_num)[None, None, :],
        q_indices=to_t(q_idx)[None, None, :, :],
        full_q_num_blocks=to_t(fq_num)[None, None, :],
        full_q_indices=to_t(fq_idx)[None, None, :, :],
        BLOCK_SIZE=(BS, BS),
        mask_mod=mask_mod,
        seq_lengths=(seq_len, seq_len),
    )
    _apply_dynamic(mask, max_per_row)
    return mask


# ---------------------------------------------------------------------------
# make_docmask_numpy — pure numpy path
# ---------------------------------------------------------------------------

def make_docmask_numpy(ntoks, attn_regions, document_ids, BLOCK_SIZE=128, max_per_row=None):
    ar, di, BS, NB, ntoks, seq_len = _prepare_inputs(ntoks, attn_regions, document_ids, BLOCK_SIZE)

    partial_sets = [set() for _ in range(NB)]
    full_sets = [set() for _ in range(NB)]

    if ntoks == 0:
        return _build_blockmask(partial_sets, full_sets, NB, BS, seq_len, ar, di, 1)

    seg_starts, seg_ends, mpr = _compute_segments_and_mpr(di, ntoks, BS, NB, max_per_row)

    for si in range(len(seg_starts)):
        s, e = int(seg_starts[si]), int(seg_ends[si])
        if di[s] == -1:
            continue

        fb = s // BS
        lb = (e - 1) // BS
        dense = _find_dense_regions_vec(ar, s, e)

        for qb in range(fb, lb + 1):
            qp = (s <= qb * BS) and ((qb + 1) * BS <= e)

            # --- Below diagonal ---
            for kvb in range(fb, qb):
                kvp = (s <= kvb * BS) and ((kvb + 1) * BS <= e)
                if qp and kvp:
                    full_sets[qb].add(kvb)
                else:
                    partial_sets[qb].add(kvb)

            # --- Diagonal ---
            diag_full = False
            if qp:
                for ds, de in dense:
                    if ds <= qb * BS and (qb + 1) * BS <= de:
                        diag_full = True
                        break
            if diag_full:
                full_sets[qb].add(qb)
            else:
                partial_sets[qb].add(qb)

            # --- Above diagonal ---
            for kvb in range(qb + 1, lb + 1):
                for ds, de in dense:
                    q_ov = (qb * BS < de) and (ds < (qb + 1) * BS)
                    kv_ov = (kvb * BS < de) and (ds < (kvb + 1) * BS)
                    if q_ov and kv_ov:
                        q_fi = (ds <= qb * BS) and ((qb + 1) * BS <= de)
                        kv_fi = (ds <= kvb * BS) and ((kvb + 1) * BS <= de)
                        if q_fi and kv_fi:
                            full_sets[qb].add(kvb)
                        else:
                            partial_sets[qb].add(kvb)
                        break

    mask = _build_blockmask(partial_sets, full_sets, NB, BS, seq_len, ar, di, mpr)
    _apply_dynamic(mask, max_per_row)
    return mask


# ---------------------------------------------------------------------------
# make_docmask_cpu — dispatcher
# ---------------------------------------------------------------------------

def make_docmask_cpu(ntoks, attn_regions, document_ids, BLOCK_SIZE=128, max_per_row=None, max_dense=32):
    if ntoks == 0:
        ar, di, BS, NB, ntoks, seq_len = _prepare_inputs(ntoks, attn_regions, document_ids, BLOCK_SIZE)
        return _build_blockmask([set() for _ in range(NB)], [set() for _ in range(NB)], NB, BS, seq_len, ar, di, 1)
    if HAS_NUMBA:
        return make_docmask_numba(ntoks, attn_regions, document_ids, BLOCK_SIZE, max_per_row, max_dense)
    return make_docmask_numpy(ntoks, attn_regions, document_ids, BLOCK_SIZE, max_per_row)


# ---------------------------------------------------------------------------
# _build_blockmask — used by make_docmask_numpy
# ---------------------------------------------------------------------------

def _build_blockmask(partial_sets, full_sets, NB, BS, seq_len, ar, di, max_per_row):
    """Convert per-row sets into a BlockMask.
    Uses compact (NB, max_per_row) indices to avoid O(NB²) memory."""
    kv_num = np.zeros(NB, dtype=np.int32)
    kv_idx = np.zeros((NB, max_per_row), dtype=np.int32)
    full_kv_num = np.zeros(NB, dtype=np.int32)
    full_kv_idx = np.zeros((NB, max_per_row), dtype=np.int32)

    for i in range(NB):
        ps = sorted(partial_sets[i])
        kv_num[i] = len(ps)
        kv_idx[i, :len(ps)] = ps

        fs = sorted(full_sets[i])
        full_kv_num[i] = len(fs)
        full_kv_idx[i, :len(fs)] = fs

    mask_mod = partial(_mask_fn, attn_regions=torch.from_numpy(ar), document_ids=torch.from_numpy(di))

    # Build the q-blocks (transpose) ourselves efficiently,
    # instead of letting from_kv_blocks do it with dense O(NB²) intermediates.
    q_partial = [[] for _ in range(NB)]
    q_full = [[] for _ in range(NB)]
    for qb in range(NB):
        for j in range(kv_num[qb]):
            q_partial[kv_idx[qb, j]].append(qb)
        for j in range(full_kv_num[qb]):
            q_full[full_kv_idx[qb, j]].append(qb)

    q_num = np.zeros(NB, dtype=np.int32)
    q_idx = np.zeros((NB, max_per_row), dtype=np.int32)
    full_q_num = np.zeros(NB, dtype=np.int32)
    full_q_idx = np.zeros((NB, max_per_row), dtype=np.int32)
    for i in range(NB):
        q_num[i] = len(q_partial[i])
        q_idx[i, :q_num[i]] = q_partial[i]
        full_q_num[i] = len(q_full[i])
        full_q_idx[i, :full_q_num[i]] = q_full[i]

    to_t = torch.from_numpy

    return BlockMask(
        kv_num_blocks=to_t(kv_num)[None, None, :],
        kv_indices=to_t(kv_idx)[None, None, :, :],
        full_kv_num_blocks=to_t(full_kv_num)[None, None, :],
        full_kv_indices=to_t(full_kv_idx)[None, None, :, :],
        q_num_blocks=to_t(q_num)[None, None, :],
        q_indices=to_t(q_idx)[None, None, :, :],
        full_q_num_blocks=to_t(full_q_num)[None, None, :],
        full_q_indices=to_t(full_q_idx)[None, None, :, :],
        BLOCK_SIZE=(BS, BS),
        mask_mod=mask_mod,
        seq_lengths=(seq_len, seq_len),
    )
