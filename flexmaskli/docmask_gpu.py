from functools import cache, partial

import torch
from torch.nn.attention.flex_attention import BlockMask, create_block_mask


@cache
def maybe_compiled_fn(fn_name, compiled):
    # Changing compile mode to reduce-overhead or max-autotune barely moved the needle.
    # compile = partial(torch.compile, mode="max-autotune") if compiled else lambda fn: fn
    compile = torch.compile if compiled else lambda fn: fn
    return {
        "create_block_mask": compile(partial(create_block_mask, B=None, H=None)),
    }[fn_name]


def mask_fn(b, h, q_idx, kv_idx, attn_regions, document_ids):
    causal = q_idx >= kv_idx
    is_padding = (attn_regions[q_idx] == -1) | (attn_regions[kv_idx] == -1)
    dense_region = (attn_regions[q_idx] > 0) & (attn_regions[kv_idx] > 0)
    same_region = attn_regions[q_idx] == attn_regions[kv_idx]
    same_document = document_ids[q_idx] == document_ids[kv_idx]
    return (causal | (dense_region & same_region)) & same_document & ~is_padding


def make_mask_fn(attn_regions, document_ids):
    return partial(mask_fn, attn_regions=attn_regions, document_ids=document_ids)


def mask_fn_v2(b, h, q_idx, kv_idx, q_positions, kv_positions, q_attn_regions, kv_attn_regions, q_document_ids, kv_document_ids):
    causal = q_positions[q_idx] >= kv_positions[kv_idx]
    is_padding = (q_attn_regions[q_idx] == -1) | (kv_attn_regions[kv_idx] == -1)
    dense_region = (q_attn_regions[q_idx] > 0) & (kv_attn_regions[kv_idx] > 0)
    same_region = q_attn_regions[q_idx] == kv_attn_regions[kv_idx]
    same_document = q_document_ids[q_idx] == kv_document_ids[kv_idx]
    return (causal | (dense_region & same_region)) & same_document & ~is_padding

def make_mask_fn_v2(q_positions, kv_positions, q_attn_regions, kv_attn_regions, q_document_ids, kv_document_ids):
    return partial(mask_fn_v2, q_positions=q_positions, kv_positions=kv_positions, q_attn_regions=q_attn_regions, kv_attn_regions=kv_attn_regions, q_document_ids=q_document_ids, kv_document_ids=kv_document_ids)


def make_docmask_gpu(ntoks, attn_regions, document_ids, BLOCK_SIZE=128, compile=True):
    """
    ntoks: int, the total sequence length, both for queries, and kvs.
    document_ids: int[ntoks] 1D tensor of ints:
        Tokens with the same document_id may attend to each other. Different IDs never may.
    attn_regions: int[ntoks] 1D tensor of ints:
        Tokens with the same integer attn_region may attend to each other in any direction,
        but if the attn_region is zero, then they only attend to each other in autoregressively.
        If the attn_region is -1, then it's actually a padding token and there is no attention.
    device: pytorch device where to compute the mask on.

    Returns a pytorch flex_attention BlockMask object that contains the block mask information,
    see the file `flex_attention.py` for what it looks like.
    """
    assert document_ids.device == attn_regions.device
    device = document_ids.device

    # This is only reasonably efficient up to a reasonable but not huge
    # seqlen (somewhere <128k). See the files tools/benchmark_docmask_gpu.py and
    # tools/batched_vmap_slow.py for more details. Switch to make_docmask_gpu.
    return maybe_compiled_fn("create_block_mask", compile)(
        make_mask_fn(attn_regions, document_ids),
        B=1, H=1, Q_LEN=ntoks, KV_LEN=ntoks, device=device, BLOCK_SIZE=BLOCK_SIZE)


def make_mask_v2(ntoks, q_positions, kv_positions, q_attn_regions, kv_attn_regions, q_document_ids, kv_document_ids, BLOCK_SIZE=128, compile=True):
    """Create a BlockMask for a sub-problem with separate q/kv position and ID arrays.

    Used internally by the superblock implementations (v2/v3) to compute masks
    for individual superblock pairs.

    q/kv_positions: int[ntoks] position indices for query/key-value tokens.
    q/kv_attn_regions: int[ntoks] attention region IDs for query/key-value tokens.
    q/kv_document_ids: int[ntoks] document IDs for query/key-value tokens.
    """
    assert q_positions.device == kv_positions.device == q_attn_regions.device == kv_attn_regions.device == q_document_ids.device == kv_document_ids.device
    device = q_positions.device

    # This is only reasonably efficient up to a reasonable but not huge
    # seqlen (somewhere 32k-1M). See the files tools/benchmark_docmask_gpu.py and
    # tools/batched_vmap_slow.py for more details. Switch to make_docmask_gpu.
    return maybe_compiled_fn("create_block_mask", compile)(
        make_mask_fn_v2(q_positions, kv_positions, q_attn_regions, kv_attn_regions, q_document_ids, kv_document_ids),
        B=1, H=1, Q_LEN=ntoks, KV_LEN=ntoks, device=device, BLOCK_SIZE=BLOCK_SIZE)


def make_docmask_gpu_v2(ntoks, attn_regions, document_ids, BLOCK_SIZE=128, SUPERBLOCK_SIZE=8192, compile=True):
    """
    This is a more efficient algorithm (~10x faster, much better)
    Same as make_docmask_gpu but using vectorized operations (Option 4).
    This version should be faster on GPU but produce identical results.
    """
    # Make it work for both torch tensors (on gpu) and numpy arrays (on cpu)
    attn_regions = torch.as_tensor(attn_regions)
    document_ids = torch.as_tensor(document_ids)

    # For now, for simplicity. Otherwise, need to write extra code for the last superblock...
    assert ntoks % SUPERBLOCK_SIZE == 0, "Simplify my life for now."
    assert ntoks % BLOCK_SIZE == 0, "Simplify my life for now."
    assert SUPERBLOCK_SIZE % BLOCK_SIZE == 0, "Simplify my life for now."
    assert document_ids.device == attn_regions.device
    device = document_ids.device

    # For each superblock, which document IDs are in that block?
    superblock_docs = [set(docs.unique().cpu().tolist()) for docs in document_ids.split(SUPERBLOCK_SIZE)]

    # Initialize block mask storage
    NB = ntoks // BLOCK_SIZE

    # These could be uint16 up to ntoks=8M (BLOCK=128), but torch uint16 support is abysmal.
    kv_num_blocks = torch.zeros(NB, dtype=torch.int32, device=device)
    kv_block_indices = torch.zeros((NB, NB), dtype=torch.int32, device=device)
    full_kv_num_blocks = torch.zeros(NB, dtype=torch.int32, device=device)
    full_kv_block_indices = torch.zeros((NB, NB), dtype=torch.int32, device=device)

    # Just 8MiB per million ntok, so no need to optimze more:
    positions = torch.arange(len(document_ids), device=device)
    subblocks = torch.arange(SUPERBLOCK_SIZE // BLOCK_SIZE, device=device)

    # Now, go over all superblocks, and either skip them entirely (if no document ID overlap),
    # or compute them using the original "base" mask creation function, and combine outputs.
    for q_superblock in range(len(superblock_docs)):
        q_docs = superblock_docs[q_superblock]

        q_block_start = q_superblock * SUPERBLOCK_SIZE // BLOCK_SIZE
        q_block_end = (q_superblock + 1) * SUPERBLOCK_SIZE // BLOCK_SIZE

        for kv_superblock in range(len(superblock_docs)):
            # Only do detailed check if there's any chance of document overlap.
            if q_docs & superblock_docs[kv_superblock]:
                q_start, q_end = q_superblock * SUPERBLOCK_SIZE, (q_superblock+1) * SUPERBLOCK_SIZE
                kv_start, kv_end = kv_superblock * SUPERBLOCK_SIZE, (kv_superblock+1) * SUPERBLOCK_SIZE

                # Most of the time is spent on this one, and most of that on launch overhead
                # as opposed to actual compute. Might be worth cuda-graph'ing it maybe?
                result = make_mask_v2(SUPERBLOCK_SIZE,
                    positions[q_start:q_end], positions[kv_start:kv_end],
                    attn_regions[q_start:q_end], attn_regions[kv_start:kv_end],
                    document_ids[q_start:q_end], document_ids[kv_start:kv_end],
                    BLOCK_SIZE=BLOCK_SIZE, compile=compile)

                offset = kv_superblock * SUPERBLOCK_SIZE // BLOCK_SIZE

                # Mixed blocks:
                cols = kv_num_blocks[q_block_start:q_block_end, None] + subblocks
                kvi_slice = kv_block_indices[q_block_start:q_block_end]  # A real view
                kvi_slice.scatter_(dim=1, index=cols, src=result.kv_indices[0, 0, :, :] + offset)

                kv_num_blocks[q_block_start:q_block_end] += result.kv_num_blocks[0, 0, :]

                # Same for full blocks:
                cols = full_kv_num_blocks[q_block_start:q_block_end, None] + subblocks
                kvi_slice = full_kv_block_indices[q_block_start:q_block_end]  # A real view
                kvi_slice.scatter_(dim=1, index=cols, src=result.full_kv_indices[0, 0, :, :] + offset)

                full_kv_num_blocks[q_block_start:q_block_end] += result.full_kv_num_blocks[0, 0, :]

    # Create BlockMask using from_kv_blocks
    return BlockMask.from_kv_blocks(
        kv_num_blocks=kv_num_blocks[None, None, :],
        kv_indices=kv_block_indices[None, None, :, :],
        full_kv_num_blocks=full_kv_num_blocks[None, None, :],
        full_kv_indices=full_kv_block_indices[None, None, :, :],
        BLOCK_SIZE=BLOCK_SIZE,
        seq_lengths=(ntoks, ntoks),
        mask_mod=make_mask_fn(attn_regions, document_ids),
    )


def make_docmask_gpu_v3(ntoks, attn_regions, document_ids, BLOCK_SIZE=128, SUPERBLOCK_SIZE=8192, compile=True, max_per_row=None):
    """
    Memory-efficient version of make_docmask_gpu_v2.
    Uses compact (NB, max_per_row) index arrays instead of (NB, NB).

    max_per_row: controls index array column width.
      "dynamic" (default): computed from data, dims marked dynamic via mark_dynamic.
      int: fixed width (asserts if too small). Use for torch.compile(dynamic=False).
      None: computed from data (shapes vary, no mark_dynamic).
    """
    attn_regions = torch.as_tensor(attn_regions)
    document_ids = torch.as_tensor(document_ids)

    assert ntoks % SUPERBLOCK_SIZE == 0
    assert ntoks % BLOCK_SIZE == 0
    assert SUPERBLOCK_SIZE % BLOCK_SIZE == 0
    assert document_ids.device == attn_regions.device
    device = document_ids.device

    NB = ntoks // BLOCK_SIZE
    SB_BLOCKS = SUPERBLOCK_SIZE // BLOCK_SIZE

    # For each superblock, which document IDs are in that block?
    q_superblock_docs = [set(docs.unique().cpu().tolist()) for docs in document_ids.split(SUPERBLOCK_SIZE)]

    # Compute needed max_per_row: for each q_superblock, count how many
    # kv_superblocks share at least one document.  The scatter writes
    # SB_BLOCKS entries per overlapping pair, so we need that many columns.
    max_overlapping = max(
        sum(1 for kv_docs in q_superblock_docs if q_docs & kv_docs)
        for q_docs in q_superblock_docs
    )
    needed_mpr = min(max_overlapping * SB_BLOCKS, NB)

    if isinstance(max_per_row, int):
        assert needed_mpr <= max_per_row, f"max_per_row={max_per_row} too small, need {needed_mpr}"
        mpr = max_per_row
    else:  # None or "dynamic"
        mpr = needed_mpr

    kv_num_blocks = torch.zeros(NB, dtype=torch.int32, device=device)
    kv_block_indices = torch.zeros((NB, mpr), dtype=torch.int32, device=device)
    full_kv_num_blocks = torch.zeros(NB, dtype=torch.int32, device=device)
    full_kv_block_indices = torch.zeros((NB, mpr), dtype=torch.int32, device=device)

    positions = torch.arange(len(document_ids), device=device)
    subblocks = torch.arange(SUPERBLOCK_SIZE // BLOCK_SIZE, device=device)

    for q_superblock in range(len(q_superblock_docs)):
        q_docs = q_superblock_docs[q_superblock]
        q_block_start = q_superblock * SUPERBLOCK_SIZE // BLOCK_SIZE
        q_block_end = (q_superblock + 1) * SUPERBLOCK_SIZE // BLOCK_SIZE

        for kv_superblock in range(len(q_superblock_docs)):
            if q_docs & q_superblock_docs[kv_superblock]:
                q_start, q_end = q_superblock * SUPERBLOCK_SIZE, (q_superblock+1) * SUPERBLOCK_SIZE
                kv_start, kv_end = kv_superblock * SUPERBLOCK_SIZE, (kv_superblock+1) * SUPERBLOCK_SIZE

                result = make_mask_v2(SUPERBLOCK_SIZE,
                    positions[q_start:q_end], positions[kv_start:kv_end],
                    attn_regions[q_start:q_end], attn_regions[kv_start:kv_end],
                    document_ids[q_start:q_end], document_ids[kv_start:kv_end],
                    BLOCK_SIZE=BLOCK_SIZE, compile=compile)

                offset = kv_superblock * SUPERBLOCK_SIZE // BLOCK_SIZE

                # Mixed blocks:
                cols = kv_num_blocks[q_block_start:q_block_end, None] + subblocks
                kvi_slice = kv_block_indices[q_block_start:q_block_end]
                kvi_slice.scatter_(dim=1, index=cols, src=result.kv_indices[0, 0, :, :] + offset)
                kv_num_blocks[q_block_start:q_block_end] += result.kv_num_blocks[0, 0, :]

                # Full blocks:
                cols = full_kv_num_blocks[q_block_start:q_block_end, None] + subblocks
                kvi_slice = full_kv_block_indices[q_block_start:q_block_end]
                kvi_slice.scatter_(dim=1, index=cols, src=result.full_kv_indices[0, 0, :, :] + offset)
                full_kv_num_blocks[q_block_start:q_block_end] += result.full_kv_num_blocks[0, 0, :]

    # Transpose (kv -> q) entirely on device using sort + scatter.
    def _transpose(num, idx):
        cols = torch.arange(idx.shape[1], device=device)
        valid = cols[None, :] < num[:, None]
        qb_all = torch.arange(NB, device=device)[:, None].expand_as(idx)
        v_qb = qb_all[valid].long()
        v_kvb = idx[valid].long()
        if len(v_kvb) == 0:
            return (torch.zeros(NB, dtype=torch.int32, device=device),
                    torch.zeros((NB, mpr), dtype=torch.int32, device=device))
        order = v_kvb.argsort(stable=True)
        s_kvb, s_qb = v_kvb[order], v_qb[order]
        t_num = torch.zeros(NB, dtype=torch.int32, device=device)
        t_num.scatter_add_(0, s_kvb, torch.ones(len(s_kvb), dtype=torch.int32, device=device))
        starts = torch.zeros(NB, dtype=torch.int64, device=device)
        starts[1:] = t_num[:-1].cumsum(0).long()
        pos = torch.arange(len(s_kvb), device=device) - starts[s_kvb]
        t_idx = torch.zeros((NB, mpr), dtype=torch.int32, device=device)
        t_idx[s_kvb, pos] = s_qb.to(torch.int32)
        return t_num, t_idx

    q_num_blocks, q_block_indices = _transpose(kv_num_blocks, kv_block_indices)
    full_q_num_blocks, full_q_block_indices = _transpose(full_kv_num_blocks, full_kv_block_indices)

    mask = BlockMask(
        kv_num_blocks=kv_num_blocks[None, None, :],
        kv_indices=kv_block_indices[None, None, :, :],
        full_kv_num_blocks=full_kv_num_blocks[None, None, :],
        full_kv_indices=full_kv_block_indices[None, None, :, :],
        q_num_blocks=q_num_blocks[None, None, :],
        q_indices=q_block_indices[None, None, :, :],
        full_q_num_blocks=full_q_num_blocks[None, None, :],
        full_q_indices=full_q_block_indices[None, None, :, :],
        BLOCK_SIZE=(BLOCK_SIZE, BLOCK_SIZE),
        mask_mod=make_mask_fn(attn_regions, document_ids),
        seq_lengths=(ntoks, ntoks),
    )
    if max_per_row == "dynamic":
        # Use positive index: mark_dynamic with negative indices is broken
        # (torch stores -1 raw, but builder.py iterates range(ndim) so -1
        # is never matched). See bv2/tools/repro_mark_dynamic_blockmask.py.
        d = mask.kv_indices.ndim - 1
        torch._dynamo.mark_dynamic(mask.kv_indices, d)
        torch._dynamo.mark_dynamic(mask.full_kv_indices, d)
        torch._dynamo.mark_dynamic(mask.q_indices, d)
        torch._dynamo.mark_dynamic(mask.full_q_indices, d)
    return mask
