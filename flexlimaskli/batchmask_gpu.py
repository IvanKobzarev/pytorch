"""GPU batched mask creation for single-document-per-batch-element sequences.

Entry point:
- make_batchmask_gpu: uses create_block_mask on GPU for batched single-document masks
"""

from functools import cache, partial

import torch
from torch.nn.attention.flex_attention import create_block_mask


@cache
def _compiled_cbm(compiled):
    compile = torch.compile if compiled else lambda fn: fn
    return compile(partial(create_block_mask, H=None))


def _mask_fn(b, h, q_idx, kv_idx, attn_regions):
    causal = q_idx >= kv_idx
    is_padding = (attn_regions[b, q_idx] == -1) | (attn_regions[b, kv_idx] == -1)
    dense_region = (attn_regions[b, q_idx] > 0) & (attn_regions[b, kv_idx] > 0)
    same_region = attn_regions[b, q_idx] == attn_regions[b, kv_idx]
    return (causal | (dense_region & same_region)) & ~is_padding


def make_batchmask_gpu(ntoks, attn_regions_batch, BLOCK_SIZE=128, compile=True):
    """GPU batchmask for single-document-per-batch-element (e.g. decoding).

    attn_regions_batch: (B, ntoks) tensor on GPU. Returns a BlockMask with batch dim B.
    Uses create_block_mask on GPU — simpler but slower than the CPU numba path
    for typical decode sizes. Mainly useful as a reference implementation.
    """
    attn_regions = torch.as_tensor(attn_regions_batch)
    B = attn_regions.shape[0]
    device = attn_regions.device
    return _compiled_cbm(compile)(
        partial(_mask_fn, attn_regions=attn_regions),
        B=B, Q_LEN=ntoks, KV_LEN=ntoks, device=device, BLOCK_SIZE=BLOCK_SIZE)
