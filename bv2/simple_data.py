from functools import partial
from importlib import import_module

import numpy as np
import torch
from torch.nn.attention.flex_attention import create_block_mask

import bv2.utils as u
from bv2.simple_input import iter_packed_examples, parallel_prefetch, to_len

# Current high-level description of input pipeline:
# 0. A dataset module `ds` defines two functions: `make_exids` and `make_example`.
# 1. The `make_exids` yields pairs of (exid, state_after), where:
#    - `exid` is a dict such that `make_example(**exid)` generates a specific example
#    - `state_after` is a dict such that make_exids(**state_after) generates the next `exid`.
#    These dictionaries can be anything, it's up to the dataset implementation to decide.
# 2. The `make_example` function generates one example for the given exid.
#    Again, the dataset decides what exactly an example is, and even what an epoch is.
#
# As for the input pipeline itself, the flow is now straightforward:
# 1. `ex_gen` iterates exids, and uses `_exid2example` to yield examples one by one,
#    where the `state_after` is also put into the example.
# 2. `parallel_prefetch` is a generic util which is similar to `map`, with bg prefetch.
# 3. `iter_packed_examples` then eagerly packs as many examples into a single sequence
#    as possible. Once full, we pad the sequence to a fixed length with `seq_padder`.
# 4. When exhausted, we (optionally) continue yielding padding examples infinitely.
#    this is inevitable; to avoid different-iterations edge-cases in multiprocessing.
# 5. `to_gpu_and_mask` computes flex-attention masks, and shifts tensors to GPU.
#    Ideally we prefetch this one step too, but that didn't work with multiprocessing.


@u.suppress_warnings("`isinstance(treespec, LeafSpec)` is deprecated", FutureWarning)
@u.suppress_warnings("`isinstance(treespec, TreeSpec)` is deprecated", FutureWarning)
def data_iter(ds, *, maxtok, device, seed=0, eagerness=16,
              rank=0, world_size=1, resume={}, pad_after=True):
    make_exids = partial(ds.make_exids, seed=seed, rank=rank, world_size=world_size, **resume)

    def cpu_data_gen():
        make_example = partial(_exid2example, make_example=ds.make_example)
        ex_gen = parallel_prefetch(make_exids(), make_example, eagerness)

        seq_padder = lambda seq: to_len(seq, to_len=maxtok, pad_values={
            # Only pad these fields, keep unmentioned fields unpadded.
            "tokens": 0,
            "loss_weights": 0.0,  # Also makes sure it's float.
            "iseq": -1,
            # attn_region -1 is ignored by our flex call.
        } | {k: -1 for k in seq if k.startswith("attn_regions")})  # fmt: skip

        yield from map(seq_padder, iter_packed_examples(ex_gen, max_seqlen=maxtok))

        if pad_after:
            # But we need to know the content/shape/dtype of sequence entries!
            # So we make one example, that we then truncate, pad, and reuse forever.
            dummy_ex = _exid2example(next(make_exids()), ds.make_example)
            dummy_ex["loss_weights"] = dummy_ex["loss_weights"][:0]
            dummy_ex["tokens"] = dummy_ex["tokens"][:0]
            # Usually added by the packer, so we need to manually add it here:
            dummy_ex["iseq"] = np.empty(0, np.int64)
            dummy_ex["lens"] = []
            dummy_ex = seq_padder(dummy_ex)
            while True:
                yield dummy_ex

    # NOTE: can't define it here inline because needs to be picklable.
    fn = partial(to_gpu_and_mask, device=device, maxtok=maxtok)
    # WARNING: Think twice before enabling this prefetch and increasing n_parallel,
    #          because it will copy the RNG to processes, introducing repeats!
    # yield from parallel_prefetch(cpu_data_gen(), fn, n_parallel=1)
    yield from map(fn, cpu_data_gen())


# This needs to be global for pickle-ability. Should just be a lambda...
def _exid2example(exid_and_state_after, make_example):
    make_example_kw, state_after = exid_and_state_after
    return {**make_example(**make_example_kw), "state_after": state_after}


create_block_mask = torch.compile(partial(create_block_mask, B=None, H=None))


@u.suppress_warnings("`isinstance(treespec, LeafSpec)` is deprecated", FutureWarning)
@u.suppress_warnings("`isinstance(treespec, TreeSpec)` is deprecated", FutureWarning)
def make_mask(ntoks, attn_regions, document_ids, device):
    def mask_mod(b, h, q_idx, kv_idx):
        causal = q_idx >= kv_idx
        is_padding = (document_ids[q_idx] == -1) | (document_ids[kv_idx] == -1)
        dense_region = (attn_regions[q_idx] > 0) & (attn_regions[kv_idx] > 0)
        same_region = attn_regions[q_idx] == attn_regions[kv_idx]
        same_document = document_ids[q_idx] == document_ids[kv_idx]
        return (causal | (dense_region & same_region)) & same_document & ~is_padding

    # TODO: This is only reasonably efficient up to a reasonable but not huge
    #       seqlen (about 1M). See the file tools/batched_vmap_slow.py for more.
    #       There are plans to fix this, reach out to qkv@ to discuss.
    with torch.no_grad():
        return create_block_mask(mask_mod, Q_LEN=ntoks, KV_LEN=ntoks, device=device)


def to_gpu_and_mask(seq, device, maxtok):
    # fmt: off
    _can_torch = {
        np.float32, np.float64, np.float16,
        np.int8, np.int16, np.int32, np.int64,
        np.uint8, np.bool, np.complex64, np.complex128,
    }
    # fmt: on

    def maybe_to_gpu(x):
        if isinstance(x, np.ndarray) and any(x.dtype == t for t in _can_torch):
            x = torch.from_numpy(x)
        if isinstance(x, torch.Tensor):
            return x.pin_memory().to(device=device, non_blocking=True)
        else:
            return x

    seq = {k: maybe_to_gpu(v) for k, v in seq.items()}

    # Turn all attention regions into flex-attention mask datastructures.
    seq["flex_masks"] = {}
    for k in (s for s in seq if s.startswith("attn_regions")):
        seq["flex_masks"][k] = make_mask(maxtok, seq[k], seq["iseq"], device)

    return seq


def from_config(data_config):
    ds = import_module(f"bv2.data.{data_config['name']}")
    return ds.Dataset(**{k: v for k, v in data_config.items() if k != "name"})
