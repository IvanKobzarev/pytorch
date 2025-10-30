from functools import partial
from importlib import import_module
from itertools import count as icount, islice

import numpy as np
import sws
import torch
from torch.nn.attention.flex_attention import create_block_mask

from bv2.simple_input import iter_packed_examples, to_len, parallel_prefetch  # fmt: skip  # usort: skip

# Current high-level description of input pipeline:
# 0. A dataset module `ds` defines two functions: `make_exids` and `make_example`.
# 1. The `make_exids` function returns something that iterates over the
#    deterministically shuffled example IDs (exids) for a given epoch number.
#    An exid can be anything, it's up to the dataset.
# 2. The `make_example` function generates one example for the given exid and epoch.
#    Again, the dataset decides what exactly this means across epochs; for example
#    a VQA dataset might define exid to be the image and cycle through questions
#    over epochs, and/or randomly (but deterministically) augment the image, or...
#
# As for the input pipeline itself, the flow is as follows:
# 1. `ex_id_gen` iterates over example IDs, potentially for an infinite number
#    of epochs. It yields the ID coupled with checkpointing-state.
# 2. `ds.make_example` is wrapped with `_with_state`, so that it's only called
#    with the actual exid and epoch, and the state is then merged into the example.
# 3. `parallel_prefetch` is a generic util which is similar to `map`, calling the
#    passed function on every item of the passed iterable (our infinite example
#    generator), but does prefetch the next ones in background processes.
# 5. `iter_packed_examples` eagerly packs as many examples into a single sequence
#    as possible, and then pads it.
# 6. `to_gpu_and_mask` computes flex-attention masks, and shifts tensors to GPU.
#    Ideally we prefetch this one step too, but that didn't work so far.


def data_iter(ds, *, maxtok, device, seed=0, eagerness=16,
              rank=0, world_size=1, resumed_ep=0,
              resumed_i=0, max_ep=None):
    make_exids = partial(ds.make_exids, seed=seed, rank=rank, world_size=world_size)

    # Generator that yields a tuple of (save states, example IDs)
    def ex_id_gen(resumed_i=resumed_i):  # arg instead of capture because =0 below.
        for epoch in count(start=resumed_ep, end=max_ep):
            ex_id_iter = enumerate(make_exids(epoch=epoch))
            ex_id_iter = islice(ex_id_iter, resumed_i, None)
            for i, ex_id in ex_id_iter:
                yield ({"ep": epoch, "i": i + 1}, ex_id, epoch)  # Yes, correct.
            resumed_i = 0

    def cpu_data_gen():
        make_example = partial(_with_state, make_example=ds.make_example)
        ex_gen = parallel_prefetch(ex_id_gen(), make_example, eagerness)

        seq_padder = lambda seq: to_len(seq, to_len=maxtok, pad_values={
            # Only pad these fields, keep unmentioned fields unpadded.
            "tokens": 0,
            "loss_weights": 0.0,  # Also makes sure it's float.
            "iseq": -1,
            # attn_region -1 is ignored by our flex call.
        } | {k: -1 for k in seq if k.startswith("attn_regions")})  # fmt: skip

        yield from map(seq_padder, iter_packed_examples(ex_gen, max_seqlen=maxtok))

        # After epochs are exhausted, we generate pad-only seqs forever.
        if max_ep:
            # But we need to know the content/shape/dtype of sequence entries!
            # So we make one example, that we then truncate, pad, and reuse forever.
            dummy_id = next(iter(ds.make_exids(seed=0, epoch=0)))
            dummy_ex = ds.make_example(dummy_id, epoch=0)
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


# This needs to be global for pickle-ability.
def _with_state(things, make_example):
    state_after, ex_id, epoch = things
    return {**make_example(ex_id, epoch), "state_after": state_after}


create_block_mask = torch.compile(partial(create_block_mask, B=None, H=None))


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



def count(start, *, end=None, step=1):
    if end is None:
        yield from icount(start, step)
    else:
        yield from range(start, end, step)
