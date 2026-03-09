from functools import partial
from importlib import import_module

import numpy as np
from flexmaskli import make_docmask_cpu

import bv2.utils as u
from bv2.simple_input import iter_packed_examples, pmap, prefetch, to_len

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
def data_iter(ds, *, maxtok, device, seed=0, rank=0, world_size=1, resume={}, pad_after=True,
              # The following defaults were tuned for steptime on a FineVision d8w2k@3136 run:
              pmap_chunksz=24, pmap_threads=16, eagerness=1, mask_block_size=128):
    make_exids = partial(ds.make_exids, seed=seed, rank=rank, world_size=world_size, **resume)

    def make_example(exid_and_state_after):
        make_example_kw, state_after = exid_and_state_after
        try:
            return {**ds.make_example(**make_example_kw), "state_after": state_after}
        except Exception as e:
            kw = ", ".join(f"{k}={v!r}" for k, v in make_example_kw.items())
            e.add_note(f"{ds} make_example({kw})")
            raise

    def cpu_data_gen():
        ex_gen = pmap(make_exids(), make_example, n_prefetch=pmap_chunksz, n_threads=pmap_threads)

        seq_padder = lambda seq: to_len(seq, to_len=maxtok, pad_values={
            # Only pad these fields, keep unmentioned fields unpadded.
            "toki": 0, "toko": 0, "lowe": 0, "iseq": -1,
            # attn_region -1 is ignored by our flex call.
        } | {k: -1 for k in seq if k.startswith("attn_regions")})  # fmt: skip

        yield from map(seq_padder, iter_packed_examples(ex_gen, max_seqlen=maxtok))

        if pad_after:
            # But we need to know the content/shape/dtype of sequence entries!
            # So we make one example, that we then truncate, pad, and reuse forever.
            dummy_ex = make_example(next(make_exids()))
            dummy_ex["lowe"] = dummy_ex["lowe"][:0]
            dummy_ex["toki"] = dummy_ex["toki"][:0]
            dummy_ex["toko"] = dummy_ex["toko"][:0]
            # Usually added by the packer, so we need to manually add it here:
            dummy_ex["iseq"] = np.empty(0, np.int64)
            dummy_ex["ntok"] = dummy_ex["ndatatoks"] = []
            dummy_ex = seq_padder(dummy_ex)
            while True:
                yield dummy_ex

    to_gpu = partial(u.to_gpu, device=device)

    def add_flexmasks_cpu(seq):
        seq["flex_masks"] = {
            k: make_docmask_cpu(maxtok, v, seq["iseq"], BLOCK_SIZE=mask_block_size,
                                max_per_row="dynamic")
            for k, v in seq.items() if k.startswith("attn_regions")
        }
        return seq

    yield from prefetch((to_gpu(add_flexmasks_cpu(s)) for s in cpu_data_gen()), n=eagerness)


def from_config(data_config):
    ds = import_module(f"bv2.data.{data_config['name']}")
    return ds.Dataset(**{k: v for k, v in data_config.items() if k != "name"})
