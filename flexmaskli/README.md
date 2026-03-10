# Flexmaskli: flex-attention BlockMask creation

## Overview

Flexmaskli is a PyTorch `flex_attention` BlockMask creation library for two use cases:
- **Docmask**: multi-document packed sequences (training)
- **Batchmask**: batched single-document sequences (decoding)

It provides CPU (numpy/numba) and GPU implementations with progressive optimizations from O(n²) baseline to memory-efficient O(n) superblock approaches.

All implementations produce compact index arrays and use `mark_dynamic` for compatibility with `torch.compile(dynamic=False)` and strict `recompile_limit=1` — shapes can vary between batches without triggering recompilation.

## Two mask types

This library provides two kinds of flex-attention BlockMask creation:

- **Docmask** — multi-document packed sequences. Multiple documents packed into one sequence, with per-document causal attention and optional dense (bidirectional) regions. Both CPU (`docmask_cpu.py`) and GPU (`docmask_gpu.py`) implementations.
- **Batchmask** — batched single-document sequences. One document per batch element (batch dim B), with causal + dense regions. CPU-only numba implementation (`batchmask_cpu.py`), designed for decoding.

## Docmask implementations

### `make_docmask_gpu` (baseline, in `docmask_gpu.py`)

Uses PyTorch's `create_block_mask` with a `mask_mod` function + `torch.compile`. Internally materializes the full NxN boolean mask then converts to block-sparse format. Scales as O(n²) — fine up to ~64k tokens, OOM at 2M+.

### `make_docmask_gpu_v2` (superblock, in `docmask_gpu.py`)

Divide-and-conquer: splits the sequence into superblocks (default 8192 tokens). For each pair of superblocks, checks whether they share any document IDs. If not, the entire pair is skipped. Otherwise calls the baseline on just that sub-problem and stitches results together. Exploits the fact that most superblock pairs share no documents → ~linear in sequence length. Uses `(NB, NB)` index arrays.

### `make_docmask_gpu_v3` (compact superblock, in `docmask_gpu.py`)

Fork of v2 with compact `(NB, max_per_row)` index arrays instead of `(NB, NB)`:

- **`max_per_row` computed from document structure**: for each q-superblock, counts how many kv-superblocks share at least one document. For 4k-token docs with 8k superblocks, this is ~192 instead of 32768.
- **GPU-native transpose**: uses argsort + scatter on device to transpose kv→q arrays without leaving the GPU.
- **Direct `BlockMask` constructor**: passes all 8 tensors directly, avoiding any NB-wide intermediates.
- **`max_per_row` parameter**:
  - `None`: computed from data (shapes vary per batch — not torch.compile compatible).
  - `"dynamic"`: computed from data, dims marked via `mark_dynamic` for `torch.compile(dynamic=False)` compatibility. Shapes can vary between batches without recompilation.
  - `int`: fixed width (asserts if too small). Always compile-safe.

### CPU docmask (in `docmask_cpu.py`)

Three entry points for single-sequence (multi-document) mask creation:

- **`make_docmask_numpy`**: pure numpy path — vectorized dense region finding, Python block iteration. No external dependencies beyond numpy.
- **`make_docmask_numba`**: numba JIT path — full block iteration + dense region finding + transpose compiled via numba. Requires numba.
- **`make_docmask_cpu`**: dispatcher — calls `make_docmask_numba` if numba is available, else falls back to `make_docmask_numpy`.

All use compact `(NB, max_per_row)` index arrays. Same `max_per_row` parameter as gpu_v3.

## Batchmask implementations

### `make_batchmask_gpu` (in `batchmask_gpu.py`)

GPU batchmask using `create_block_mask` with `B=batch_size`. This was the original decoding codepath before the CPU numba version was added. Simpler but slower than the CPU numba path for typical decode sizes (1k-32k tokens). Useful as a reference implementation and when GPU is preferred over CPU preprocessing.

### `make_batchmask_cpu` (in `batchmask_cpu.py`)

Specialized for the **decoding iterator** use-case: one document per batch element, batch dimension B. Uses a single numba JIT call over the full `(B, ntoks)` input instead of per-element calls + pad + stack.

Key simplifications over the multi-document docmask:
- **No document segmentation**: single doc per element, so all below-diagonal blocks are trivially full.
- **Compact per-type trimming**: each of the 4 index array types (kv, full_kv, q, full_q) is trimmed to its actual needed width. Partial arrays (kv, q) shrink to ~1-3 columns (just the diagonal), while full arrays (full_kv, full_q) stay at ~NB (the causal triangle). All dims are marked dynamic via `mark_dynamic`.
- **No padding/stacking overhead**: one JIT call produces all B masks with uniform shapes, eliminating the per-element pad + `torch.cat` that caused GPU memory fragmentation in `decode_lib.py`.

The `mask_mod` closure uses 2D indexing (`attn_regions[b, q_idx]`) which is compatible with `torch.compile`'s pointwise subgraph lowering (chained indexing `attn_regions[b][q_idx]` would fail).

## Utilities

### `blockmask_to_gpu` (in `to_gpu.py`)

Standalone utility for moving a BlockMask to a device. Handles three things that `BlockMask.to(device)` alone does not:

- **Closure tensors**: `mask_mod` is a `functools.partial` whose `.keywords` may contain CPU tensors. Iterates them and calls `.to(device)`.
- **`_dynamo_dynamic_indices`**: `.to()` creates new tensors that lose dynamic marking. Re-applies from the originals.

### torch.compile compatibility

`train.py` uses `torch.compile(dynamic=False)` with `recompile_limit=1`, so BlockMask tensor shapes must be stable across batches.

- **Docmask `max_per_row="dynamic"`**: dims marked dynamic via `mark_dynamic` → no recompilation when index width varies between batches. Recommended for gpu_v3 and cpu.
- **Docmask `max_per_row=<int>`**: fixed shape, no dynamism. Use NB (always safe) or a tighter bound.
- **Batchmask**: compact per-type arrays with `mark_dynamic` on all index tensors. Partial arrays (kv, q) are narrow, full arrays (full_kv, full_q) are ~NB-wide.
- `make_docmask_gpu` and `gpu_v2` produce stable shapes inherently (NB-wide arrays).

**Note**: `mark_dynamic` must be called with **positive** dimension indices. Negative indices (e.g. `-1`) are silently ignored due to a PyTorch bug. Our code uses `ndim - 1`.

## Benchmark results (H200, torch 2.9.1+cu126)

### Docmask: GPU (superblock=8k, docs 1k-4k)

| ntoks | make_docmask_gpu | gpu_v2 | gpu_v3 | v3 vs v2 | v3 mem | v2 mem | mem reduction |
|------:|----------:|---------:|---------:|---------:|-------:|-------:|--------------:|
|   32k |     3.2ms |    6.0ms |    6.0ms |    1.00x |  2 MiB |  4 MiB |           2x |
|   64k |    11.9ms |   10.7ms |   11.5ms |    0.93x |  3 MiB | 15 MiB |         **5x** |
|  128k |    46.4ms |   20.6ms |   22.4ms |    0.92x |  6 MiB | 56 MiB |         **9x** |
|  256k |   195.4ms |   41.1ms |   44.0ms |    0.93x | 12 MiB |218 MiB |        **18x** |
|  512k |   634.2ms |   83.5ms |   86.8ms |    0.96x | 23 MiB |860 MiB |        **37x** |
|    1M |  2671.5ms |  178.8ms |  172.3ms | **1.04x** | 46 MiB |4120 MiB |        **90x** |
|    2M |       OOM |  404.1ms |  331.7ms | **1.22x** | 93 MiB |16432 MiB |       **177x** |
|    4M |       OOM |  987.5ms |  696.3ms | **1.42x** |185 MiB |65633 MiB |       **355x** |

### Docmask: CPU (docs 1k-4k)

| ntoks | numpy |  numba |
|------:|------:|-------:|
|   32k | 2.5ms |  0.2ms |
|   64k | 5.0ms |  0.3ms |
|  128k |10.4ms |  0.5ms |
|  512k |42.6ms |  1.9ms |
|    1M |90.4ms | 10.2ms |
|    4M |395.7ms| 45.3ms |

### Compiled flex_attention: compact (dynamic) vs full-width (fixed) BlockMask

Benchmark script: `bv2/tools/bench_flex_compile.py`.

Setup: `torch.compile(flex_attention, dynamic=False, fullgraph=True)` with `recompile_limit=1`. Two masks with different document structures (docs 2k-4k and docs 256-512) alternated across iterations. Both compile once and run without recompilation. H200, 4 heads, head_dim=128, bf16.

| ntoks | dynamic (idx=192) | fixed (idx=NB) | NB |
|------:|------------------:|---------------:|-----:|
|   32k |            0.5ms |         0.5ms |  256 |
|   64k |            0.8ms |         0.8ms |  512 |
|  128k |            1.3ms |         1.3ms | 1024 |
|  256k |            2.5ms |         2.5ms | 2048 |
|  512k |            4.7ms |         4.9ms | 4096 |

No performance difference in the attention kernel. `flex_attention` scans only up to `kv_num_blocks` per row regardless of index array width, so extra padding columns cost nothing. The benefit of compact arrays is **memory**: at 512k tokens, 192 cols vs 4096 cols per index array. At 4M tokens with NB-wide arrays, the indices alone consume ~64 GiB — compact arrays are essential to fit in memory.

### Batchmask: `make_batchmask_cpu` vs per-element + stack

Benchmark script: `python -m flexmaskli.benchmarks.batchmask_cpu`.

Setup: single-document inputs with random dense prefix (30-70% of ntoks) per batch element. "per-elem+stack" is the old approach (B calls to `make_docmask_numba` + pad indices to max width + `torch.cat`). "batched" is a single `make_batchmask_cpu` call.

| ntoks |   B | per-elem+stack |  batched | speedup |
|------:|----:|---------------:|---------:|--------:|
|  1024 |   1 |          0.1ms |    0.0ms |    3.5x |
|  1024 |   8 |          0.7ms |    0.0ms | **14.6x** |
|  1024 | 128 |          9.6ms |    0.2ms | **47.1x** |
|  4096 |   1 |          0.1ms |    0.0ms |    3.0x |
|  4096 |   8 |          0.7ms |    0.1ms |    8.0x |
|  4096 | 128 |         11.2ms |    0.9ms | **12.4x** |
|  8192 |   8 |          0.9ms |    0.2ms |    4.9x |
|  8192 | 128 |         15.3ms |    8.0ms |    1.9x |
| 16384 |   8 |          1.6ms |    0.5ms |    3.1x |
| 16384 | 128 |         32.2ms |   10.5ms |    3.1x |
| 32768 |   8 |          5.3ms |    1.8ms |    3.0x |
| 32768 | 128 |        169.1ms |   97.4ms |    1.7x |

Speedups are largest at decode-relevant sizes (1k-8k tokens, batch 8-128): **3-47x**. The batched version also eliminates GPU memory fragmentation from variable-width per-element index tensors.

### Block alignment

Both `make_docmask_cpu` and `make_batchmask_cpu` handle non-block-aligned `ntoks` internally by padding arrays before processing. Callers do not need to round up to BLOCK_SIZE. This matches the behavior of GPU `create_block_mask`.

## Recommendations

- **Training (docmask)**: use `make_docmask_cpu` (auto-dispatches to numba if available) with `max_per_row="dynamic"`. 45ms at 4M tokens. Prefetchable off GPU.
- **Decoding (batchmask)**: use `make_batchmask_cpu`. Single JIT call, fixed output shapes, no GPU memory fragmentation. 3-47x faster than per-element + stack.
- **GPU docmask**: use `make_docmask_gpu_v3` with `max_per_row="dynamic"`. Same or faster speed as v2, O(n) memory instead of O(n²). At 4M tokens: 185 MiB vs 64 GiB.
- **GPU transfer**: use `blockmask_to_gpu` (in `to_gpu.py`). Handles BlockMask closure tensors and `_dynamo_dynamic_indices` preservation.
