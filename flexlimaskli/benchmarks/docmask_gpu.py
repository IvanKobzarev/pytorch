#!/usr/bin/env python3
"""
Benchmark for faster document-based mask creation.

Usage:
    python3 -m flexlimaskli.benchmarks.docmask_gpu --fn make_docmask_gpu --device cuda --ntoks 32768

Some results on a H200 machine, with random documents of sizes 1024 to 4096:

Reference `make_docmask_gpu`:

for s in 8 16 32 64 128 256 512 1024; echo -n "s="(math "$s * 1024")" " ; python3 -m flexlimaskli.benchmarks.docmask_gpu --fn make_docmask_gpu --device cuda --ntoks (math "$s * 1024"); end
s=8192    -> Median:    0.3ms, Compile:  20.2s, Peak mem: 60MiB
s=16384   -> Median:    0.8ms, Compile:   7.5s, Peak mem: 61MiB
s=32768   -> Median:    3.1ms, Compile:  23.1s, Peak mem: 63MiB
s=65536   -> Median:   11.8ms, Compile:  38.1s, Peak mem: 70MiB
s=131072  -> Median:   46.2ms, Compile:  11.8s, Peak mem: 96MiB
s=262144  -> Median:  195.1ms, Compile:  18.0s, Peak mem: 197MiB
s=524288  -> Median:  633.9ms, Compile:  45.2s, Peak mem: 525MiB
s=1048576 -> Median: 2688.4ms, Compile: 134.6s, Peak mem: 4130MiB

And for superblocks with superblock size 8k (which seems optimal for this setup):
s=8192 -> Median: 1.5ms, Compile: 8.7s, Peak mem: 60MiB
s=16384 -> Median: 2.2ms, Compile: 5.2s, Peak mem: 1MiB
s=32768 -> Median: 3.9ms, Compile: 5.3s, Peak mem: 4MiB
s=65536 -> Median: 7.7ms, Compile: 4.6s, Peak mem: 15MiB
s=131072 -> Median: 15.9ms, Compile: 4.4s, Peak mem: 56MiB
s=262144 -> Median: 31.8ms, Compile: 3.5s, Peak mem: 218MiB
s=524288 -> Median: 64.9ms, Compile: 3.9s, Peak mem: 860MiB
s=1048576 -> Median: 140.8ms, Compile: 4.0s, Peak mem: 4120MiB

More extra timings documented at the end of this file. TL;DR from them:
- superblock size 16k seems near optimal.
- going beyond 1M takes a lot of GPU RAM, 4M works, 8M OOMs.
- CPU is between 2 to 5 times slower than GPU (but is pre-fetchable.)
"""

import argparse
import random
import time

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

import flexlimaskli.docmask_gpu


def create_random_documents(ntoks, nmin=1024, nmax=4096, seed=42):
    """Create random document structure with documents of length 1k-4k."""
    random.seed(seed)

    document_ids = []
    attn_regions = []
    current_pos = 0
    doc_id = 0

    while ntoks - current_pos > nmin:
        doc_length = random.randint(nmin, min(nmax, ntoks - current_pos))

        # Don't exceed total sequence length
        if current_pos + doc_length > ntoks:
            break

        # Add document tokens
        document_ids.extend([doc_id] * doc_length)

        # Dense prefix then causal, random split in 30-70%.
        nprefix = int(random.uniform(0.3, 0.7) * doc_length)
        attn_regions.extend([1] * nprefix)
        attn_regions.extend([0] * (doc_length - nprefix))

        current_pos += doc_length
        doc_id += 1

    # Fill remaining with padding tokens
    while len(document_ids) < ntoks:
        document_ids.append(-1)
        attn_regions.append(-1)

    return torch.tensor(document_ids), torch.tensor(attn_regions)


def main_bench(fn):
    ts, ms = [], []
    for i in range(10):
        if args.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.monotonic_ns()

        fn()

        if args.device == "cuda":
            torch.cuda.synchronize()
            ms.append(torch.cuda.max_memory_allocated() / 1024**2)  # MiB

        ts.append((time.monotonic_ns() - t0) / 1_000_000)
        print(f"{ts[-1]:.1f}ms", end="", flush=True)

        if args.device == "cuda":
            print(f"/{ms[-1]:.0f}MiB", end="", flush=True)

        print(" ", end="", flush=True)
    print("")
    print(f" -> Median: {np.median(ts[1:]):.1f}ms, Compile: {ts[0]/1000:.1f}s", end="")
    if ms:
        print(f", Peak mem: {max(ms):.0f}MiB", end="")
    print("", flush=True)


def main_prof(fn):
    fn() ; fn()  # Compile and warmup

    if args.device == "cuda":
        torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if args.device == "cuda" else []),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_modules=True
    ) as prof:
        fn()
        if args.device == "cuda":
            torch.cuda.synchronize()
    prof.export_chrome_trace("/tmp/trace.json")
    print("Profile saved to /tmp/trace.json. Open in perfetto.dev or chrome://trace")
    print("Consider whether you may want to run gzip /tmp/trace.json first.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark document mask creation")
    parser.add_argument("--fn", default="make_docmask_gpu",
                        help="Which function to benchmark")
    parser.add_argument("--ntoks", type=int, default=32768,
                        help="Number of tokens")
    parser.add_argument("--block", type=int, default=128,
                        help="Size of flex block (default: 128)")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                        help="Device to run on")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--docmin", type=int, default=1024,
                        help="Shortest acceptable document.")
    parser.add_argument("--docmax", type=int, default=4096,
                        help="Longest acceptable document.")
    parser.add_argument("--profile", action="store_true", default=False,
                        help="Store profiler trace (makes timing invalid).")
    args = parser.parse_args()

    document_ids, attn_regions = create_random_documents(args.ntoks, args.docmin, args.docmax, args.seed)
    document_ids, attn_regions = document_ids.to(args.device), attn_regions.to(args.device)
    mask_fn = getattr(flexlimaskli.docmask_gpu, args.fn)
    fn = lambda: mask_fn(args.ntoks, attn_regions, document_ids)

    if args.profile:  # NOTE: once I get a 2nd benchmark, I'll make them generic.
        main_prof(fn)
    else:
        main_bench(fn)

"""
Longer maxtok:
s=2097152 -> Median: 446.3ms, Compile: 6.6s, Peak mem: 16465MiB
s=4194304 -> Median: 1028.2ms, Compile: 4.7s, Peak mem: 65697MiB
s=8388608 -> OOM

superblock == block == 128:
s=8192 -> Median: 403.6ms, Compile: 7.4s, Peak mem: 1MiB
s=16384 -> Median: 690.3ms, Compile: 4.9s, Peak mem: 1MiB
s=32768 -> Median: 1475.4ms, Compile: 5.2s, Peak mem: 5MiB

superblock = 1024:
s=8192 -> Median: 9.7ms, Compile: 5.9s, Peak mem: 1MiB
s=16384 -> Median: 16.1ms, Compile: 4.0s, Peak mem: 1MiB
s=32768 -> Median: 30.9ms, Compile: 3.8s, Peak mem: 5MiB
s=65536 -> Median: 64.0ms, Compile: 3.8s, Peak mem: 16MiB
s=131072 -> Median: 128.3ms, Compile: 4.3s, Peak mem: 58MiB

superblock = 4096:
s=8192 -> Median: 2.2ms, Compile: 14.2s, Peak mem: 60MiB
s=16384 -> Median: 4.0ms, Compile: 4.2s, Peak mem: 61MiB
s=32768 -> Median: 6.8ms, Compile: 5.1s, Peak mem: 62MiB
s=65536 -> Median: 12.6ms, Compile: 3.4s, Peak mem: 65MiB
s=131072 -> Median: 23.6ms, Compile: 4.6s, Peak mem: 73MiB
s=262144 -> Median: 46.6ms, Compile: 3.9s, Peak mem: 222MiB

superblock = 16k:
s=8192   n/a
s=16384  -> Median:   1.6ms, Compile: 5.1s, Peak mem: 2MiB
s=32768  -> Median:   4.5ms, Compile: 3.6s, Peak mem: 5MiB
s=65536  -> Median:  10.6ms, Compile: 4.8s, Peak mem: 16MiB
s=131072 -> Median:  23.0ms, Compile: 5.2s, Peak mem: 58MiB
s=262144 -> Median:  46.9ms, Compile: 4.0s, Peak mem: 222MiB
s=524288 -> Median:  95.7ms, Compile: 3.7s, Peak mem: 868MiB
s=1048576-> Median: 204.7ms, Compile: 3.7s, Peak mem: 4137MiB

superblock = 32k:
s=32768 -> Median: 4.9ms, Compile: 7.8s, Peak mem: 62MiB
s=65536 -> Median: 17.6ms, Compile: 3.5s, Peak mem: 65MiB
s=131072 -> Median: 43.1ms, Compile: 4.1s, Peak mem: 74MiB
s=262144 -> Median: 94.7ms, Compile: 4.5s, Peak mem: 224MiB
s=524288 -> Median: 198.7ms, Compile: 3.7s, Peak mem: 870MiB
s=1048576 -> Median: 416.2ms, Compile: 5.5s, Peak mem: 4138MiB

superblock = 4k ON cpu:
s=8192 -> Median: 2.5ms, Compile: 25.5s
s=16384 -> Median: 5.7ms, Compile: 19.4s
s=32768 -> Median: 12.8ms, Compile: 22.2s
s=65536 -> Median: 32.4ms, Compile: 22.3s
s=131072 -> Median: 55.1ms, Compile: 22.3s
s=262144 -> Median: 146.2ms, Compile: 21.2s
s=524288 -> Median: 535.5ms, Compile: 21.8s
s=1048576 -> Median: 1665.7ms, Compile: 23.0s

superblock = 16k ON cpu:
s=16384 -> Median: 5.3ms, Compile: 24.7s
s=32768 -> Median: 25.1ms, Compile: 22.0s
s=65536 -> Median: 55.3ms, Compile: 20.7s
s=131072 -> Median: 115.1ms, Compile: 20.2s
s=262144 -> Median: 281.9ms, Compile: 20.7s
s=524288 -> Median: 788.3ms, Compile: 23.9s
s=1048576 -> Median: 1925.6ms, Compile: 23.6s

make_mask (no superblock) on cpu:
s=8192 -> Median: 1.2ms, Compile: 25.2s
s=16384 -> Median: 7.9ms, Compile: 24.1s
s=32768 -> Median: 23.2ms, Compile: 23.3s
s=65536 -> Median: 64.1ms, Compile: 22.8s
s=131072 -> Median: 228.4ms, Compile: 24.0s
s=262144 -> Median: 910.9ms, Compile: 24.6s
s=524288 -> Median: 3918.9ms, Compile: 27.3s
s=1048576 -> Median: 15564.2ms, Compile: 36.7s

Changing compile to reduce-overhead or max-autotune barely moved the needle.
"""
