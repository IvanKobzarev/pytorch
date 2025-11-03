#!/usr/bin/env python3
"""
Benchmark for sequence packing code, which seems to bottleneck us.

Usage:
    python3 -m bv2.tools.benchmark_seq_pack --ntoks 32768

Some results on a H200 machine, with random documents of sizes 1024 to 4096:
"""

import argparse
import time

import torch
import numpy as np
from torch.profiler import profile, ProfilerActivity

from bv2.simple_input import iter_packed_examples


def main_bench(fn, repeats=10, device="cpu"):
    ts, ms = [], []
    for i in range(repeats):
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.monotonic_ns()

        fn()

        if device == "cuda":
            torch.cuda.synchronize()
            ms.append(torch.cuda.max_memory_allocated() / 1024**2)  # MiB

        ts.append((time.monotonic_ns() - t0) / 1_000_000)
        print(f"{ts[-1]:.1f}ms", end="", flush=True)

        if device == "cuda":
            print(f"/{ms[-1]:.0f}MiB", end="", flush=True)

        print(" ", end="", flush=True)
    print("")
    print(f" -> Median: {np.median(ts[1:]):.1f}ms, Warmup/Compile: {ts[0]/1000:.1f}s", end="")
    if ms:
        print(f", Peak mem: {max(ms):.0f}MiB", end="")
    print("", flush=True)


def main_prof(fn, device="cpu"):
    fn() ; fn()  # Compile and warmup

    if device == "cuda":
        torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device == "cuda" else []),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_modules=True
    ) as prof:
        fn()
        if device == "cuda":
            torch.cuda.synchronize()
    prof.export_chrome_trace("/tmp/trace.json")
    print("Profile saved to /tmp/trace.json. Open in perfetto.dev or chrome://trace")
    print("Consider whether you may want to run gzip /tmp/trace.json first.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark sequence packing")
    parser.add_argument("--ntoks", type=int, default=32768,
                        help="Number of tokens")
    parser.add_argument("--tokbytes", type=int, default=16*16*4,
                        help="Number of bytes per token.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--docmin", type=int, default=1024,
                        help="Shortest acceptable document.")
    parser.add_argument("--docmax", type=int, default=4096,
                        help="Longest acceptable document.")
    parser.add_argument("--profile", action="store_true", default=False,
                        help="Store profiler trace (makes timing invalid).")
    parser.add_argument("--repeats", type=int, default=10,
                        help="How many times to run the timing (not profiling).")
    args = parser.parse_args()

    # Create a large pool of documents from which to sample.
    rng = np.random.default_rng(args.seed)
    seqlens = rng.integers(args.docmin, args.docmax, 1000)
    documents = [{
        "tokens": rng.integers(0, 255, size=(seqlen, args.tokbytes), dtype=np.uint8),
        "loss_weights":  np.ones(seqlen, np.int64),
        "attn_regions":  np.ones(seqlen, np.int64),
        "attn_regions2": np.ones(seqlen, np.int64),
        "src": "Pure random bro",
        "id": i,
    } for i, seqlen in enumerate(seqlens)]

    mk_example_generator = lambda: (rng.choice(documents) for _ in range(1_000))

    fn = lambda: list(iter_packed_examples(mk_example_generator(), args.ntoks))

    if args.profile:  # NOTE: once I get a 3rd benchmark, I'll make them generic.
        main_prof(fn)
    else:
        main_bench(fn, repeats=args.repeats)
