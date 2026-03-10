#!/usr/bin/env python3
"""
Benchmark for batched CPU mask creation: per-element+stack vs batched numba.

Usage:
    python3 -m flexmaskli.benchmarks.batchmask_cpu
    python3 -m flexmaskli.benchmarks.batchmask_cpu --runs 10
"""

import argparse
import random
import time

import numpy as np
import torch

from flexmaskli.batchmask_cpu import make_batchmask_cpu
from flexmaskli.docmask_cpu import make_docmask_numba


def create_random_documents(ntoks, nmin=1024, nmax=4096, seed=42):
    """Create random document structure with documents of length 1k-4k."""
    random.seed(seed)
    document_ids, attn_regions = [], []
    current_pos, doc_id = 0, 0

    while ntoks - current_pos > nmin:
        doc_length = random.randint(nmin, min(nmax, ntoks - current_pos))
        if current_pos + doc_length > ntoks:
            break
        document_ids.extend([doc_id] * doc_length)
        nprefix = int(random.uniform(0.3, 0.7) * doc_length)
        attn_regions.extend([1] * nprefix)
        attn_regions.extend([0] * (doc_length - nprefix))
        current_pos += doc_length
        doc_id += 1

    while len(document_ids) < ntoks:
        document_ids.append(-1)
        attn_regions.append(-1)

    return torch.tensor(document_ids), torch.tensor(attn_regions)


def bench_one(fn, runs=5):
    """Run fn `runs` times, return median ms."""
    ts = []
    for _ in range(runs):
        t0 = time.monotonic_ns()
        fn()
        ts.append((time.monotonic_ns() - t0) / 1_000_000)
    return np.median(ts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark batched CPU mask creation")
    parser.add_argument("--block", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--docmin", type=int, default=1024)
    parser.add_argument("--docmax", type=int, default=4096)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    # Warmup numba JIT
    print("Warming up numba JIT...", end="", flush=True)
    d_w, a_w = create_random_documents(8192, args.docmin, args.docmax, args.seed)
    make_docmask_numba(8192, a_w, d_w, BLOCK_SIZE=args.block)
    make_batchmask_cpu(8192, a_w.numpy()[None], BLOCK_SIZE=args.block)
    print(" done.")

    sizes = [1024, 2048, 4096, 8192, 16384, 32768]
    batches = [1, 2, 4, 8, 16, 32, 64, 128]
    print(f"\n{'ntoks':>8s} {'B':>4s}  {'per-elem+stack':>16s}  {'batched':>12s}  {'speedup':>8s}")
    print("-" * 58)

    for ntoks in sizes:
        for B in batches:
            # Single-document inputs: each element has a dense prefix of random length.
            ar_np = np.zeros((B, ntoks), dtype=np.int64)
            di_np = np.zeros((B, ntoks), dtype=np.int64)
            for b in range(B):
                prefix = int(np.random.RandomState(args.seed + b).uniform(0.3, 0.7) * ntoks)
                ar_np[b, :prefix] = 1

            def per_elem(ar=ar_np, di=di_np):
                masks = [make_docmask_numba(ntoks, ar[b], di[b], BLOCK_SIZE=args.block)
                         for b in range(B)]
                max_mpr = max(m.kv_indices.shape[-1] for m in masks)
                for attr in ("kv_num_blocks", "kv_indices", "full_kv_num_blocks", "full_kv_indices",
                              "q_num_blocks", "q_indices", "full_q_num_blocks", "full_q_indices"):
                    parts = [getattr(m, attr) for m in masks]
                    if "indices" in attr:
                        parts = [torch.cat([p, p.new_zeros(*p.shape[:-1], max_mpr - p.shape[-1])], -1)
                                 if p.shape[-1] < max_mpr else p for p in parts]
                    torch.cat(parts)

            def batched(ar=ar_np):
                make_batchmask_cpu(ntoks, ar, BLOCK_SIZE=args.block)

            t_elem = bench_one(per_elem, args.runs)
            t_batch = bench_one(batched, args.runs)
            print(f"{ntoks:>8d} {B:>4d}  {t_elem:>14.1f}ms  {t_batch:>10.1f}ms  {t_elem/t_batch:>7.2f}x")
