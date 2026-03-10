#!/usr/bin/env python3
"""
Benchmark for numpy-based document mask creation.

Usage:
    python3 -m flexmaskli.benchmarks.docmask_cpu --ntoks 32768
    python3 -m flexmaskli.benchmarks.docmask_cpu --ntoks 1048576
    python3 -m flexmaskli.benchmarks.docmask_cpu --ntoks 32768 --verify
    python3 -m flexmaskli.benchmarks.docmask_cpu --compare   # compare numpy vs numba
"""

import argparse
import random
import time

import numpy as np
import torch

from flexmaskli.docmask_cpu import make_docmask_cpu, make_docmask_numpy, make_docmask_numba


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


def blockmask_to_dense(bm, NB):
    """Convert BlockMask to NB x NB dense bool, works with compact indices."""
    dense = torch.zeros(NB, NB, dtype=torch.bool)
    for name in ['kv', 'full_kv']:
        nums = getattr(bm, f'{name}_num_blocks')
        idxs = getattr(bm, f'{name}_indices')
        if nums is None:
            continue
        n, idx = nums[0, 0], idxs[0, 0]
        for qb in range(NB):
            for j in range(n[qb]):
                dense[qb, idx[qb, j]] = True
    return dense[None, None, :, :]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark numpy document mask creation")
    parser.add_argument("--ntoks", type=int, default=32768)
    parser.add_argument("--block", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--docmin", type=int, default=1024)
    parser.add_argument("--docmax", type=int, default=4096)
    parser.add_argument("--verify", action="store_true",
                        help="Verify against torch create_block_mask")
    parser.add_argument("--compare", action="store_true",
                        help="Compare all opt variants across sizes")
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    if args.compare:
        from flexmaskli.docmask_cpu import HAS_NUMBA

        sizes = [32768, 65536, 131072, 524288, 1048576, 4194304]
        labels = ["numpy", "numba"]
        fns_for_compare = [make_docmask_numpy, make_docmask_numba]

        if not HAS_NUMBA:
            print("WARNING: numba not available, skipping numba column")
            labels = labels[:1]
            fns_for_compare = fns_for_compare[:1]
        else:
            # Warmup numba JIT on a small input
            print("Warming up numba JIT...", end="", flush=True)
            d_w, a_w = create_random_documents(8192, args.docmin, args.docmax, args.seed)
            make_docmask_numba(8192, a_w, d_w, BLOCK_SIZE=args.block)
            print(" done.")

        print(f"{'ntoks':>10s}", end="")
        for l in labels:
            print(f"  {l:>10s}", end="")
        print()

        for ntoks in sizes:
            document_ids, attn_regions = create_random_documents(
                ntoks, args.docmin, args.docmax, args.seed)
            print(f"{ntoks:>10d}", end="", flush=True)
            for fn in fns_for_compare:
                bench_fn = lambda fn=fn: fn(
                    ntoks, attn_regions, document_ids, BLOCK_SIZE=args.block)
                med = bench_one(bench_fn, args.runs)
                print(f"  {med:>8.1f}ms", end="", flush=True)
            print()
    else:
        document_ids, attn_regions = create_random_documents(
            args.ntoks, args.docmin, args.docmax, args.seed)

        fn = lambda: make_docmask_cpu(
            args.ntoks, attn_regions, document_ids, BLOCK_SIZE=args.block)

        ts = []
        for i in range(args.runs):
            t0 = time.monotonic_ns()
            result = fn()
            ts.append((time.monotonic_ns() - t0) / 1_000_000)
            print(f"{ts[-1]:.1f}ms ", end="", flush=True)
        print()
        print(f" -> Median: {np.median(ts):.1f}ms, Min: {min(ts):.1f}ms")

        if args.verify:
            from functools import partial as fpartial
            from torch.nn.attention.flex_attention import create_block_mask
            from flexmaskli.docmask_cpu import _mask_fn

            NB = args.ntoks // args.block
            print("\nVerifying against create_block_mask (ground truth)...")
            mask_mod = fpartial(_mask_fn, attn_regions=attn_regions, document_ids=document_ids)
            ref = create_block_mask(mask_mod, B=1, H=1, Q_LEN=args.ntoks, KV_LEN=args.ntoks,
                                    device="cpu", BLOCK_SIZE=args.block)

            ref_dense = ref.to_dense()
            our_dense = blockmask_to_dense(result, NB)

            if torch.equal(ref_dense, our_dense):
                print("PASS: Dense block masks match!")
            else:
                diff = (ref_dense != our_dense).sum().item()
                total = ref_dense.numel()
                print(f"FAIL: {diff}/{total} blocks differ")
                w = torch.where(ref_dense != our_dense)
                for j in range(min(10, len(w[0]))):
                    coords = tuple(x[j].item() for x in w)
                    print(f"  Block {coords}: ref={ref_dense[coords].item()}, ours={our_dense[coords].item()}")
