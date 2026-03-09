#!/usr/bin/env python3
"""
Comprehensive benchmark: compact=True vs compact=False for both batchmask and docmask.

Measures CPU mask creation time and memory for diverse scenarios.

Usage:
    NCCL_SOCKET_IFNAME=lo python -m flexlimaskli.benchmarks.compact_vs_fullwidth
    NCCL_SOCKET_IFNAME=lo python -m flexlimaskli.benchmarks.compact_vs_fullwidth --gpu
"""

import argparse
import time

import numpy as np
import torch


def bench_one(fn, runs=5, warmup=2):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(runs):
        t0 = time.monotonic_ns()
        fn()
        ts.append((time.monotonic_ns() - t0) / 1_000_000)
    return np.median(ts)


def index_memory_bytes(mask):
    total = 0
    for attr in ("kv_num_blocks", "kv_indices", "full_kv_num_blocks", "full_kv_indices",
                 "q_num_blocks", "q_indices", "full_q_num_blocks", "full_q_indices"):
        t = getattr(mask, attr)
        if t is not None:
            total += t.nelement() * t.element_size()
    return total


def widths(mask):
    return tuple(getattr(mask, a).shape[-1] for a in
                 ["kv_indices", "full_kv_indices", "q_indices", "full_q_indices"])


# ---- Batchmask scenarios ----

def batchmask_scenarios():
    scenarios = []
    for ntoks in [512, 1024, 2048, 4096, 8192]:
        for B in [4, 16, 32]:
            ar = np.zeros((B, ntoks), dtype=np.int64)
            for b in range(B):
                prefix = int(np.random.RandomState(42 + b).uniform(0.3, 0.7) * ntoks)
                ar[b, :prefix] = 1
            scenarios.append((f"decode-{ntoks}-B{B}", ntoks, ar))
    for ntoks in [512, 1024, 2048, 4096]:
        for B in [4, 16, 32]:
            ar = np.zeros((B, ntoks), dtype=np.int64)
            for b in range(B):
                rng = np.random.RandomState(42 + b)
                q_len = rng.randint(20, 80)
                img_len = rng.randint(100, min(400, ntoks - q_len - 20))
                reg_len = rng.randint(10, 30)
                ar[b, :q_len] = 1
                ar[b, q_len:q_len+img_len] = -1
                ar[b, q_len+img_len:q_len+img_len+reg_len] = 1
            scenarios.append((f"ar2-{ntoks}-B{B}", ntoks, ar))
    for ntoks in [1024, 4096]:
        for B in [8, 32]:
            ar = np.zeros((B, ntoks), dtype=np.int64)
            for b in range(B):
                if b < B // 2:
                    ar[b, :ntoks // 10] = 1
                else:
                    ar[b, :ntoks * 9 // 10] = 1
            scenarios.append((f"variable-{ntoks}-B{B}", ntoks, ar))
    return scenarios


# ---- Docmask scenarios ----

def docmask_scenarios():
    scenarios = []
    rng = np.random.RandomState(42)
    for ntoks in [8192, 32768, 131072, 524288]:
        for nmin, nmax, label in [(1024, 4096, "med"), (128, 512, "short"), (8192, 32768, "long")]:
            if nmin >= ntoks:
                continue
            di, ar = [], []
            pos, doc = 0, 0
            while ntoks - pos > nmin:
                dlen = rng.randint(nmin, min(nmax, ntoks - pos))
                if pos + dlen > ntoks:
                    break
                di.extend([doc] * dlen)
                nprefix = int(rng.uniform(0.3, 0.7) * dlen)
                ar.extend([1] * nprefix + [0] * (dlen - nprefix))
                pos += dlen
                doc += 1
            while len(di) < ntoks:
                di.append(-1)
                ar.append(-1)
            scenarios.append((f"doc-{ntoks//1024}k-{label}",
                              ntoks, torch.tensor(ar, dtype=torch.int32), torch.tensor(di, dtype=torch.int64)))
    return scenarios


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--block", type=int, default=128)
    args = parser.parse_args()

    from flexlimaskli.batchmask_cpu import make_batchmask_cpu
    from flexlimaskli.docmask_cpu import make_docmask_cpu

    # Warmup
    print("Warming up numba...", end="", flush=True)
    ar_w = np.zeros((1, 1024), dtype=np.int64); ar_w[0, :300] = 1
    make_batchmask_cpu(1024, ar_w, BLOCK_SIZE=args.block)
    make_batchmask_cpu(1024, ar_w, BLOCK_SIZE=args.block, compact=True)
    di_w = torch.zeros(8192, dtype=torch.int64); ar_dw = torch.zeros(8192, dtype=torch.int32)
    make_docmask_cpu(8192, ar_dw, di_w, BLOCK_SIZE=args.block)
    make_docmask_cpu(8192, ar_dw, di_w, BLOCK_SIZE=args.block, compact=True)
    print(" done.\n")

    # ---- Batchmask ----
    print("=" * 100)
    print("BATCHMASK (batched single-doc decode masks, B > 1)")
    print("=" * 100)

    b_scenarios = batchmask_scenarios()
    hdr = f"{'scenario':<22s}  {'full_ms':>7s}  {'comp_ms':>7s}  {'ratio':>6s}  {'full_KB':>8s}  {'comp_KB':>8s}  {'mem%':>6s}  {'NB':>3s}  {'comp_w':>6s}"
    print(hdr)
    print("-" * len(hdr))

    for name, ntoks, ar in b_scenarios:
        NB = (ntoks + args.block - 1) // args.block
        t_f = bench_one(lambda ar=ar: make_batchmask_cpu(ntoks, ar, BLOCK_SIZE=args.block), args.runs)
        t_c = bench_one(lambda ar=ar: make_batchmask_cpu(ntoks, ar, BLOCK_SIZE=args.block, compact=True), args.runs)
        m_f = make_batchmask_cpu(ntoks, ar, BLOCK_SIZE=args.block)
        m_c = make_batchmask_cpu(ntoks, ar, BLOCK_SIZE=args.block, compact=True)
        mem_f = index_memory_bytes(m_f) / 1024
        mem_c = index_memory_bytes(m_c) / 1024
        w_c = widths(m_c)[0]
        print(f"{name:<22s}  {t_f:>5.1f}ms  {t_c:>5.1f}ms  {t_c/t_f:>5.2f}x  {mem_f:>6.1f}KB  {mem_c:>6.1f}KB  {mem_c/mem_f*100:>5.1f}%  {NB:>3d}  {w_c:>6d}")

    # ---- Docmask ----
    print()
    print("=" * 100)
    print("DOCMASK (packed multi-doc training masks, B = 1)")
    print("=" * 100)

    d_scenarios = docmask_scenarios()
    hdr = f"{'scenario':<22s}  {'full_ms':>8s}  {'comp_ms':>8s}  {'ratio':>6s}  {'full_KB':>9s}  {'comp_KB':>9s}  {'mem%':>6s}  {'NB':>5s}  {'comp_w':>6s}"
    print(hdr)
    print("-" * len(hdr))

    for name, ntoks, ar, di in d_scenarios:
        NB = (ntoks + args.block - 1) // args.block
        t_f = bench_one(lambda: make_docmask_cpu(ntoks, ar, di, BLOCK_SIZE=args.block), args.runs)
        t_c = bench_one(lambda: make_docmask_cpu(ntoks, ar, di, BLOCK_SIZE=args.block, compact=True), args.runs)
        m_f = make_docmask_cpu(ntoks, ar, di, BLOCK_SIZE=args.block)
        m_c = make_docmask_cpu(ntoks, ar, di, BLOCK_SIZE=args.block, compact=True)
        mem_f = index_memory_bytes(m_f) / 1024
        mem_c = index_memory_bytes(m_c) / 1024
        w_c = widths(m_c)[0]
        print(f"{name:<22s}  {t_f:>6.1f}ms  {t_c:>6.1f}ms  {t_c/t_f:>5.2f}x  {mem_f:>7.1f}KB  {mem_c:>7.1f}KB  {mem_c/mem_f*100:>5.1f}%  {NB:>5d}  {w_c:>6d}")

    if args.gpu:
        print()
        print("=" * 100)
        print("GPU COMPILED FLEX_ATTENTION KERNEL TIME")
        print("=" * 100)
        from torch.nn.attention.flex_attention import flex_attention
        from flexlimaskli.to_gpu import blockmask_to_gpu
        head_dim = 64

        # Batchmask GPU
        print("\nBatchmask GPU:")
        gpu_bm = [(n, nt, ar) for n, nt, ar in b_scenarios
                   if any(x in n for x in ["decode-1024-B4", "decode-4096-B16",
                                             "ar2-1024-B16", "variable-4096-B32"])]
        print(f"{'scenario':<22s}  {'full_ms':>8s}  {'compact_ms':>10s}")
        print("-" * 46)
        for name, ntoks, ar in gpu_bm:
            B = ar.shape[0]
            q = torch.randn(B, 1, ntoks, head_dim, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(B, 1, ntoks, head_dim, device="cuda", dtype=torch.bfloat16)
            v = torch.randn(B, 1, ntoks, head_dim, device="cuda", dtype=torch.bfloat16)
            results = {}
            for label, kw in [("full", {}), ("compact", {"compact": True})]:
                mask = blockmask_to_gpu(make_batchmask_cpu(ntoks, ar, BLOCK_SIZE=args.block, **kw), "cuda")
                torch._dynamo.reset()
                cflex = torch.compile(flex_attention, dynamic=False, fullgraph=True)
                with torch.no_grad():
                    for _ in range(3):
                        cflex(q, k, v, block_mask=mask)
                torch.cuda.synchronize()
                ts = []
                with torch.no_grad():
                    for _ in range(args.runs):
                        torch.cuda.synchronize()
                        t0 = time.monotonic_ns()
                        cflex(q, k, v, block_mask=mask)
                        torch.cuda.synchronize()
                        ts.append((time.monotonic_ns() - t0) / 1_000_000)
                results[label] = np.median(ts)
            print(f"{name:<22s}  {results['full']:>6.1f}ms  {results['compact']:>8.1f}ms")

        # Docmask GPU
        print("\nDocmask GPU:")
        gpu_dm = [(n, nt, ar, di) for n, nt, ar, di in d_scenarios
                   if any(x in n for x in ["doc-8k-med", "doc-32k-short", "doc-32k-med"])]
        print(f"{'scenario':<22s}  {'full_ms':>8s}  {'compact_ms':>10s}")
        print("-" * 46)
        for name, ntoks, ar, di in gpu_dm:
            q = torch.randn(1, 1, ntoks, head_dim, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(1, 1, ntoks, head_dim, device="cuda", dtype=torch.bfloat16)
            v = torch.randn(1, 1, ntoks, head_dim, device="cuda", dtype=torch.bfloat16)
            results = {}
            for label, kw in [("full", {}), ("compact", {"compact": True})]:
                mask = blockmask_to_gpu(make_docmask_cpu(ntoks, ar, di, BLOCK_SIZE=args.block, **kw), "cuda")
                torch._dynamo.reset()
                cflex = torch.compile(flex_attention, dynamic=False, fullgraph=True)
                with torch.no_grad():
                    for _ in range(3):
                        cflex(q, k, v, block_mask=mask)
                torch.cuda.synchronize()
                ts = []
                with torch.no_grad():
                    for _ in range(args.runs):
                        torch.cuda.synchronize()
                        t0 = time.monotonic_ns()
                        cflex(q, k, v, block_mask=mask)
                        torch.cuda.synchronize()
                        ts.append((time.monotonic_ns() - t0) / 1_000_000)
                results[label] = np.median(ts)
            print(f"{name:<22s}  {results['full']:>6.1f}ms  {results['compact']:>8.1f}ms")
