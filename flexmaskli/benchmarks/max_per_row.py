#!/usr/bin/env python
"""
Benchmark docmask max_per_row modes (dynamic/auto/fixed) and batchmask reference.

Usage:
    NCCL_SOCKET_IFNAME=lo python -m flexmaskli.benchmarks.max_per_row
    NCCL_SOCKET_IFNAME=lo python -m flexmaskli.benchmarks.max_per_row --gpu
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


def docmask_scenarios():
    scenarios = []

    def pack_docs(ntoks, rng, nmin, nmax, make_ar):
        di, ar = [], []
        pos, doc = 0, 0
        while ntoks - pos > nmin:
            dlen = rng.randint(nmin, min(nmax, ntoks - pos))
            if pos + dlen > ntoks:
                break
            di.extend([doc] * dlen)
            ar.extend(make_ar(rng, dlen))
            pos += dlen
            doc += 1
        while len(di) < ntoks:
            di.append(-1)
            ar.append(-1)
        return torch.tensor(ar, dtype=torch.int32), torch.tensor(di, dtype=torch.int64)

    def standard_ar(rng, dlen):
        nprefix = int(rng.uniform(0.3, 0.7) * dlen)
        return [1] * nprefix + [0] * (dlen - nprefix)

    def ar2_ar(_, dlen):
        nq = max(10, int(0.10 * dlen))
        nimg = max(10, int(0.40 * dlen))
        nreg = max(5, int(0.05 * dlen))
        return [1] * nq + [-1] * nimg + [1] * nreg + [0] * (dlen - nq - nimg - nreg)

    rng = np.random.RandomState(42)
    for ntoks in [8192, 32768, 131072, 524288]:
        for nmin, nmax, label in [(1024, 4096, "med"), (128, 512, "short"), (8192, 32768, "long")]:
            if nmin >= ntoks:
                continue
            ar, di = pack_docs(ntoks, rng, nmin, nmax, standard_ar)
            scenarios.append((f"doc-{ntoks//1024}k-{label}", ntoks, ar, di))

    rng2 = np.random.RandomState(99)
    for ntoks in [8192, 32768, 131072]:
        ar, di = pack_docs(ntoks, rng2, 1024, 4096, ar2_ar)
        scenarios.append((f"ar2-{ntoks//1024}k", ntoks, ar, di))
    return scenarios


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--block", type=int, default=128)
    args = parser.parse_args()

    from flexmaskli.batchmask_cpu import make_batchmask_cpu
    from flexmaskli.docmask_cpu import make_docmask_cpu

    print("Warming up numba...", end="", flush=True)
    ar_w = np.zeros((1, 1024), dtype=np.int64); ar_w[0, :300] = 1
    make_batchmask_cpu(1024, ar_w, BLOCK_SIZE=args.block)
    di_w = torch.zeros(8192, dtype=torch.int64); ar_dw = torch.zeros(8192, dtype=torch.int32)
    make_docmask_cpu(8192, ar_dw, di_w, BLOCK_SIZE=args.block, max_per_row="dynamic")
    make_docmask_cpu(8192, ar_dw, di_w, BLOCK_SIZE=args.block, max_per_row=None)
    make_docmask_cpu(8192, ar_dw, di_w, BLOCK_SIZE=args.block, max_per_row=8192 // args.block)
    print(" done.\n")

    print("=" * 110)
    print("BATCHMASK (reference, fixed NB width)")
    print("=" * 110)

    b_scenarios = batchmask_scenarios()
    hdr = f"{'scenario':<22s}  {'ms':>7s}  {'mem_KB':>8s}  {'NB':>3s}  {'w':>4s}"
    print(hdr)
    print("-" * len(hdr))
    for name, ntoks, ar in b_scenarios:
        NB = (ntoks + args.block - 1) // args.block
        t = bench_one(lambda ar=ar: make_batchmask_cpu(ntoks, ar, BLOCK_SIZE=args.block), args.runs)
        m = make_batchmask_cpu(ntoks, ar, BLOCK_SIZE=args.block)
        mem = index_memory_bytes(m) / 1024
        w = widths(m)[0]
        print(f"{name:<22s}  {t:>5.1f}ms  {mem:>6.1f}KB  {NB:>3d}  {w:>4d}")

    print()
    print("=" * 110)
    print("DOCMASK (max_per_row modes)")
    print("=" * 110)

    d_scenarios = docmask_scenarios()
    hdr = (
        f"{'scenario':<22s}  "
        f"{'dyn_ms':>7s}  {'auto_ms':>7s}  {'fixed_ms':>8s}  "
        f"{'dyn_KB':>8s}  {'auto_KB':>8s}  {'fixed_KB':>9s}  "
        f"{'NB':>5s}  {'dyn_w':>6s}  {'auto_w':>6s}"
    )
    print(hdr)
    print("-" * len(hdr))

    for name, ntoks, ar, di in d_scenarios:
        NB = (ntoks + args.block - 1) // args.block
        t_dyn = bench_one(lambda: make_docmask_cpu(ntoks, ar, di, BLOCK_SIZE=args.block, max_per_row="dynamic"), args.runs)
        t_auto = bench_one(lambda: make_docmask_cpu(ntoks, ar, di, BLOCK_SIZE=args.block, max_per_row=None), args.runs)
        t_fix = bench_one(lambda: make_docmask_cpu(ntoks, ar, di, BLOCK_SIZE=args.block, max_per_row=NB), args.runs)

        m_dyn = make_docmask_cpu(ntoks, ar, di, BLOCK_SIZE=args.block, max_per_row="dynamic")
        m_auto = make_docmask_cpu(ntoks, ar, di, BLOCK_SIZE=args.block, max_per_row=None)
        m_fix = make_docmask_cpu(ntoks, ar, di, BLOCK_SIZE=args.block, max_per_row=NB)

        mem_dyn = index_memory_bytes(m_dyn) / 1024
        mem_auto = index_memory_bytes(m_auto) / 1024
        mem_fix = index_memory_bytes(m_fix) / 1024
        w_dyn = widths(m_dyn)[0]
        w_auto = widths(m_auto)[0]
        print(
            f"{name:<22s}  "
            f"{t_dyn:>5.1f}ms  {t_auto:>5.1f}ms  {t_fix:>6.1f}ms  "
            f"{mem_dyn:>6.1f}KB  {mem_auto:>6.1f}KB  {mem_fix:>7.1f}KB  "
            f"{NB:>5d}  {w_dyn:>6d}  {w_auto:>6d}"
        )

    if args.gpu:
        print()
        print("=" * 110)
        print("GPU COMPILED FLEX_ATTENTION KERNEL TIME (docmask dynamic vs fixed)")
        print("=" * 110)
        from torch.nn.attention.flex_attention import flex_attention
        from flexmaskli.to_gpu import blockmask_to_gpu
        head_dim = 64

        gpu_dm = [(n, nt, ar, di) for n, nt, ar, di in d_scenarios
                  if any(x in n for x in ["doc-8k-med", "doc-32k-short", "doc-32k-med"])]
        print(f"{'scenario':<22s}  {'dynamic_ms':>10s}  {'fixed_ms':>8s}")
        print("-" * 46)
        for name, ntoks, ar, di in gpu_dm:
            q = torch.randn(1, 1, ntoks, head_dim, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(1, 1, ntoks, head_dim, device="cuda", dtype=torch.bfloat16)
            v = torch.randn(1, 1, ntoks, head_dim, device="cuda", dtype=torch.bfloat16)
            results = {}
            for label, kw in [("dynamic", {"max_per_row": "dynamic"}), ("fixed", {"max_per_row": (ntoks + args.block - 1) // args.block})]:
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
            print(f"{name:<22s}  {results['dynamic']:>8.1f}ms  {results['fixed']:>6.1f}ms")
