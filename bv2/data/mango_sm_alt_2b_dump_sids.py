"""Dump sample IDs from rigi_mango_2b04 into bagz sharded files.

Multiprocessing version — no GPU needed, configurable number of workers.
Every worker exhausts its airstore shard, then all IDs are globally
shuffled and split into non-overlapping train/val/val_mini.
A split with n=0 takes all remaining IDs.

Usage:
    python -m bv2.data.mango_sm_alt_2b_dump_sids --workers 128

Output: /checkpoint/rigi/data/mango_2b04_sids/{split}-{shard}-of-{total}.bag

Collected 2,209,156,586 IDs out of 2,209,186,816 total. skip_empty=0 (0.00%), skip_cv2=30,230 (0.00%). 66869s.
  val_mini: 5,120 entries -> /checkpoint/rigi/data/mango_2b04_sids/val_mini@1.bag
    [0]: 1!122853/s001228976316_e001228976331_c00016_b569344!6
    [-1]: 1!122853/s000242857914_e000242857929_c00016_b1745408!14
  val: 51,200 entries -> /checkpoint/rigi/data/mango_2b04_sids/val@4.bag
    [0]: 1!122853/s001800242534_e001800242549_c00016_b1796096!7
    [-1]: 1!122853/s001315498761_e001315498776_c00016_b1761280!12
  train: 2,209,100,266 entries -> /checkpoint/rigi/data/mango_2b04_sids/train@64.bag
    [0]: 1!122853/s002151227563_e002151227578_c00016_b793088!4
    [-1]: 1!122853/s001459091035_e001459091050_c00016_b674304!3

"""

import os
from multiprocessing import Process, Queue
from time import perf_counter

import cv2
import numpy as np
import sackli

import bv2.utils as u

DATASET = "rigi_mango_2b04"
OUT_DIR = "/checkpoint/rigi/data/mango_2b04_sids"

# After shuffle, split sequentially. n=0 means "take all remaining".
# At most one split should have n=0.
SPLITS = [
    # name        n          shards
    ("val_mini",  5_120,     1),
    ("val",       51_200,    4),
    ("train",     0,         64),
]


def _worker(rank, world_size, max_per_worker, queue):
    from airstore.client.airstore_tabular import AIRStorePathHandler
    from iopath.common.file_io import PathManager

    pm = PathManager()
    pm.register_handler(AIRStorePathHandler())

    label = f"{max_per_worker:,}" if max_per_worker else "all"
    print(f"[worker {rank}/{world_size}] Collecting {label} IDs from shard...", flush=True)
    t0 = perf_counter()

    ids = []
    skip_empty = 0
    skip_cv2 = 0
    with pm.opent(f"airstore://{DATASET}", rank=rank, world_size=world_size, shuffle_window=1) as stream:
        for row in stream:
            sid = row.get("__airstore_sample_id")
            alt = str(row.get("alt_text", "") or "")
            img = row.get("storage_handle", b"") or b""
            if not (sid and alt and img):
                skip_empty += 1
                continue
            if cv2.imdecode(np.frombuffer(bytes(img), np.uint8), cv2.IMREAD_COLOR) is None:
                skip_cv2 += 1
                continue
            ids.append(sid)
            n_total = len(ids) + skip_empty + skip_cv2
            if n_total % 100_000 == 0 and n_total > 0:
                elapsed = perf_counter() - t0
                if max_per_worker:
                    eta_s = (max_per_worker - len(ids)) / (len(ids) / elapsed)
                    print(f"[worker {rank}] {len(ids):,}/{max_per_worker:,} IDs, "
                          f"skip_empty={skip_empty:,} skip_cv2={skip_cv2:,} "
                          f"({n_total/elapsed:.0f}/s, ETA {eta_s/60:.0f}min)", flush=True)
                else:
                    print(f"[worker {rank}] {len(ids):,} IDs, "
                          f"skip_empty={skip_empty:,} skip_cv2={skip_cv2:,} "
                          f"({n_total/elapsed:.0f}/s)", flush=True)
            if max_per_worker and len(ids) >= max_per_worker:
                break

    dt = perf_counter() - t0
    print(f"[worker {rank}] Collected {len(ids):,} IDs, "
          f"skip_empty={skip_empty:,} skip_cv2={skip_cv2:,} in {dt:.0f}s", flush=True)
    queue.put((ids, skip_empty, skip_cv2))


def _write_split(name, ids, n_shards):
    writers = [sackli.Writer(os.path.join(OUT_DIR, f"{name}-{s:05d}-of-{n_shards:05d}.bag"))
               for s in range(n_shards)]
    for i, sid in enumerate(ids):
        writers[i % n_shards].write(sid.encode())
    for w in writers:
        w.close()

    fspec = os.path.join(OUT_DIR, f"{name}@{n_shards}.bag")
    reader = sackli.Reader(fspec)
    print(f"  {name}: {len(reader):,} entries -> {fspec}")
    print(f"    [0]: {bytes(reader[0]).decode()}")
    print(f"    [-1]: {bytes(reader[len(reader)-1]).decode()}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", "-w", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    world_size = args.workers

    # If all splits have fixed sizes, cap each worker to avoid scanning the whole dataset.
    n_total = sum(n for _, n, _ in SPLITS)
    has_open_split = any(n == 0 for _, n, _ in SPLITS)
    max_per_worker = 0 if has_open_split else -(-n_total // world_size)  # ceil div, 0 = no cap

    label = "all" if has_open_split else f"{n_total:,}"
    print(f"Collecting {label} IDs with {world_size} workers", flush=True)
    t0 = perf_counter()

    queue = Queue()
    procs = []
    for rank in range(world_size):
        p = Process(target=_worker, args=(rank, world_size, max_per_worker, queue))
        p.start()
        procs.append(p)

    results = [queue.get() for _ in range(world_size)]
    for p in procs:
        p.join()

    ids = [sid for worker_ids, _, _ in results for sid in worker_ids]
    total_skip_empty = sum(se for _, se, _ in results)
    total_skip_cv2 = sum(sc for _, _, sc in results)
    total_seen = len(ids) + total_skip_empty + total_skip_cv2

    dt = perf_counter() - t0
    n_fixed = sum(n for _, n, _ in SPLITS if n > 0)
    assert len(ids) >= n_fixed, f"Only got {len(ids):,} IDs, need at least {n_fixed:,} for fixed splits"
    print(f"Collected {len(ids):,} IDs out of {total_seen:,} total. "
          f"skip_empty={total_skip_empty:,} ({total_skip_empty/total_seen*100:.2f}%), "
          f"skip_cv2={total_skip_cv2:,} ({total_skip_cv2/total_seen*100:.2f}%). "
          f"{dt:.0f}s. Shuffling and splitting...", flush=True)

    # Global shuffle.
    rng = u.rng("dump_sids", args.seed)
    rng.shuffle(ids)

    # Split sequentially; n=0 means "take all remaining".
    os.makedirs(OUT_DIR, exist_ok=True)
    offset = 0
    for name, n, n_shards in SPLITS:
        if n == 0:
            split_ids = ids[offset:]
        else:
            split_ids = ids[offset:offset + n]
            assert len(split_ids) == n, f"{name}: wanted {n:,} but only {len(split_ids):,} left"
        _write_split(name, split_ids, n_shards)
        offset += len(split_ids)

    print("Done!", flush=True)


if __name__ == "__main__":
    main()
