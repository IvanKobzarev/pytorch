"""
This script converts and re-shards just a single original shard.
There's originally 32 of them, so run 32 times in parallel.

Run like this:

python ~/rigi/bv2/data/deduped_code_import.py --inshard_idx 3

for inshard_idx ranging from 0 to 31 (inclusive).
I made a tmux script running them in panes in parallel, see end of file.

Statistics:

Random 1k samples stats. Numbers are min|mean/median|max:
    Size: [78|4388.2/1752|109538]
    Lines: [4|135.4/58|4189]

len(bagz.Reader("train@256.bag")) == 170492035
len(bagz.Reader("val@32.bag")) == 32768
"""

import argparse
import io
import json
import zipfile
from collections import deque
from statistics import mean, median

import bagz
import numpy as np

INDIR = "/datasets/llama/codegen/shuffled/deduped_code/"
OUTDIR = "/checkpoint/rigi/data/deduped_code/"


def zip_datum(txt, meta):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        # For this data, the compression tradeoffs are as follows. For 500k rows:
        # lzma-9:    93MB in 1076s
        # deflate-9: 92MB in  142s
        # store:    322MB in   53s
        z.writestr("txt.json", json.dumps(txt), zipfile.ZIP_DEFLATED, 9)
        z.writestr("meta.json", json.dumps(meta), zipfile.ZIP_DEFLATED, 9)
    return buf.getvalue()


def getfirst(d, *keys):
    for k in keys:
        if k in d:
            return d[k]


def normalize_meta(row):
    # There appears to be multiple different formats to be present. Guess.
    meta = {
        "path": getfirst(row, "path", "max_stars_repo_path"),
        "repo": getfirst(row, "repo_name", "max_stars_repo_name"),
    }

    if "licenses" in row:
        meta["licenses"] = row["licenses"]
    elif "license" in row:
        meta["licenses"] = [row["license"]]

    if v := getfirst(row, "lang", "language"):
        meta["lang"] = v

    if "max_stars_count" in row:
        meta["stars"] = row["max_stars_count"]

    if "max_issues_count" in row:
        meta["issues"] = row["max_issues_count"]

    if "max_forks_count" in row:
        meta["forks"] = row["max_forks_count"]

    if "is_minified_js" in row:
        meta["is_minjs"] = row["is_minified_js"]

    if len(meta) == 2:
        raise ValueError("Unknown type of entry:\n" + json.dumps(row, indent=1))
    return meta


def mmmm(l):
    return f"[{min(l)}|{mean(l):.1f}/{median(l):.0f}|{max(l)}]"


def convert(args):
    rng = np.random.default_rng(args.shuffle_seed)  # noqa:TID251 - This file is bv2-free.

    # First, select which lines to set aside for val. In the original data, each shard
    # has at least 5.3M rows, so this works.
    val_indices = set(rng.choice(5_000_000, args.nval, replace=False).tolist())

    assert args.shard_total % 32 == 0
    shard_factor = args.shard_total // 32
    first_shard = args.inshard_idx * shard_factor
    writers = [
        bagz.Writer(f"{OUTDIR}/train-{i:05d}-of-{args.shard_total:05d}.bag")
        for i in range(first_shard, first_shard + shard_factor)
    ]
    val_writer = bagz.Writer(f"{OUTDIR}/val-{args.inshard_idx:05d}-of-{32:05d}.bag")

    sizes, lines = deque(maxlen=1000), deque(maxlen=1000)  # Stats while we're at it.
    try:
        with open(f"{INDIR}/deduped_code.chunk.{args.inshard_idx:02d}.jsonl") as f:
            for i, raw_row in enumerate(f):
                writer = val_writer if i in val_indices else rng.choice(writers)
                row = json.loads(raw_row)
                if i % 100 == 0:
                    sizes.append(len(row["content"]))
                    lines.append(row["content"].count("\n"))
                    print(f"\rRow {i}", flush=True, end="")
                    print(f" Size: {mmmm(sizes):<24s}", flush=True, end="")
                    print(f" Lines: {mmmm(lines):<20s}", flush=True, end="")
                writer.write(zip_datum(txt=row["content"], meta=normalize_meta(row)))
    finally:
        for w in writers:
            w.close()
        val_writer.close()

    print("\nAll done!", flush=True)


if __name__ == "__main__":
    # fmt:off
    parser = argparse.ArgumentParser()
    parser.add_argument("--shuffle_seed", default=42, type=int,
                        help="The seed to use for shuffling the data.")
    parser.add_argument("--nval", default=32_768 // 32, type=int,
                        help="Number of examples to set aside for validation.")
    parser.add_argument("--shard_total", default=256, type=int,
                        help="Number of output shards to distribute ALL examples across.")
    parser.add_argument("--inshard_idx", default=0, type=int,
                        help="Number of the input shard.")
    args = parser.parse_args()
    # fmt:on

    convert(args)
