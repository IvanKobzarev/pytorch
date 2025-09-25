"""
FineVision Dataset Downloader

with-proxy pip install datasets
with-proxy python data/finevision_downloader.py --download_dir=/home/zhai/tmp/data/FineVision
"""

import argparse
from collections import defaultdict

from huggingface_hub import HfApi, snapshot_download


def main():
    # fmt:off
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", default="HuggingFaceM4/FineVision", type=str,
                        help="Huggingface dataset repo name.")
    parser.add_argument("--download_dir", required=True, type=str,
                        help="Directory to download the dataset.")
    args = parser.parse_args()
    # fmt:on

    api = HfApi()
    files = api.list_repo_files(repo_id=args.repo_id, repo_type="dataset")
    parquet_files = [f for f in files if f.endswith(".parquet") and "/" in f]

    subset_stats = defaultdict(int)
    for filepath in parquet_files:
        if "/" in filepath and not filepath.startswith("."):
            subset = filepath.split("/")[0]
            subset_stats[subset] += 1

    sorted_subsets = sorted(subset_stats.items(), key=lambda x: (x[1], x[0]))

    START = 0
    END = 140  # Fit to a single devbox, ~420GB size.
    download_subsets = [subset for subset, _ in sorted_subsets[START:END]]
    print(download_subsets)

    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision="refs/convert/parquet",  # parquet data revision
        local_dir=args.download_dir,
        local_dir_use_symlinks=False,
        allow_patterns=[f"{s}/*" for s in download_subsets],
        max_workers=1,  # actually runs smoother than larger numbers
    )


if __name__ == "__main__":
    main()
