"""
Convert FineVision datasets from parquet format to sharded bagz format.

Usage:
python finevision_to_bagz.py --data_path /home/zhai/tmp/data/FineVision --output_base /home/zhai/tmp/data/t1
"""

import argparse
import io
import json
import os
import tempfile
import time
import zipfile
from pathlib import Path

import bagz
import numpy as np
import pyarrow.parquet as pq


def array_tolist(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: array_tolist(v) for k, v in obj.items()}
    return obj


def get_shard_count(input_count):
    if input_count <= 2:
        return 1
    elif input_count <= 8:
        return 4
    elif input_count <= 64:
        return 32
    return 256


def is_already_converted(output_dir, shard_count):
    for shard_idx in range(shard_count):
        bagz_filename = f"train-{shard_idx:05d}-of-{shard_count:05d}.bag"
        if not (output_dir / bagz_filename).exists():
            return False

    return True


def create_zipfile_data(data, image_bytes_list):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("data.json", json.dumps(array_tolist(data)), zipfile.ZIP_DEFLATED, 6)
        if len(image_bytes_list) == 1:
            z.writestr("image", image_bytes_list[0], zipfile.ZIP_STORED)
        elif len(image_bytes_list) > 1:
            for i, img_bytes in enumerate(image_bytes_list):
                z.writestr(f"images/{i}", img_bytes, zipfile.ZIP_STORED)

    return buf.getvalue()


def process_row_texts(texts):
    qas = {}
    for i, text_item in enumerate(texts):
        if text_item.keys() != {"user", "assistant"}:
            raise Exception(f"Unexpected keys: {text_item.keys()} from {text_item}")
        qas[i] = (text_item["user"], [text_item["assistant"]])

    if not qas:
        raise Exception(f"No Q&A found {texts}")

    return qas


def extract_image_bytes(images):
    return [img["bytes"] for img in images if "bytes" in img]


def process_row(row, parquet_path, idx, output_shards, shard_writers):
    qas = process_row_texts(row["texts"])

    has_images = len(row["images"]) > 0
    image_bytes_list = extract_image_bytes(row["images"]) if has_images else []

    data = {
        "qas": qas,
        "source": row["source"],
        "id": f"{parquet_path.stem}_{idx}",
        "relevance_ratings": row["relevance_ratings"],
        "formatting_ratings": row["formatting_ratings"],
    }

    if has_images:
        data["image_correspondence_ratings"] = row["image_correspondence_ratings"]
        data["visual_dependency_ratings"] = row["visual_dependency_ratings"]

    shard_idx = hash(data["id"]) % output_shards
    zipfile_data = create_zipfile_data(data, image_bytes_list)
    shard_writers[shard_idx].write(zipfile_data)


def convert_parquets_to_sharded_bagz(parquet_paths, output_dir):
    start_time = time.time()
    output_shards = get_shard_count(len(parquet_paths))

    print(
        f"Processing {output_dir.name}: {len(parquet_paths)} files -> {output_shards} shards"
    )

    if is_already_converted(output_dir, output_shards):
        print(f"Skipping {output_dir.name} (already converted)")
        return

    shard_writers = {}
    temp_and_final_files = []

    for shard_idx in range(output_shards):
        bagz_filename = f"train-{shard_idx:05d}-of-{output_shards:05d}.bag"
        temp_file = tempfile.NamedTemporaryFile(
            dir=output_dir, delete=False, suffix=f".{shard_idx}.tmp"
        )
        temp_path = Path(temp_file.name)
        temp_and_final_files.append((temp_path, output_dir / bagz_filename))
        shard_writers[shard_idx] = bagz.Writer(temp_path)

    total_rows = 0
    for parquet_path in parquet_paths:
        table = pq.read_table(parquet_path)

        for batch in table.to_batches():
            df = batch.to_pandas()
            for idx, row in df.iterrows():
                process_row(row, parquet_path, idx, output_shards, shard_writers)
                total_rows += 1

        elapsed = time.time() - start_time
        file_size_mb = parquet_path.stat().st_size / (1024 * 1024)
        print(f"    {parquet_path.name}: {file_size_mb:.2f}MB, {elapsed:.2f}s")

    for shard_idx in range(output_shards):
        shard_writers[shard_idx].close()

    for temp_path, final_path in temp_and_final_files:
        os.rename(temp_path, final_path)

    elapsed = time.time() - start_time
    print(f"  * Converted: {total_rows} rows, {elapsed:.2f}s")


def main():
    start_time = time.time()
    # fmt:off
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, type=str,
                        help="Directory to the finevision dataset with parquet files.")
    parser.add_argument("--output_base", required=True, type=str,
                        help="Output directory to the converted bag dataset.")
    args = parser.parse_args()
    # fmt:on

    processed_shards = 0
    # Only train splits available for FineVision.
    total_shards = len(list(Path(args.data_path).glob("*/train/*.parquet")))
    for dataset_dir in Path(args.data_path).iterdir():
        parquet_files = sorted(dataset_dir.glob("train/*.parquet"))
        if not parquet_files:
            continue

        print(f"[{processed_shards}/{total_shards}] {time.time() - start_time:.2f}s")
        output_dir = Path(args.output_base) / dataset_dir.name
        output_dir.mkdir(parents=True, exist_ok=True)

        convert_parquets_to_sharded_bagz(parquet_files, output_dir)
        processed_shards += len(parquet_files)

    total_elapsed = time.time() - start_time
    print(f"\nTotal processing time: {total_elapsed:.2f}s")


if __name__ == "__main__":
    main()
