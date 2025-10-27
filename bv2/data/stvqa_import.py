"""
STVQA dataset import script.

Official STVQA dataset files:
- Train: /checkpoint/rigi/data/stvqa/task3/train_task_3.json
- Test: /checkpoint/rigi/data/stvqa/task3/test_task_3.json
- Images: Extract /checkpoint/rigi/data/stvqa/ST-VQA.tar.gz first using:
  cd /checkpoint/rigi/data/stvqa && tar -xzf ST-VQA.tar.gz

High resolution COCO images train2014.zip:
  cd /checkpoint/rigi/data/stvqa && unzip train2014.zip

Train and validation sub-splits:
- Train images: /checkpoint/rigi/data/stvqa/imdb_subtrain.npy
- Val images: /checkpoint/rigi/data/stvqa/imdb_subval.npy

cd /checkpoint/rigi/data/docvqa
python ~/rigi/bv2/data/stvqa_import.py

train: 17028 images
val: 1893 images
test: 2971 images
"""

import io
import json
import os
import random
import zipfile

import bagz
import numpy as np


def load_split_image_sets(train_npy_path, val_npy_path):
    train_images = {item["image_path"] for item in np.load(train_npy_path, allow_pickle=True) if "image_path" in item}
    val_images = {item["image_path"] for item in np.load(val_npy_path, allow_pickle=True) if "image_path" in item}
    print(f"Train: {len(train_images)}, Val: {len(val_images)}, Overlap: {len(train_images & val_images)}")
    return train_images, val_images


def convert_split(outname, inname, split_name, shuffle_seed, image_set=None, image_dir=None):
    data = json.load(open(inname))["data"]
    filtered_data = [ex for ex in data if ex["file_path"] in image_set] if image_set else data
    print(f"\r{split_name}: {len(filtered_data)}/{len(data)} examples", flush=True, end="")

    mtdata = {}
    for ex in filtered_data:
        image_id = ex["file_path"].replace("coco-text/", "train2014/")
        if image_id not in mtdata:
            mtdata[image_id] = {
                "qas": {},
                "image_name": ex["file_name"],
                "image_path": image_id,
                "dataset": ex["dataset"],
                "set_name": split_name,
                "id": image_id,
            }
        mtdata[image_id]["qas"][ex["question_id"]] = (ex["question"], ex.get("answers", []))

    mtdata = list(mtdata.values())
    random.seed(shuffle_seed)
    random.shuffle(mtdata)

    with bagz.Writer(outname) as writer:
        for i, ex in enumerate(mtdata):
            print(f"\rWriting {split_name} ex {i+1}/{len(mtdata)}", flush=True, end="")
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as z:
                z.writestr("data.json", json.dumps(ex), zipfile.ZIP_LZMA, 9)
                path = os.path.join(image_dir, ex["image_path"])
                z.write(path, "image", zipfile.ZIP_STORED)
            writer.write(buf.getvalue())

    print(f"\n{split_name} split done!")


if __name__ == "__main__":
    data_dir = "/checkpoint/rigi/data/stvqa"
    train_images, val_images = load_split_image_sets(f"{data_dir}/imdb_subtrain.npy", f"{data_dir}/imdb_subval.npy")
    convert_split("train.bag", f"{data_dir}/task3/train_task_3.json", "train", 42, train_images, data_dir)
    convert_split("val.bag", f"{data_dir}/task3/train_task_3.json", "val", 42, val_images, data_dir)
    convert_split("test.bag", f"{data_dir}/task3/test_task_3.json", "test", 42, image_dir=data_dir)
