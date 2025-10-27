"""
TextVQA dataset import script.

cd /checkpoint/rigi/data/textvqa
python ~/rigi/bv2/data/textvqa_import.py

train: 21953 images
val: 3166 images
test: 3289 images
"""

import io
import json
import random
import zipfile
from collections import defaultdict

import bagz


def convert(outname, inname, image_dir, shuffle_seed):
    with open(inname) as f:
        data = json.load(f)["data"]

    mtdata = defaultdict(lambda: {"qas": {}, "image_name": None, "id": None})

    for ex in data:
        image_id = ex["image_id"]
        if not mtdata[image_id]["id"]:
            mtdata[image_id].update({
                "id": image_id,
                "image_name": f"{image_id}.jpg",
            })
        mtdata[image_id]["qas"][ex["question_id"]] = (ex["question"], ex.get("answers", []))

    mtdata = list(mtdata.values())
    random.seed(shuffle_seed)
    random.shuffle(mtdata)

    with bagz.Writer(outname) as writer:
        for i, ex in enumerate(mtdata):
            print(f"\r{outname}: {i+1}/{len(mtdata)}", flush=True, end="")
            buf = io.BytesIO()

            with zipfile.ZipFile(buf, "w") as z:
                z.writestr("data.json", json.dumps(ex), zipfile.ZIP_LZMA, 9)
                z.write(f"{image_dir}/{ex['image_name']}", "image", zipfile.ZIP_STORED)

            writer.write(buf.getvalue())

    print(f"\nCompleted {outname}!")


if __name__ == "__main__":
    data_dir = "/checkpoint/rigi/data/textvqa"
    convert("train.bag", f"{data_dir}/TextVQA_0.5.1_train.json", f"{data_dir}/train_images", 42)
    convert("val.bag", f"{data_dir}/TextVQA_0.5.1_val.json", f"{data_dir}/train_images", 42)
    convert("test.bag", f"{data_dir}/TextVQA_0.5.1_test.json", f"{data_dir}/test_images", 42)
