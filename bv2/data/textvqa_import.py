"""
TextVQA dataset import script.

cd /checkpoint/rigi/data_orig/textvqa
python ~/rigi/bv2/data/textvqa_import.py

train: 21953 images
val: 3166 images
test: 3289 images
"""

import argparse
import copy
import io
import json
import random
import zipfile
from collections import defaultdict

import sackli


def convert(outname, inname, image_dir, args):
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
    random.seed(args.shuffle_seed)
    random.shuffle(mtdata)

    with sackli.Writer(outname) as writer:
        for i, ex in enumerate(mtdata if not args.flatten else flatten(mtdata)):
            print(f"\r{outname}: {i+1}/{len(mtdata)}", flush=True, end="")
            buf = io.BytesIO()

            with zipfile.ZipFile(buf, "w") as z:
                z.writestr("data.json", json.dumps(ex), zipfile.ZIP_LZMA, 9)
                z.write(f"{image_dir}/{ex['image_name']}", "image", zipfile.ZIP_STORED)

            writer.write(buf.getvalue())

    print(f"\nCompleted {outname}!")


def flatten(mtdata):
    for ex in mtdata:
        for i, (q_id, qa) in enumerate(ex["qas"].items()):
            ex_single_q = copy.deepcopy(ex)
            ex_single_q["qas"] = {q_id: qa}
            ex_single_q["id"] = f"{ex['id']}_{i:02d}"
            yield ex_single_q


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shuffle_seed", default=42, type=int,
                        help="The seed to use for shuffling the data.")
    parser.add_argument("--flatten", action="store_true", help="Flatten the dataset: a single question per image.")
    args = parser.parse_args()

    convert("train.bag", "TextVQA_0.5.1_train.json", "train_images", args)
    convert("val.bag", "TextVQA_0.5.1_val.json", "train_images", args)
    convert("test.bag", "TextVQA_0.5.1_test.json", "test_images", args)
