"""
DocVQA dataset import script.

cd /checkpoint/rigi/data/docvqa
python ~/rigi/bv2/data/docvqa_import.py

train: 10194 images
val: 1286 images
test: 1287 images
"""

import argparse
import io
import json
import random
import zipfile
from collections import defaultdict
from pathlib import Path

import bagz


def convert(outname, inname, args):
    docvqa_path = Path("/checkpoint/rigi/data/docvqa")

    with open(docvqa_path / inname) as f:
        data = json.load(f)["data"]

    mtdata = defaultdict(lambda: {"qas": {}, "id": None, "image_name": None})

    for ex in data:
        img_name = ex["image"]
        if not mtdata[img_name]["id"]:
            mtdata[img_name].update({
                "id": ex["docId"],
                "image_name": img_name,
                "ucsf_document_id": ex.get("ucsf_document_id", ""),
                "ucsf_document_page_no": ex.get("ucsf_document_page_no", ""),
            })
        mtdata[img_name]["qas"][ex["questionId"]] = (ex["question"], ex.get("answers", []))

    mtdata = list(mtdata.values())
    random.seed(args.shuffle_seed)
    random.shuffle(mtdata)

    with bagz.Writer(outname) as writer:
        for i, ex in enumerate(mtdata):
            print(f"\r{outname}: {i+1}/{len(mtdata)}", flush=True, end="")
            buf = io.BytesIO()

            with zipfile.ZipFile(buf, "w") as z:
                z.writestr("data.json", json.dumps(ex), zipfile.ZIP_LZMA, 9)
                z.write(str(docvqa_path / ex["image_name"]), "image", zipfile.ZIP_STORED)

                if args.ocr:
                    ocr_name = ex["image_name"].replace(".png", ".json").replace(".jpg", ".json")
                    z.write(str(docvqa_path / "spdocvqa_ocr" / ocr_name), "ocr.json", zipfile.ZIP_LZMA, 9)

            writer.write(buf.getvalue())

    print(f"\nCompleted {outname}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr", action="store_true", help="Include OCR metadata")
    parser.add_argument("--shuffle_seed", default=42, type=int, help="Shuffle seed")
    args = parser.parse_args()

    convert("train.bag", "train_v1.0_withQT.json", args)
    convert("val.bag", "val_v1.0_withQT.json", args)
    convert("test.bag", "test_v1.0.json", args)
