"""
First, download and unzip the dataset from https://rrc.cvc.uab.es/?ch=17
Then, run this as `python convert.py` in the folder with the files.
This generates the `.bag` files to be copied wherever training data resides.

To run on fair-sc-3:
$ cd /checkpoint/rigi/data_orig/infovqa
$ python3 ~/rigi/bv2/data/infovqa_import.py

Statistics:

Number of images:
len(sackli.Reader("train.bag")) == 4406
len(sackli.Reader("test.bag")) == 579
len(sackli.Reader("val.bag")) == 500

Number of questions:
23946 == sum(len(json.loads(zipfile.ZipFile(io.BytesIO(i)).read("data.json"))["qas"]) for i in sackli.Reader("train.bag"))
 2801 == sum(len(json.loads(zipfile.ZipFile(io.BytesIO(i)).read("data.json"))["qas"]) for i in sackli.Reader("val.bag"))
 3288 == sum(len(json.loads(zipfile.ZipFile(io.BytesIO(i)).read("data.json"))["qas"]) for i in sackli.Reader("test.bag"))
"""

import argparse
import copy
import io
import json
import random
import zipfile

import sackli


def convert(outname, inname, args):
    print("\rReading QAs", flush=True, end="")
    with zipfile.ZipFile("infographicsvqa_qas.zip", "r") as z:
        with z.open(inname) as f:
            data = json.load(f)["data"]

    print("\rCollating", flush=True, end="")
    # The original data is "flat" in terms of question IDs, but multiple question
    # IDs may share the same image. We turn it into "multi-turn" image-keyed format here.
    mtdata = {}
    for ex in data:
        exid = ex["image_local_name"]
        if exid not in mtdata:
            mtdata[exid] = {
                # No answers on test, up to 8 answers on train.
                "qas": {ex["questionId"]: (ex["question"], ex.get("answers", []))},
                "image_name": ex["image_local_name"],  # 20471.jpeg
                "image_source": ex["image_url"],
                "ocr_name": ex["ocr_output_file"],
                "id": exid,  # Happens to be image_name, but we should always have one.
            }
        else:
            mtdata[exid]["qas"][ex["questionId"]] = (ex["question"], ex.get("answers", []))  # fmt: skip

    print("\rShuffling", flush=True, end="")
    mtdata = list(mtdata.values())
    random.seed(args.shuffle_seed)
    random.shuffle(mtdata)

    with sackli.Writer(outname) as writer:
        for i, ex in enumerate(mtdata if not args.flatten else flatten(mtdata)):
            print(f"\rWriting ex {i}", flush=True, end="")
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as z:
                z.writestr("data.json", json.dumps(ex), zipfile.ZIP_LZMA, 9)
                z.write(f"img/{ex["image_name"]}", "image", zipfile.ZIP_STORED)
                if args.ocr:
                    z.write(f"ocr/{ex["ocr_name"]}", "ocr.json", zipfile.ZIP_LZMA, 9)
            writer.write(buf.getvalue())

    print("\nAll done!", flush=True)


def flatten(mtdata):
    for ex in mtdata:
        for i, (q_id, qa) in enumerate(ex["qas"].items()):
            ex_single_q = copy.deepcopy(ex)
            ex_single_q["qas"] = {q_id: qa}
            ex_single_q["id"] = f"{ex['id']}_{i:02d}"
            yield ex_single_q


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr", action="store_true",
                        help="Store the (raw) OCR metadata.")
    parser.add_argument("--shuffle_seed", default=42, type=int,
                        help="The seed to use for shuffling the data.")
    parser.add_argument("--flatten", action="store_true", help="Flatten the dataset: a single question per image.")
    args = parser.parse_args()

    convert("val.bag", "infographicsVQA_val_v1.0_withQT.json", args)
    convert("test.bag", "infographicsVQA_test_v1.0.json", args)
    convert("train.bag", "infographicsVQA_train_v1.0.json", args)
