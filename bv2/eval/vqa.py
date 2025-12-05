import functools
import json
from io import BytesIO

import editdistance
import numpy as np

import bv2.utils as u
from bv2.eval.decode_lib import decoding_iterator


def run(predict_fn, ds, decode, **comms):
    """VQA evaluator."""

    anls, acc, acc_any, preds = [], [], [], {}
    for ex in decoding_iterator(predict_fn, ds, **decode, **comms):
        if all(u.all_gather_object(obj=ex["done"])):
            break

        if u.about_to_get_killed():
            return  # Do not yield any metrics, we didn't finish!

        pred = ds.tt.decode(ex["suffix"])
        preds[f"{ex["id"].item()}/suffix"] = pred

        gt = ds.ground_truth(ex["id"])

        assert len(gt['qas']) == 1
        question, answers = list(gt['qas'].values())[0]
        preds[f"{ex["id"].item()}/question"] = question
        preds[f"{ex["id"].item()}/answers"] = answers
        num_match = sum([ans == pred for ans in answers])
        acc.append(min(1.0, num_match / 3.0))
        acc_any.append(min(1.0, float(num_match)))
        anls.append(max([anls_metric(ans, pred) for ans in answers]))
        print(".", end="", flush=True)

    if res := u.gather_object_to(rank=0, obj=(acc, acc_any, anls, preds)):
        acc, acc_any, anls, preds = zip(*res)

        anls = np.concat(anls)
        acc = np.concat(acc)
        acc_any = np.concat(acc_any)

        preds = functools.reduce(lambda x, y: x | y, preds, {})
        buf = BytesIO()
        buf.write(json.dumps(preds, ensure_ascii=False, indent=2).encode('utf-8'))

        return {"anls": np.mean(anls), "acc": np.mean(acc), "acc_any": np.mean(acc_any), "predictions.json": buf}
    else:
        return {}


def anls_metric(target, prediction, r=0.5):
    """See https://arxiv.org/abs/1907.00490."""
    if target:
        edit_distance = editdistance.eval(target, prediction)
        normalized_ld = edit_distance / max(len(target), len(prediction))
        return 1.0 - normalized_ld if normalized_ld < r else 0.0
    else:
        return float(prediction == "")
