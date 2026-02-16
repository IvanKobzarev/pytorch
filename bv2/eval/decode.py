"""
Decoding evaluator.

It explicitly supports decoding text modality only.
"""

import functools
import io
import json

import bv2.data.dpack as dpack
import bv2.utils as u
from bv2.eval.decode_lib import decoding_iterator


def run(predict_fn, ds, decode={}, **comms):
    pred_iter = decoding_iterator(predict_fn, ds, **decode, **comms)

    all_preds = {}
    for ex in pred_iter:
        if all(u.all_gather_object(obj=ex["done"])):
            break

        if u.about_to_get_killed():
            return  # Do not yield any metrics, we didn't finish!

        if not ex["done"]:

            # Decode prefix for visualization/debugging purposes
            prefix, _, mask_prefix = dpack.unpack_as_text(ex["packed_prefix_torch"])
            all_preds[f"{ex["id"]}/prefix"] = (prefix := ds.tt.decode(prefix[mask_prefix].cpu().numpy()))

            all_preds[f"{ex["id"]}/suffix"] = ds.tt.decode(ex["suffix"])
        print(".", end="", flush=True)

    if all_preds := u.gather_object_to(rank=0, obj=all_preds):
        all_preds = functools.reduce(lambda x, y: x | y, all_preds, {})
        buf = io.BytesIO()
        buf.write(json.dumps(all_preds, ensure_ascii=False, indent=2).encode('utf-8'))
        return {"predictions.json": buf}
