"""
Decoding evaluator.

It explicitly supports decoding text modality only.
"""

import functools
import io
import json

import wandb

import bv2.data.dpack as dpack
import bv2.utils as u
from bv2.data.common import vis_image_text_unpack
from bv2.eval.decode_lib import decoding_iterator


def run(predict_fn, ds, ps=16, num_ex_to_vis=32, decode={}, **comms):
    pred_iter = decoding_iterator(predict_fn, ds, **decode, **comms)

    all_preds = {}
    vis_ex_to_wandb = []
    for ex in pred_iter:

        if all(u.all_gather_object(obj=ex["done"])):
            break

        if not ex["done"]:

            # Decode prefix for visualization/debugging purposes
            prefix, _, mask_prefix = dpack.unpack_as_text(ex["packed_prefix_torch"])
            all_preds[f"{ex["id"]}/prefix"] = (prefix := ds.tt.decode(prefix[mask_prefix].cpu().numpy()))

            all_preds[f"{ex["id"]}/suffix"] = (suffix := ds.tt.decode(ex["suffix"]))

            if num_ex_to_vis > 0:
                _, images = vis_image_text_unpack(ex["packed_prefix_torch"].cpu(), ph=ps, pw=ps)
                images = [wandb.Image(img) for img in images]
                vis_ex_to_wandb.append((ex["id"], prefix, images, suffix))
                num_ex_to_vis -= 1

    if all_preds := u.gather_object_to(rank=0, obj=all_preds):
        all_preds = functools.reduce(lambda x, y: x | y, all_preds, {})
        buf = io.BytesIO()
        buf.write(json.dumps(all_preds, ensure_ascii=False, indent=2).encode('utf-8'))

        vis = {}
        if vis_ex_to_wandb:
            table = wandb.Table(["id", "prefix", "images", "suffix"])
            for ex in vis_ex_to_wandb:
                table.add_data(*ex)
            vis = {"vis": table}

        return {"predictions.json": buf} | vis
