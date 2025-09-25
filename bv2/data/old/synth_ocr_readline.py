#!/usr/bin/env python3
"""
"<|bos|>Read line {line number}<|sep|>{Image}<|sep|>{the line}<|eos|>"
"""

import numpy as np

import data.dpack as d
from data.pp import patchify, sanity_check
from data.synth_ocr import get_tiktoken, render, vis_data_wandb, vocab_size


def make_example(seed, args=None, ps=16, **kw):
    # Format is [BOS, prefix, SEP, full text, SEP, suffix, EOS].
    img, txt, _ = render(seed, ps=ps, **kw)

    lines = [" " + line for line in txt.split("\n")]
    ln = np.random.default_rng(seed).integers(len(lines))

    t = get_tiktoken()
    prefix = t.encode(f"Read line {ln+1}")
    suffix = t.encode(lines[ln])

    # TODO: also do a "resize to min/max" in the future.
    patches, positions = patchify(img, pw=ps, ph=ps)

    npre = len(prefix)
    nsuf = len(suffix)
    nreg = args.nreg
    nimg = len(patches)

    nbytes = max(d.nbytes_text(), d.nbytes_image(ph=ps, pw=ps), d.nbytes_reg())
    tokens = np.zeros((1 + npre + 1 + nimg + nreg + 1 + nsuf + 1, nbytes), np.uint8)

    txtpos = np.arange(1 + npre + 1 + 1 + nsuf + 1)
    d.pack_text([t.bos, prefix, t.sep], positions=txtpos[:1 + npre + 1], out=tokens[:1 + npre + 1])
    d.pack_image(patches, positions, out=tokens[1 + npre + 1 : -(nreg + 1 + nsuf + 1)])
    d.pack_regs(nreg, out=tokens[1 + npre + 1 + nimg : -(1 + nsuf + 1)])
    d.pack_text([t.sep, suffix, t.eos], positions=txtpos[-(1 + nsuf + 1):], out=tokens[-(1 + nsuf + 1):])

    return sanity_check({
        "tokens": tokens,

        # NOTE: cast to int64, because if nreg == 0, then [] causes float in np.r_
        "loss_weights": np.r_[0, [0] * npre, 0, [0] * nimg, [0] * nreg, 0, [1] * nsuf, 1].astype(np.int64),

        "attn_regions": np.r_[1, [1] * npre, 1, [1] * nimg, [1] * nreg, 1, [0] * nsuf, 0].astype(np.int64),

        # Our initial
        "attn_regions2": np.r_[1, [1] * npre, 1, [-1] * nimg, [1] * nreg, 1, [0] * nsuf, 0].astype(np.int64),

        # regonly
        # "attn_regions2": np.r_[-1, [-1] * npre, -1, [-1] * nimg, [1] * nreg, 1, [0] * nsuf, 0].astype(np.int64),

        # NOTE: for attn_regions, 0 = AR, >0 = dense region ID.
        "id": seed,
    })  # fmt: skip


from data.synth_ocr import vis_data_wandb
