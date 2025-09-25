"""
"Read line {line number}" + "{full text}" -> "{the line}"
"""

import numpy as np

import data.dpack as d
from data.pp import sanity_check
from data.synth_ocr import get_tiktoken, render, vocab_size
from data.synth_text_nih import vis_data_wandb


def make_example(seed, args=None, ps=16, **kw):
    # Format is [BOS, prefix, SEP, full text, SEP, suffix, EOS].
    _, txt, _ = render(seed, ps=ps, draw_img=False, **kw)

    # For more "uniform" tokenization, prepend a space before each line's first word.
    lines = [" " + line for line in txt.split("\n")]
    ln = np.random.default_rng(seed).integers(len(lines))

    t = get_tiktoken()
    prefix = t.encode(f"Read line {ln+1}")
    contex = t.encode("\n".join(lines))
    suffix = t.encode(lines[ln])

    npre = len(prefix)
    nctx = len(contex)
    nsuf = len(suffix)

    # Simpler case (code-wise) without registers:
    nreg = getattr(args, "nreg", 0)
    if nreg == 0:
        return sanity_check({
            "tokens": d.pack_text(np.r_[t.bos, prefix, t.sep, contex, t.sep, suffix, t.eos], positions="auto"),
            "loss_weights": np.r_[0, [0] * npre, 0, [0] * nctx, 0, [1] * nsuf, 1],
            "attn_regions": np.r_[1, [1] * npre, 1, [1] * nctx, 1, [0] * nsuf, 0],
            # NOTE: for attn_regions, 0 = AR, >0 = dense region ID.
            "id": seed,
        })  # fmt: skip

    # More complicated case with registers:
    nbytes = max(d.nbytes_text(), d.nbytes_image(ph=ps, pw=ps), d.nbytes_reg())
    tokens = np.zeros((1 + npre + 1 + nctx + 1 + nreg + 1 + nsuf + 1, nbytes), np.uint8)

    txtpos = np.arange(1 + npre + 1 + nctx + 1 + 1 + nsuf + 1)
    d.pack_text([t.bos, prefix, t.sep, contex, t.sep], positions=txtpos[:1 + npre + 1 + nctx + 1], out=tokens[:1 + npre + 1 + nctx + 1])
    d.pack_text([t.sep, suffix, t.eos], positions=txtpos[-(1 + nsuf + 1):], out=tokens[-(1 + nsuf + 1):])
    d.pack_regs(nreg, out=tokens[1 + npre + 1 + nctx + 1 : -(1 + nsuf + 1)])

    return sanity_check({
        "tokens": tokens,
        # no loss on registers and sep after registers.
        "loss_weights": np.r_[0, [0] * npre, 0, [0] * nctx, 0, [0] * nreg, 0, [1] * nsuf, 1],

        "attn_regions": np.r_[1, [1] * npre, 1, [1] * nctx, 1, [1] * nreg, 1, [0] * nsuf, 0],

        # Our initial
        "attn_regions2": np.r_[1, [1] * npre, 1, [-1] * nctx, -1, [1] * nreg, 1, [0] * nsuf, 0],

        # Keep only registers, nothing else:
        # "attn_regions2": np.r_[-1, [-1] * npre, -1, [-1] * nctx, -1, [1] * nreg, 1, [0] * nsuf, 0],

        # NOTE: for attn_regions, 0 = AR, >0 = dense region ID.

        "id": seed,
    })  # fmt: skip
