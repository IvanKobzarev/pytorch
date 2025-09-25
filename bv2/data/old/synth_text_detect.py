"""
"On which {line|column|location} is the word {word}" + "{full text}" -> "{line|column|line column}"
"""

import numpy as np

import data.dpack as d
from data.pp import sanity_check
from data.synth_ocr import get_tiktoken, render, vocab_size
from data.synth_text_nih import vis_data_wandb


def make_example(seed, args=None, mode="row", **kw):
    # Format is [BOS, prefix, SEP, full text, SEP, suffix, EOS].
    _, txt, _ = render(seed, unique=True, draw_img=False, **kw)

    mode = getattr(args, "data_mode", mode)

    rng = np.random.default_rng(seed)

    lines = txt.split("\n")
    i = rng.choice(len(lines))
    words = lines[i].split()
    j = rng.choice(len(words))

    t = get_tiktoken()
    prefix = t.encode(f"On which {mode} is the word {words[j]}")
    contex = t.encode("\n".join([" " + l for l in lines]))  # Uniform tokenization.
    suffix = t.encode(dict(row=f"{i}", col=f"{j}", loc=f"{i} {j}", loc2=f"{j} {i}")[mode])

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
    nbytes = max(d.nbytes_text(), d.nbytes_reg())
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
