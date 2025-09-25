#!/usr/bin/env python3
"""
Needle in a haystack task:
"Is this word present: {query_word}" + Image -> "yes|no"

"""

import numpy as np

from data.dpack import nbytes_image, nbytes_text, pack_image, pack_text
from data.noun_vocab import VOCAB
from data.pp import patchify, sanity_check
from data.synth_ocr import get_tiktoken, render, vocab_size


def make_example(seed, args=None, ps=16, **kw):
    # Format is [BOS, prefix, SEP, img, SEP, suffix, EOS].
    assert getattr(args, "nreg", 0) == 0, "Registers unsupported for this task."

    img, _, words_in_image = render(seed, ps=ps, **kw)

    rng = np.random.default_rng(seed)
    if rng.random() < 0.5:
        query_word = rng.choice(words_in_image).item()
        answer = "yes"
    else:
        words_not_in_image = list(set(VOCAB) - set(words_in_image))
        query_word = rng.choice(words_not_in_image).item()
        answer = "no"
    question = f"Is this word present: {query_word}"

    t = get_tiktoken()
    prefix = np.array(t.encode(question))
    suffix = np.array(t.encode(answer))

    # TODO: also do a "resize to min/max" in the future.
    patches, positions = patchify(img, pw=ps, ph=ps)

    npre = 1 + len(prefix) + 1
    nsuf = 1 + len(suffix) + 1
    nimg = len(patches)

    nbytes = max(nbytes_text(), nbytes_image(ph=ps, pw=ps))
    tokens = np.zeros((npre + nimg + nsuf, nbytes), np.uint8)

    txtpos = np.arange(npre + nsuf)
    pack_text([t.bos, prefix, t.sep], positions=txtpos[:npre], out=tokens[:npre])
    pack_image(patches, positions, out=tokens[npre:-nsuf])
    pack_text([t.sep, suffix, t.eos], positions=txtpos[-nsuf:], out=tokens[-nsuf:])

    return sanity_check({
        "tokens": tokens,
        # no loss on sep after image.
        "loss_weights": np.r_[[0] * npre, [0] * (nimg+1), [1] * (nsuf-1)],
        "attn_regions": np.r_[[1] * npre, [1] * (nimg+1), [0] * (nsuf-1)],
        # NOTE: for attn_regions, 0 = AR, >0 = dense region ID.
        "id": seed,
    })  # fmt: skip


from data.synth_ocr import vis_data_wandb
