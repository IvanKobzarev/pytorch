#!/usr/bin/env python3
"""
"What is the word after: {sequence}" + Image -> "next word"

Configurable sequence length: 3-10 consecutive words by default.
"""

import numpy as np

from data.dpack import nbytes_image, nbytes_text, pack_image, pack_text
from data.pp import patchify, sanity_check
from data.synth_ocr import get_tiktoken, render, vocab_size


def make_example(seed, args=None, ps=16, min_preword_length=1, max_preword_length=1, **kw):
    # Format is [BOS, prefix, SEP, img, SEP, suffix, EOS].
    assert getattr(args, "nreg", 0) == 0, "Registers unsupported for this task."

    img, txt, words_in_image = render(seed, unique=True, ps=ps, **kw)
    rng = np.random.default_rng(seed)

    assert len(words_in_image) > min_preword_length, "Not enough words in image."
    max_possible_length = min(max_preword_length, len(words_in_image) - 1)
    sequence_length = rng.integers(min_preword_length, max_possible_length + 1)

    start_idx = rng.integers(0, len(words_in_image) - sequence_length)
    sequence_words = words_in_image[start_idx : start_idx + sequence_length]
    answer = words_in_image[start_idx + sequence_length]
    sequence_text = " ".join(sequence_words)
    question = f"What is the word after: {sequence_text}"

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
