#!/usr/bin/env python3
"""
"What is the word after: {sequence}" + "Full text: {full text}" -> "next word"

Configurable sequence length: 3-10 consecutive words by default.
"""

import numpy as np

from data.dpack import pack_text
from data.pp import sanity_check
from data.synth_ocr import get_tiktoken, render, vocab_size
from data.synth_text_nih import vis_data_wandb


def make_example(seed, args=None, ps=16, min_preword_length=1, max_preword_length=1, **kw):
    # Format is [BOS, prefix, SEP, full text, SEP, suffix, EOS].
    assert getattr(args, "nreg", 0) == 0, "Registers unsupported for this task."

    _, _, words_in_image = render(seed, ps=ps, draw_img=False, unique=True, **kw)
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
    suffix = np.array(t.encode(" " + answer))
    full_text = np.array(t.encode(" " + " ".join(words_in_image)))  # with a leading " "

    npre = 1 + len(prefix) + 1
    nsuf = len(suffix) + 1
    ntxt = len(full_text) + 1

    return sanity_check({
        "tokens": pack_text(np.r_[t.bos, prefix, t.sep, full_text, t.sep, suffix, t.eos], positions="auto"),
        "loss_weights": np.r_[[0] * npre, [0] * ntxt, [1] * nsuf],
        "attn_regions": np.r_[[1] * npre, [1] * ntxt, [0] * nsuf],
        # NOTE: for attn_regions, 0 = AR, >0 = dense region ID.
        "id": seed,
    })  # fmt: skip
