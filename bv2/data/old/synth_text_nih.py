#!/usr/bin/env python3
"""
Needle in a haystack task:
"Is this word present: {query_word}" + "Full text: {full text}" -> "yes|no"

"""

import numpy as np

import wandb
from data import dpack

from data.dpack import pack_text
from data.noun_vocab import VOCAB
from data.pp import sanity_check
from data.synth_ocr import get_tiktoken, render, vocab_size


def make_example(seed, args=None, **kw):
    # Format is [BOS, prefix, SEP, full text, SEP, suffix, EOS].
    assert getattr(args, "nreg", 0) == 0, "Registers unsupported for this task."

    _, _, words_in_image = render(seed, draw_img=False, **kw)

    rng = np.random.default_rng(seed)
    if rng.random() < 0.5:
        query_word = rng.choice(words_in_image).item()
        answer = "yes"
    else:
        words_not_in_image = list(set(VOCAB) - set(words_in_image))
        query_word = rng.choice(words_not_in_image).item()
        answer = "no"

    # Create question and answer
    question = f"Is this word present: {query_word}"

    t = get_tiktoken()
    prefix = np.array(t.encode(question))
    suffix = np.array(t.encode(answer))
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


def vis_example(tokens, loss_weights):
    txt, _, mask = dpack.unpack_as_text(tokens)
    txt, mask = txt.numpy(), mask.numpy()
    txt = txt[mask]
    loss_weights = loss_weights[mask]

    prompt = get_tiktoken().decode(txt[loss_weights == 0])
    prompt_tokens = txt[loss_weights == 0]
    target = get_tiktoken().decode(txt[loss_weights != 0])
    target_tokens = txt[loss_weights != 0]

    return prompt, prompt_tokens, target, target_tokens


def vis_data_wandb(data):
    table = wandb.Table(["id", "prompt", "target", "prompt tokens", "target tokens"])

    tokens = data["tokens"].cpu()
    iseq = data["iseq"].cpu()
    loss_weights = data["loss_weights"].cpu()

    for _id in range(iseq.max() + 1):
        ex_mask = iseq == _id
        ex_tokens = tokens[ex_mask]
        ex_weights = loss_weights[ex_mask]
        prompt, ptok, target, ttok = vis_example(ex_tokens, ex_weights)
        ptok = " ".join(str(x.item()) for x in ptok)
        ttok = " ".join(str(x.item()) for x in ttok)
        table.add_data(_id, prompt, target, ptok, ttok)

    return {"vis/data": table}
