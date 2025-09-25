"""
Needle in a haystack task:
"Is this word present: {query_word}" + "Full text: {full text}" -> "yes|no"

"""

import numpy as np

import wandb
from data import dpack

from data.pp import sanity_check


def make_example(seed, args=None):
    assert getattr(args, "nreg", 0) == 0, "Registers unsupported for this task."
    rng = np.random.default_rng(seed)

    t1 = rng.integers(0, vocab_size())
    t2 = rng.integers(0, vocab_size())
    eq = rng.integers(0, 2)

    tokens = np.array([t1, (t1 if eq else t2), eq])
    tokens = dpack.pack_text(tokens, positions="auto")

    return sanity_check({
        "tokens": tokens,
        "loss_weights": np.array([0, 0, 1]),
        "attn_regions": np.array([1, 1, 0]),
        # NOTE: for attn_regions, 0 = AR, >0 = dense region ID.
        "id": seed,
    })  # fmt: skip


def vocab_size():
    return 1024


def vis_data_wandb(data):
    table = wandb.Table(["id", "prompt", "target"])

    tokens = data["tokens"].cpu()
    iseq = data["iseq"].cpu()
    loss_weights = data["loss_weights"].cpu()

    tokens, _, mask = dpack.unpack_as_text(tokens)
    tokens, mask = tokens.numpy(), mask.numpy()
    tokens, iseq, loss_weights = tokens[mask], iseq[mask], loss_weights[mask]

    for _id in range(iseq.max() + 1):
        ex_mask = iseq == _id

        ex_tokens = tokens[ex_mask]
        ex_weights = loss_weights[ex_mask]

        prompt = ex_tokens[ex_weights == 0.0]
        target = ex_tokens[ex_weights > 0.0]

        prompt = " ".join(str(x) for x in prompt)
        target = " ".join(str(x) for x in target)

        table.add_data(_id, prompt, target)

    return {"vis/data": table}
