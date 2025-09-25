from itertools import count

import numpy as np

from bv2.data.common import infinite_random_exids  # isort: skip
from bv2.data.dpack import pack_text
from bv2.data.noun_vocab import VOCAB
from bv2.data.pp import sanity_check


def render(seed, tiktoken, *, min_nouns=128, max_nouns=256):
    rng = np.random.default_rng(seed)
    n_nouns = rng.integers(min_nouns, max_nouns + 1)

    sampled_nouns = rng.choice(VOCAB, size=n_nouns, replace=False)
    text = " ".join(sampled_nouns)
    tokens = np.array(tiktoken.encode(text))

    return tokens


class Dataset:
    def __init__(self, **kw):
        self.render_kw = kw

    def make_exids(self, *a, **kw):
        return infinite_random_exids(*a, epoch_size=150, **kw)

    def make_example(self, exid, epoch):
        t = _get_tiktoken()
        noun_tokens = render(exid, t, **self.render_kw)

        return sanity_check({
            "tokens": pack_text(np.r_[t.bos, noun_tokens, t.eos], positions="auto"),
            "loss_weights": np.r_[0, [1] * len(noun_tokens), 1],
            "attn_regions": np.zeros(2 + len(noun_tokens), int),
            # NOTE: for attn_regions, 0 = AR, >0 = dense region ID.
            "id": exid,
        })  # fmt: skip

    def vocab_size(self):
        return _get_tiktoken().n_vocab


def _get_tiktoken(first_N=10_000):
    import bv2.data.tokenizer

    return bv2.data.tokenizer.get_tiktoken(first_N=first_N)
