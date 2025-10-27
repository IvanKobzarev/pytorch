from itertools import count

import numpy as np

from bv2.data.common import infinite_random_exids  # isort: skip
from bv2.data.dpack import pack_text
from bv2.data.noun_vocab import VOCAB
from bv2.data.pp import sanity_check
from bv2.data.tokenizer import get_tiktoken


def render(seed, tiktoken, *, min_nouns=128, max_nouns=256):
    rng = np.random.default_rng(seed)
    n_nouns = rng.integers(min_nouns, max_nouns + 1)

    sampled_nouns = rng.choice(VOCAB, size=n_nouns, replace=False)
    text = " ".join(sampled_nouns)
    tokens = np.array(tiktoken.encode(text))

    return tokens


class Dataset:
    def __init__(self, tokenizer={}, **kw):
        self.tt = get_tiktoken(**tokenizer)
        self.render_kw = kw

    def make_exids(self, *a, **kw):
        return infinite_random_exids(*a, epoch_size=150, **kw)

    def make_example(self, exid, epoch):
        noun_tokens = render(exid, self.tt, **self.render_kw)

        return sanity_check({
            "tokens": pack_text(np.r_[self.tt.bos, noun_tokens, self.tt.eos], positions="auto"),
            "loss_weights": np.r_[0, [1] * len(noun_tokens), 1],
            "attn_regions": np.zeros(2 + len(noun_tokens), int),
            # NOTE: for attn_regions, 0 = AR, >0 = dense region ID.
            "id": exid,
        })  # fmt: skip

    def vocab_size(self):
        return self.tt.n_vocab
