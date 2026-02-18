import numpy as np

import bv2.utils as u
from bv2.data.common import random_exids
from bv2.data.dpack import pack_text
from bv2.data.noun_vocab import VOCAB
from bv2.data.pp import sanity_check
from bv2.data.tokenizer import get_tiktoken


def render(seed, tiktoken, *, min_nouns=128, max_nouns=256):
    n_nouns = u.rng(seed, "n_nouns").integers(min_nouns, max_nouns + 1)

    sampled_nouns = u.rng(seed, "nouns").choice(VOCAB, size=n_nouns, replace=False)
    text = " ".join(sampled_nouns)
    tokens = np.array(tiktoken.encode(text))

    return tokens


class Dataset:
    def __init__(self, tokenizer=None, seed=0, n=None, **kw):
        self.tt = get_tiktoken(**tokenizer or {})
        self.render_kw = kw
        self.data_seed = seed
        self.n = n

    def make_exids(self, **kw):
        yield from random_exids(n=self.n, **kw)

    def make_example(self, exid):
        noun_tokens = render((self.data_seed, exid, "render"), self.tt, **self.render_kw)

        return sanity_check({
            "toki": pack_text(np.r_[self.tt.bos, noun_tokens], positions="auto"),
            "toko": pack_text(np.r_[noun_tokens, self.tt.eos], positions="zero"),
            "lowe": np.ones(len(noun_tokens) + 1, np.float32),
            "attn_regions": np.zeros(1 + len(noun_tokens), int),
            # NOTE: for attn_regions, 0 = AR, >0 = dense region ID.
            "ndatatoks": len(noun_tokens),
            "id": exid,
        })  # fmt: skip

    def vocab_size(self):
        return self.tt.n_vocab
