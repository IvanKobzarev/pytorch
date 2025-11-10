import numpy as np

import bv2.utils as u
from bv2.data.common import infinite_random_exids
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
    def __init__(self, tokenizer=None, seed=0, **kw):
        self.ttkw = tokenizer or {}
        self.render_kw = kw

    def make_exids(self, *a, **kw):
        return infinite_random_exids(*a, epoch_size=150, **kw)

    def make_example(self, exid, epoch):
        noun_tokens = render((exid, "render"), self.tt, **self.render_kw)

        return sanity_check({
            "tokens": pack_text(np.r_[self.tt.bos, noun_tokens, self.tt.eos], positions="auto"),
            "loss_weights": np.r_[0, [1] * len(noun_tokens), 1],
            "attn_regions": np.zeros(2 + len(noun_tokens), int),
            # NOTE: for attn_regions, 0 = AR, >0 = dense region ID.
            "id": exid,
        })  # fmt: skip

    @property  # Not a cached_property because we don't want to pickle/unpickle tokenizer.
    def tt(self):
        return get_tiktoken(**self.ttkw)  # But this is functools.cache'd per process.

    def vocab_size(self):
        return self.tt.n_vocab
