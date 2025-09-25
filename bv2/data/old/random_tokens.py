import numpy as np
from data.dpack import pack_text

from data.pp import sanity_check


BOS, SEP, EOS = 0, 1, 2


def render(seed, *, prefix_min=256, prefix_max=1024, suffix_min=100, suffix_max=300):
    rng = np.random.default_rng(seed)
    n_prefix = rng.integers(prefix_min, prefix_max + 1)
    n_suffix = rng.integers(suffix_min, suffix_max + 1)
    prefix_tokens = rng.integers(3, vocab_size(), size=n_prefix)  # Reserve 0,1,2
    suffix_tokens = rng.integers(3, vocab_size(), size=n_suffix)
    return prefix_tokens, suffix_tokens


def make_example(seed, args=None, **kw):
    assert getattr(args, "nreg", 0) == 0, "Registers unsupported for this task."
    pre, suf = render(seed, **kw)  # prefix, suffix tokens
    return sanity_check(
        {
            "tokens": pack_text(np.r_[BOS, pre, SEP, suf, EOS], positions="auto"),
            "loss_weights": np.r_[0, [0] * len(pre), 0, [1] * len(suf), 1],
            "attn_regions": np.r_[1, [1] * len(pre), 1, [0] * len(suf), 0],
            # NOTE: for attn_regions, 0 = AR, >0 = dense region ID.
            "id": seed,
        }
    )


def vocab_size():
    return 32_768
