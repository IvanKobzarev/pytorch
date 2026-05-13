import json
from io import BytesIO
from itertools import pairwise
from zipfile import ZipFile

import numpy as np
import regex

import bv2.data.dpack as d
import bv2.utils as u
from bv2.data.common import get_sackli_reader, shuffled_iota_exids
from bv2.data.pp import sanity_check
from bv2.data.tokenizer import get_tiktoken

PATH = {
    "train": "/checkpoint/rigi/data/deduped_code/train@256.bag",
    "val": "/checkpoint/rigi/data/deduped_code/val@32.bag",
    "codewall_train_0.25M": "/checkpoint/rigi/data/codewall/0.25M_256_in_64k/train.bag",
    "codewall_train_0.5M": "/checkpoint/rigi/data/codewall/0.5M_256_in_64k/train.bag",
    "codewall_train_1M": "/checkpoint/rigi/data/codewall/1M_256_in_64k/train.bag",
    "codewall_train_2M": "/checkpoint/rigi/data/codewall/2M_256_in_64k/train.bag",
    "codewall_train_4M": "/checkpoint/rigi/data/codewall/4M_256_in_64k/train.bag",
    "codewall_train_16M": "/checkpoint/rigi/data/codewall/16M_256_in_64k/train.bag",
    "codewall_val": "/checkpoint/rigi/data/codewall/val_256_in_64k/val.bag",
}


class Dataset:
    def __init__(self, split, first_N=float("inf"), tokenizer=None, seed=0, epochs=None, cache=False,
                 bpe_drop_p=0.0, bpe_drop_frac=0.0):
        # Idea: here or in pp: randomize sub-seqlen, because many are >32k!
        self.reader = get_sackli_reader(PATH[split], cache)
        extras = ("bos_drop",) if bpe_drop_frac > 0 else ()
        self.tt = get_tiktoken(**(tokenizer or {}), extras=extras)
        self.first_N = first_N
        self.epochs = epochs
        self.seed = seed
        self.bpe_drop_p = bpe_drop_p
        self.bpe_drop_frac = bpe_drop_frac
        if bpe_drop_frac > 0:
            self.ranks = self.tt.mergeable_ranks
            self._pat = regex.compile(self.tt.pat_str)

    def _encode_with_dropout(self, text, rng):
        # Per pretoken drop here. To explore per merge drop and per sequence drop variants.
        result = []
        for word in self._pat.findall(text):
            parts = [bytes([b]) for b in word.encode("utf-8")]
            skipped = set()
            while len(parts) > 1:
                best = min((r for a, b in pairwise(parts)
                            if (r := self.ranks.get(a + b)) is not None and r not in skipped),
                           default=None)
                if best is None:
                    break
                if rng.random() < self.bpe_drop_p:
                    skipped.add(best)
                    continue
                merged, i = [], 0
                while i < len(parts):
                    if i + 1 < len(parts) and self.ranks.get(parts[i] + parts[i + 1]) == best:
                        merged.append(parts[i] + parts[i + 1])
                        i += 2
                    else:
                        merged.append(parts[i])
                        i += 1
                parts = merged
            result.extend(self.ranks[p] for p in parts)
        return result

    def make_example(self, exid, epoch):
        with ZipFile(BytesIO(self.reader[exid])) as zf:
            data = json.load(zf.open("txt.json"))
            # NOTE: Not using "meta.json" here yet.

        rng = u.rng(self.seed, "bpe_drop", exid, epoch)
        used_drop = rng.random() < self.bpe_drop_frac
        toks = self._encode_with_dropout(data, rng) if used_drop else self.tt.encode(data)
        first = self.tt.bos_drop if used_drop else self.tt.bos

        return sanity_check({
            "toki": d.pack_text(np.r_[first, toks], positions="auto"),
            "toko": d.pack_text(np.r_[toks, self.tt.eos], positions="zero"),
            "lowe": np.ones(len(toks) + 1, np.float32),
            "attn_regions": np.zeros(1 + len(toks), int),  # 0 = AR
            "ndatatoks": len(toks),
            "id": exid,
        })

    def make_exids(self, **kw):
        return shuffled_iota_exids(min(len(self.reader), self.first_N), epochs=self.epochs, **kw)

    def vocab_size(self):
        return self.tt.n_vocab
