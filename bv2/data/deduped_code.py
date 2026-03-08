import json
from io import BytesIO
from zipfile import ZipFile

import numpy as np

import bv2.data.dpack as d
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
    def __init__(self, split, first_N=float("inf"), tokenizer=None, seed=0, epochs=None):
        # Idea: here or in pp: randomize sub-seqlen, because many are >32k!
        self.reader = get_sackli_reader(PATH[split])
        self.tt = get_tiktoken(**tokenizer or {})
        self.first_N = first_N
        self.epochs = epochs

    def make_example(self, exid, epoch):
        with ZipFile(BytesIO(self.reader[exid])) as zf:
            data = json.load(zf.open("txt.json"))
            # NOTE: Not using "meta.json" here yet.

        toks = self.tt.encode(data)

        return sanity_check({
            "toki": d.pack_text(np.r_[self.tt.bos, toks], positions="auto"),
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
