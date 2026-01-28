import json
from io import BytesIO
from zipfile import ZipFile

import numpy as np

import bv2.data.dpack as d
from bv2.data.common import get_bagz_reader, shuffled_iota_exids
from bv2.data.pp import sanity_check
from bv2.data.tokenizer import get_tiktoken

PATH = {
    "train": "/checkpoint/rigi/data/deduped_code/train@256.bag",
    "val": "/checkpoint/rigi/data/deduped_code/val@32.bag",
}


class Dataset:
    def __init__(self, split, first_N=float("inf"), tokenizer=None, seed=0, epochs=None):
        # Idea: here or in pp: randomize sub-seqlen, because many are >32k!
        self.reader = get_bagz_reader(PATH[split])
        self.tt = get_tiktoken(**tokenizer or {})
        self.first_N = first_N
        self.epochs = epochs

    def make_example(self, exid, epoch):
        with ZipFile(BytesIO(self.reader[exid])) as zf:
            data = json.load(zf.open("txt.json"))
            # NOTE: Not using "meta.json" here yet.

        toks = self.tt.encode(data)

        return sanity_check({
            "tokens": d.pack_text(np.r_[self.tt.bos, toks, self.tt.eos], positions="auto"),
            "loss_weights": np.r_[0, [1] * len(toks), 1],
            "attn_regions": np.zeros(2 + len(toks), int),  # 0 = AR
            "id": exid,
        })

    def make_exids(self, **kw):
        return shuffled_iota_exids(min(len(self.reader), self.first_N), epochs=self.epochs, **kw)

    def vocab_size(self):
        return self.tt.n_vocab

    def vis_data_wandb(self, data):
        import wandb  # Local import to not pollute tests with silly warnings.
        table = wandb.Table(["id", "text"])

        tokens = data["tokens"].cpu()
        iseq = data["iseq"].cpu()

        for _id in range(iseq.max() + 1):
            txt, _, mask = d.unpack_as_text(tokens[iseq == _id])
            txt = txt.numpy()[mask.numpy()]
            table.add_data(_id, self.tt.decode(txt))

        return table
