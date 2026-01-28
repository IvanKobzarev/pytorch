"""
"<|bos|>Question<|sep|>{Image}<|sep|>{answer}<|eos|>"
"""

import json
from io import BytesIO
from zipfile import ZipFile

import numpy as np
from PIL import Image

import bv2.data.dpack as d
import bv2.utils as u
from bv2.data.common import cycle_qas, get_bagz_reader, shuffled_iota_exids, vis_image_text_wandb
from bv2.data.pp import patchify, rand_resize, sanity_check
from bv2.data.tokenizer import get_tiktoken

PATH = "/checkpoint/rigi/data/{split}.bag"


class Dataset:
    def __init__(self, split, basepath=PATH, ps=16, max_patches=16_384, rand_resize=None, nreg=0, greyout_frac=0.0, tokenizer=None, qfmt="{q}", lower_q=False, lower_a=False, seed=0, epochs=None):
        self.reader = get_bagz_reader(basepath.format(split=split))
        self.ps = dict(ph=ps, pw=ps)
        self.max_patches = max_patches
        self.rand_resize = rand_resize
        self.nreg = nreg
        self.tt = get_tiktoken(**tokenizer or {})
        self.greyout_frac = greyout_frac
        self.data_seed = seed
        self.epochs = epochs
        self.qfmt = qfmt
        self.lower_q = lower_q
        self.lower_a = lower_a

    def ground_truth(self, exid):
        with ZipFile(BytesIO(self.reader[exid])) as zf:
            return json.load(zf.open("data.json"))

    def make_example(self, exid, epoch):
        with ZipFile(BytesIO(self.reader[exid])) as zf:
            data = json.load(zf.open("data.json"))
            img = Image.open(zf.open("image"))
            img.load()  # Ensure it's actually fully read.
            img = img if img.mode == "RGB" else img.convert("RGB")
            # NOTE: Not using "ocr.json" here yet.

        qid, question, answer = cycle_qas(data["qas"], epoch, seed=(self.data_seed, exid, "cycle_qas"))
        answer = answer.lower() if self.lower_a else answer
        question = question.lower() if self.lower_q else question
        question = self.qfmt.format(q=question)
        prefix = self.tt.encode(question)
        suffix = self.tt.encode(answer)

        key = (self.data_seed, exid, epoch)
        img = rand_resize(img, self.max_patches, key=(key, "resize"), **self.rand_resize or {}, **self.ps)
        if u.rng(key, "greyout").random() < self.greyout_frac:
            img.paste((128, 128, 128), box=(0, 0) + img.size)
        patches, positions = patchify(img, **self.ps)

        npre = len(prefix)
        nsuf = len(suffix)
        nimg = np.prod(patches.shape[:2])
        nreg = self.nreg

        nbytes = max(d.nbytes_text(), d.nbytes_image(**self.ps), d.nbytes_reg())
        tokens = np.zeros((1 + npre + 1 + nimg + nreg + 1 + nsuf + 1, nbytes), np.uint8)

        txtpos = np.arange(1 + npre + 1 + 1 + nsuf + 1)
        d.pack_text([self.tt.bos, prefix, self.tt.sep], positions=txtpos[:1 + npre + 1], out=tokens[:1 + npre + 1])  # fmt: skip
        d.pack_image(patches, positions, out=tokens[1 + npre + 1 : 1 + npre + 1 + nimg])
        d.pack_regs(nreg, out=tokens[1 + npre + 1 + nimg : -(1 + nsuf + 1)])
        d.pack_text([self.tt.sep, suffix, self.tt.eos], positions=txtpos[-(1 + nsuf + 1):], out=tokens[-(1 + nsuf + 1):])  # fmt: skip

        return sanity_check({
            "tokens": tokens,

            # NOTE: cast to int64, because if nreg == 0, then [] causes float in np.r_
            "loss_weights": np.r_[0, [0] * npre, 0, [0] * nimg, [0] * nreg, 0, [1] * nsuf, 1].astype(np.int64),

            "attn_regions": np.r_[1, [1] * npre, 1, [1] * nimg, [1] * nreg, 1, [0] * nsuf, 0].astype(np.int64),

            # Our initial
            "attn_regions2": np.r_[1, [1] * npre, 1, [-1] * nimg, [1] * nreg, 1, [0] * nsuf, 0].astype(np.int64),

            # regonly
            # "attn_regions2": np.r_[-1, [-1] * npre, -1, [-1] * nimg, [1] * nreg, 1, [0] * nsuf, 0].astype(np.int64),

            # NOTE: for attn_regions, 0 = AR, >0 = dense region ID.
            "id": exid,
        })  # fmt: skip

    def make_exids(self, **kw):
        yield from shuffled_iota_exids(len(self.reader), epochs=self.epochs, **kw)

    def vocab_size(self):
        return self.tt.n_vocab

    def vis_data_wandb(self, data):
        return vis_image_text_wandb(data, self.tt, **self.ps)
