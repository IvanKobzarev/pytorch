"""
"<|bos|>Question<|sep|>{Image}<|sep|>{answer}<|eos|>"
"""

import json
from io import BytesIO
from zipfile import ZipFile

import cv2
import numpy as np

import bv2.data.dpack as d
import bv2.utils as u
from bv2.data import pp
from bv2.data.common import cycle_qas, get_sackli_reader, iota_exids, shuffled_iota_exids
from bv2.data.tokenizer import get_tiktoken

PATH = "/checkpoint/rigi/data/{split}.bag"


class Dataset:
    def __init__(self, split, basepath=PATH, ps=16, max_patches=16_384, rand_max_patches=None, nreg=0, greyout_frac=0.0, tokenizer=None, qfmt="{q}", lower_q=False, lower_a=False, seed=0, epochs=None):
        self._name = f"vqa({split})"
        self.reader = get_sackli_reader(basepath.format(split=split))
        self.ps = dict(ph=ps, pw=ps)
        self.max_patches = max_patches
        self.rand_max_patches = rand_max_patches or {}
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

    def make_example(self, exid, epoch=0):
        with ZipFile(BytesIO(self.reader[exid])) as zf:
            data = json.load(zf.open("data.json"))
            img = cv2.imdecode(np.frombuffer(zf.open("image").read(), np.uint8), cv2.IMREAD_COLOR)
            img = img[:, :, ::-1]  # BGR -> RGB
            # NOTE: Not using "ocr.json" here yet.

        qid, question, answer = cycle_qas(data["qas"], epoch, seed=(self.data_seed, exid, "cycle_qas"))
        answer = answer.lower() if self.lower_a else answer
        question = question.lower() if self.lower_q else question
        question = self.qfmt.format(q=question)
        prefix = self.tt.encode(question)
        suffix = self.tt.encode(answer)

        key = (self.data_seed, exid, epoch)
        img = pp.reasonable_resize(img, pp.rand_max_patches(
            img.shape[:2], self.max_patches, key=(key, "resize"), **self.rand_max_patches, **self.ps))
        if u.rng(key, "greyout").random() < self.greyout_frac:
            img[...] = 128
        patches, positions = pp.patchify(img, **self.ps)

        npre = len(prefix)
        nsuf = len(suffix)
        nimg = patches.shape[0] * patches.shape[1]
        nreg = self.nreg

        nbytes = max(d.nbytes_text(), d.nbytes_image(**self.ps), d.nbytes_reg())
        tokens = np.zeros((1 + npre + 1 + nimg + nreg + 1 + nsuf + 1, nbytes), np.uint8)

        txtpos = np.arange(1 + npre + 1 + 1 + nsuf + 1)
        d.pack_text([self.tt.bos, prefix, self.tt.sep], positions=txtpos[:1 + npre + 1], out=tokens[:1 + npre + 1])  # fmt: skip
        d.pack_image(patches, positions, out=tokens[1 + npre + 1 : 1 + npre + 1 + nimg])
        d.pack_regs(nreg, out=tokens[1 + npre + 1 + nimg : -(1 + nsuf + 1)])
        d.pack_text([self.tt.sep, suffix, self.tt.eos], positions=txtpos[-(1 + nsuf + 1):], out=tokens[-(1 + nsuf + 1):])  # fmt: skip

        # TODO: Actually we could have `toko` be only non-packed text => smaller and faster.
        example = {
            "toki": tokens[..., :-1, :],
            "toko": tokens[..., 1:, :],
            "lowe": np.r_[[0] * npre, 0, [0] * nimg, [0] * nreg, 0, [1] * nsuf, 1].astype(np.float32),
            # NOTE: for attn_regions, 0 = AR, >0 = dense region ID. Cast needed when nreg == 0.
            "attn_regions": np.r_[1, [1] * npre, 1, [1] * nimg, [1] * nreg, 1, [0] * nsuf].astype(np.int64),
            "ndatatoks": npre + nimg + nsuf,
            "id": exid,
        }
        if nreg:  # Only add if needed, because mask creation is expensive.
            example["attn_regions2"] = np.r_[1, [1] * npre, 1, [-1] * nimg, [1] * nreg, 1, [0] * nsuf].astype(np.int64)
            # regonly
            # example["attn_regions2"] = np.r_[-1, [-1] * npre, -1, [-1] * nimg, [1] * nreg, 1, [0] * nsuf, 0].astype(np.int64)
        return pp.sanity_check(example)

    def __str__(self):
        return self._name

    def make_exids(self, seed, **kw):
        if self.epochs == 1:
            yield from iota_exids(len(self.reader), **kw)
        else:
            yield from shuffled_iota_exids(len(self.reader), epochs=self.epochs, seed=seed, **kw)

    def vocab_size(self):
        return self.tt.n_vocab
