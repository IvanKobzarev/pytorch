"""
"<|bos|>Question<|sep|>{Image}<|sep|>{answer}<|eos|>"
"""

import json
from io import BytesIO
from zipfile import ZipFile

import numpy as np
from PIL import Image

import bv2.data.dpack as d
from bv2.data.common import get_bagz_reader, sharded_iota_exids, vis_image_text_wandb
from bv2.data.pp import patchify, resize_max_patches, sanity_check
from bv2.data.tokenizer import get_tiktoken

PATH = "/checkpoint/rigi/data/{split}.bag"


class Dataset:
    def __init__(self, split, basepath=PATH, ps=16, max_patches=16_384, nreg=0, greyout_frac=0.0, tokenizer=None):
        self.fspec = basepath.format(split=split)
        self.ps = dict(ph=ps, pw=ps)
        self.max_patches = max_patches
        self.nreg = nreg
        self.ttkw = tokenizer or {}
        self.greyout_frac = greyout_frac

    @property  # Not a cached_property because BagzReader is not picklable.
    def reader(self):  # which would make the whole class unpicklable.
        return get_bagz_reader(self.fspec)  # But this is functools.cache'd per process.

    @property
    def tt(self):  # Same story as for the bagz reader above.
        return get_tiktoken(**self.ttkw)

    def make_example(self, exid, epoch):
        # NOTE: Could further optimize by having each rank go only to a subset of all indices.
        # But let's keep things simple as long as they are fast enough!
        with ZipFile(BytesIO(self.reader[exid])) as zf:
            data = json.load(zf.open("data.json"))
            img = Image.open(zf.open("image"))
            img.load()  # Ensure it's actually fully read.
            img = img if img.mode == "RGB" else img.convert("RGB")
            # NOTE: Not using "ocr.json" here yet.

        # Cycle through the questions and its answers by epochs
        q_cycle, q_idx = divmod(epoch, len(data["qas"]))
        question, answers = data["qas"][list(data["qas"])[q_idx]]
        answer = answers[q_cycle % len(answers)]

        prefix = self.tt.encode(question.lower())
        suffix = self.tt.encode(answer.lower())

        rng = np.random.default_rng([exid, epoch])
        if rng.random() < self.greyout_frac:
            img.paste((128, 128, 128), box=(0, 0) + img.size)
        img = resize_max_patches(img, self.max_patches, **self.ps)
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

    def make_exids(self, *a, **kw):
        return sharded_iota_exids(len(self.reader), *a, **kw)

    def vocab_size(self):
        return self.tt.n_vocab

    def vis_data_wandb(self, data):
        return vis_image_text_wandb(data, self.tt, **self.ps)
