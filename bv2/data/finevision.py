import json
import os
import re
from io import BytesIO
from zipfile import ZipFile

import numpy as np
from PIL import Image

import bv2.data.dpack as d
import bv2.data.finevision_info as fvi
import bv2.utils as u
from bv2.data.common import get_bagz_reader, shuffled_iota_exids, vis_image_text_wandb
from bv2.data.pp import patchify, resize_max_patches, sanity_check
from bv2.data.tokenizer import get_tiktoken


class Dataset:
    def __init__(self, ps=16, max_patches=16_384, nreg=0, include=[".*"], exclude=[], tokenizer=None, greyout_frac=0.0, seed=0, epochs=None):
        base_path = "/checkpoint/rigi/data/FineVision-1.0.1"

        paths = []
        re_inc = [re.compile(p) for p in include]
        re_exc = [re.compile(p) for p in exclude]
        for name, bag_pattern in fvi.BAG_FILES.items():
            if not any(r.fullmatch(name) for r in re_inc) or any(r.fullmatch(name) for r in re_exc):
                continue
            paths.append(os.path.join(base_path, name, bag_pattern))

        self.fspec = ",".join(paths)
        self.ps = {"ph": ps, "pw": ps}
        self.max_patches = max_patches
        self.nreg = nreg
        self.ttkw = tokenizer or {}
        self.greyout_frac = greyout_frac
        self.data_seed = seed
        self.epochs = epochs

    def vis_data_wandb(self, data):
        return vis_image_text_wandb(data, self.tt, **self.ps)

    @property  # Not a cached_property because BagzReader is not picklable.
    def reader(self):  # which would make the whole class unpicklable.
        return get_bagz_reader(self.fspec)  # But this is functools.cache'd per process.

    @property
    def tt(self):  # Same story as for the bagz reader above.
        return get_tiktoken(**self.ttkw)

    def make_example(self, exid, epoch):
        with ZipFile(BytesIO(self.reader[exid])) as zf:
            data = json.load(zf.open("data.json"))

            def _read_img(f):
                img = Image.open(zf.open(f))
                img.load()
                return img if img.mode == "RGB" else img.convert("RGB")

            images = []
            if "image" in zf.namelist():
                images.append(_read_img("image"))
            else:
                image_files = [n for n in zf.namelist() if n.startswith("images/")]
                image_files.sort(key=lambda x: int(x.split("/")[1]))
                for image_file in image_files:
                    images.append(_read_img(image_file))

        # TODO: some datasets contain a sequence of QAs, with follow up questions like:
        # question - answer; follow q - answer; follow q - answer. In this case, we should
        # concat all the QAs instead of picking a random one.
        q_cycle, q_idx = divmod(epoch, len(data["qas"]))
        question, answers = data["qas"][list(data["qas"])[q_idx]]
        answer = answers[q_cycle % len(answers)]

        prefix = self.tt.encode(question)
        suffix = self.tt.encode(answer)
        npre, nsuf = len(prefix), len(suffix)

        all_patches, all_positions = [], []
        for i, img in enumerate(images):
            if u.rng(self.data_seed, exid, epoch, i, "greyout").random() < self.greyout_frac:
                img.paste((128, 128, 128), box=(0, 0) + img.size)
            img_resized = resize_max_patches(img, self.max_patches, **self.ps)
            patches, positions = patchify(img_resized, **self.ps)
            ny, nx, ph, pw, c = patches.shape
            patches_flat = patches.reshape(ny * nx, ph, pw, c)
            positions_flat = positions.reshape(ny * nx, 4)
            all_patches.append(patches_flat)
            all_positions.append(positions_flat)

        # These are token counts, so they include the corresponding separator tokens too:
        nimg = sum(map(len, all_patches)) + len(images)  # + 1 separator per image.
        nreg = self.nreg + (self.nreg > 0)  # plus one separator, if regs are present at all.

        nbytes = max(d.nbytes_text(), d.nbytes_image(**self.ps), d.nbytes_reg())
        tokens = np.zeros((1 + npre + 1 + nimg + nreg + nsuf + 1, nbytes), np.uint8)

        # The separators are still of text modality and posembs though, so that's len(images) + (nreg > 0) here:
        txtpos = np.arange(1 + npre + 1 + (len(images) + (self.nreg > 0)) + nsuf + 1)
        d.pack_text([self.tt.bos, prefix, self.tt.sep], positions=txtpos[: 1 + npre + 1], out=tokens[: 1 + npre + 1])

        pos = 1 + npre + 1
        for i_img, img_patches in enumerate(all_patches):
            n_patches = img_patches.shape[0]
            d.pack_image(img_patches, all_positions[i_img], out=tokens[pos:pos + n_patches])
            pos += n_patches

            d.pack_text([self.tt.sep], positions=[txtpos[1 + npre + 1 + i_img]], out=tokens[pos:pos + 1])
            pos += 1

            # Optional: pack regs after each image here, for cases with multiple images only.

        if nreg:
            d.pack_regs(nreg - 1, out=tokens[pos:pos + nreg - 1])  # Subtract the separator.
            d.pack_text([self.tt.sep], positions=txtpos[-(1 + nsuf + 1) : -(nsuf + 1)], out=tokens[pos + nreg - 1 : pos + nreg])

        d.pack_text([suffix, self.tt.eos], positions=txtpos[-(nsuf + 1) :], out=tokens[-(nsuf + 1) :])

        return sanity_check({
            "tokens": tokens,
            "loss_weights":  np.r_[0, [0] * npre, 0,  [0] * nimg, [0] * nreg, [1] * nsuf, 1].astype(np.int64),
            "attn_regions":  np.r_[1, [1] * npre, 1,  [1] * nimg, [1] * nreg, [0] * nsuf, 0].astype(np.int64),
            "attn_regions2": np.r_[1, [1] * npre, 1, [-1] * nimg, [1] * nreg, [0] * nsuf, 0].astype(np.int64),
            "src": data["source"][0],
            "id": exid,
        })

    def make_exids(self, **kw):
        yield from shuffled_iota_exids(len(self.reader), epochs=self.epochs, **kw)

    def vocab_size(self):
        return self.tt.n_vocab
