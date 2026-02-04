import json
import os
import re
from io import BytesIO
from zipfile import ZipFile

import cv2
import numpy as np

import bv2.data.dpack as d
import bv2.data.finevision_info as fvi
import bv2.utils as u
from bv2.data import pp
from bv2.data.common import cycle_qas, get_sackli_reader, shuffled_iota_exids
from bv2.data.tokenizer import get_tiktoken


class Dataset:
    def __init__(self, ps=16, max_patches=16_384, rand_max_patches=None, nreg=0, include=[".*"], exclude=[], tokenizer=None, greyout_frac=0.0, seed=0, epochs=None, cache=False):
        base_path = "/checkpoint/rigi/data/FineVision-1.0.1"
        self._name = f"finevision({','.join(include)})"

        paths = []
        re_inc = [re.compile(p) for p in include]
        re_exc = [re.compile(p) for p in exclude]
        for name, bag_pattern in fvi.BAG_FILES.items():
            if not any(r.fullmatch(name) for r in re_inc) or any(r.fullmatch(name) for r in re_exc):
                continue
            paths.append(os.path.join(base_path, name, bag_pattern))

        self.reader = get_sackli_reader(",".join(paths), cache)
        self.ps = {"ph": ps, "pw": ps}
        self.max_patches = max_patches
        self.rand_max_patches = rand_max_patches or {}
        self.nreg = nreg
        self.tt = get_tiktoken(**tokenizer or {})
        self.greyout_frac = greyout_frac
        self.data_seed = seed
        self.epochs = epochs

    def make_example(self, exid, epoch):
        with ZipFile(BytesIO(self.reader[exid])) as zf:
            data = json.load(zf.open("data.json"))

            def _read_img(f):
                img = cv2.imdecode(np.frombuffer(zf.open(f).read(), np.uint8), cv2.IMREAD_COLOR)
                return img[:, :, ::-1]  # BGR -> RGB

            images = []
            if "image" in zf.namelist():
                images.append(_read_img("image"))
            else:
                image_files = [n for n in zf.namelist() if n.startswith("images/")]
                image_files.sort(key=lambda x: int(x.split("/")[1]))
                for image_file in image_files:
                    images.append(_read_img(image_file))

        # TODO: some datasets contain a sequence of QAs that are follow-ups:
        # question - answer; follow q - answer; follow q - answer. In this case, we should
        # concat all the QAs instead of picking a random one.
        qid, question, answer = cycle_qas(data["qas"], epoch, seed=(self.data_seed, exid, "cycle_qas"))
        prefix = self.tt.encode(question)
        suffix = self.tt.encode(answer)
        npre, nsuf = len(prefix), len(suffix)

        all_patches, all_positions = [], []
        for i, img in enumerate(images):
            key = (self.data_seed, exid, epoch, i)
            img = pp.reasonable_resize(img, pp.rand_max_patches(
                img.shape[:2], self.max_patches, key=(key, "resize"), **self.rand_max_patches, **self.ps),
                warning_exid=f"finevision/{exid}/{i} ({data['source'][0]})")
            if u.rng(key, "greyout").random() < self.greyout_frac:
                img[...] = 128
            patches, positions = pp.patchify(img, **self.ps)
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

        # TODO: Actually we could have `toko` be only non-packed text => smaller and faster.
        example = {
            "toki": tokens[..., :-1, :],
            "toko": tokens[..., 1:, :],
            "lowe":  np.r_[[0] * npre, 0,  [0] * nimg, [0] * nreg, [1] * nsuf, 1].astype(np.float32),
            "attn_regions":  np.r_[1, [1] * npre, 1,  [1] * nimg, [1] * nreg, [0] * nsuf].astype(np.int64),
            "ndatatoks": npre + nimg + nsuf,
            "src": data["source"][0],
            "id": exid,
        }
        if nreg:  # Only add if needed, because mask creation is expensive.
            example["attn_regions2"] = np.r_[1, [1] * npre, 1, [-1] * nimg, [1] * nreg, [0] * nsuf].astype(np.int64)
        return pp.sanity_check(example)

    def __str__(self):
        return self._name

    def make_exids(self, **kw):
        yield from shuffled_iota_exids(len(self.reader), epochs=self.epochs, **kw)

    def vocab_size(self):
        return self.tt.n_vocab
