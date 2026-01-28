#!/usr/bin/env python3
"""
Renders random words on white background.

Bento: https://fburl.com/anp/9dej90z4
"""

from functools import cache
from itertools import count

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

import bv2.data.dpack as d
import bv2.utils as u
from bv2.data.common import random_exids, vis_image_text_wandb
from bv2.data.noun_vocab import VOCAB
from bv2.data.pp import patchify, sanity_check
from bv2.data.tokenizer import get_tiktoken


@cache
def font(size=18):
    font = ImageFont.truetype("DejaVuSans.ttf", size=size)
    ascent, descent = font.getmetrics()
    info = {
        "space_w": font.getlength(" "),
        "line_h": ascent + descent,  # `multiline_text` adds 4, but +0 looks neater.
    }
    return font, info


def render(seed, *, min_h=256, min_w=256, max_h=768, max_w=768, ps=16,
           fs=24, random_pad=16, random_angle=20, fs_jitter=8,
           unique=False, draw_img=True):

    if fs_jitter:
        fs += u.rng(seed, "fs_jitter").integers(-fs_jitter, fs_jitter+1, ()).item()
    ft, info = font(fs)

    line_h, space_w = info["line_h"], info["space_w"]

    # Note we add ps to make sure size_max is inclusive
    img_w = (u.rng(seed, "w").integers(min_w, max_w + ps) // ps) * ps
    img_h = (u.rng(seed, "h").integers(min_h, max_h + ps) // ps) * ps

    img = None
    if draw_img:
        img = Image.new("RGB", (img_w, img_h), "white")
        draw = ImageDraw.Draw(img)

    cur_line, cur_width, y = "", 0.0, 0.0
    all_words, all_text = [], ""
    for iword in count():
        candidates = VOCAB if not unique else list(set(VOCAB) - set(all_words))
        word = u.rng(seed, "word", iword).choice(candidates).item()
        w_len = ft.getlength(word)
        add_w = w_len if not cur_line else space_w + w_len
        if cur_width + add_w <= img_w:
            cur_line += (" " if cur_line else "") + word
            cur_width += add_w
            all_words.append(word)
        else:
            if draw_img:
                draw.text((0, y), cur_line, font=ft, fill="black")
            all_text += f"\n{cur_line}" if all_text else cur_line
            y += line_h
            if y + line_h > img_h:
                break
            cur_line, cur_width = "", 0.0

    if random_angle:
        angle = u.rng(seed, "rotate").integers(-random_angle, random_angle + 1)
        # resample=2 means bilinear
        rotated = img.rotate(angle, expand=True, fillcolor=(255, 255, 255), resample=2)
        img = rotated.resize(img.size, resample=Image.LANCZOS)

    if random_pad:
        pad_vals = u.rng(seed, "pad").integers(0, random_pad+1, (4,))
        padded = ImageOps.expand(img, border=tuple(pad_vals), fill=(255, 255, 255))
        img = padded.resize(img.size, resample=Image.LANCZOS)

    return img, all_text, all_words


class Dataset:
    def __init__(self, add_row_sep=False, add_hw=False, tiptoi=0, ps=16, tokenizer=None, seed=0, n=None, **kw):
        self.ps = ps
        self.add_row_sep = add_row_sep
        self.add_hw = add_hw
        self.tiptoi = tiptoi
        self.render_kw = kw
        self.tt = get_tiktoken(**tokenizer or {})
        self.data_seed = seed
        self.n = n

    def make_exids(self, **kw):
        yield from random_exids(n=self.n, **kw)

    def ground_truth(self, exid):
        img, txt, _ = render((self.data_seed, exid, "render"), ps=self.ps, **self.render_kw)
        # VQA format
        return {"qas": {"0": ("ocr?", [txt])}, "img": img}

    def make_example(self, exid):
        img, txt, _ = render((self.data_seed, exid, "render"), ps=self.ps, **self.render_kw)

        prefix = np.array(self.tt.encode("ocr"))
        suffix = np.array(self.tt.encode(txt))

        # TODO: also do a "resize to min/max" in the future.
        patches, positions = patchify(img, pw=self.ps, ph=self.ps)

        npre = 1 + len(prefix) + 1
        nsuf = 1 + len(suffix) + 1
        nimg = np.prod(patches.shape[:2])

        nbytes = max(d.nbytes_text(), d.nbytes_image_with_extras(ph=self.ps, pw=self.ps, tiptoi=self.tiptoi))
        tokens = np.zeros((npre + nimg + nsuf, nbytes), np.uint8)

        txtpos = np.arange(npre + nsuf)
        d.pack_text([self.tt.bos, prefix, self.tt.sep], positions=txtpos[:npre], out=tokens[:npre])
        d.pack_image_with_extras(patches, positions, out=tokens[npre:-nsuf],
                                 add_hw=self.add_hw, add_row_sep=self.add_row_sep, tiptoi=self.tiptoi)
        d.pack_text([self.tt.sep, suffix, self.tt.eos], positions=txtpos[-nsuf:], out=tokens[-nsuf:])

        return sanity_check({
            "tokens": tokens,
            #                     Prefix      Image       Sep  Suffix
            "loss_weights": np.r_[[0] * npre, [0] * nimg, [0], [1] * (nsuf - 1)],
            "attn_regions": np.r_[[1] * npre, [1] * nimg, [1], [0] * (nsuf - 1)],
            # NOTE: for attn_regions, 0 = AR, >0 = dense region ID.
            "id": exid,
        })  # fmt: skip

    def vocab_size(self):
        return self.tt.n_vocab

    def vis_data_wandb(self, data):
        return vis_image_text_wandb(data, self.tt, ph=self.ps, pw=self.ps)
