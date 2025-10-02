#!/usr/bin/env python3
"""
Renders random words on white background.

Bento: https://fburl.com/anp/9dej90z4
"""

from functools import cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import bv2.data.dpack as d  # isort: skip
from bv2.data.noun_vocab import VOCAB  # isort: skip
from bv2.data.pp import patchify, sanity_check, unpatchify  # isort: skip
from bv2.data.common import infinite_random_exids, vis_image_text_wandb  # isort: skip


@cache
def font(size=18):
    font = ImageFont.truetype("DejaVuSans.ttf", size=size)
    ascent, descent = font.getmetrics()
    info = {
        "space_w": font.getlength(" "),
        "line_h": ascent + descent,  # `multiline_text` adds 4, but +0 looks neater.
    }
    return font, info


def render(seed, *, min_h=224, min_w=224, max_h=288, max_w=288, ps=16, fs=18, unique=False, draw_img=True):  # fmt: skip
    ft, info = font(fs)
    line_h, space_w = info["line_h"], info["space_w"]

    rng = np.random.default_rng(seed)

    # Note we add ps to make sure size_max is inclusive
    img_w = (rng.integers(min_w, max_w + ps) // ps) * ps
    img_h = (rng.integers(min_h, max_h + ps) // ps) * ps

    img = None
    if draw_img:
        img = Image.new("RGB", (img_w, img_h), "white")
        draw = ImageDraw.Draw(img)

    cur_line, cur_width, y = "", 0.0, 0.0
    all_words, all_text = [], ""
    while True:
        candidates = VOCAB if not unique else list(set(VOCAB) - set(all_words))
        word = rng.choice(candidates).item()
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

    return img, all_text, all_words


class Dataset:
    def __init__(self, ps=16, **kw):
        self.ps = ps
        self.render_kw = kw

    def make_exids(self, *a, **kw):
        return infinite_random_exids(*a, **kw)

    def make_example(self, exid, epoch):
        img, txt, _ = render(exid, ps=self.ps, **self.render_kw)
        t = _get_tiktoken()

        prefix = np.array(t.encode("ocr"))
        suffix = np.array(t.encode(txt))

        # TODO: also do a "resize to min/max" in the future.
        patches, positions = patchify(img, pw=ps, ph=ps)

        npre = 1 + len(prefix) + 1
        nsuf = 1 + len(suffix) + 1
        nimg = len(patches)

        nbytes = max(d.nbytes_text(), d.nbytes_image(ph=ps, pw=ps))
        tokens = np.zeros((npre + nimg + nsuf, nbytes), np.uint8)

        txtpos = np.arange(npre + nsuf)
        d.pack_text([t.bos, prefix, t.sep], positions=txtpos[:npre], out=tokens[:npre])
        d.pack_image(patches, positions, out=tokens[npre:-nsuf])
        d.pack_text([t.sep, suffix, t.eos], positions=txtpos[-nsuf:], out=tokens[-nsuf:])

        return sanity_check({
            "tokens": tokens,
            "loss_weights": np.r_[[0] * npre, [0] * nimg, [1] * nsuf],
            "attn_regions": np.r_[[1] * npre, [1] * nimg, [0] * nsuf],
            # NOTE: for attn_regions, 0 = AR, >0 = dense region ID.
            "id": exid,
        })  # fmt: skip

    def vocab_size(self):
        return _get_tiktoken().n_vocab

    def vis_data_wandb(self, data):
        return vis_image_text_wandb(data, _get_tiktoken(), self.ps, self.ps)


def _get_tiktoken(first_N=10_000):
    import bv2.data.tokenizer

    return bv2.data.tokenizer.get_tiktoken(first_N=first_N)
