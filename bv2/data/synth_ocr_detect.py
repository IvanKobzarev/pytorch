"""
"Where is the word {word}" + Image -> "x y width height"
"""
# ruff: noqa: E701

import numpy as np
from PIL import Image, ImageDraw

import bv2.data.dpack as d
import bv2.utils as u
from bv2.data.common import random_exids
from bv2.data.pp import patchify, sanity_check
from bv2.data.synth_ocr import font, render
from bv2.data.tokenizer import get_tiktoken


class Dataset:
    def __init__(self, mode="tlwh", add_row_sep=False, add_hw=False, tiptoi=0, fs=18, ps=16, tokenizer=None, seed=0, n=None, **kw):
        self.fs = fs
        self.ps = ps
        self.render_kw = kw
        self.mode = mode
        self.add_row_sep = add_row_sep
        self.add_hw = add_hw
        self.tiptoi = tiptoi
        self.tt = get_tiktoken(**tokenizer or {})
        self.data_seed = seed
        self.n = n

    def make_exids(self, **kw):
        yield from random_exids(n=self.n, **kw)

    def make_example(self, exid):
        # Format is [BOS, prefix, SEP, img, SEP, suffix, EOS].
        img, text, _ = render(exid, ps=self.ps, unique=True, **self.render_kw)
        lines = text.split("\n")
        line_idx = u.rng(self.data_seed, exid, "row").integers(0, len(lines))
        words_in_line = lines[line_idx].split()
        word_idx = u.rng(self.data_seed, exid, "col").integers(0, len(words_in_line))
        query_word = words_in_line[word_idx]

        ft, info = font(self.fs)
        h, space_w = info["line_h"], info["space_w"]
        w = ft.getlength(query_word)
        y = line_idx * h
        x = sum(ft.getlength(word) + space_w for word in words_in_line[:word_idx])

        cx, cy = round(x + w / 2), round(y + h / 2)
        x2, y2 = round(x + w), round(y + h)
        x, y, w, h = round(x), round(y), round(w), round(h)
        match self.mode:
            case "ltwh": coords_str = f"{x} {y} {w} {h}"
            case "tlwh": coords_str = f"{y} {x} {w} {h}"
            case "lt": coords_str = f"{x} {y}"
            case "tl": coords_str = f"{y} {x}"
            case "rb": coords_str = f"{x2} {y2}"
            case "cc": coords_str = f"{cx} {cy}"
            case "ccwh": coords_str = f"{cx} {cy} {w} {h}"
            case "ltrb": coords_str = f"{x} {y} {x2} {y2}"
            case "lrtb": coords_str = f"{x} {x2} {y} {y2}"
            case _: raise ValueError(f"Unknown mode {self.mode}. See code")

        prefix = np.array(self.tt.encode(f"Where is the word {query_word}"))
        suffix = np.array(self.tt.encode(coords_str))

        # TODO: also do a "resize to min/max" in the future.
        patches, positions = patchify(img, pw=self.ps, ph=self.ps)

        npre = 1 + len(prefix) + 1
        nsuf = 1 + len(suffix) + 1
        nimg = np.prod(patches.shape[:2]) + patches.shape[0] * self.add_row_sep + 2 * self.add_hw  # fmt: skip

        nbytes = max(
            d.nbytes_text(),
            d.nbytes_image_with_extras(ph=self.ps, pw=self.ps, tiptoi=self.tiptoi),
        )
        tokens = np.zeros((npre + nimg + nsuf, nbytes), np.uint8)

        txtpos = np.arange(npre + nsuf)

        d.pack_text([self.tt.bos, prefix, self.tt.sep], positions=txtpos[:npre], out=tokens[:npre])
        d.pack_image_with_extras(
            patches, positions, out=tokens[npre:-nsuf],
            add_row_sep=self.add_row_sep, add_hw=self.add_hw, tiptoi=self.tiptoi)
        d.pack_text([self.tt.sep, suffix, self.tt.eos], positions=txtpos[-nsuf:], out=tokens[-nsuf:])

        return sanity_check({
            "toki": tokens[..., :-1, :],
            "toko": tokens[..., 1:, :],
            # no loss on sep after image.
            "lowe": np.r_[[0] * (npre-1), [0] * (nimg+1), [1] * (nsuf-1)].astype(np.float32),  # -1 removes bos+sep
            "attn_regions": np.r_[[1] * npre, [1] * (nimg+1), [0] * (nsuf-2)],  # -2 removes sep+eos
            # NOTE: for attn_regions, 0 = AR, >0 = dense region ID.
            "ndatatoks": len(prefix) + len(suffix) + nimg,
            "id": exid,
        })  # fmt: skip

    def draw_bbox(self, image, x1, y1, x2, y2, color="red", width=2):
        image = Image.fromarray(np.array(image))  # Make a copy for sure.
        ImageDraw.Draw(image).rectangle([x1, y1, x2, y2], outline=color, width=width)
        return image


    def draw_cross(self, image, x, y, color="red", width=2, size=5):
        image = Image.fromarray(np.array(image))  # Make a copy for sure.
        draw = ImageDraw.Draw(image)
        draw.line([x - size, y, x + size, y], fill=color, width=width)
        draw.line([x, y - size, x, y + size], fill=color, width=width)
        return image


    def parse_and_draw(self, image, coords_str, color="red"):
        """Parse coordinates based on mode and draw appropriate visualization."""
        try:
            coords = [int(x) for x in coords_str.removesuffix("<|eos|>").split()]
        except:
            return image  # Return original image if parsing fails

        if self.mode == "ltwh" and len(coords) == 4:
            x, y, width, height = coords
            return self.draw_bbox(image, x, y, x + width, y + height, color)
        elif self.mode == "tlwh" and len(coords) == 4:
            y, x, width, height = coords
            return self.draw_bbox(image, x, y, x + width, y + height, color)
        elif self.mode == "lt" and len(coords) == 2:
            x, y = coords
            return self.draw_cross(image, x, y, color)
        elif self.mode == "tl" and len(coords) == 2:
            y, x = coords
            return self.draw_cross(image, x, y, color)
        elif self.mode == "rb" and len(coords) == 2:
            x, y = coords
            return self.draw_cross(image, x, y, color)
        elif self.mode == "cc" and len(coords) == 2:
            x, center_y = coords
            return self.draw_cross(image, x, center_y, color)
        elif self.mode == "ccwh" and len(coords) == 4:
            cx, cy, width, height = coords
            x = cx - width / 2
            y = cy - height / 2
            return self.draw_bbox(image, x, y, x + width, y + height, color)
        elif self.mode == "ltrb" and len(coords) == 4:
            x1, y1, x2, y2 = coords
            return self.draw_bbox(image, x1, y1, x2, y2, color)
        elif self.mode == "lrtb" and len(coords) == 4:
            x1, x2, y1, y2 = coords
            return self.draw_bbox(image, x1, y1, x2, y2, color)

        print(f"skip mode: {self.mode} coords: {len(coords)} {coords}")
        return image

    def vocab_size(self):
        return self.tt.n_vocab
