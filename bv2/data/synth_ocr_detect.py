"""
"Where is the word {word}" + Image -> "x y width height"
"""

import numpy as np
from PIL import Image, ImageDraw

import bv2.data.dpack as d  # isort: skip
from bv2.data.pp import patchify, sanity_check  # isort: skip
from bv2.data.synth_ocr import font, render  # isort: skip
from bv2.data.common import infinite_random_exids, vis_image_text_unpack  # isort: skip

class Dataset:
    def __init__(self, mode="tlwh", add_row_sep=False, add_hw=False, tiptoi=0, fs=18, ps=16, **kw):
        self.fs = fs
        self.ps = ps
        self.render_kw = kw
        self.mode = mode
        self.add_row_sep = add_row_sep
        self.add_hw = add_hw
        self.tiptoi = tiptoi

    def make_exids(self, *a, **kw):
        return infinite_random_exids(*a, epoch_size=2048, **kw)

    def make_example(self, exid, epoch):
        # Format is [BOS, prefix, SEP, img, SEP, suffix, EOS].

        # In this dataset, we consider one image to be an example, and which word in it is
        # to be detected changes from epoch to epoch. Hence, we fold the epoch into the RNG
        # below, but not into the number given to `render`.
        # Note, however, that currently we do generate new independent exids each epoch.
        img, text, _ = render(exid, ps=self.ps, unique=True, **self.render_kw)
        epoch_rng = np.random.default_rng([exid, epoch])
        lines = text.split("\n")
        line_idx = epoch_rng.integers(0, len(lines))
        words_in_line = lines[line_idx].split()
        word_idx = epoch_rng.integers(0, len(words_in_line))
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

        t = _get_tiktoken()
        prefix = np.array(t.encode(f"Where is the word {query_word}"))
        suffix = np.array(t.encode(coords_str))

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

        # fmt:off
        d.pack_text([t.bos, prefix, t.sep], positions=txtpos[:npre], out=tokens[:npre])
        d.pack_image_with_extras(
            patches, positions, out=tokens[npre:-nsuf],
            add_row_sep=self.add_row_sep, add_hw=self.add_hw, tiptoi=self.tiptoi)
        d.pack_text([t.sep, suffix, t.eos], positions=txtpos[-nsuf:], out=tokens[-nsuf:])
        # fmt:on

        return sanity_check({
            "tokens": tokens,
            # no loss on sep after image.
            "loss_weights": np.r_[[0] * npre, [0] * (nimg+1), [1] * (nsuf-1)],
            "attn_regions": np.r_[[1] * npre, [1] * (nimg+1), [0] * (nsuf-1)],
            # NOTE: for attn_regions, 0 = AR, >0 = dense region ID.
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

    def vis_output_wandb(self, data, preds, max_examples=20):
        import wandb  # Local import to not pollute tests with silly warnings.
        t = _get_tiktoken()
        table = wandb.Table([
            "input_text",
            "ground_truth",
            "TF prediction",
            "image_gt",
            "image_pred",
        ])  # fmt: skip

        tokens = data["tokens"].cpu()
        iseq = data["iseq"].cpu()
        loss_mask = data["loss_weights"].cpu() > 0

        for i in range(min(max_examples, iseq.max() + 1)):
            seq_mask = iseq == i

            txt, img = vis_image_text_unpack(tokens[seq_mask], ph=self.ps, pw=self.ps)
            txt = t.decode(txt)
            prefix, _, suffix = txt.split("<|sep|>")
            prefix = prefix.removeprefix("<|bos|>")
            suffix = suffix.removesuffix("<|eos|>")

            query = prefix.replace("Where is the word ", "")
            img_gt = self.parse_and_draw(img, suffix, color="red")

            tgt_mask = seq_mask & loss_mask
            seq_preds = preds[tgt_mask[1:]].numpy()

            pred_suffix = t.decode(seq_preds)

            if seq_preds[-1] == t.eos:
                pred_str = t.decode(seq_preds[:-1])
                img_pred = self.parse_and_draw(img, pred_str, color="blue")
            else:
                img_pred = img

            table.add_data(
                txt,
                suffix,
                pred_suffix,
                wandb.Image(img_gt, caption=f"{query}: {suffix}"),
                wandb.Image(img_pred, caption=f"{query}: {pred_suffix}"),
            )
        return table

    def vocab_size(self):
        return _get_tiktoken().n_vocab


def _get_tiktoken(first_N=30_000):
    import bv2.data.tokenizer

    return bv2.data.tokenizer.get_tiktoken(first_N=first_N)
