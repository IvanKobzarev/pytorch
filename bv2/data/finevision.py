import json
import os
import re
from io import BytesIO
from zipfile import ZipFile

import numpy as np
from PIL import Image

import bv2.data.dpack as d
import bv2.utils as u
from bv2.data.common import get_bagz_reader, sharded_iota_exids, vis_image_text_wandb
from bv2.data.pp import patchify, resize_max_patches, sanity_check
from bv2.data.tokenizer import get_tiktoken


class Dataset:
    def __init__(self, ps=16, max_patches=16_384, nreg=0, include=[".*"], exclude=[], tokenizer=None, greyout_frac=0.0, seed=0):
        base_path = "/checkpoint/rigi/data/FineVision-1.0.1"

        paths = []
        re_inc = [re.compile(p) for p in include]
        re_exc = [re.compile(p) for p in exclude]
        for name, bag_pattern in DATA_TO_BAG.items():
            if not any(r.match(name) for r in re_inc) or any(r.match(name) for r in re_exc):
                continue
            paths.append(os.path.join(base_path, name, bag_pattern))

        self.fspec = ",".join(paths)
        self.ps = {"ph": ps, "pw": ps}
        self.max_patches = max_patches
        self.nreg = nreg
        self.ttkw = tokenizer or {}
        self.greyout_frac = greyout_frac
        self.seed = seed

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
            if u.rng(exid, epoch, self.seed, i, "greyout").random() < self.greyout_frac:
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
        return sharded_iota_exids(len(self.reader), **kw)

    def vocab_size(self):
        return self.tt.n_vocab


DATA_TO_BAG = {
    "aguvis-stage-1": "train@256.bag",
    "ai2d_merged": "train@1.bag",
    "alfworldgpt": "train@4.bag",
    "allava_laion": "train@256.bag",
    "allava_vflan": "train@256.bag",
    "aokvqa": "train@1.bag",
    "a_okvqa": "train@32.bag",
    "art": "train@32.bag",
    "arxivqa": "train@256.bag",
    "bentham": "train@4.bag",
    "blockdiagramcomputerized": "train@1.bag",
    "blockdiagramhandwritten": "train@1.bag",
    "cambrian(filtered)_processed": "train@256.bag",
    "captcha": "train@4.bag",
    "chart2text": "train@4.bag",
    "chartqa": "train@1.bag",
    "chinesememe": "train@32.bag",
    "chrome_writting": "train@1.bag",
    "clevr_math(mathv360k)": "train@1.bag",
    "clevr_math": "train@32.bag",
    "clevr": "train@32.bag",
    "coco_colors": "train@256.bag",
    "cocoqa": "train@4.bag",
    "cocotext": "train@32.bag",
    "CoSyn_400k_chart": "train@32.bag",
    "CoSyn_400k_chemical": "train@1.bag",
    "CoSyn_400k_circuit": "train@1.bag",
    "CoSyn_400k_diagram": "train@32.bag",
    "CoSyn_400k_document": "train@32.bag",
    "CoSyn_400k_graphic": "train@1.bag",
    "CoSyn_400k_math": "train@32.bag",
    "CoSyn_400k_music": "train@1.bag",
    "CoSyn_400k_nutrition": "train@4.bag",
    "CoSyn_400k_table": "train@32.bag",
    "ctw": "train@32.bag",
    "datik": "train@4.bag",
    "datikz": "train@1.bag",
    "densefusion_1m": "train@256.bag",
    "diagram_image_to_text": "train@1.bag",
    "DoclingMatix": "train@256.bag",
    "docvqa": "train@32.bag",
    "drivelm": "train@4.bag",
    "dvqa": "train@32.bag",
    "est_vqa": "train@32.bag",
    "figureqa(mathv360k)": "train@1.bag",
    "figureqa": "train@4.bag",
    "finqa": "train@1.bag",
    "funsd": "train@1.bag",
    "geo170k(align)": "train@1.bag",
    "geo170k(qa)": "train@1.bag",
    "geo3k": "train@1.bag",
    "geometry3k(mathv360k)": "train@1.bag",
    "geomverse": "train@4.bag",
    "geoqa+(mathv360k)": "train@1.bag",
    "geos(mathv360k)": "train@1.bag",
    "google_landmarks": "train@256.bag",
    "groundui": "train@32.bag",
    "handwriting_forms": "train@1.bag",
    "hateful_memes": "train@4.bag",
    "hitab": "train@1.bag",
    "hme100k": "train@4.bag",
    "hw_squad": "train@32.bag",
    "iam": "train@4.bag",
    "iconqa(mathv360k)": "train@1.bag",
    "iconqa": "train@1.bag",
    "idk": "train@32.bag",
    "iiit5k": "train@1.bag",
    "image_textualization(filtered)": "train@256.bag",
    "imgur5k": "train@32.bag",
    "indoor_qa": "train@1.bag",
    "infographic(gpt4v)": "train@4.bag",
    "infographic_vqa_llava_format": "train@4.bag",
    "infographic_vqa": "train@32.bag",
    "intergps": "train@1.bag",
    "invoices_receipts": "train@4.bag",
    "k12_printing": "train@32.bag",
    "laion_gpt4v": "train@4.bag",
    "latexformulas": "train@32.bag",
    "latex_handwritten": "train@32.bag",
    "LLaVA_Instruct_150K": "train@256.bag",
    "llavar_gpt4_20k": "train@32.bag",
    "lnqa": "train@256.bag",
    "localized_narratives": "train@32.bag",
    "lrv_chart": "train@1.bag",
    "lrv_normal(filtered)": "train@4.bag",
    "lvis_instruct4v": "train@256.bag",
    "mapqa(mathv360k)": "train@1.bag",
    "mapqa": "train@4.bag",
    "maptext": "train@1.bag",
    "mathwriting-google": "train@32.bag",
    "mavis_math_metagen": "train@4.bag",
    "mavis_math_rule_geo": "train@32.bag",
    "memotion": "train@4.bag",
    "mimic_cgd": "train@32.bag",
    "mmc_instruct": "train@32.bag",
    "mmevol": "train@32.bag",
    "mmra": "train@4.bag",
    "mmsoc_memotion": "train@4.bag",
    "multihiertt": "train@4.bag",
    "nlvr2": "train@32.bag",
    "objects365_qa": "train@256.bag",
    "ocrvqa": "train@32.bag",
    "olmOCR-mix-0225-books": "train@32.bag",
    "olmOCR-mix-0225-documents": "train@256.bag",
    "oodvqa": "train@32.bag",
    "orand_car_a": "train@1.bag",
    "pathvqa": "train@32.bag",
    "pdfvqa": "train@4.bag",
    "plotqa": "train@32.bag",
    "pmc_vqa(mathv360k)": "train@4.bag",
    "raven": "train@4.bag",
    "rendered_text": "train@32.bag",
    "robut_sqa": "train@1.bag",
    "robut_wikisql": "train@32.bag",
    "robut_wtq": "train@32.bag",
    "scienceqa(nona_context)": "train@4.bag",
    "scienceqa": "train@1.bag",
    "screen2words": "train@4.bag",
    "screenqa": "train@256.bag",
    "sharegpt4o": "train@256.bag",
    "sharegpt4v(coco)": "train@32.bag",
    "sharegpt4v(knowledge)": "train@4.bag",
    "sharegpt4v(llava)": "train@32.bag",
    "sharegpt4v(sam)": "train@4.bag",
    "sketchyvqa": "train@1.bag",
    "slidevqa": "train@32.bag",
    "spark": "train@4.bag",
    "spatialsense": "train@4.bag",
    "spot_the_diff": "train@4.bag",
    "sroie": "train@1.bag",
    "st_vqa": "train@1.bag",
    "sujet_finance": "train@32.bag",
    "super_clevr(mathv360k)": "train@4.bag",
    "svrd": "train@32.bag",
    "SynthChartNet": "train@32.bag",
    "SynthCodeNet": "train@256.bag",
    "synthdog": "train@256.bag",
    "SynthFormulaNet": "train@4.bag",
    "tabmwp(mathv360k)": "train@1.bag",
    "tabmwp": "train@1.bag",
    "tallyqa": "train@32.bag",
    "tal_ocr_eng": "train@32.bag",
    "tat_dqa": "train@1.bag",
    "tat_qa": "train@1.bag",
    "textcaps": "train@32.bag",
    "text_codefeedback_filtered_instruction": "train@1.bag",
    "text_code_feedback": "train@1.bag",
    "text_infinitymath": "train@1.bag",
    "text_mathinstruct": "train@1.bag",
    "text_mathqa": "train@1.bag",
    "text_mathstepdpo10k": "train@1.bag",
    "text_numinamath_cot": "train@4.bag",
    "textocr(gpt4v)": "train@32.bag",
    "text_openhermes_2_5": "train@4.bag",
    "text_OpenMathInstruct-2": "train@4.bag",
    "text_openorca": "train@32.bag",
    "text_orcamath": "train@1.bag",
    "text_pythoncode25k": "train@1.bag",
    "text_pythoncodealpaca": "train@1.bag",
    "text_ruozhiba": "train@1.bag",
    "text_theoremqa": "train@1.bag",
    "textvqa": "train@4.bag",
    "text_wizardlm_evol": "train@1.bag",
    "tqa": "train@1.bag",
    "Unichart": "train@32.bag",
    "unigeo(mathv360k)": "train@1.bag",
    "ureader_cap": "train@256.bag",
    "ureader_ie": "train@32.bag",
    "ureader_kg_processed": "train@32.bag",
    "ureader_qa_processed": "train@256.bag",
    "vision_flan(filtered)": "train@256.bag",
    "vistext": "train@1.bag",
    "visual7w": "train@32.bag",
    "visualmrc": "train@4.bag",
    "visualwebinstruct(filtered)": "train@256.bag",
    "vizwiz(mathv360k)": "train@32.bag",
    "vqaonbd": "train@32.bag",
    "vqarad": "train@1.bag",
    "vqav2": "train@32.bag",
    "vsr": "train@1.bag",
    "websight": "train@32.bag",
    "wildvision": "train@1.bag",
    "wordart": "train@4.bag",
    "yesbut": "train@4.bag",
}
