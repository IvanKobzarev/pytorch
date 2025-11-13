# torchrun --nproc_per_node=gpu -m bv2.train --config bv2/config/finevision.py

import sws


def get_config():
    c = sws.Config()
    c.seed = 0

    c.maxtok = 32_768

    c.data.name = "finevision"
    c.data.max_patches = 784  # (448/16)^2 = 784; (2048/16)^2 = 16_384
    c.data.nreg = lambda: c.model.reg.nreg
    c.data.greyout_frac = 0.03
    c.data.include = [".*"]  # List of regex patterns to match dataset names
    # Exclude datasets according to:
    # https://docs.google.com/spreadsheets/d/1m58LOIj0E_Nfwy6QGJHmFd7Sx4A-pMJZshcSoXNYPe0
    c.data.exclude = [
        "text_openorca", "objects365_qa", "text_openhermes_2_5",
        "text_OpenMathInstruct-2", "text_numinamath_cot",
        "synthdog", "text_mathqa",
    ]

    c.iter.maxtok = lambda: c.maxtok
    c.iter.eagerness = 24

    # 784 patches: 35m examples / 100k steps
    # 16384 patches: 13m examples / 100k steps
    c.nsteps = 100_000
    c.warmup_nsteps = 1000
    c.lr = 1e-4
    c.wd = lambda: c.lr * 0.1

    c.model.dim = 2048
    c.model.depth = 12
    c.model.reg.nreg = 10
    c.model.stages = lambda: "half" if c.model.reg.nreg > 0 else "single"
    c.model.txt_unemb.chunks = 8

    for greyout_frac in [0.0, 1.0]:
        suffix = "_blind" if greyout_frac else ""

        c.evals[f"pplx_st_vqa{suffix}"].type = "pplx"
        c.evals[f"pplx_st_vqa{suffix}"].steps = 2000
        c.evals[f"pplx_st_vqa{suffix}"].data.name = "vqa"
        c.evals[f"pplx_st_vqa{suffix}"].data.split = "stvqa/val"
        c.evals[f"pplx_st_vqa{suffix}"].data.max_patches = lambda: c.data.max_patches
        c.evals[f"pplx_st_vqa{suffix}"].data.nreg = lambda: c.model.reg.nreg
        c.evals[f"pplx_st_vqa{suffix}"].data.greyout_frac = greyout_frac
        c.evals[f"pplx_st_vqa{suffix}"].iter.maxtok = lambda: c.maxtok

        c.evals[f"pplx_info_vqa{suffix}"].type = "pplx"
        c.evals[f"pplx_info_vqa{suffix}"].steps = 2000
        c.evals[f"pplx_info_vqa{suffix}"].data.name = "vqa"
        c.evals[f"pplx_info_vqa{suffix}"].data.split = "infovqa/val"
        c.evals[f"pplx_info_vqa{suffix}"].data.max_patches = lambda: c.data.max_patches
        c.evals[f"pplx_info_vqa{suffix}"].data.nreg = lambda: c.model.reg.nreg
        c.evals[f"pplx_info_vqa{suffix}"].data.greyout_frac = greyout_frac
        c.evals[f"pplx_info_vqa{suffix}"].iter.maxtok = lambda: c.maxtok

        c.evals[f"pplx_text_vqa{suffix}"].type = "pplx"
        c.evals[f"pplx_text_vqa{suffix}"].steps = 2000
        c.evals[f"pplx_text_vqa{suffix}"].data.name = "vqa"
        c.evals[f"pplx_text_vqa{suffix}"].data.split = "textvqa/val"
        c.evals[f"pplx_text_vqa{suffix}"].data.max_patches = lambda: c.data.max_patches
        c.evals[f"pplx_text_vqa{suffix}"].data.nreg = lambda: c.model.reg.nreg
        c.evals[f"pplx_text_vqa{suffix}"].data.greyout_frac = greyout_frac
        c.evals[f"pplx_text_vqa{suffix}"].iter.maxtok = lambda: c.maxtok

        c.evals[f"pplx_doc_vqa{suffix}"].type = "pplx"
        c.evals[f"pplx_doc_vqa{suffix}"].steps = 2000
        c.evals[f"pplx_doc_vqa{suffix}"].data.name = "vqa"
        c.evals[f"pplx_doc_vqa{suffix}"].data.split = "docvqa/val"
        c.evals[f"pplx_doc_vqa{suffix}"].data.max_patches = lambda: c.data.max_patches
        c.evals[f"pplx_doc_vqa{suffix}"].data.nreg = lambda: c.model.reg.nreg
        c.evals[f"pplx_doc_vqa{suffix}"].data.greyout_frac = greyout_frac
        c.evals[f"pplx_doc_vqa{suffix}"].iter.maxtok = lambda: c.maxtok

    # VQA evals section
    special_tokens = 64 # rough estimate of special tokens count: bos, eos, sep, image line sep.

    c.evals.info_vqa.type = "vqa"
    c.evals.info_vqa.steps = 5_000
    c.evals.info_vqa.data.name = "vqa"
    c.evals.info_vqa.data.split = "infovqa_flat/val"
    c.evals.info_vqa.data.max_patches = lambda: c.data.max_patches
    c.evals.info_vqa.data.nreg = lambda: c.model.reg.nreg
    c.evals.info_vqa.decode.max_prefix = lambda: c.data.max_patches + special_tokens + 28  # covers 99%, 38 for all
    c.evals.info_vqa.decode.batch_size = 32
    c.evals.info_vqa.decode.max_decode = 1 + 11  # covers 99%, 20 for all
    c.evals.info_vqa.decode.T = 0.01
    c.evals.info_vqa.decode.omit_eos = True

    c.evals.doc_vqa.type = "vqa"
    c.evals.doc_vqa.steps = 5_000
    c.evals.doc_vqa.data.name = "vqa"
    c.evals.doc_vqa.data.split = "docvqa_flat/val"
    c.evals.doc_vqa.data.max_patches = lambda: c.data.max_patches
    c.evals.doc_vqa.data.nreg = lambda: c.model.reg.nreg
    c.evals.doc_vqa.decode.max_prefix = lambda: c.data.max_patches + special_tokens + 25  # covers 99%, 40 for all
    c.evals.doc_vqa.decode.batch_size = 32
    c.evals.doc_vqa.decode.max_decode = 1 + 16  # covers 99%, 33 for all
    c.evals.doc_vqa.decode.T = 0.01
    c.evals.doc_vqa.decode.omit_eos = True

    c.evals.st_vqa.type = "vqa"
    c.evals.st_vqa.steps = 5_000
    c.evals.st_vqa.data.name = "vqa"
    c.evals.st_vqa.data.split = "stvqa_flat/val"
    c.evals.st_vqa.data.max_patches = lambda: c.data.max_patches
    c.evals.st_vqa.data.nreg = lambda: c.model.reg.nreg
    c.evals.st_vqa.decode.max_prefix = lambda: c.data.max_patches + special_tokens + 18  # covers 99%, 27 for all
    c.evals.st_vqa.decode.batch_size = 32
    c.evals.st_vqa.decode.max_decode = 1 + 11  # covers 99%, 23 for all
    c.evals.st_vqa.decode.T = 0.01
    c.evals.st_vqa.decode.omit_eos = True

    # Nice to visualize predictions in W&B periodically
    c.evals.decode_info_vqa.type = "decode"
    c.evals.decode_info_vqa.steps = 2_500
    c.evals.decode_info_vqa.data.name = "vqa"
    c.evals.decode_info_vqa.data.split = "infovqa_flat/val"
    c.evals.decode_info_vqa.data.max_patches = lambda: c.data.max_patches
    c.evals.decode_info_vqa.data.nreg = lambda: c.model.reg.nreg
    c.evals.decode_info_vqa.decode.max_prefix = lambda: c.data.max_patches + special_tokens + 28
    c.evals.decode_info_vqa.decode.batch_size = 32
    c.evals.decode_info_vqa.decode.max_decode = 8  # For visualization/qualitative purposes only, so intentially extra short.
    c.evals.decode_info_vqa.decode.T = 0.01

    return c
