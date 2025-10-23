# torchrun --nproc_per_node=gpu -m bv2.train --config bv2/config/finevision.py

import sws


def get_config():
    c = sws.Config()
    c.seed = 0

    c.maxtok = 32_768

    c.data.name = "finevision"
    # c.data.max_patches = 16_384  # FineVision resized to max 2048
    c.data.max_patches = 784  # (448/16)^2 = 784; (224/16)^2 = 196
    c.data.nreg = lambda: c.model.reg.nreg
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

    c.nsteps = 100_000
    c.warmup_nsteps = 1000
    c.lr = 1e-4
    c.wd = lambda: c.lr * 0.1

    c.model.dim = 2048
    c.model.depth = 12
    c.model.reg.nreg = 10
    c.model.stages = lambda: "half" if c.model.reg.nreg > 0 else "single"
    c.model.txt_unemb.chunks = 8

    c.evals.pplx_st_vqa.type = "pplx"
    c.evals.pplx_st_vqa.steps = 2000
    c.evals.pplx_st_vqa.data.name = "vqa"
    c.evals.pplx_st_vqa.data.split = "stvqa/val"
    c.evals.pplx_st_vqa.data.max_patches = lambda: c.data.max_patches
    c.evals.pplx_st_vqa.data.nreg = lambda: c.model.reg.nreg
    c.evals.pplx_st_vqa.iter.maxtok = lambda: c.maxtok

    c.evals.pplx_info_vqa.type = "pplx"
    c.evals.pplx_info_vqa.steps = 2000
    c.evals.pplx_info_vqa.data.name = "vqa"
    c.evals.pplx_info_vqa.data.split = "infovqa/val"
    c.evals.pplx_info_vqa.data.max_patches = lambda: c.data.max_patches
    c.evals.pplx_info_vqa.data.nreg = lambda: c.model.reg.nreg
    c.evals.pplx_info_vqa.iter.maxtok = lambda: c.maxtok

    c.evals.pplx_text_vqa.type = "pplx"
    c.evals.pplx_text_vqa.steps = 2000
    c.evals.pplx_text_vqa.data.name = "vqa"
    c.evals.pplx_text_vqa.data.split = "textvqa/val"
    c.evals.pplx_text_vqa.data.max_patches = lambda: c.data.max_patches
    c.evals.pplx_text_vqa.data.nreg = lambda: c.model.reg.nreg
    c.evals.pplx_text_vqa.iter.maxtok = lambda: c.maxtok

    c.evals.pplx_doc_vqa.type = "pplx"
    c.evals.pplx_doc_vqa.steps = 2000
    c.evals.pplx_doc_vqa.data.name = "vqa"
    c.evals.pplx_doc_vqa.data.split = "docvqa/val"
    c.evals.pplx_doc_vqa.data.max_patches = lambda: c.data.max_patches
    c.evals.pplx_doc_vqa.data.nreg = lambda: c.model.reg.nreg
    c.evals.pplx_doc_vqa.iter.maxtok = lambda: c.maxtok

    return c
