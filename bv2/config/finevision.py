# torchrun --nproc_per_node=gpu -m bv2.train --config bv2/config/finevision.py

import sws


def get_config():
    c = sws.Config()
    c.seed = 0

    c.maxtok = 32_768

    c.data.name = "finevision"
    c.data.split = "train"
    c.data.max_patches = 16_384  # FineVision resized to max 2048
    c.data.nreg = lambda: c.model.reg.nreg
    c.data.include = [".*"]  # List of regex patterns to match dataset names
    c.data.exclude = ["st_vqa", "text_theoremqa", "wordart"]  # List of regex patterns to exclude from dataset names

    c.iter.maxtok = lambda: c.maxtok

    c.nsteps = 500_000
    c.warmup_nsteps = 500
    # lr and wd are not tuned yet.
    c.lr = 1e-5
    c.wd = lambda: c.lr * 0.1

    c.model.dim = 4096
    c.model.depth = 4
    c.model.reg.nreg = 10
    c.model.stages = lambda: "half" if c.model.reg.nreg > 0 else "single"
    c.model.txt_unemb.chunks = 8

    c.evals.pplx_st_vqa.type = "pplx"
    c.evals.pplx_st_vqa.steps = 5000
    c.evals.pplx_st_vqa.data.name = "finevision"
    c.evals.pplx_st_vqa.data.split = "train"
    c.evals.pplx_st_vqa.data.include = ["st_vqa"]
    c.evals.pplx_st_vqa.data.max_patches = lambda: c.data.max_patches
    c.evals.pplx_st_vqa.data.nreg = lambda: c.model.reg.nreg
    c.evals.pplx_st_vqa.iter.maxtok = lambda: c.maxtok

    c.evals.pplx_text_theoremqa.type = "pplx"
    c.evals.pplx_text_theoremqa.steps = 5000
    c.evals.pplx_text_theoremqa.data.name = "finevision"
    c.evals.pplx_text_theoremqa.data.split = "train"
    c.evals.pplx_text_theoremqa.data.include = ["text_theoremqa"]
    c.evals.pplx_text_theoremqa.data.max_patches = lambda: c.data.max_patches
    c.evals.pplx_text_theoremqa.data.nreg = lambda: c.model.reg.nreg
    c.evals.pplx_text_theoremqa.iter.maxtok = lambda: c.maxtok

    c.evals.pplx_wordart.type = "pplx"
    c.evals.pplx_wordart.steps = 5000
    c.evals.pplx_wordart.data.name = "finevision"
    c.evals.pplx_wordart.data.split = "train"
    c.evals.pplx_wordart.data.include = ["wordart"]
    c.evals.pplx_wordart.data.max_patches = lambda: c.data.max_patches
    c.evals.pplx_wordart.data.nreg = lambda: c.model.reg.nreg
    c.evals.pplx_wordart.iter.maxtok = lambda: c.maxtok

    return c
