# torchrun --nproc_per_node=gpu -m bv2.train --config bv2/config/finevision.py

import sws


def get_config():
    c = sws.Config()
    c.seed = 0

    c.maxtok = 32_768 + 1

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
    c.lr = 3e-4
    c.wd = lambda: c.lr * 0.1
    c.beta2 = 0.99

    c.model.dim = 2048
    c.model.depth = 12
    c.model.reg.nreg = 0
    c.model.stages = lambda: "half" if c.model.reg.nreg > 0 else "single"
    c.model.txt_unemb.chunksz = 4096

    def pplx_eval(split, blind=False):
        k = sws.Config()
        k.type = "pplx"
        k.steps = 5000 if blind else 2500
        k.data.name = "vqa"
        k.data.split = split
        k.data.max_patches = lambda: c.data.max_patches
        k.data.epochs = 1
        k.data.nreg = lambda: c.model.reg.nreg
        k.data.greyout_frac = 1.0 if blind else 0.0
        k.iter.maxtok = lambda: c.maxtok
        return k

    c.evals["docvqa/pplx"] = pplx_eval("docvqa/val")
    c.evals["docvqa/pplx_blind"] = pplx_eval("docvqa/val", blind=True)
    c.evals["infovqa/pplx"] = pplx_eval("infovqa/val")
    c.evals["infovqa/pplx_blind"] = pplx_eval("infovqa/val", blind=True)
    c.evals["stvqa/pplx"] = pplx_eval("stvqa/val")
    c.evals["stvqa/pplx_blind"] = pplx_eval("stvqa/val", blind=True)
    c.evals["textvqa/pplx"] = pplx_eval("textvqa/val")
    c.evals["textvqa/pplx_blind"] = pplx_eval("textvqa/val", blind=True)

    special_tokens = 64 # rough estimate of special tokens count: bos, eos, sep, image line sep.
    def vqa_eval(split, max_q, max_a, blind=False, suffix=""):
        k = sws.Config()
        k.type = "vqa"
        k.steps = lambda: range(5000, c.nsteps, 20_000 if blind else 5000)  # Skip first, then every 5k
        k.data.name = "vqa"
        k.data.split = split
        k.data.max_patches = lambda: c.data.max_patches
        k.data.epochs = 1
        k.data.nreg = lambda: c.model.reg.nreg
        k.data.greyout_frac = 1.0 if blind else 0.0
        k.data.question_suffix = suffix
        k.lower_a = True
        k.decode.max_prefix = lambda: c.data.max_patches + special_tokens + max_q
        k.decode.batch_size = 32
        k.decode.max_decode = 1 + max_a
        k.decode.T = 0.01
        k.decode.omit_eos = True
        return k

    c.evals['docvqa/vqa'] = vqa_eval("docvqa_flat/val", max_q=25, max_a=16, suffix="\nOffer a terse response.")    # covers 99% ; do 40, 33 for all
    c.evals['docvqa/vqa_blind'] = vqa_eval("docvqa_flat/val", max_q=25, max_a=16, blind=True, suffix="\nOffer a terse response.")
    c.evals['infovqa/vqa'] = vqa_eval("infovqa_flat/val", max_q=28, max_a=11, suffix="\nAnswer the question with a short phrase.")  # covers 99% ; do 38, 11 for all
    c.evals['infovqa/vqa_blind'] = vqa_eval("infovqa_flat/val", max_q=28, max_a=11, blind=True, suffix="\nAnswer the question with a short phrase.")
    c.evals['stvqa/vqa'] = vqa_eval("stvqa_flat/val", max_q=18, max_a=11)      # covers 99% ; do 27, 23 for all
    c.evals['stvqa/vqa_blind'] = vqa_eval("stvqa_flat/val", max_q=18, max_a=11, blind=True)

    # Nice to visualize predictions in W&B periodically
    c.evals.decode_doc_vqa.type = "decode"
    c.evals.decode_doc_vqa.steps = lambda: range(5000, c.nsteps, 20_000)
    c.evals.decode_doc_vqa.data.name = "vqa"
    c.evals.decode_doc_vqa.data.split = "docvqa_flat/val"
    c.evals.decode_doc_vqa.data.max_patches = lambda: c.data.max_patches
    c.evals.decode_doc_vqa.data.nreg = lambda: c.model.reg.nreg
    c.evals.decode_doc_vqa.decode.max_prefix = lambda: c.data.max_patches + special_tokens + 28
    c.evals.decode_doc_vqa.decode.batch_size = 32
    c.evals.decode_doc_vqa.decode.max_decode = 8  # For visualization/qualitative purposes only, so intentially extra short.
    c.evals.decode_doc_vqa.decode.T = 0.01

    return c
