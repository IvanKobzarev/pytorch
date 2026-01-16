# torchrun --nproc_per_node=gpu -m bv2.train --config bv2/config/docvqa_ft.py
# test run: https://fairwandb.org/rigi/bv2/runs/dtf2q4au

import sws


def get_config():
    c = sws.Config()
    c.seed = 0

    c.maxtok = 32_768 + 1

    c.data.name = "vqa"
    c.data.split = "docvqa/train"
    c.data.max_patches = 3136
    c.data.nreg = lambda: c.model.reg.nreg
    c.data.greyout_frac = 0.0

    c.iter.maxtok = lambda: c.maxtok
    c.iter.eagerness = 24

    c.nsteps = 8_000
    c.warmup_nsteps = 1000
    c.lr = 3e-5
    c.wd = lambda: c.lr * 0.1

    c.muon.regexps = [r".*mlp.l[12].weight", r".*att.[qkvo].weight", r'.*unemb.head.weight', r'.*img_emb.proj.weight']
    c.muon.beta2 = 0.999

    c.model.dim = 2048
    c.model.depth = 12
    c.model.reg.nreg = 0
    c.model.stages = lambda: "half" if c.model.reg.nreg > 0 else "single"
    c.model.txt_unemb.chunksz = 4096
    c.init = '/checkpoint/rigi/bv2/workdirs/1212_103646/zhai-h100-8n-1212_103646-1448561-0.0006-0.99-0/ckpt-535000_bak'  # finevision + official datasets
    # c.init = '/checkpoint/rigi/bv2/workdirs/1125_145646/zhai-h100-8n-1125_145646-0-0.0006-(0.9, 0.99)-0/ckpt-179617'  # finevision only

    def pplx_eval(split, blind=False):
        k = sws.Config()
        k.type = "pplx"
        k.steps = 2000 if blind else 200
        k.data.name = "vqa"
        k.data.split = split
        k.data.max_patches = lambda: c.data.max_patches
        k.data.nreg = lambda: c.model.reg.nreg
        k.data.greyout_frac = 1.0 if blind else 0.0
        k.iter.maxtok = lambda: c.maxtok
        return k

    c.evals["docvqa/pplx"] = pplx_eval("docvqa/val")
    c.evals["docvqa/pplx_blind"] = pplx_eval("docvqa/val", blind=True)

    special_tokens = 64 # rough estimate of special tokens count: bos, eos, sep, image line sep.
    def vqa_eval(split, max_q, max_a, blind=False, suffix=""):
        k = sws.Config()
        k.type = "vqa"
        k.steps = lambda: range(0, c.nsteps, 2000 if blind else 500)  # Skip first, then every 5k
        k.data.name = "vqa"
        k.data.split = split
        k.data.max_patches = lambda: c.data.max_patches
        k.data.nreg = lambda: c.model.reg.nreg
        k.data.greyout_frac = 1.0 if blind else 0.0
        k.data.question_suffix = suffix
        k.decode.max_prefix = lambda: c.data.max_patches + special_tokens + max_q
        k.decode.batch_size = 32
        k.decode.max_decode = 1 + max_a
        k.decode.T = 0.01
        k.decode.omit_eos = True
        k.lower = True
        return k

    c.evals['docvqa/vqa'] = vqa_eval("docvqa_flat/val", max_q=25, max_a=16)    # covers 99% ; do 40, 33 for all
    c.evals['docvqa/vqa_suffix'] = vqa_eval("docvqa_flat/val", max_q=25, max_a=16, suffix="\nOffer a terse response.")    # covers 99% ; do 40, 33 for all
    c.evals['docvqa/vqa_blind'] = vqa_eval("docvqa_flat/val", max_q=25, max_a=16, blind=True)

    # Nice to visualize predictions in W&B periodically
    c.evals.decode_doc_vqa.type = "decode"
    c.evals.decode_doc_vqa.steps = lambda: [0, c.nsteps]
    c.evals.decode_doc_vqa.data.name = "vqa"
    c.evals.decode_doc_vqa.data.split = "docvqa_flat/val"
    c.evals.decode_doc_vqa.data.max_patches = lambda: c.data.max_patches
    c.evals.decode_doc_vqa.data.nreg = lambda: c.model.reg.nreg
    c.evals.decode_doc_vqa.decode.max_prefix = lambda: c.data.max_patches + special_tokens + 28
    c.evals.decode_doc_vqa.decode.batch_size = 32
    c.evals.decode_doc_vqa.decode.max_decode = 8  # For visualization/qualitative purposes only, so intentially extra short.
    c.evals.decode_doc_vqa.decode.T = 0.01

    return c
