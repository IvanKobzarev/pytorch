# bv2/tools/local_run -m bv2.train --config bv2/config/finevision.py nsteps:=250 c.evals=None

import sws

import bv2.data.finevision_info as fv


def get_config():
    c = sws.Config()
    c.seed = 0

    c.maxtok = 32_768 + 1

    c.data.name = "mix"
    c.data.mix = {
        **fv.weights(exclude=("docvqa", "infographic_vqa", "st_vqa", "textvqa", *fv.RIGI_EXCLUDES)),
        "docvqa": 10194//2, "docvqa_fmt": 10194//2,
        "infovqa": 4406//2, "infovqa_fmt": 4406//2,
        "stvqa": 17028//2, "stvqa_fmt": 17028//2,
        "textvqa": 21953//2, "textvqa_fmt": 21953//2,
    }
    c.data.common.max_patches = 784  # (448/16)^2 = 784; (2048/16)^2 = 16_384
    c.data.common.rand_max_patches.exp = None  # Disables this.
    c.data.common.rand_max_patches.mode = None
    c.data.common.nreg = lambda: c.model.reg.nreg
    c.data.common.greyout_frac = 0.03

    fv.add_to_config_(c.data, exclude=("docvqa", "infographic_vqa", "st_vqa", "textvqa", *fv.RIGI_EXCLUDES))

    CUSTOM_QFMT = {  # NOTE: Not using suffix here anymore, since not using FV for these.
        # TODO: Adapt max decode settings for this!
        "docvqa": "docvqa\n{q}",
        "infovqa": "infovqa\n{q}",
        "stvqa": "stvqa\n{q}",
        "textvqa": "textvqa\n{q}",
    }
    # However, the eval harness that FineVision used uses a different suffix ??
    # https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/lmms_eval/tasks/docvqa/_default_template_docvqa_yaml#L16C20-L16C71

    for name in ("docvqa", "infovqa", "stvqa", "textvqa"):
        c.data[name].name = "vqa"
        c.data[name].split = f"{name}/train"

        c.data[f"{name}_fmt"].name = "vqa"
        c.data[f"{name}_fmt"].split = f"{name}/train"
        c.data[f"{name}_fmt"].seed = 12345  # Sample different Q's than the no_fmt one.
        c.data[f"{name}_fmt"].qfmt = CUSTOM_QFMT[name]
        c.data[f"{name}_fmt"].lower_q = True
        c.data[f"{name}_fmt"].lower_a = True

    c.iter.maxtok = lambda: c.maxtok
    c.iter.eagerness = 24

    # 784 patches: 35m examples / 100k steps
    # 16384 patches: 13m examples / 100k steps
    c.nsteps = 100_000
    c.warmup_nsteps = 1000
    c.lr = 6e-4
    c.wd = lambda: c.lr * 0.1

    c.muon.param_modes = {"muon": [r".*mlp.l[12].weight", r".*att.[qkvo].weight", r".*img_emb.proj.weight", r".*txt_unemb.head.weight"],
                          "adam": [r".*"]}

    c.model.dim = 2048
    c.model.depth = 12
    c.model.reg.nreg = 0
    c.model.stages = lambda: "half" if c.model.reg.nreg > 0 else "single"
    c.model.txt_unemb.chunksz = 4096

    def eval_data(name, max_p=None, blind=False, qfmt=None, lower_q=True, lower_a=True):
        k = sws.Config()
        k.name = "vqa"
        k.split = name
        k.epochs = 1
        k.max_patches = max_p or (lambda: min(c.data.common.max_patches, 3136))
        k.nreg = lambda: c.model.reg.nreg
        k.greyout_frac = 1.0 if blind else 0.0
        if qfmt:
            k.qfmt = qfmt
        k.lower_q = lower_q
        k.lower_a = lower_a
        return k

    def pplx_eval(name, max_p=None, blind=False, qfmt=None):
        k = sws.Config()
        k.type = "pplx"
        k.steps = 5000 if not blind and qfmt else 20_000
        k.data = eval_data(name, max_p=max_p, blind=blind, qfmt=qfmt)
        k.iter.maxtok = lambda: c.maxtok
        return k

    for max_p in (196, 784, 3136):  # 224/448/896
        c.evals[f"docvqa/{max_p}/pplx"] = pplx_eval("docvqa/val", max_p)
        c.evals[f"docvqa_fmt/{max_p}/pplx"] = pplx_eval("docvqa/val", max_p, qfmt=CUSTOM_QFMT["docvqa"])
        c.evals[f"docvqa_fmt/{max_p}/blind/pplx"] = pplx_eval("docvqa/val", max_p, qfmt=CUSTOM_QFMT["docvqa"], blind=True)
        c.evals[f"infovqa/{max_p}/pplx"] = pplx_eval("infovqa/val", max_p)
        c.evals[f"infovqa_fmt/{max_p}/pplx"] = pplx_eval("infovqa/val", max_p, qfmt=CUSTOM_QFMT["infovqa"])
        c.evals[f"infovqa_fmt/{max_p}/blind/pplx"] = pplx_eval("infovqa/val", max_p, qfmt=CUSTOM_QFMT["infovqa"], blind=True)
        c.evals[f"stvqa/{max_p}/pplx"] = pplx_eval("stvqa/val", max_p)
        c.evals[f"stvqa_fmt/{max_p}/pplx"] = pplx_eval("stvqa/val", max_p, qfmt=CUSTOM_QFMT["stvqa"])
        c.evals[f"stvqa_fmt/{max_p}/blind/pplx"] = pplx_eval("stvqa/val", max_p, qfmt=CUSTOM_QFMT["stvqa"], blind=True)
        c.evals[f"textvqa/{max_p}/pplx"] = pplx_eval("textvqa/val", max_p)
        c.evals[f"textvqa_fmt/{max_p}/pplx"] = pplx_eval("textvqa/val", max_p, qfmt=CUSTOM_QFMT["textvqa"])
        c.evals[f"textvqa_fmt/{max_p}/blind/pplx"] = pplx_eval("textvqa/val", max_p, qfmt=CUSTOM_QFMT["textvqa"], blind=True)

    special_tokens = 64 # rough estimate of special tokens count: bos, eos, sep, image line sep.
    def vqa_eval(name, max_q, max_a, max_p=None, bs=32, blind=False, qfmt=None):
        k = sws.Config()
        k.type = "vqa"
        k.steps = lambda: range(5000, c.nsteps, 5000 if not blind and qfmt else 20_000)  # Skip first, then every 5k
        k.data = eval_data(name, max_p=max_p, blind=blind, qfmt=qfmt)
        k.lower_a = True
        k.decode.max_prefix = max_p + special_tokens + max_q if max_p else lambda: min(c.data.common.max_patches, 3136) + special_tokens + max_q
        k.decode.max_decode = 1 + max_a
        k.decode.batch_size = bs
        k.decode.T = 0.01
        k.decode.omit_eos = True
        return k

    # NOTE: The max_q was increased to cover the fmt!
    for max_p, bs in [(196, 128), (784, 64), (3136, 8)]:  # 224/448/896
        c.evals[f"docvqa/{max_p}/vqa"] = vqa_eval("docvqa_flat/val", max_q=25, max_a=16, max_p=max_p, bs=bs)    # covers 99% ; do 40, 33 for all
        c.evals[f"docvqa_fmt/{max_p}/vqa"] = vqa_eval("docvqa_flat/val", max_q=28, max_a=16, max_p=max_p, bs=bs, qfmt=CUSTOM_QFMT["docvqa"])    # covers 99% ; do 44, 33 for all
        c.evals[f"docvqa_fmt/{max_p}/blind/vqa"] = vqa_eval("docvqa_flat/val", max_q=28, max_a=16, max_p=max_p, bs=bs, qfmt=CUSTOM_QFMT["docvqa"], blind=True)
        c.evals[f"infovqa/{max_p}/vqa"] = vqa_eval("infovqa_flat/val", max_q=28, max_a=11, max_p=max_p, bs=bs)  # covers 99% ; do 38, 11 for all
        c.evals[f"infovqa_fmt/{max_p}/vqa"] = vqa_eval("infovqa_flat/val", max_q=33, max_a=11, max_p=max_p, bs=bs, qfmt=CUSTOM_QFMT["infovqa"])  # covers 99% ; do 46, 11 for all
        c.evals[f"infovqa_fmt/{max_p}/blind/vqa"] = vqa_eval("infovqa_flat/val", max_q=33, max_a=11, max_p=max_p, bs=bs, qfmt=CUSTOM_QFMT["infovqa"], blind=True)
        c.evals[f"stvqa/{max_p}/vqa"] = vqa_eval("stvqa_flat/val", max_q=18, max_a=11, max_p=max_p, bs=bs)      # covers 99% ; do 27, 23 for all
        c.evals[f"stvqa_fmt/{max_p}/vqa"] = vqa_eval("stvqa_flat/val", max_q=20, max_a=11, max_p=max_p, bs=bs, qfmt=CUSTOM_QFMT["stvqa"])      # covers 99% ; do 30, 23 for all
        c.evals[f"stvqa_fmt/{max_p}/blind/vqa"] = vqa_eval("stvqa_flat/val", max_q=20, max_a=11, max_p=max_p, bs=bs, qfmt=CUSTOM_QFMT["stvqa"], blind=True)

    # Nice to visualize predictions in W&B periodically. Very small/short decode for sanity-check only.
    # Single resolution to avoid bugginess.
    c.evals["decode_docvqa_fmt"].type = "decode"
    c.evals["decode_docvqa_fmt"].steps = lambda: range(5000, c.nsteps, 20_000)
    c.evals["decode_docvqa_fmt"].data = eval_data("docvqa_flat/val", qfmt=CUSTOM_QFMT["docvqa"])
    c.evals["decode_docvqa_fmt"].max_prefix = lambda: min(c.data.common.max_patches, 3136) + special_tokens + 28
    c.evals["decode_docvqa_fmt"].max_decode = 8
    c.evals["decode_docvqa_fmt"].batch_size = bs
    c.evals["decode_docvqa_fmt"].T = 0.01

    return c

def nosweep():
    for lr in (3e-4, 6e-4,):
        # Baselines:
        yield f"{lr=}", "c.data.common.max_patches=196"
        yield f"{lr=}", "c.data.common.max_patches=784"
        yield f"{lr=}", "c.data.common.max_patches=3136"

        # Randomized max-patches
        for exp, mode in [
            (1.616, None),  # 448px² in expectation
            (1.12, 196),  # 224px² as mode, 448px² in expectation
        ]:
            yield f"{lr=}", f"{exp=}", f"{mode=}", "c.data.common.max_patches=16_384"
