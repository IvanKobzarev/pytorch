# torchrun --nproc_per_node=gpu -m bv2.train --config bv2/config/infovqa.py c.model.reg.nreg=10

import sws


def get_config():
    c = sws.Config()
    c.seed = 0

    c.maxtok = 32_768

    c.data.name = "vqa"
    c.data.split = "infovqa/train"
    c.data.max_patches = 16_384
    c.data.nreg = lambda: c.model.reg.nreg
    c.data.greyout_frac = 0.03

    c.iter.maxtok = lambda: c.maxtok
    c.iter.eagerness = 24  # Amount of CPUs per GPU.

    c.nsteps = 30_000
    c.warmup_nsteps = 500
    c.lr = 1e-5
    c.wd = lambda: c.lr * 0.1

    c.model.dim = 2048
    c.model.depth = 12
    c.model.reg.nreg = 10
    c.model.stages = lambda: "half" if c.model.reg.nreg > 0 else "single"
    c.model.txt_unemb.chunks = 8

    # These are defined here so they are easy to set on commandline or sweep. Just an example.
    c.decode_max_patches = lambda: c.data.max_patches
    c.decode_batch_size = 32

    special_tokens = 64 # rough estimate of special tokens count: bos, eos, sep, image line sep.

    for s, frac in [("", 0.0), ("_blind", 1.0)]:
        data = lambda max_patches, frac=frac: dict(
            name = "vqa",
            split = "infovqa_flat/val",
            max_patches = max_patches,
            nreg = lambda: c.model.reg.nreg,
            greyout_frac = frac,
        )

        c.evals[f"val_pplx{s}"].type = "pplx"
        c.evals[f"val_pplx{s}"].steps = 180  # ~1ep for 8gpus maxpatch=16k, maxtok=32k
        c.evals[f"val_pplx{s}"].data = data(max_patches=lambda: c.data.max_patches)
        c.evals[f"val_pplx{s}"].iter.maxtok = lambda: c.maxtok

        c.evals[f"val_vqa{s}"].type = "vqa"
        c.evals[f"val_vqa{s}"].steps = 180
        c.evals[f"val_vqa{s}"].data = data(max_patches=lambda: c.decode_max_patches)
        c.evals[f"val_vqa{s}"].decode.max_prefix = lambda: c.decode_max_patches + special_tokens + 28  # covers 99%, 38 for all
        c.evals[f"val_vqa{s}"].decode.batch_size = lambda: c.decode_batch_size
        c.evals[f"val_vqa{s}"].decode.max_decode = 1 + 11  # covers 99%, 20 for all
        c.evals[f"val_vqa{s}"].decode.T = 0.01
        c.evals[f"val_vqa{s}"].decode.omit_eos = True

    # Nice to visualize predictions in W&B periodically
    c.evals.decode.type = "decode"
    c.evals.decode.steps = 900  # ~5ep
    c.evals.decode.data.name = "vqa"
    c.evals.decode.data.split = "infovqa_flat/val"
    c.evals.decode.data.max_patches = lambda: c.decode_max_patches
    c.evals.decode.data.nreg = lambda: c.model.reg.nreg
    c.evals.decode.decode.max_prefix = lambda: c.decode_max_patches + special_tokens + 28
    c.evals.decode.decode.batch_size = lambda: c.decode_batch_size
    c.evals.decode.decode.max_decode = 8  # For visualization/qualitative purposes only, so intentionally extra short.
    c.evals.decode.decode.T = 0.01

    return c
