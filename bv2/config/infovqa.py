import sws


def get_config():
    c = sws.Config()
    c.seed = 0

    c.maxtok = 32_768

    c.data.name = "infovqa"
    c.data.split = "train"
    c.data.max_patches = 16_384
    c.data.nreg = lambda: c.model.reg.nreg

    c.iter.eagerness = 16  # The default, just as an example.
    c.iter.maxtok = lambda: c.maxtok

    c.nsteps = 30_000
    c.warmup_nsteps = 500
    c.lr = 1e-5
    c.wd = lambda: c.lr * 0.1

    c.model.dim = 4096
    c.model.depth = 4
    c.model.reg.nreg = 10
    c.model.stages = lambda: "half" if c.model.reg.nreg > 0 else "single"

    c.evals.pplx_val.type = "pplx"
    c.evals.pplx_val.steps = 180  # ~1ep for 8gpus maxpatch=16k, maxtok=32k.
    c.evals.pplx_val.data.name = "infovqa"
    c.evals.pplx_val.data.split = "val"
    c.evals.pplx_val.data.max_patches = lambda: c.data.max_patches
    c.evals.pplx_val.data.nreg = lambda: c.data.nreg
    c.evals.pplx_val.iter.eagerness = 2
    c.evals.pplx_val.iter.maxtok = lambda: c.maxtok

    return c
