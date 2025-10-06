import sws


def get_config():
    c = sws.Config()
    c.seed = 0

    c.data_name = "deduped_code"
    c.data.split = "train"

    c.maxtok = 32_768

    # TODO: None of these are tuned.
    c.nsteps = 100_000
    c.warmup_nsteps = 2000
    c.lr = 1e-5
    c.wd = 1e-5

    c.model.dim = 4096
    c.model.depth = 4

    c.model.txt.posemb = True
    c.model.glope = 0  # No glope by deafult

    c.evals.pplx.type = "pplx"
    c.evals.pplx.steps = 500
    c.evals.pplx.data_name = "deduped_code"
    c.evals.pplx.data.split = "val"

    return c
