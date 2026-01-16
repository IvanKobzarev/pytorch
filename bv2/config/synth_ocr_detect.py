import sws


def get_config():
    c = sws.Config()
    c.seed = 0

    c.maxtok = 2048 + 1

    c.data.name = "synth_ocr_detect"
    c.data.mode = "ltwh"

    c.iter.eagerness = 16  # The default, just as an example.
    c.iter.maxtok = lambda: c.maxtok

    # By default no tiptoi and no row separators/hw
    c.data.tiptoi = 0
    c.data.add_hw = False
    c.data.add_row_sep = False

    c.nsteps = 150_000
    c.warmup_nsteps = 2000
    c.lr = 1e-5
    c.wd = 1e-5

    c.muon.regexps = [r".*mlp.l[12].weight", r".*att.[qkvo].weight", r'.*unemb.head.weight', r'.*img_emb.proj.weight']

    c.model.dim = 4096
    c.model.depth = 4

    c.model.txt.posemb = True
    c.model.img.posemb = True

    # No glope / image separators by deafult
    c.model.glope = 0
    c.model.sep.nreg = 0

    c.model.img.tiptoi = lambda: c.data.tiptoi

    c.evals.pplx.type = "pplx"
    c.evals.pplx.steps = 500
    c.evals.pplx.iter.maxtok = lambda: c.maxtok
    c.evals.pplx.iter.seed = 31337  # Defines the "fixed val split".
    c.evals.pplx.data.name = "synth_ocr_detect"
    c.evals.pplx.data.n = 2048
    # Carry over all other settings from train
    c.evals.pplx.data.mode = lambda: c.data.mode
    c.evals.pplx.data.tiptoi = lambda: c.data.tiptoi
    c.evals.pplx.data.add_hw = lambda: c.data.add_hw
    c.evals.pplx.data.add_row_sep = lambda: c.data.add_row_sep

    return c
