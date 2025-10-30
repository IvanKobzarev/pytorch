from datetime import datetime
from getpass import getuser

import sws


def get_config():
    c = sws.Config()
    c.seed = 0

    c.maxtok = 2048

    c.data.name = "synth_ocr"
    c.data.tokenizer.first_N = 10_000
    c.data.tiptoi = 0
    c.data.add_hw = False
    c.data.add_row_sep = False

    c.iter.eagerness = 16  # The default, just as an example.
    c.iter.maxtok = lambda: c.maxtok

    c.nsteps = 20_000
    c.warmup_nsteps = 2000
    c.lr = 1e-5
    c.wd = 1e-5

    c.model.dim = 4096
    c.model.depth = 4

    c.model.txt.posemb = True
    c.model.img.posemb = True

    # No glope / image separators by default
    c.model.glope = 0
    c.model.sep.nreg = 0

    c.model.img.tiptoi = lambda: c.data.tiptoi

    c.evals.pplx.type = "pplx"
    c.evals.pplx.steps = 500
    c.evals.pplx.iter.maxtok = lambda: c.maxtok
    c.evals.pplx.iter.seed = 31337  # Defines the "fixed val split".
    c.evals.pplx.data.name = "synth_ocr"
    # Carry over all other settings from train
    c.evals.pplx.data.tokenizer.first_N = lambda: c.data.tokenizer.first_N
    c.evals.pplx.data.tiptoi = lambda: c.data.tiptoi
    c.evals.pplx.data.add_hw = lambda: c.data.add_hw
    c.evals.pplx.data.add_row_sep = lambda: c.data.add_row_sep

    c.evals.inference.type = "decode"
    c.evals.inference.steps = 5000
    # Decoding happens with batched, not packed data, so maxtok can be smaller
    c.evals.inference.iter.max_prefix = 256  # stress test filtering at test time
    c.evals.inference.iter.seed = 31337  # Defines the "fixed val split".
    c.evals.inference.iter.batch_size = 32
    c.evals.inference.data.name = "synth_ocr"
    c.evals.inference.data.tokenizer.first_N = lambda: c.data.tokenizer.first_N
    c.evals.inference.data.tiptoi = lambda: c.data.tiptoi
    c.evals.inference.data.add_hw = lambda: c.data.add_hw
    c.evals.inference.data.add_row_sep = lambda: c.data.add_row_sep
    c.evals.inference.args.max_decode = 128
    # Can't be zero yet (plan: optimize with argmax).
    c.evals.inference.args.T = 1e-3

    return c
