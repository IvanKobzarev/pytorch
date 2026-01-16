import sws


def get_config():
    c = sws.Config()
    c.seed = 0

    c.maxtok = 8192 + 1

    def get_data_config(seed=0, n=None):
        dc = sws.Config()
        dc.name = "synth_ocr"
        dc.seed = seed
        dc.n = n

        dc.tokenizer.first_N = 10_000

        dc.tiptoi = 0
        dc.add_hw = False
        dc.add_row_sep = False

        dc.min_h = 256
        dc.min_w = 256
        dc.max_h = 512
        dc.max_w = 512
        dc.random_angle = 20
        dc.random_pad = 16
        dc.fs_jitter = 8

        return dc

    c.data = get_data_config()

    c.iter.eagerness = 16  # The default, just as an example.
    c.iter.maxtok = lambda: c.maxtok

    c.nsteps = 20_000
    c.warmup_nsteps = 2000
    c.lr = 3e-4
    c.wd = 1e-4

    c.muon.regexps = [r".*mlp.l[12].weight", r".*att.[qkvo].weight", r".*txt_unemb.head.weight", r".*img_emb.proj.weight"]

    c.model.dim = 2048
    c.model.depth = 8

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
    c.evals.pplx.data = get_data_config(n=128)

    c.evals.vqa.type = "vqa"
    c.evals.vqa.steps = lambda: range(1000, c.nsteps, 2000)
    c.evals.vqa.data = get_data_config(seed=31337, n=128)
    c.evals.vqa.decode.max_prefix = 1024
    c.evals.vqa.decode.batch_size = 32
    c.evals.vqa.decode.max_decode = 256
    c.evals.vqa.decode.T = 1e-3
    c.evals.vqa.decode.omit_eos = True

    c.evals.decode.type = "decode"
    c.evals.decode.steps = lambda: range(1000, c.nsteps, 2000)
    c.evals.decode.data = get_data_config(seed=31337, n=128)
    c.evals.decode.decode.max_prefix = 1024
    c.evals.decode.decode.batch_size = 32
    c.evals.decode.decode.max_decode = 256
    c.evals.decode.decode.T = 1e-3

    c.ckpt_steps = 1000

    return c
