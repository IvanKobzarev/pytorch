# bv2/tools/local_run -m bv2.train --config bv2/config/mango_sm_alt_2b.py nsteps:=250
# python -m bv2.launch bv2/config/mango_sm_alt_2b.py --qos h200_lowest --gpus-per-node 8 --nodes 1 'name:=f"mango_sm_alt_2b-{c.xid}-{c.wid}"'

import sws


def get_config():
    c = sws.Config()
    c.seed = 0

    c.cache = True
    c.maxtok = 128*1024

    c.data.name = "mango_sm_alt_2b"
    c.data.max_patches = 784
    c.data.rand_max_patches.exp = None
    c.data.rand_max_patches.mode = None
    c.data.nreg = lambda: c.model.reg.nreg
    c.data.greyout_frac = 0.0
    c.data.tokenizer.first_N = 4096
    c.iter.maxtok = lambda: c.maxtok

    c.nsteps = 1_000_000
    c.warmup_nsteps = 1000

    c.lr_adam = 1e-2
    c.lr_muon = 3e-3
    c.wd = 1e-5

    c.muon.param_modes = {"muon_h": [r".*mlp.l[12].weight", r".*att.[qkvo].weight", r".*img_emb.proj.weight", r".*txt_unemb.head.weight"],
                          "embedding": [r".*txt_emb.emb.weight"],
                          "adam": [r".*"]}

    c.model.dim = 2048
    c.model.depth = 12
    c.model.reg.nreg = 0
    c.model.stages = lambda: "half" if c.model.reg.nreg > 0 else "single"

    def eval_data(split="val", max_p=None):
        k = sws.Config()
        k.name = "mango_sm_alt_2b"
        k.split = split
        k.epochs = 1
        k.max_patches = max_p or (lambda: c.data.max_patches)
        k.tokenizer.first_N = lambda: c.data.tokenizer.first_N
        k.nreg = lambda: c.model.reg.nreg
        return k

    # Perplexity on val split.
    c.evals.pplx_val.type = "pplx"
    c.evals.pplx_val.steps = 2000
    c.evals.pplx_val.data = eval_data("val")
    c.evals.pplx_val.iter.maxtok = lambda: c.maxtok

    special_tokens = 64  # rough estimate of special tokens count: bos, eos, sep, image line sep.
    def decode_eval(split="val_mini", T=0.01, max_p=None):
        k = sws.Config()
        k.type = "decode"
        k.steps = 5000
        k.data = eval_data(split, max_p=max_p)
        k.decode.max_prefix = lambda: c.data.max_patches + special_tokens + 1
        k.decode.batch_size = 32
        k.decode.max_decode = 64
        k.decode.T = T
        return k

    c.evals.decode_val01 = decode_eval(T=0.01)
    c.evals.decode_val3 = decode_eval(T=0.3)

    return c

def sweep():
    for lr in (0.0003, 0.0006, 0.001, 0.003, 0.006):
        yield f"c.lr_muon={lr}"
