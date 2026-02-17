# bv2/tools/local_run -m bv2.train --config bv2/config/code.py nsteps:=250

import sws


def get_config():
    c = sws.Config()
    c.seed = 0

    c.maxtok = 2*32_768 + 1

    c.data.name = "deduped_code"
    c.data.split = "codewall_train_0.25M"
    c.iter.maxtok = lambda: c.maxtok

    # Empirically, with maxtok=32k, we get on avg 28.22 examples per gpu.
    # So this is roughly an epoch of 16M on 8 GPUs for this situation:
    # For the 0.25M filtered subset, it's 31.06 for 32k and 64.28 for 65k.
    c.nsteps = int(16_000_000 / (8 * 64.28))

    c.warmup_nsteps = 2000
    c.lr = 1e-3  # NEEDS TO BE TUNED
    c.wd = lambda: 0.1*c.lr

    c.muon.param_modes = {"muon": [r".*mlp.l[12].weight", r".*att.[qkvo].weight", r".*img_emb.proj.weight", r".*txt_unemb.head.weight"],
                          "adam": [r".*"]}

    c.model.dim = 2048
    c.model.depth = 8

    c.model.txt.posemb = True
    c.model.glope = 0  # No glope by default
    c.model.txt_unemb.chunksz = 4096

    c.evals.pplx.type = "pplx"
    c.evals.pplx.steps = 250  # 500 steps is ~1ep of the 0.25M subset.
    c.evals.pplx.data.epochs = 1
    c.evals.pplx.data.name = "deduped_code"
    c.evals.pplx.data.split = "codewall_val"
    c.evals.pplx.iter.maxtok = lambda: c.maxtok

    # One example is on average 1k tokens. So we can go for 1M, 4M, 16M, 64M examples.
    # This way, we grow by 4x each time, which is also the same as going 2gpu -> 8gpu.
    # And if we were to continue 4x'ing, we'd get to 256M and then 1B, which are nice.
    # Also, all these numbers are nicely divisible by 32 or 64, so going to 4-8 hosts is nice too.
    # Finally, at least 1-16M are nicely sweepable with a single node, 64M starts to take a while.
    # c.data.first_N = 16_000_000

    return c

def nosweep():
    for seed in (0, 1, 2):
        for maxtok in (32_768 + 1, 2*32_768 + 1):
            yield "nsteps=2000",  f"c.{seed=}", f"c.{maxtok=}"
