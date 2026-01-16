import sws


def get_config():
    c = sws.Config()
    c.seed = 0

    c.maxtok = 32_768 + 1

    c.data.name = "deduped_code"
    c.data.split = "train"
    c.iter.eagerness = 24
    c.iter.maxtok = lambda: c.maxtok

    # Empirically, with maxtok=32k, we get on avg 28.22 examples per gpu.
    # So this is roughly an epoch of 16M on 8 GPUs for this situation:
    c.nsteps = int(16_000_000 / (8 * 28.22))

    c.warmup_nsteps = 2000
    c.lr = 1e-5
    c.wd = lambda: 0.1*c.lr

    c.muon.regexps = [r".*mlp.l[12].weight", r".*att.[qkvo].weight", r".*txt_unemb.head.weight", r".*img_emb.proj.weight"]

    c.model.dim = 4096
    c.model.depth = 4

    c.model.txt.posemb = True
    c.model.glope = 0  # No glope by deafult
    c.model.txt_unemb.chunksz = 4096

    c.evals.pplx.type = "pplx"
    c.evals.pplx.steps = 500
    c.evals.pplx.data.name = "deduped_code"
    c.evals.pplx.data.split = "val"
    c.evals.pplx.iter.maxtok = lambda: c.maxtok

    # contains 170_492_035 examples
    # a weekend d4/4096 on 2 GPUs went over 55mio examples or 6B tokens
    # this means on average only 1k tokens per example (?) although I see an example dropped for being >32k every other step
    # keeping in mind this was for 2-gpu, it means on a single full machine we'd have 4x that, so 24B

    # One example is on average 1k tokens. So we can go for 1M, 4M, 16M, 64M examples.
    # This way, we grow by 4x each time, which is also the same as going 2gpu -> 8gpu.
    # And if we were to continue 4x'ing, we'd get to 256M and then 1B, which are nice.
    # Also, all these numbers are nicely divisible by 32 or 64, so going to 4-8 hosts is nice too.
    # Finally, at least 1-16M are nicely sweepable with a single node, 64M starts to take a while.
    c.data.first_N = 16_000_000

    return c
