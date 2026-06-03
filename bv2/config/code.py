# bv2/tools/launch_local bv2.train bv2/config/code.py nexamples=10_000

import sws


def _int(s):
    assert s[-1] == "M"
    return round(float(s[:-1]) * 1000000)


# Valid `c.voc` values are "bytes", "code_<N>k" tokenizers such as "code_4k",
# "l4_full" for the complete L4 tokenizer, or "l4_<N>k" for its first_N version
# after leaving room for 16 reserved IDs (as the code_Nk already do).
def make_tokenizer(c, ctok):
    ctok.regex = lambda: "o200k" if c.voc.startswith("l4_") else "code"
    ctok.path = lambda: (
        "/checkpoint/rigi/bv2/l4_200k_base.model" if c.voc.startswith("l4_")
        else "/checkpoint/rigi/bv2/code_4k.tt" if c.voc == "bytes"
        else f"/checkpoint/rigi/bv2/{c.voc}.tt"
    )
    ctok.first_N = lambda: (
        256 if c.voc == "bytes"
        else None if c.voc == "l4_full" or not c.voc.startswith("l4_")
        else int(c.voc[3:-1]) * 1000 - 16
    )


def get_config():
    c = sws.Config()
    c.seed = 0

    c.maxtok = 256 * 1024

    c.data.name = "deduped_code"
    c.dataset_size = "0.25M"
    c.data.split = lambda: f"codewall_train_{c.dataset_size}"
    c.data.cache = True
    c.voc = "code_4k"
    make_tokenizer(c, c.data.tokenizer)
    c.iter.maxtok = lambda: c.maxtok

    c.nexamples = lambda: min(_int(c.dataset_size), 1_000_000) * 64
    c.warmup_nexamples = 150_000
    c.ckpt_at_examples = 1_000_000

    c.lr_muon = 1e-3  # ALWAYS SWEEP
    c.lr_adam = lambda: 1.0 * c.lr_muon  # 0.3 or 1.0
    c.wd = 0
    c.muon.param_modes = {"muon_h": [r".*mlp.l[12].weight", r".*att.[qkvo].weight", r".*img_emb.proj.weight", r".*txt_unemb.head.weight"],
                          "embedding": [r".*txt_emb.emb.weight"],
                          "adam": [r".*"]}

    c.model.dim = 2048
    c.model.depth = 8

    c.model.txt.posemb = True
    c.model.glope = 0  # No glope by default
    c.model.txt_unemb.chunksz = 8192

    c.evals.pplx.type = "pplx"
    c.evals.pplx.at_examples = 100_000
    c.evals.pplx.data.epochs = 1
    c.evals.pplx.data.name = "deduped_code"
    c.evals.pplx.data.cache = lambda: c.data.cache
    c.evals.pplx.data.split = "codewall_val"
    c.evals.pplx.data.tokenizer = lambda: getattr(c.data, "tokenizer", None)
    c.evals.pplx.iter.maxtok = 2**16

    return c
