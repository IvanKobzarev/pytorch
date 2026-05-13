# bv2/tools/launch_local bv2.train bv2/config/code.py nsteps:=250

import sws


def _int(s):
    assert s[-1] == "M"
    return round(float(s[:-1]) * 1000000)


def get_config():
    c = sws.Config()
    c.seed = 0

    c.maxtok = 256 * 1024

    c.data.name = "deduped_code"
    c.dataset_size = "0.25M"
    c.data.split = lambda: f"codewall_train_{c.dataset_size}"
    c.data.cache = True
    c.data.tokenizer.regex = "code"
    c.voc = "code_4k"
    c.data.tokenizer.path = lambda: f"/checkpoint/rigi/bv2/{c.voc if c.voc != 'bytes' else 'code_4k'}.tt"  # For bytes, use any file, say 4k.
    c.data.tokenizer.first_N = lambda: 256 if c.voc == "bytes" else None
    c.iter.maxtok = lambda: c.maxtok

    c.nsteps = lambda: int(min(_int(c.dataset_size), 1000000) * 64 * _AVGSEQLEN[c.dataset_size][c.voc]) // (c.maxtok * 8)
    c.warmup_nsteps = 100

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
    c.evals.pplx.steps = 250  # 500 steps is ~1ep of the 0.25M subset.
    c.evals.pplx.data.epochs = 1
    c.evals.pplx.data.name = "deduped_code"
    c.evals.pplx.data.cache = lambda: c.data.cache
    c.evals.pplx.data.split = "codewall_val"
    c.evals.pplx.data.tokenizer = lambda: getattr(c.data, "tokenizer", None)
    c.evals.pplx.iter.maxtok = 2**16
    return c


# Mean tokens/example per (subset, tokenizer), measured on N=4000 random samples
# (seed=0) of each codewall_train_{subset} split. l4_<v> = l4_200k_base with
# o200k regex and first_N = <vocab>-16. See bv2/config/code.py history / chat.
_AVGSEQLEN = {
    "0.25M": {"bytes": 4022.34, "code_1k": 1876.94, "code_2k": 1612.66, "code_3k": 1501.79, "code_4k": 1434.23, "code_8k": 1303.57, "code_16k": 1206.67, "code_32k": 1139.12, "code_64k": 1090.69, "code_128k": 1056.44, "l4_1k": 2012.23, "l4_2k": 1726.66, "l4_4k": 1511.62, "l4_8k": 1348.82, "l4_16k": 1220.06, "l4_32k": 1127.21, "l4_64k": 1060.38, "l4_128k": 1014.65},
    "0.5M":  {"bytes": 3891.33, "code_1k": 1835.60, "code_2k": 1575.19, "code_3k": 1464.17, "code_4k": 1398.26, "code_8k": 1270.37, "code_16k": 1175.61, "code_32k": 1111.03, "code_64k": 1064.59, "code_128k": 1032.45, "l4_1k": 1958.88, "l4_2k": 1681.12, "l4_4k": 1469.99, "l4_8k": 1311.03, "l4_16k": 1186.09, "l4_32k": 1094.99, "l4_64k": 1029.66, "l4_128k":  985.19},
    "1M":    {"bytes": 3876.62, "code_1k": 1819.73, "code_2k": 1561.71, "code_3k": 1454.20, "code_4k": 1388.84, "code_8k": 1261.87, "code_16k": 1166.53, "code_32k": 1101.82, "code_64k": 1057.87, "code_128k": 1027.38, "l4_1k": 1949.34, "l4_2k": 1677.05, "l4_4k": 1467.89, "l4_8k": 1309.65, "l4_16k": 1179.26, "l4_32k": 1089.07, "l4_64k": 1023.96, "l4_128k":  979.49},
    "4M":    {"bytes": 4196.83, "code_1k": 1951.94, "code_2k": 1674.65, "code_3k": 1557.37, "code_4k": 1486.70, "code_8k": 1349.57, "code_16k": 1249.02, "code_32k": 1180.40, "code_64k": 1133.75, "code_128k": 1100.78, "l4_1k": 2091.22, "l4_2k": 1794.40, "l4_4k": 1569.56, "l4_8k": 1398.24, "l4_16k": 1264.22, "l4_32k": 1167.76, "l4_64k": 1098.63, "l4_128k": 1051.57},
    "16M":   {"bytes": 4074.18, "code_1k": 1919.03, "code_2k": 1646.13, "code_3k": 1531.90, "code_4k": 1461.36, "code_8k": 1326.26, "code_16k": 1225.54, "code_32k": 1156.33, "code_64k": 1110.58, "code_128k": 1077.54, "l4_1k": 2047.12, "l4_2k": 1759.27, "l4_4k": 1537.12, "l4_8k": 1368.06, "l4_16k": 1233.55, "l4_32k": 1139.16, "l4_64k": 1071.53, "l4_128k": 1025.33},
}
