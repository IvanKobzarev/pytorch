# bv2/tools/launch_local bv2.train x/zhai/config/code_bpe_drop.py nsteps:=20
# bv2/tools/launch_slurm bv2.train x/zhai/config/code_bpe_drop.py --qos h100_rigi_high --gpus-per-node 8 --nodes 1 'name:=f"zhai-code_bpe_drop-{c.xid}-{c.wid}"'
# bv2/tools/launch_slurm bv2.train x/zhai/config/code_bpe_drop.py --qos lowest --account fair_amaia_cw_explore --gpus-per-node 8 --nodes 1 'name:=f"zhai-code_bpe_drop-{c.xid}-{c.wid}"'

import sws


def get_config():
    c = sws.Config()
    c.seed = 0

    c.maxtok = 1024*256

    c.data.name = "deduped_code"
    c.data.split = "codewall_train_0.25M"
    c.data.cache = True
    c.data.bpe_drop_p = 0.063
    c.data.bpe_drop_frac = 0.25
    c.data.tokenizer.first_N = None
    c.iter.maxtok = lambda: c.maxtok

    # rough estimation of ~64 epochs
    c.nsteps = 7601

    c.warmup_nsteps = 2000
    c.lr_muon = 1e-2
    c.lr_adam = lambda: c.lr_muon
    c.wd = lambda: c.lr_adam * 0.01

    c.muon.param_modes = {"muon_h": [r".*mlp.l[12].weight", r".*att.[qkvo].weight", r".*img_emb.proj.weight", r".*txt_unemb.head.weight"],
                          "embedding": [r".*txt_emb.emb.weight"],
                          "adam": [r".*"]}

    c.model.dim = 2048
    c.model.depth = 8

    c.model.txt.posemb = True
    c.model.glope = 0
    c.model.txt_unemb.chunksz = 4096

    # Eval with standard BPE (no dropout)
    c.evals.pplx.type = "pplx"
    c.evals.pplx.steps = 500
    c.evals.pplx.data.epochs = 1
    c.evals.pplx.data.name = "deduped_code"
    c.evals.pplx.data.cache = lambda: c.data.cache
    c.evals.pplx.data.split = "codewall_val"
    c.evals.pplx.data.bpe_drop_p = 0.0
    c.evals.pplx.data.bpe_drop_frac = 0.0
    c.evals.pplx.data.tokenizer.first_N = lambda: c.data.tokenizer.first_N
    c.evals.pplx.iter.maxtok = 1024*64

    # Byte-level eval: force full BPE-dropout so every token is split to bytes.
    c.evals.pplx_byte.type = "pplx"
    c.evals.pplx_byte.steps = 5000  # infrequent byte-level evals
    c.evals.pplx_byte.data.epochs = 1
    c.evals.pplx_byte.data.name = "deduped_code"
    c.evals.pplx_byte.data.cache = lambda: c.data.cache
    c.evals.pplx_byte.data.split = "codewall_val"
    c.evals.pplx_byte.data.bpe_drop_p = 1.0
    c.evals.pplx_byte.data.bpe_drop_frac = 1.0
    c.evals.pplx_byte.data.tokenizer.first_N = lambda: c.data.tokenizer.first_N
    c.evals.pplx_byte.iter.maxtok = 1024*64

    return c


def sweep():
    # average sequence length

    # scenario                           vocab      mean
    # --------------------------------------------------
    # bpe_drop p=0.063 frac=0.25       200,004     996.4
    # bpe_drop p=0.117 frac=0.25       200,004    1029.1
    # bpe_drop p=0.1803 frac=0.25      200,004    1070.0
    # bpe_drop p=0.3407 frac=0.25      200,004    1185.9
    # bpe_drop p=0.5266 frac=0.25      200,004    1337.5
    # bpe_drop p=1.0 frac=0.25         200,004    1722.6
    # bpe_drop p=0.063 frac=0.50       200,004    1027.0
    # bpe_drop p=0.117 frac=0.50       200,004    1089.1
    # bpe_drop p=0.1803 frac=0.50      200,004    1167.0
    # bpe_drop p=0.3407 frac=0.50      200,004    1389.0
    # bpe_drop p=0.5266 frac=0.50      200,004    1678.8
    # bpe_drop p=1.0 frac=0.50         200,004    2417.6
    # baseline (no dropout)            200,004     962.3
    # byte tokenizer (first_N=256)         260    3901.5
    # byte tokenizer (first_N=4096)      4,100    1471.6
    # byte tokenizer (first_N=16000)    16,004    1187.9
    # byte tokenizer (first_N=32000)    32,004    1096.0
    # byte tokenizer (first_N=64000)    64,004    1029.0

    # c.nsteps = lambda: (min(_int(c.dataset_size), 1000000) * 64 * average_sequence_length) // (c.maxtok * 8)

    # Sweep on 0.25M data.
    for depth, dim in [(12, 3072), (12, 2048), (8, 2048), (8, 1024)]:
        for lr_muon in [3e-3, 6e-3, 1e-2, 3e-2]:
            # BPE dropout variants
            for frac, p, nsteps in [(0.25, 0.063, 7601), (0.25, 0.117, 7851), (0.25, 0.1803, 8163), (0.25, 0.3407, 9047), (0.25, 0.5266, 10204), (0.25, 1.0, 13141),
                                    (0.50, 0.063, 7835), (0.50, 0.117, 8309), (0.50, 0.1803, 8903), (0.50, 0.3407, 10597), (0.50, 0.5266, 12808), (0.50, 1.0, 18444)]:
                yield f"{depth=}", f"c.lr_muon={lr_muon}", f"{dim=}", f"c.nsteps={nsteps}", f"c.data.bpe_drop_p={p}", f"c.data.bpe_drop_frac={frac}"

            # baselines (no BPE dropout)
            yield f"{depth=}", f"c.lr_muon={lr_muon}", f"{dim=}", "c.nsteps=7341", "c.data.bpe_drop_p=0.0", "c.data.bpe_drop_frac=0.0", "c.evals.pplx_byte=None"

            # byte tokenizer baselines
            for first_N, nsteps in [(256, 29766),]:
                yield f"c.data.tokenizer.first_N={first_N}", f"c.nsteps={nsteps}", f"{depth=}", f"c.lr_muon={lr_muon}", f"{dim=}", "c.data.bpe_drop_p=0.0", "c.data.bpe_drop_frac=0.0", "c.evals.pplx_byte=None"

    # Same sweep but on 1M data (4x more data -> 4x more steps).
    for depth, dim in [(12, 3072), (12, 2048), (8, 2048), (8, 1024)]:
        for lr_muon in [3e-3, 6e-3, 1e-2, 3e-2]:
            for frac, p, nsteps in [(0.25, 0.063, 7601*4), (0.25, 0.117, 7851*4), (0.25, 0.1803, 8163*4), (0.25, 0.3407, 9047*4), (0.25, 0.5266, 10204*4), (0.25, 1.0, 13141*4),
                                    (0.50, 0.063, 7835*4), (0.50, 0.117, 8309*4), (0.50, 0.1803, 8903*4), (0.50, 0.3407, 10597*4), (0.50, 0.5266, 12808*4), (0.50, 1.0, 18444*4)]:
                yield 'c.data.split="codewall_train_1M"', f"{depth=}", f"c.lr_muon={lr_muon}", f"{dim=}", f"c.nsteps={nsteps}", f"c.data.bpe_drop_p={p}", f"c.data.bpe_drop_frac={frac}"

            yield 'c.data.split="codewall_train_1M"', f"{depth=}", f"c.lr_muon={lr_muon}", f"{dim=}", f"c.nsteps={7341*4}", "c.data.bpe_drop_p=0.0", "c.data.bpe_drop_frac=0.0", "c.evals.pplx_byte=None"

            for first_N, nsteps in [(256, 29766*4),]:
                yield 'c.data.split="codewall_train_1M"', f"c.data.tokenizer.first_N={first_N}", f"c.nsteps={nsteps}", f"{depth=}", f"c.lr_muon={lr_muon}", f"{dim=}", "c.data.bpe_drop_p=0.0", "c.data.bpe_drop_frac=0.0", "c.evals.pplx_byte=None"
