# bv2/tools/launch_local bv2.train bv2/config/code_bpe_drop.py nsteps:=20
# bv2/tools/launch_slurm bv2.train bv2/config/code_bpe_drop.py --qos h200_lowest --gpus-per-node 8 --nodes 1 'name:=f"zhai-code_bpe_drop-{c.xid}-{c.wid}"'

#   Vocab Size   Avg Tokens  ≈ BPE dropout p
# --------------------------------------------
#      200,000        542.7           0.0000
#      100,000        562.1           0.0353
#       64,000        579.0           0.0633
#       32,000        616.5           0.1168

#        8,000        741.0           0.2599
#        4,000        831.6           0.3407
#        1,000       1107.4           0.5266
#          256       2206.1           1.0000

#       16,000        667.5           0.1803
#        2,000        950.8           0.4301
#          500       1339.3           0.6441

import sws


def get_config():
    c = sws.Config()
    c.seed = 0

    c.maxtok = 1024*256

    c.data.name = "deduped_code"
    c.data.split = "codewall_train_0.25M"
    c.data.bpe_drop_p = 0.1
    c.data.bpe_drop_frac = 0.5
    c.data.mode_tokens = True
    c.iter.maxtok = lambda: c.maxtok

    # rough estimation of ~64 epochs
    c.nsteps = 7630

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
    c.evals.pplx.data.split = "codewall_val"
    c.evals.pplx.data.bpe_drop_p = 0.0
    c.evals.pplx.data.bpe_drop_frac = 0.0
    c.evals.pplx.data.mode_tokens = False
    c.evals.pplx.iter.maxtok = 1024*64

    return c


def sweep():
      for depth, dim in [(12, 2048), (8, 1024),]:
          # BPE dropout variants
          for frac, p in [(0.25, 0.063), (0.25, 0.117), (0.5, 0.063)]:
              for lr_muon in [1e-3, 2e-3, 3e-3, 4e-3, 6e-3, 1e-2]:
                  yield f"{depth=}", f"c.lr_muon={lr_muon}", f"{dim=}", f"c.data.bpe_drop_p={p}", f"c.data.bpe_drop_frac={frac}", "c.data.mode_tokens=True"

          # baselines (no BPE dropout)
          for lr_muon in [1e-3, 2e-3, 3e-3, 4e-3, 6e-3, 1e-2]:
              yield f"{depth=}", f"c.lr_muon={lr_muon}", f"{dim=}", "c.data.bpe_drop_p=0.0", "c.data.bpe_drop_frac=0.0", "c.data.mode_tokens=False"
