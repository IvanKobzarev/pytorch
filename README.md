# Rigi.

Meant for internal R&D, not well documented.

PyTorch folks, see: [this doc](https://docs.google.com/document/d/15gCm7Zn9ghmyQNUaBnLufn2BRRHbpW6zge9_JbLIkfA)

```
pip install -r bv2/requirements-gpu-nightly.txt  # Or -stable, or -cpu-(nightly|stable)
bv2/tools/local_run -m bv2.train

# If you're not in `rigi`, then:
bv2/tools/local_run -m bv2.train ...tokenizer.path=$HOME/pci-wsf/tokenizers/tiktoken/l4_200k_base ckpt_steps:=999999 workdir_base:=/tmp/rigi_workdir
```

A little bit more details in [bv2/README.md](bv2/README.md).
