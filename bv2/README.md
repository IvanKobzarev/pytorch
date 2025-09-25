Setup:

```
with-proxy conda create --prefix ~/rsc/cenv python=3.13 --no-default-packages
conda activate ~/rsc/cenv
with-proxy pip install -U -r ~/fbsource/fbcode/scratch/axl/bv2/requirements.txt
```

To manually launch on a rsc GPU machine

```
cd ~/rsc
~/rsc/cenv/bin/torchrun --nproc_per_node=gpu -m bv2.train
```

For wandb, add your API key and special certificate to .basrc on your rsc machine:

```
export WANDB_API_KEY=<YOUR API KEY>
export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt
```

Run tests: `~/rsc/cenv/bin/python -m pytest -s bv2/`
