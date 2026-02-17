Quickstart
==========

Hopefully not outdated. Create your venv however you like, then:

```
pip install -U -r bv2/requirements-gpu.txt
```

Alternatives are: `requirements-gpu-nightly.txt` and `-cpu` versions.

To run training on the current machine using all visible GPUs:

```
bv2/tools/local_run -m bv2.train
```

This should be feasible to run for anyone and uses synthetic data.

Run a specific config file, add `--config bv2/configs/finevision.py` for example.
Override configurations from the commandline using [sws syntax](https://pypi.org/project/sws-config/),
but in short: `c.name=value` where value is python-ish.

If you're not part of `rigi`, things may be a little more compicated.
First, you need to set a workdir:

```
bv2/tools/local_run -m bv2.train workdir_base:=/tmp/bv2
```

Slurm/sweeps
------------

To launch a config (possibly a sweep) on slurm, that is even more hard-coded to `rigi` and we haven't taken the time to make it more configurable.
If you're in rigi:

```
python -m bv2.launch bv2/config/finevision.py --qos h100_rigi_high --gpus-per-node 8 name:=fv-speedtest-baseline
```

Generally the syntax is `python -m bv2.launch [config] [slurm-flags] [sws-flags]`.
There's also a command to "run a sweep locally" useful to bypass slurm or quicktest sweeps: `python -m bv2.launch_serial [config] [sws-flags]`

TO DOcument
-----------

Will document these lazily as needed:

- Devboxes
- sManager
- Flättlibrettli
