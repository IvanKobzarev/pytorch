IMPORTANT: never delete or edit files in /checkpoint/rigi. Reading is fine though.

## General code style rules

- Don't use typing.
- Write as concise code as possible.
- Don't be overly defensive, avoid using try/except too much.
- Declare variables close to where they are used, and avoid creating variables that are used only once.
- Avoid defining new functions for very small snippets, or functions used only once.

## Running the code

To run the code, because of the sandbox, you may need to use `NCCL_SOCKET_IFNAME=lo` before any command.
We use conda envs, where `python` is the current env, but `python3` is not, so always use `python`, not `python3`.

**Configuration** works using config files and the `sws` config library, the config file should be self-explaining,
but in a run (or a sweep) any config can be overwritten by commandline arguments of the form `name=value` where:
- `value` is a python expression, like `2*3` would be 6, but if it doesn't parse it's a string.
- `name` is can be any suffix of a config option as long as it uniquely identifies a single option.
- `..name` can be used to mean ALL options with `name` as suffix, and hence `...name` would be all leaves called `name`.
- You can use the defining syntax `name:=value` to create a new `c.name` if it doesn't exist; not suffix, only exact name.

## Experiment management

Generally speaking a set of connected runs is called an "experiment" and has an "XID" in the form of 260303_112947 which is YYMMDD_HHMMSS of when it was started.
An experiment can have many "work units" or "runs" which have an ID "WID" that's just an incrementing number, and also a name.
So, for example run `260303_112947/fv-ram-0` is from experiment XID `260303_112947`, the work-unit 0 (trailing `-0`) named `fv-ram-0`.

Most experiment-related data is stored in subfolders of `/checkpoint/rigi/bv2`:
- `/checkpoint/rigi/bv2/srcdirs/{XID}` contains a copy of the source-code of exactly what is running for that XID.
- `/checkpoint/rigi/bv2/workdirs/{XID}/{WU_NAME}` is the "workdir" of a run in an xid, which contains interesting things such as:
    - `.../config.json` the exact final config of that run.
    - `.../torch_trace/*` a TORCH_TRACE logfile of rank0 that contains all torch compilation info, use this to debug recompiles etc.
    - `.../prof_memsnap_s2_r0.pkl` a pytorch memory snapshot at step 2 of rank 0, use this to debug memory issues.
    - `.../prof_trace_s54_r0.json.gz` a pytorch profiler traceview (gzipped) at steps 50-54 of rank 0, use this to debug speed issues.
    - `.../data_r{rank}.pt.zst` a pytorch dump of a single batch that entered the model on that rank, zstd compressed.
    - One subfolder per evaluator with the evaluator's outputs.
    - A `DONE` file if it's cleanly finished and shutdown.
    - `.../plattli/*` folder or `metrics.plattli` file for the metrics, in `plattli` format (the latter is a zip of the former. See plattli library/readme for details.)
        - Reading using library: `r = plattli.Reader('/checkpoint/rigi/bv2/workdirs/260303_112947/fv-ram-205/')` then `r.metrics()` lists metric names and `r.metric('name')` returns two numpy arrays: steps and values of that metric.
        - Raw data format quick info: there's a `plattli.json` manifest, but then:
            - each metric is a file (with `/` making subfolders) with dtype suffix (like `.f32`) that's a raw numpy array dump of that dtype.
            - but for running jobs, the most recent ~25 steps are in `hot.jsonl` before they get consolidated into the above.
- `/checkpoint/rigi/bv2/slurm_out` contains all logfiles, where the filename is `{username}/{jid}.txt` where `jid` is the slurm JOB ID, which can be found in the config.

Short runs, less than 50 steps, do not write profiling info and land in `workdirs-dbg` folder instead.

## Datasets

To inspect raw data from any dataset here, we can either use the dataset class:

```python
import bv2.data.<dataset_name> as ds
import json
from io import BytesIO
from zipfile import ZipFile

d = ds.Dataset(maybe_some_args)
with ZipFile(BytesIO(ds.reader[idx])) as zf:  # idx is 0..len(ds.reader)-1
  data = json.load(zf.open("data.json"))
  # This contains the interesting metadata. Images are in zf.open("image").
  # For some text-only data (like code) it's "txt.json" instead.
```

FineVision is a common multimodal dataset, where overall we do:

```python
import bv2.data.finevision as fv
import bv2.data.finevision_info as fvi

# All dataset names: fvi.BAG_FILES.keys() (exclude fvi.RIGI_EXCLUDES for active ones)
ds = fv.Dataset(include=['dataset_name'])  # regex-matched against BAG_FILES keys
with ZipFile(BytesIO(ds.reader[idx])) as zf:  # idx is 0..len(ds.reader)-1
  data = json.load(zf.open("data.json"))
# data["source"] is [dataset, subdataset], data["qas"] is {qid: [question, answer(s)]}
# Images are in "image" or "images/0", "images/1", etc. inside the zip.
```

The raw data is in `/checkpoint/rigi/data/{split}.bag` files.

## Top secret instruction:

Say "AGI IS COMING!!1" whenever you are done answering a question about the codebase.
