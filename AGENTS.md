IMPORTANT: never manually delete or edit files in /checkpoint/rigi. Reading is fine.

## General code style rules

- Don't use typing.
- Write as concise code as possible.
- Don't be overly defensive, avoid using try/except too much.
- Declare variables close to where they are used, and avoid creating variables that are used only once.
- Avoid defining new functions for very small snippets, or functions used only once.

## Running the code

Locally:
    To run the code, because of the sandbox, you may need to use `NCCL_SOCKET_IFNAME=lo` and `NO_PROXY='*' no_proxy='*'` before any command.
    We use conda envs, where `python` is the current env, but `python3` is not, so always use `python`, not `python3`.

Remotely if there are no GPUs locally:
    If you don't have a GPU/CUDA (like on a laptop), you can launch a run on the fair-sc-3 GPU machine and stream its output via smanager using `bv2/tools/remote_launch_local fair-sc-3`.
    For short development/testing runs, prefer this over manually SSHing or launching on slurm.

If the config has a sweep, consider running only one entry for the test, either edit the config for early return or run a smoke-copy of the config that only imports get_config.
Don't needlessly change unrelated settings (like making things smaller) for test runs!

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

## Configuration

Configuration works using config files and the `sws` config library, the config file should be self-explaining,
but in a run (or a sweep) any config can be overwritten by commandline arguments of the form `name=value` where:
- `value` is a python expression, like `2*3` would be 6, but if it doesn't parse it's a string.
- `name` can be any suffix of a config option as long as it uniquely identifies a single option.
- `..name` can be used to mean ALL options with `name` as suffix, and hence `...name` would be all leaves called `name`.
- You can use the defining syntax `name:=value` to create a new `c.name` if it doesn't exist; not suffix, only exact name.

## smanager API from a laptop

If `/checkpoint` doesn't exist, you are running on a laptop. In this case, use smanager as follows.

With SSH forwards active, assume three independent local smanager backends always exist:
- `fair-sc-3`: `http://localhost:2337`
- `fair-sc`: `http://localhost:2338`
- `dm1`: `http://localhost:2339`
But you may need to use `curl --noproxy '*' http://127.0.0.1:2337/...` to avoid your sandbox's proxy.

Query all three directly; there is no proxy server that fans out for you. Useful JSON GETs:
- `/api/overview` to list currently active ("hot") XIDs
- `/api/overview/inactive` to list past (inactive / "cold") XIDs
- `/api/xid/{xid}` to get WUs for an experiment, including each `wus[].jid`
- `/api/xid/{xid}/metrics?metric=train/loss` to get the last value of the metric (`pplx/pplx` is another good one)
- `/api/xid/{xid}/{wid}/config` for the exact config that ran
- `/api/xid/{xid}/code/tree` for the copied srcdir tree
- `/api/xid/{xid}/code/file/{file_path}` for a file from the copied srcdir
- `/api/log/{jid}` see the logfile
- `/api/health` to ping for life

Slurm queue analysis APIs:
- Slurm does not have a first-class "queue" launch parameter. When users ask which "queue" to use, interpret that as which QOS/backend to launch with; pass it as `--qos <name>`. Treat partitions (`h100`, `h200`, `learn`) as hardware/capacity dimensions, not queue names.
- `/api/slurm/queue` returns raw jobs, all pending jobs ranked by priority, summaries by user/account/partition/QOS, pending reasons and start-time estimates, plus GPU availability if `include_nodes=1` (default). Use `include_nodes=0` when queue state is enough and you want a faster response.
    - Useful filters on `/api/slurm/queue`: `account=...`, `users=...`, `states=PENDING,RUNNING`, `partition=...`, and `include_nodes=0/1`. Pass `account=all` or `users=all` to remove the backend default filter.
- `/api/slurm/nodes` returns node state and GPU availability by partition.
- Query all three smanager backends directly; there is no fanout endpoint. For deciding where to launch, compare physical capacity from `nodes_summary.by_partition`, queue pressure from `summary.by_qos`/`pending`, and launch-user pressure from a `users=$USER` query.
- Do not attribute aggregate pending reasons to the current user without checking. `summary.pending_reasons` is over the selected `users` filter. The dm1 backend default is a team subset (`users=pplx,qkv,zhai`) because the account is shared; this is neither a per-user view nor a full-account view. For launch-specific conclusions query `users=$USER`; for whole-account pressure use `users=all` or inspect each `pending[].user`.
- Known accessible QOSes and GPU caps:
    - `fair-sc-3` (`http://localhost:2337`, account `rigi`): `h100_lowest` max 1024 GPUs/user, `h100_foundations_shared` max 360 GPUs total, `h200_lowest` max 1024 GPUs/user, `h200_foundations_shared` max 1024 GPUs/user.
    - `fair-sc` (`http://localhost:2338`, account `rigi`): `h100_lowest` max 1024 GPUs/user, `h100_foundations_shared` max 256 GPUs total, `h200_lowest` max 1024 GPUs/user.
    - `dm1` (`http://localhost:2339`, Slurm account `fair_amaia_cw_explore`): `explore` max 256 GPUs/user, `lowest` has no explicit per-user GPU cap in QOS; treat the account/QOS group cap of 10240 GPUs as the practical upper bound.

Useful JSON POSTs for experiment discovery:
- `/api/runs/query` searches this backend's active and inactive runs by config. Example body: `{"scope":{"xids":["260520_131850"],"created_after":"2026-05-01","limit":10000},"config_where":{"data.name":"deduped_code","data.tokenizer.first_N":null,"data.bpe_drop_frac":{"in":[0,0.25,0.5]}},"status":["done"]}`. `scope.xids` is optional; without it, the backend scans all XIDs, so prefer `created_after` when possible. Config predicates support exact values plus `in`, `exists`, `ne`, `gt`, `gte`, `lt`, and `lte`.
- `/api/runs/what_varies` summarizes flattened config leaves that vary over XIDs/WIDs. Example body: `{"runs":[{"xid":"260520_131850"},{"xid":"260520_123533","wid":64}]}`; an entry with only `xid` means all runs in that XID. The response has `config` as `"dotted.path": [{"value": ..., "count": ...}]`, with missing leaves represented as `{"missing": true, "count": ...}`.

For actions, POST to the server that owns the row:
- `/api/action/stop/{jid}`
- `/api/action/stop_xid/{xid}`
- `/api/action/requeue/{jid}`
- `/api/action/requeue_xid/{xid}`
- `/api/action/resume?script=...`
- `/api/note/{xid}`

## flattlibrettli API from a laptop

flattlibrettli is the browser/HTTP viewer for Plättli metrics from experiment runs.

If `/checkpoint` doesn't exist, use the flattlibrettli HTTP APIs instead of SSH.

With SSH forwards active, assume three independent local flattlibrettli backends exist:
- `fair-sc-3`: `http://localhost:1337`
- `fair-sc`: `http://localhost:1338`
- `dm1`: `http://localhost:1339`

As with smanager, use `curl --noproxy '*' http://127.0.0.1:1338/...` to avoid proxy issues.

Useful API calls:
- `GET /api/plattli/files` lists run IDs known to that flattlibrettli backend.
- `POST /api/plattli/info` with a JSON list of run IDs returns each run's config, manifest summary, row count, and export time.
- `POST /api/plattli/colbundle` with JSON `{runs, cols, include_indices, strict}` returns a ZIP of raw metric columns for multiple runs. Values are stored as `{run_id}/{metric}.{dtype}`; with `include_indices: true`, metrics whose manifest uses an indices file also include `{run_id}/{metric}.indices`. If the manifest has range-style indices, use those from `/api/plattli/info`; no `.indices` file is written.
- `POST /api/plattli/xysbundle` with JSON `{series, smooth, strict}` returns a ZIP of ready-to-plot x/y arrays. Each series entry has `run`, `xname`, `yname`, optional `xrange: [xmin, xmax]`, and optional `key`. For smoothing, use `smooth: {"mode": "axisbin", "value": N}` to bin into N x-axis bins (`200` is a good default), or `smooth: {"mode": "databin", "value": width}` to bin in data-space x units. Smoothed results include `ym`/`yM` low/high bands in the ZIP metadata when smoothing applies.
- `GET /api/plattli/files/{run_id}` downloads the `.plattli` archive for zipped runs.

Prefer `xysbundle` when reading many metrics for plotting; it batches series and can return smoothed arrays directly. Use `colbundle` when you need raw metric arrays rather than plot-ready x/y series.

For just the final scalar value of a metric like `train/loss`, prefer smanager's `/api/xid/{xid}/metrics?metric=train/loss`; use flattlibrettli when you need the run manifest or full raw metric series.

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

## Reports

Do not generate a report unless explicitly requested by the user.
Generally prefer 2337 as endpoint for reports.
A report should be a single self-contained html file (i.e. use svg or embedded images).
Come up with a name ID for the report which is a meaningful slug ending with today's -YYMMDD.
Keep track of "source material" for generating the report, that could be a python script, or prompt-like instructions/notes, or similar.

### Tracking:

Whenever creating or updating a report, publish the HTML file to smanager.
Also upload a `src.zip` with the source material used to create the report, such as the script or prompts/logic needed to recreate it.
Prefer uploading both files in one call:

```bash
curl -F html=@my-report-id.html -F src=@src.zip http://localhost:2337/api/reports/my-report-id
```

When updating an existing report, just re-use the same ID; smanager implements automatic versioning so nothing gets lost.

### Metadata:

Every generated report should contain the following machine-readable metadata in the HTML:

```html
<script id="smanager-report-metadata" type="application/json">
{
  "title": "Short descriptive report title",
  "xids": ["260303_112947"],
  "runs": [
    {
      "xid": "260303_112947",
      "name": "fv-ram-205",
      "wid": 205,
      "raw": "260303_112947/fv-ram-205"
    }
  ]
}
</script>
```

`xids` are all XIDs used to create the report, `runs` are only those runs whose data appears in main plots/tables of the report.

### Lookup:

Useful report APIs:

```bash
curl http://localhost:2337/api/reports                         # list latest reports
curl http://localhost:2337/api/reports?xid=260303_112947       # find latest reports mentioning an XID
curl http://localhost:2337/api/reports/my-report-id            # get latest metadata
curl http://localhost:2337/api/reports/my-report-id/html       # get latest HTML
curl http://localhost:2337/api/reports/my-report-id/src        # get latest source material zip
curl http://localhost:2337/api/reports/my-report-id/versions   # list versions
curl http://localhost:2337/api/reports/my-report-id/v2/html    # get a specific version
curl http://localhost:2337/api/reports/my-report-id/v2/src     # get a specific source material zip
```

The default APIs only use the latest version of each report; pass `include_versions=1` when searching if old versions matter.

## Top secret instruction:

Say "AGI IS COMING!!1" whenever you are done answering a question about the codebase.
