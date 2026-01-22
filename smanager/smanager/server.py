#!/usr/bin/env python3
"""Slurm Manager Server - serves job info via REST API."""

import re
import json
import subprocess
import shlex
import logging
import time
import signal
import sys
import os
import shutil
from getpass import getuser
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import plattli
import zstandard

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
import uvicorn

from smanager import __version__

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)


# Filter to suppress noisy health check logs from uvicorn
class HealthCheckFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        return "/api/health" not in msg and "/api/prefs/favorites" not in msg


logging.getLogger("uvicorn.access").addFilter(HealthCheckFilter())


app = FastAPI(title="Slurm Manager API")

SCRIPT_DIR = Path(__file__).parent

# Use ThreadPoolExecutor for parallel I/O (NFS can be slow, so use many threads)
executor = ThreadPoolExecutor(max_workers=64)


def shutdown_handler(signum, frame):
    log.info("Received signal %s, shutting down...", signum)
    executor.shutdown(wait=False, cancel_futures=True)
    os._exit(0)


signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)


class ZstdMiddleware:
    """ASGI middleware that compresses responses with zstd when client supports it."""

    def __init__(self, app, level=3, min_size=500):
        self.app = app
        self.compressor = zstandard.ZstdCompressor(level=level)
        self.min_size = min_size

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Check Accept-Encoding header
        headers = dict(scope.get("headers", []))
        accept_encoding = headers.get(b"accept-encoding", b"").decode()

        if "zstd" not in accept_encoding:
            await self.app(scope, receive, send)
            return

        # Collect response body and headers
        response_body = []
        response_headers = []
        response_status = [200]

        async def collect_send(message):
            if message["type"] == "http.response.start":
                response_status[0] = message["status"]
                response_headers.extend(message.get("headers", []))
            elif message["type"] == "http.response.body":
                body = message.get("body", b"")
                if body:
                    response_body.append(body)

        await self.app(scope, receive, collect_send)

        # Combine body
        body = b"".join(response_body)

        # Only compress if body is large enough
        if len(body) >= self.min_size:
            body = self.compressor.compress(body)
            # Update headers: remove Content-Length, add Content-Encoding
            new_headers = [(k, v) for k, v in response_headers if k.lower() != b"content-length"]
            new_headers.append((b"content-encoding", b"zstd"))
            new_headers.append((b"content-length", str(len(body)).encode()))
        else:
            new_headers = response_headers

        # Send response
        await send({
            "type": "http.response.start",
            "status": response_status[0],
            "headers": new_headers,
        })
        await send({
            "type": "http.response.body",
            "body": body,
        })


app.add_middleware(ZstdMiddleware, level=3, min_size=500)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
GROUP = "rigi"
BASEDIR = Path("/checkpoint/rigi/bv2/workdirs")
SRCDIR = Path("/checkpoint/rigi/bv2/srcdirs")
FBIDIR = Path("/checkpoint/rigi/fbi")
SLURM_OUT_DIR = Path("/checkpoint/rigi/bv2/slurm_out")
PREFS_DIR = Path(f"/checkpoint/rigi/{getuser()}")  # Set via --prefs-dir flag
ARCHIVE_DIR = Path("/checkpoint/rigi/bv2/workdirs-archive")  # Set via --archive-dir flag
NUM_RECENT = 50
ACTIONS_ENABLED = True  # Set via --no-actions flag

_xid_re = re.compile(r'\d{4,6}_\d{6}')


def run_cmd(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout.strip().split('\n')


def extract_xid(name):
    if firstmatch := _xid_re.search(name):
        return firstmatch.group()
    return None


def get_jobs(group=GROUP):
    lines = run_cmd(f"squeue -A {group} -O JobId:20,Name:20,UserName:20,State:20,TimeUsed:20,NumCPUs:20,QOS:20,NumNodes:20,GRES:20,RestartCnt:20,Reason:20")
    jobs = [[j[i*20:(i+1)*20].strip() for i in range(11)] for j in lines if j.strip()]
    if len(jobs) < 2:
        return [], []
    return jobs[0], jobs[1:]


def extract_common_name(workdir_names, xid):
    """Extract a common name from workdir names by finding common prefix and removing XID."""
    if not workdir_names:
        return ""
    names = list(workdir_names)
    if len(names) == 1:
        common = names[0]
    else:
        # Find common prefix
        common = names[0]
        for name in names[1:]:
            while common and not name.startswith(common):
                common = common[:-1]
    # Remove the XID from the common part
    common = common.replace(xid, "")
    # Strip common separators from both ends
    return common.strip(" -_")


def _extra_info(xid_info):
    """Get extra info for an XID.

    Args:
        xid_info: Tuple of (xid, info) or (xid, info, skip_config_loading)
            If skip_config_loading is True, skip loading config.json files
            (faster but won't detect duplicate workdirs for warnings).
    """
    if len(xid_info) == 3:
        xid, info, skip_config_loading = xid_info
    else:
        xid, info = xid_info
        skip_config_loading = False

    wd_path = BASEDIR / info["wd"]
    try:
        info["user"] = wd_path.owner()
    except:
        info["user"] = "?"
    # Count total WUs from launch scripts (includes pending jobs without workdirs)
    info["total_wus"] = len(list(wd_path.glob("launch_*.sh")))
    launchinfo = wd_path / 'launchinfo.txt'
    workdir_names = []
    wid_counts = {}  # wid -> count (for duplicate detection)
    if launchinfo.is_file():
        info["wus"] = {}
        for wuwd in wd_path.iterdir():
            if wuwd.is_dir():
                info["wus"][str(wuwd.name)] = (wuwd / "DONE").exists()
                workdir_names.append(wuwd.name)
                if not skip_config_loading:
                    # Load config to get wid for duplicate detection
                    config = load_config(wuwd)
                    wid = config.get("wid", wuwd.name)
                    wid_counts[wid] = wid_counts.get(wid, 0) + 1
        try:
            info["config"] = next(re.finditer(r"bv2/config/(.*?) ", launchinfo.read_text())).group(1)
        except:
            info["config"] = "?"
    else:
        info["wus"] = {}
        info["config"] = "?"
        for wuwd in wd_path.iterdir():
            if wuwd.is_dir():
                workdir_names.append(wuwd.name)
                if not skip_config_loading:
                    # Load config to get wid for duplicate detection
                    config = load_config(wuwd)
                    wid = config.get("wid", wuwd.name)
                    wid_counts[wid] = wid_counts.get(wid, 0) + 1

    # Calculate Done-ish count if jobs are available
    # Note: A job might write DONE but get stuck running in Slurm without quitting.
    # Without explicit xid-jid mapping, we must load DONE workdir configs to get their JID and compare to running jobs.
    info["done_ish_count"] = 0
    info["actual_done_count"] = 0

    if "jobs" in info and info.get("wus"):
        jobs_by_jid = {job.get("JOBID", ""): job for job in info["jobs"] if job.get("JOBID")}
        done_wuwd_names = [name for name, is_done in info["wus"].items() if is_done]

        if done_wuwd_names:
            try:
                config_args = [(wd_path, wuwd_name) for wuwd_name in done_wuwd_names]
                config_results = list(executor.map(_load_config_only, config_args))

                done_ish = 0
                actual_done = 0

                for wuwd_name, config in config_results:
                    jid = config.get("jid")
                    if jid and str(jid) in jobs_by_jid:
                        slurm_state = jobs_by_jid[str(jid)].get("STATE", "")
                        if slurm_state == "RUNNING":
                            done_ish += 1
                        else:
                            actual_done += 1
                    else:
                        actual_done += 1

                info["done_ish_count"] = done_ish
                info["actual_done_count"] = actual_done
            except Exception as e:
                log.warning("Failed to calculate done_ish for %s: %s", xid, e)

    info["name"] = extract_common_name(workdir_names, xid)
    # Count WUs with duplicate workdirs (warning_count)
    info["warning_count"] = sum(1 for c in wid_counts.values() if c > 1)
    # Read note if exists
    note_file = wd_path / "NOTE.md"
    info["note"] = note_file.read_text().strip() if note_file.exists() else ""
    return xid, info


@app.get("/api/health")
def health():
    return {"status": "ok", "group": GROUP, "user": getuser(), "actions_enabled": ACTIONS_ENABLED}


def _read_xid_file(filename):
    path = PREFS_DIR / filename
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _write_xid_file(filename, xids):
    PREFS_DIR.mkdir(parents=True, exist_ok=True)
    path = PREFS_DIR / filename
    path.write_text("\n".join(sorted(xids)) + "\n" if xids else "")


@app.get("/api/prefs/favorites")
def get_favorites():
    log.info("GET /api/prefs/favorites")
    return {"favorites": _read_xid_file("xid_favs.txt")}


@app.post("/api/prefs/favorites")
def set_favorites(favorites: list[str] = Body(...)):
    log.info("POST /api/prefs/favorites (%d items)", len(favorites))
    _write_xid_file("xid_favs.txt", favorites)
    return {"status": "ok"}


@app.post("/api/note/{xid}")
def set_note(xid: str, note: str = Body(..., embed=True)):
    """Set or delete a note for an XID."""
    log.info("POST /api/note/%s", xid)
    # Find the XID's workdir
    wd_path = BASEDIR / xid
    if not wd_path.exists():
        for d in BASEDIR.iterdir():
            if d.is_dir() and xid in d.name:
                wd_path = d
                break
        else:
            raise HTTPException(status_code=404, detail=f"XID {xid} not found")
    note_file = wd_path / "NOTE.md"
    note = note.strip()
    if note:
        note_file.write_text(note + "\n")
        log.info("Saved note for %s", xid)
    elif note_file.exists():
        note_file.unlink()
        log.info("Deleted note for %s", xid)
    return {"status": "ok", "note": note}


@app.get("/")
def serve_index():
    log.info("GET / (serving index.html)")
    html = (SCRIPT_DIR / "index.html").read_text()
    html = html.replace("{{VERSION}}", __version__)
    return Response(content=html, media_type="text/html")


@app.get("/api/overview")
def get_overview():
    """Get overview of hot (active) experiments only - fast path.

    Returns only XIDs with active jobs in squeue.
    """
    t0 = time.time()
    log.info("GET /api/overview - fetching hot only...")

    t1 = time.time()
    headers, job_rows = get_jobs()
    log.info("  - squeue took %.2fs (%d jobs)", time.time() - t1, len(job_rows))
    if not headers:
        log.info("GET /api/overview - no jobs found (%.2fs)", time.time() - t0)
        return {"hot": {}}

    # Build jobs lookup and track misc (non-XID) jobs
    jobs_by_name = {}
    misc_jobs = []
    for row in job_rows:
        job = dict(zip(headers, row))
        name = job.get("NAME", "")
        if extract_xid(name):
            if name not in jobs_by_name:
                jobs_by_name[name] = []
            jobs_by_name[name].append(job)
        else:
            misc_jobs.append(job)

    # Only get workdirs that have active jobs
    t2 = time.time()
    active_xids = set(jobs_by_name.keys())
    workdirs = [d.name for d in BASEDIR.iterdir() if d.is_dir() and extract_xid(d.name) in active_xids]
    wd_by_xid = {extract_xid(wd): wd for wd in workdirs}
    log.info("  - iterdir took %.2fs (%d workdirs matched)", time.time() - t2, len(workdirs))

    # Build hot xids
    hot_xids = {}
    for xid, wd in wd_by_xid.items():
        xid_jobs = jobs_by_name.get(xid, [])
        states = Counter(j.get("STATE", "") for j in xid_jobs)
        if states:
            hot_xids[xid] = {"states": dict(states), "wd": wd, "jobs": xid_jobs}

    # Add extra info with threading
    t3 = time.time()
    hot_items = [(xid, {"states": info["states"], "wd": info["wd"], "jobs": info["jobs"]}) for xid, info in hot_xids.items()]
    hot_results = list(executor.map(_extra_info, hot_items))
    hot_xids = {}
    for xid, info in hot_results:
        hot_xids[xid] = info
    log.info("  - extra_info took %.2fs (%d xids)", time.time() - t3, len(hot_results))

    # Compute summary stats for hot xids
    nGPUs = {f'gres/gpu:{i}': i for i in range(1, 9)}
    for xid, info in hot_xids.items():
        xid_jobs = info.get("jobs", [])
        if xid_jobs:
            qos_set = set(j.get("QOS", "") for j in xid_jobs)
            info["qos"] = " ".join(qos_set)
            users_set = set(j.get("USER", "") for j in xid_jobs)
            info["users"] = " ".join(users_set)
            nodes = xid_jobs[0].get("NODES", "1")
            tres = xid_jobs[0].get("TRES_PER_NODE", "")
            try:
                gpus_per_job = int(nodes) * nGPUs.get(tres, 0)
            except:
                gpus_per_job = 0
            info["gpus_per_job"] = gpus_per_job
            info["total_gpus"] = gpus_per_job * info["states"].get("RUNNING", 0)
            info["max_restarts"] = max(int(j.get("RESTART_COUNT", 0)) for j in xid_jobs)
        else:
            info["qos"] = ""
            info["users"] = ""
            info["gpus_per_job"] = 0
            info["total_gpus"] = 0
            info["max_restarts"] = 0
        # Remove raw jobs from response
        if "jobs" in info:
            del info["jobs"]

    # Compute misc stats (non-XID jobs)
    misc = None
    if misc_jobs:
        misc_gpus = 0
        misc_states = Counter(j.get("STATE", "") for j in misc_jobs)
        for job in misc_jobs:
            if job.get("STATE") != "RUNNING":
                continue
            nodes = job.get("NODES", "1")
            tres = job.get("TRES_PER_NODE", "")
            try:
                misc_gpus += int(nodes) * nGPUs.get(tres, 0)
            except:
                pass
        misc = {"count": len(misc_jobs), "gpus": misc_gpus, "states": dict(misc_states)}

    log.info("GET /api/overview - done: %d hot, %d misc (%.2fs)", len(hot_xids), len(misc_jobs), time.time() - t0)
    return {"hot": hot_xids, "misc": misc}


@app.get("/api/overview/inactive")
def get_overview_inactive():
    """Get overview of cold and frozen experiments - slower path.

    Returns ALL inactive workdirs. Client should filter out hot XIDs.
    """
    t0 = time.time()
    log.info("GET /api/overview/inactive - fetching...")

    # Read all workdirs (client will filter out hot XIDs)
    workdirs = [d.name for d in BASEDIR.iterdir() if d.is_dir()]
    wd_by_xid = {xid: wd for wd in workdirs if (xid := extract_xid(wd))}

    # Build list of all XIDs (client filters out hot ones)
    all_xids = {}
    for xid, wd in sorted(wd_by_xid.items(), reverse=True):
        all_xids[xid] = {"wd": wd}

    # Split into cold (first NUM_RECENT) and frozen (rest)
    all_list = list(all_xids.keys())
    cold_xids = {xid: all_xids[xid] for xid in all_list[:NUM_RECENT]}
    frozen_xids = {xid: all_xids[xid] for xid in all_list[NUM_RECENT:]}

    # Add extra info with threading (skip config loading for faster response)
    cold_items = [(xid, info, True) for xid, info in cold_xids.items()]
    cold_results = list(executor.map(_extra_info, cold_items))
    cold_xids = dict(cold_results)

    frozen_items = [(xid, info, True) for xid, info in frozen_xids.items()]
    frozen_results = list(executor.map(_extra_info, frozen_items))
    frozen_xids = dict(frozen_results)

    log.info("GET /api/overview/inactive - done: %d cold, %d frozen (%.2fs)",
             len(cold_xids), len(frozen_xids), time.time() - t0)
    return {"cold": cold_xids, "frozen": frozen_xids}


def load_config(wd_path):
    config_file = wd_path / "config.json"
    if not config_file.exists():
        return {}
    try:
        return json.loads(config_file.read_text())
    except Exception as e:
        log.debug("load_config failed for %s: %s", wd_path, e)
        # Fallback to raw JSON without sws
        try:
            return json.loads((wd_path / "config.json").read_text())
        except:
            return {}


def last_metric(wd_path, metric_name="train/loss"):
    # Try plattli format first
    if plattli.is_run_dir(wd_path):
        try:
            with plattli.Reader(wd_path) as r:
                result = {}
                metrics = r.metrics()
                # Get step from first metric's last index
                if metrics:
                    indices = r.metric_indices(metrics[0])
                    if len(indices) > 0:
                        result["step"] = int(indices[-1])
                # Only fetch the requested metric
                if metric_name in metrics:
                    values = r.metric_values(metric_name)
                    if len(values) > 0:
                        v = values[-1]
                        result[metric_name] = v.item() if hasattr(v, 'item') else v
                return result
        except Exception as e:
            log.debug("plattli read failed for %s: %s", wd_path, e)

    # Fallback to jsonl
    fname = wd_path / "metrics.jsonl"
    if not fname.exists():
        return {}
    try:
        res = subprocess.run(["tail", "-n", "1", str(fname)], capture_output=True, text=True)
        if res.returncode == 0 and res.stdout.strip():
            return json.loads(res.stdout.strip())
    except:
        pass
    return {}


def load_sacct(jid):
    try:
        result = run_cmd(f"sacct --long --jobs={jid} --json")
        data = json.loads('\n'.join(result))
        if data.get('jobs'):
            return data['jobs'][0]
    except Exception as e:
        log.debug("load_sacct failed for %s: %s", jid, e)
    return {}


def _load_config_only(args):
    wd_path, wuwd_name = args
    return wuwd_name, load_config(wd_path / wuwd_name)


def _load_metric_only(args):
    wd_path, wuwd_name, metric_name = args
    return wuwd_name, last_metric(wd_path / wuwd_name, metric_name)


def _find_xid_path(xid):
    """Find the workdir path for an XID."""
    wd_path = BASEDIR / xid
    if wd_path.exists():
        return wd_path
    # Slow path: iterate BASEDIR to find partial match
    t0 = time.time()
    for d in BASEDIR.iterdir():
        if d.is_dir() and xid in d.name:
            log.info("  - _find_xid_path fallback took %.2fs", time.time() - t0)
            return d
    log.info("  - _find_xid_path fallback took %.2fs (not found)", time.time() - t0)
    return None


@app.get("/api/xid/{xid}")
def get_xid_info(xid: str):
    """Get detailed info for a specific XID (without metrics for faster response)."""
    t0 = time.time()
    log.info("GET /api/xid/%s - fetching...", xid)
    wd_path = _find_xid_path(xid)
    if not wd_path:
        log.warning("GET /api/xid/%s - not found (%.2fs)", xid, time.time() - t0)
        raise HTTPException(status_code=404, detail=f"XID {xid} not found")
    log.info("  - found path in %.2fs", time.time() - t0)

    # Get current jobs for this xid
    t1 = time.time()
    lines = run_cmd(f"squeue -n {xid} -O JobId:20,Name:20,UserName:20,State:20,TimeUsed:20,NumCPUs:20,QOS:20,NumNodes:20,GRES:20,RestartCnt:20,Reason:20")
    log.info("  - squeue took %.2fs", time.time() - t1)
    xid_jobs = [[j[i*20:(i+1)*20].strip() for i in range(11)] for j in lines if j.strip()]
    if len(xid_jobs) >= 2:
        headers, job_rows = xid_jobs[0], xid_jobs[1:]
        jobs_by_jid = {row[0]: dict(zip(headers, row)) for row in job_rows}
    else:
        jobs_by_jid = {}

    # Get workdirs
    t2 = time.time()
    workdirs = [d.name for d in wd_path.iterdir() if d.is_dir()]
    log.info("  - found %d workdirs (iterdir took %.2fs)", len(workdirs), time.time() - t2)

    # Load configs in parallel
    t1 = time.time()
    config_results = list(executor.map(_load_config_only, [(wd_path, wd) for wd in workdirs]))
    log.info("  - loaded configs in %.2fs", time.time() - t1)

    configs = {}
    warnings = {}  # wid -> list of warning strings
    for wuwd_name, config in config_results:
        wid = config.get("wid", wuwd_name)
        new_jid = config.get("jid")

        # If wid already exists, prefer the "better" config
        if wid in configs:
            old_jid = configs[wid].get("jid")
            new_in_squeue = str(new_jid) in jobs_by_jid
            old_in_squeue = str(old_jid) in jobs_by_jid

            # Record warning about duplicate workdir
            if wid not in warnings:
                warnings[wid] = []
            warnings[wid].append(f"Duplicate workdir detected (JIDs: {old_jid}, {new_jid})")

            # Prefer: in squeue > not in squeue; higher jid > lower jid
            if old_in_squeue and not new_in_squeue:
                continue  # Keep old
            if not old_in_squeue and new_in_squeue:
                pass  # Replace with new
            elif (new_jid or 0) <= (old_jid or 0):
                continue  # Keep old (higher or equal jid)
            # Otherwise fall through to replace

        configs[wid] = config

    # Get sacct info for all jids
    jids = set()
    for wid, config in configs.items():
        if "jid" in config:
            jids.add(config["jid"])
    jids.update(int(jid) for jid in jobs_by_jid.keys() if jid.isdigit())

    t2 = time.time()
    saccts = {}
    jids_list = list(jids)
    sacct_results = list(executor.map(load_sacct, jids_list))
    for jid, sacct in zip(jids_list, sacct_results):
        saccts[jid] = sacct
    log.info("  - loaded sacct in %.2fs", time.time() - t2)

    # Determine status for each wid
    status = {}
    for wid, config in configs.items():
        name = config.get("name", "")
        has_done = name and (wd_path / name / "DONE").is_file()
        jid = config.get("jid")
        slurm_state = jobs_by_jid.get(str(jid), {}).get("STATE") if jid else None

        if has_done and slurm_state == "RUNNING":
            # DONE file exists but Slurm still shows RUNNING (laggy teardown)
            status[wid] = "DONE_ISH"
        elif has_done:
            status[wid] = "DONE"
        elif jid and str(jid) in jobs_by_jid:
            status[wid] = slurm_state or "UNKNOWN"
        elif jid and jid in saccts and saccts[jid].get('state', {}).get('current'):
            status[wid] = saccts[jid]['state']['current'][-1]
        else:
            status[wid] = "UNKNOWN"

    # Add pending jobs from squeue that don't have workdirs yet
    existing_jids = {str(c.get("jid")) for c in configs.values() if c.get("jid")}
    pending_unknown_idx = 0
    for jid_str, job_info in jobs_by_jid.items():
        if jid_str in existing_jids:
            continue  # Already have this job from workdirs
        jid = int(jid_str) if jid_str.isdigit() else None
        # Try to extract wid from sacct submit_line
        wid = None
        if jid and jid in saccts:
            submit_line = saccts[jid].get("submit_line", "")
            # Try wid:=N pattern first (direct arg), then launch_N.sh
            m = re.search(r'wid:=(\d+)', submit_line)
            if not m:
                m = re.search(r'launch_(\d+)\.sh', submit_line)
            if m:
                wid = int(m.group(1))  # Convert to int to match configs keys
        # If we couldn't extract wid, use "??" placeholder
        if wid is None:
            wid = f"??{pending_unknown_idx}"
            pending_unknown_idx += 1
        # If wid exists but old job is not in squeue, replace with new pending job
        if wid in configs:
            old_jid = configs[wid].get("jid")
            old_in_squeue = str(old_jid) in jobs_by_jid
            if not old_in_squeue:
                # Old job finished, new job is pending - update to new job
                configs[wid]["jid"] = jid
                configs[wid]["pending_only"] = True
                status[wid] = job_info.get("STATE", "PENDING")
        else:
            configs[wid] = {"jid": jid, "name": "", "pending_only": True}
            status[wid] = job_info.get("STATE", "PENDING")

    # Format for response
    wus = []
    for wid in sorted(configs.keys(), key=lambda x: int(x) if str(x).isdigit() else x):
        config = configs[wid]
        jid = config.get("jid", "")
        sacct = saccts.get(jid, {})

        # Extract submit line args
        submit_line = sacct.get("submit_line", "")
        sws_args = []
        if submit_line:
            ignore = ["xid:=", "wid:=", "jid:=", "name:="]
            for a in shlex.split(submit_line):
                if "=" in a and not a.startswith("--") and not any(x in a for x in ignore):
                    sws_args.append(a)

        # Time formatting
        def hms(s):
            if not s:
                return "n/a"
            m, s = divmod(s, 60)
            h, m = divmod(m, 60)
            d, h = divmod(h, 24)
            parts = []
            if d:
                parts.append(f"{d}d")
                parts.append(f"{h:02d}h{m:02d}m{s:02d}s")
            elif h:
                parts.append(f"{h}h{m:02d}m{s:02d}s")
            elif m:
                parts.append(f"{m}m{s:02d}s")
            else:
                parts.append(f"{s}s")
            return "".join(parts)

        time_info = sacct.get("time", {})
        elapsed = time_info.get("elapsed", 0)
        eligible = time_info.get("eligible", 0)
        start = time_info.get("start", 0)

        if start in [0, 4294967294]:
            qwait = "never"
        else:
            qwait = hms(start - eligible)

        wus.append({
            "wid": wid,
            "jid": jid,
            "restarts": sacct.get("restart_cnt", 0),
            "exit_code": sacct.get("exit_code", {}).get("return_code", {}).get("number", 0),
            "status": status.get(wid, "UNKNOWN"),
            "nsteps": config.get("nsteps"),
            "config_args": sws_args,
            "name": config.get("name", ""),
            "qwait": qwait,
            "runtime": hms(elapsed),
            "workdir": str(config.get("workdir", "")),
            "launch_script": str(wd_path / f"launch_{wid}.sh"),
            "warnings": warnings.get(wid, []),
        })

    result = {
        "xid": xid,
        "wus": wus,
        "launch_command": (wd_path / "launchinfo.txt").read_text() if (wd_path / "launchinfo.txt").exists() else "",
    }
    log.info("GET /api/xid/%s - done: %d work units (%.2fs)", xid, len(wus), time.time() - t0)
    return result


@app.get("/api/xid/{xid}/metrics")
def get_xid_metrics(xid: str, metric: str = "train/loss"):
    """Get metrics for all WUs of an XID (separate from main xid-detail for lazy loading)."""
    t0 = time.time()
    log.info("GET /api/xid/%s/metrics (metric=%s)", xid, metric)
    wd_path = _find_xid_path(xid)
    if not wd_path:
        raise HTTPException(status_code=404, detail=f"XID {xid} not found")

    # Get workdirs
    workdirs = [d.name for d in wd_path.iterdir() if d.is_dir()]

    # Load configs and metrics in parallel
    config_results = list(executor.map(_load_config_only, [(wd_path, wd) for wd in workdirs]))
    metric_results = list(executor.map(_load_metric_only, [(wd_path, wd, metric) for wd in workdirs]))

    # Build wid -> metrics mapping (same duplicate resolution as main endpoint)
    metric_by_wuwd = {wuwd_name: metrics for wuwd_name, metrics in metric_results}
    wid_to_wuwd = {}
    for wuwd_name, config in config_results:
        wid = config.get("wid", wuwd_name)
        new_jid = config.get("jid") or 0
        if wid in wid_to_wuwd:
            old_jid = wid_to_wuwd[wid][1]
            if new_jid <= old_jid:
                continue
        wid_to_wuwd[wid] = (wuwd_name, new_jid)

    # Build response: wid -> {step, metric_name: value}
    result = {}
    for wid, (wuwd_name, _) in wid_to_wuwd.items():
        result[wid] = metric_by_wuwd.get(wuwd_name, {})

    log.info("GET /api/xid/%s/metrics - done: %d WUs (%.2fs)", xid, len(result), time.time() - t0)
    return result


@app.get("/api/xid/{xid}/{wid}/config")
def get_wu_config(xid: str, wid: int):
    """Get full config.json for a specific work unit."""
    wd_path = BASEDIR / xid
    if not wd_path.exists():
        # Try to find it
        for d in BASEDIR.iterdir():
            if d.is_dir() and xid in d.name:
                wd_path = d
                break
        else:
            raise HTTPException(status_code=404, detail=f"XID {xid} not found")

    # Find the work unit directory by checking each config's wid field
    for d in wd_path.iterdir():
        if not d.is_dir():
            continue
        config_path = d / "config.json"
        if not config_path.exists():
            continue
        try:
            config = json.loads(config_path.read_text())
            if config.get("wid") == wid:
                return Response(
                    content=json.dumps(config, indent=2),
                    media_type="application/json",
                )
        except Exception:
            continue

    raise HTTPException(status_code=404, detail=f"Config not found for WU {wid}")


def _get_srcdir(xid: str) -> Path:
    """Get and validate the source directory for an XID."""
    src_path = SRCDIR / xid
    if not src_path.exists():
        raise HTTPException(status_code=404, detail=f"Source directory not found for XID {xid}")
    return src_path


def _safe_path(base: Path, user_path: str) -> Path:
    """Safely resolve a user-provided path within a base directory."""
    # Resolve the full path and ensure it's within the base
    try:
        full_path = (base / user_path).resolve()
        base_resolved = base.resolve()
        if not str(full_path).startswith(str(base_resolved) + "/") and full_path != base_resolved:
            raise HTTPException(status_code=403, detail="Access denied: path outside source directory")
        return full_path
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid path: {e}")


def _build_tree(path: Path, base: Path) -> list:
    """Recursively build a file tree structure."""
    items = []
    try:
        for entry in sorted(path.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower())):
            rel_path = str(entry.relative_to(base))
            if entry.is_dir():
                items.append({
                    "name": entry.name,
                    "path": rel_path,
                    "type": "dir",
                    "children": _build_tree(entry, base)
                })
            else:
                items.append({
                    "name": entry.name,
                    "path": rel_path,
                    "type": "file",
                    "size": entry.stat().st_size
                })
    except PermissionError:
        pass
    return items


@app.get("/api/xid/{xid}/code/tree")
def get_code_tree(xid: str):
    """Get the file tree for an XID's source directory."""
    src_path = _get_srcdir(xid)
    return {"xid": xid, "tree": _build_tree(src_path, src_path)}


@app.get("/api/xid/{xid}/code/file/{file_path:path}")
def get_code_file(xid: str, file_path: str):
    """Get the content of a file from an XID's source directory."""
    src_path = _get_srcdir(xid)
    full_path = _safe_path(src_path, file_path)

    if not full_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    if not full_path.is_file():
        raise HTTPException(status_code=400, detail=f"Not a file: {file_path}")

    # Read file content (with size limit for safety)
    max_size = 10 * 1024 * 1024  # 10MB
    if full_path.stat().st_size > max_size:
        raise HTTPException(status_code=400, detail=f"File too large (max {max_size // 1024 // 1024}MB)")

    try:
        content = full_path.read_text(errors="replace")
        return Response(content=content, media_type="text/plain; charset=utf-8")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read file: {e}")


@app.get("/code/{xid}")
@app.get("/code/{xid}/{file_path:path}")
def code_browser_page(xid: str, file_path: str = ""):
    """Serve the code browser page for an XID."""
    # Verify the source directory exists
    _get_srcdir(xid)
    html = (SCRIPT_DIR / "code.html").read_text()
    html = html.replace("{{VERSION}}", __version__)
    return Response(content=html, media_type="text/html")


@app.get("/log/{jid}")
def log_viewer_page(jid: int):
    """Serve the log viewer page for a job."""
    html = (SCRIPT_DIR / "log.html").read_text()
    html = html.replace("{{VERSION}}", __version__)
    return Response(content=html, media_type="text/html")


@app.get("/api/log/{jid}")
def get_log(jid: int):
    """Get raw log content for a Slurm job."""
    # Search for the log file in all user directories
    log_filename = f"{jid}.txt"
    for user_dir in SLURM_OUT_DIR.iterdir():
        if not user_dir.is_dir():
            continue
        log_path = user_dir / log_filename
        if log_path.exists():
            content = log_path.read_text(errors="replace")
            return Response(content=content, media_type="text/plain; charset=utf-8")
    raise HTTPException(status_code=404, detail=f"Log file not found for job {jid}")


@app.post("/api/action/stop/{jid}")
def stop_job(jid: int):
    """Stop a job using scancel."""
    if not ACTIONS_ENABLED:
        raise HTTPException(status_code=403, detail="Actions are disabled (--no-actions)")
    log.info("POST /api/action/stop/%s", jid)
    result = subprocess.run(["scancel", str(jid)], capture_output=True, text=True)
    if result.returncode != 0:
        log.error("scancel failed: %s", result.stderr)
        raise HTTPException(status_code=500, detail=f"scancel failed: {result.stderr}")
    log.info("scancel %s succeeded", jid)
    return {"status": "ok", "jid": jid}


@app.post("/api/action/stop_xid/{xid}")
def stop_xid(xid: str):
    """Stop all jobs for an XID using scancel -n."""
    if not ACTIONS_ENABLED:
        raise HTTPException(status_code=403, detail="Actions are disabled (--no-actions)")
    log.info("POST /api/action/stop_xid/%s", xid)
    result = subprocess.run(["scancel", "-n", xid], capture_output=True, text=True)
    if result.returncode != 0:
        log.error("scancel -n %s failed: %s", xid, result.stderr)
        raise HTTPException(status_code=500, detail=f"scancel failed: {result.stderr}")
    log.info("scancel -n %s succeeded", xid)
    return {"status": "ok", "xid": xid}


@app.post("/api/action/requeue/{jid}")
def requeue_job(jid: int):
    """Requeue a job using scontrol requeue."""
    if not ACTIONS_ENABLED:
        raise HTTPException(status_code=403, detail="Actions are disabled (--no-actions)")
    log.info("POST /api/action/requeue/%s", jid)
    result = subprocess.run(["scontrol", "requeue", str(jid)], capture_output=True, text=True)
    if result.returncode != 0:
        log.error("scontrol requeue %s failed: %s", jid, result.stderr)
        raise HTTPException(status_code=500, detail=f"scontrol requeue failed: {result.stderr}")
    log.info("scontrol requeue %s succeeded", jid)
    return {"status": "ok", "jid": jid}


@app.post("/api/action/requeue_xid/{xid}")
def requeue_xid(xid: str):
    """Requeue all non-pending jobs for an XID."""
    if not ACTIONS_ENABLED:
        raise HTTPException(status_code=403, detail="Actions are disabled (--no-actions)")
    log.info("POST /api/action/requeue_xid/%s", xid)
    # Get all job IDs with state for this XID
    result = subprocess.run(["squeue", "-n", xid, "-h", "-o", "%i %T"], capture_output=True, text=True)
    if result.returncode != 0:
        log.error("squeue -n %s failed: %s", xid, result.stderr)
        raise HTTPException(status_code=500, detail=f"squeue failed: {result.stderr}")
    # Parse job IDs and filter out pending jobs (can't requeue pending)
    jids = []
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[1] != "PENDING":
            jids.append(parts[0])
    if not jids:
        raise HTTPException(status_code=404, detail=f"No requeuable jobs found for XID {xid}")
    # Requeue each job
    failed = []
    for jid in jids:
        r = subprocess.run(["scontrol", "requeue", jid], capture_output=True, text=True)
        if r.returncode != 0:
            log.error("scontrol requeue %s failed: %s", jid, r.stderr)
            failed.append(jid)
        else:
            log.info("scontrol requeue %s succeeded", jid)
    if failed:
        raise HTTPException(status_code=500, detail=f"Failed to requeue jobs: {failed}")
    return {"status": "ok", "xid": xid, "requeued": jids}


@app.post("/api/action/resume")
def resume_job(script: str):
    """Resume a job by running its launch script."""
    if not ACTIONS_ENABLED:
        raise HTTPException(status_code=403, detail="Actions are disabled (--no-actions)")
    log.info("POST /api/action/resume script=%s", script)
    script_path = Path(script)
    if not script_path.exists():
        raise HTTPException(status_code=404, detail=f"Launch script not found: {script}")
    if not script_path.is_file():
        raise HTTPException(status_code=400, detail=f"Not a file: {script}")
    result = subprocess.run(["bash", str(script_path)], capture_output=True, text=True, cwd=script_path.parent)
    if result.returncode != 0:
        log.error("Launch script failed: %s", result.stderr)
        raise HTTPException(status_code=500, detail=f"Launch failed: {result.stderr}")
    log.info("Launch script succeeded: %s", result.stdout.strip())
    return {"status": "ok", "output": result.stdout.strip()}


@app.post("/api/action/delete/{xid}")
def delete_xid(xid: str):
    """Delete an XID and all its associated directories."""
    if not ACTIONS_ENABLED:
        raise HTTPException(status_code=403, detail="Actions are disabled (--no-actions)")
    if not xid or not _xid_re.fullmatch(xid):
        raise HTTPException(status_code=400, detail="Invalid XID format")
    log.info("POST /api/action/delete/%s", xid)
    deleted = []
    for base in [BASEDIR, SRCDIR, FBIDIR]:
        path = base / xid
        if path.exists():
            shutil.rmtree(path)
            deleted.append(str(path))
            log.info("Deleted %s", path)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"XID {xid} not found in any directory")
    return {"status": "ok", "xid": xid, "deleted": deleted}


@app.post("/api/action/archive/{xid}")
def archive_xid(xid: str):
    """Archive an XID by removing checkpoints and moving workdir to archive directory."""
    if not ACTIONS_ENABLED:
        raise HTTPException(status_code=403, detail="Actions are disabled (--no-actions)")
    if ARCHIVE_DIR is None:
        raise HTTPException(status_code=400, detail="Archive directory not configured (use --archive-dir)")
    if not xid or not _xid_re.fullmatch(xid):
        raise HTTPException(status_code=400, detail="Invalid XID format")
    log.info("POST /api/action/archive/%s", xid)
    wd_path = BASEDIR / xid
    if not wd_path.exists():
        raise HTTPException(status_code=404, detail=f"XID {xid} not found")
    # Remove ckpt-* directories/files from workdir and subdirectories
    for item in wd_path.rglob("ckpt-*"):
        if item.is_symlink():
            item.unlink()
            log.info("Removed checkpoint symlink %s", item)
        elif item.is_dir():
            shutil.rmtree(item)
            log.info("Removed checkpoint dir %s", item)
        else:
            item.unlink()
            log.info("Removed checkpoint file %s", item)
    # Move workdir to archive directory
    dest = ARCHIVE_DIR / xid
    shutil.move(str(wd_path), str(dest))
    log.info("Moved %s to %s", wd_path, dest)
    return {"status": "ok", "xid": xid, "archived_to": str(dest)}


def main():
    global PREFS_DIR, ACTIONS_ENABLED, ARCHIVE_DIR
    import argparse
    parser = argparse.ArgumentParser(description="sManager - Slurm job management web UI")
    parser.add_argument("--version", action="version", version=f"smanager {__version__}")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=2337)
    parser.add_argument("--prefs-dir", help=f"Directory for preferences files (default: /checkpoint/rigi/USER)")
    parser.add_argument("--no-actions", action="store_true", help="Disable all action endpoints (stop, resume, delete)")
    parser.add_argument("--archive-dir", help="Directory where archived XIDs are moved to")
    args = parser.parse_args()
    if args.prefs_dir:
        PREFS_DIR = Path(args.prefs_dir)
        log.info("Preferences directory: %s", PREFS_DIR)
    if args.no_actions:
        ACTIONS_ENABLED = False
        log.info("Actions are DISABLED (--no-actions flag)")
    if args.archive_dir:
        ARCHIVE_DIR = Path(args.archive_dir)
        log.info("Archive directory: %s", ARCHIVE_DIR)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
