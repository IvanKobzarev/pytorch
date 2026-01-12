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

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
import uvicorn

from smanager import __version__

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

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
PREFS_DIR = Path(f"/checkpoint/rigi/{getuser()}")  # Set via --prefs-dir flag
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
    xid, info = xid_info
    wd_path = BASEDIR / info["wd"]
    try:
        info["user"] = wd_path.owner()
    except:
        info["user"] = "?"
    # Count total WUs from launch scripts (includes pending jobs without workdirs)
    info["total_wus"] = len(list(wd_path.glob("launch_*.sh")))
    launchinfo = wd_path / 'launchinfo.txt'
    workdir_names = []
    if launchinfo.is_file():
        info["wus"] = {}
        for wuwd in wd_path.iterdir():
            if wuwd.is_dir():
                info["wus"][str(wuwd.name)] = (wuwd / "DONE").exists()
                workdir_names.append(wuwd.name)
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
    info["name"] = extract_common_name(workdir_names, xid)
    # Read note if exists
    note_file = wd_path / "NOTE.md"
    info["note"] = note_file.read_text().strip() if note_file.exists() else ""
    return xid, info


@app.get("/api/health")
def health():
    log.info("GET /api/health")
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


@app.get("/api/prefs/hidden")
def get_hidden():
    log.info("GET /api/prefs/hidden")
    return {"hidden": _read_xid_file("xid_hide.txt")}


@app.post("/api/prefs/hidden")
def set_hidden(hidden: list[str] = Body(...)):
    log.info("POST /api/prefs/hidden (%d items)", len(hidden))
    _write_xid_file("xid_hide.txt", hidden)
    return {"status": "ok"}


@app.post("/api/prefs/hide/{xid}")
def hide_xid(xid: str):
    log.info("POST /api/prefs/hide/%s", xid)
    current = set(_read_xid_file("xid_hide.txt"))
    current.add(xid)
    _write_xid_file("xid_hide.txt", current)
    return {"status": "ok", "hidden": True}


@app.post("/api/prefs/unhide/{xid}")
def unhide_xid(xid: str):
    log.info("POST /api/prefs/unhide/%s", xid)
    current = set(_read_xid_file("xid_hide.txt"))
    current.discard(xid)
    _write_xid_file("xid_hide.txt", current)
    return {"status": "ok", "hidden": False}


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
def get_overview(include_hidden: bool = False):
    """Get overview of all active and recent experiments.

    Args:
        include_hidden: If true, include hidden XIDs in cold/frozen (marked with hidden=true).
    """
    t0 = time.time()
    hidden_xids = set(_read_xid_file("xid_hide.txt"))
    log.info("GET /api/overview - fetching jobs... (%d hidden, include=%s)", len(hidden_xids), include_hidden)
    headers, job_rows = get_jobs()
    if not headers:
        log.info("GET /api/overview - no jobs found (%.2fs)", time.time() - t0)
        return {"hot": {}, "cold": {}, "frozen_count": 0}

    # Build jobs lookup
    jobs_by_name = {}
    for row in job_rows:
        job = dict(zip(headers, row))
        name = job.get("NAME", "")
        if name not in jobs_by_name:
            jobs_by_name[name] = []
        jobs_by_name[name].append(job)

    # Read workdirs
    workdirs = [d.name for d in BASEDIR.iterdir() if d.is_dir()]
    wd_by_xid = {xid: wd for wd in workdirs if (xid := extract_xid(wd))}

    # Split into hot/cold/frozen
    hot_xids = {}
    inactive_xids = {}
    for xid, wd in sorted(wd_by_xid.items(), reverse=True):
        xid_jobs = jobs_by_name.get(xid, [])
        states = Counter(j.get("STATE", "") for j in xid_jobs)
        if states:
            hot_xids[xid] = {"states": dict(states), "wd": wd, "jobs": xid_jobs}
        else:
            is_hidden = xid in hidden_xids
            if include_hidden or not is_hidden:
                inactive_xids[xid] = {"wd": wd, "hidden": is_hidden}

    # Split inactive into cold (first NUM_RECENT) and frozen (rest)
    inactive_list = list(inactive_xids.keys())
    cold_xids = {xid: inactive_xids[xid] for xid in inactive_list[:NUM_RECENT]}
    frozen_xids = {xid: inactive_xids[xid] for xid in inactive_list[NUM_RECENT:]}

    # Add extra info with threading
    hot_items = [(xid, {"states": info["states"], "wd": info["wd"], "jobs": info["jobs"]}) for xid, info in hot_xids.items()]
    hot_results = list(executor.map(_extra_info, hot_items))
    hot_xids = {}
    for xid, info in hot_results:
        hot_xids[xid] = info

    cold_items = [(xid, info) for xid, info in cold_xids.items()]
    cold_results = list(executor.map(_extra_info, cold_items))
    cold_xids = dict(cold_results)

    frozen_items = [(xid, info) for xid, info in frozen_xids.items()]
    frozen_results = list(executor.map(_extra_info, frozen_items))
    frozen_xids = dict(frozen_results)

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

    result = {
        "hot": hot_xids,
        "cold": cold_xids,
        "frozen": frozen_xids,
    }
    log.info("GET /api/overview - done: %d hot, %d cold, %d frozen (%.2fs)",
             len(hot_xids), len(cold_xids), len(frozen_xids), time.time() - t0)
    return result


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


def last_metric(wd_path):
    fname = wd_path / "metrics.jsonl"
    if not fname.exists():
        return {}
    try:
        # Use tail -n 1 to efficiently read last line (seeks from end, fast on NFS)
        result = subprocess.run(["tail", "-n", "1", str(fname)], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            return json.loads(result.stdout.strip())
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
    wd_path, wuwd_name = args
    return wuwd_name, last_metric(wd_path / wuwd_name)


@app.get("/api/xid/{xid}")
def get_xid_info(xid: str):
    """Get detailed info for a specific XID."""
    t0 = time.time()
    log.info("GET /api/xid/%s - fetching...", xid)
    wd_path = BASEDIR / xid
    if not wd_path.exists():
        # Try to find it
        for d in BASEDIR.iterdir():
            if d.is_dir() and xid in d.name:
                wd_path = d
                break
        else:
            log.warning("GET /api/xid/%s - not found (%.2fs)", xid, time.time() - t0)
            raise HTTPException(status_code=404, detail=f"XID {xid} not found")

    # Get current jobs for this xid
    lines = run_cmd(f"squeue -n {xid} -O JobId:20,Name:20,UserName:20,State:20,TimeUsed:20,NumCPUs:20,QOS:20,NumNodes:20,GRES:20,RestartCnt:20,Reason:20")
    xid_jobs = [[j[i*20:(i+1)*20].strip() for i in range(11)] for j in lines if j.strip()]
    if len(xid_jobs) >= 2:
        headers, job_rows = xid_jobs[0], xid_jobs[1:]
        jobs_by_jid = {row[0]: dict(zip(headers, row)) for row in job_rows}
    else:
        jobs_by_jid = {}

    # Get workdirs
    workdirs = [d.name for d in wd_path.iterdir() if d.is_dir()]
    log.info("  - found %d workdirs", len(workdirs))

    # Load configs in parallel
    t1 = time.time()
    config_results = list(executor.map(_load_config_only, [(wd_path, wd) for wd in workdirs]))
    log.info("  - loaded configs in %.2fs", time.time() - t1)

    # Load metrics in parallel
    t2 = time.time()
    metric_results = list(executor.map(_load_metric_only, [(wd_path, wd) for wd in workdirs]))
    log.info("  - loaded metrics in %.2fs", time.time() - t2)

    configs = {}
    last_metrics = {}
    metric_by_wuwd = {wuwd_name: metrics for wuwd_name, metrics in metric_results}
    for wuwd_name, config in config_results:
        wid = config.get("wid", wuwd_name)
        configs[wid] = config
        last_metrics[wid] = metric_by_wuwd.get(wuwd_name, {})

    # Debug: log first config's keys
    if configs:
        first_config = next(iter(configs.values()))
        log.info("  - sample config keys: %s", list(first_config.keys())[:15])

    # Get sacct info for all jids
    jids = set()
    for wid, config in configs.items():
        if "jid" in config:
            jids.add(config["jid"])
    jids.update(int(jid) for jid in jobs_by_jid.keys() if jid.isdigit())
    log.info("  - found %d jids to fetch sacct for (configs have jid: %d)",
             len(jids), sum(1 for c in configs.values() if "jid" in c))

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
        if name and (wd_path / name / "DONE").is_file():
            status[wid] = "DONE"
        elif (jid := config.get("jid")) and str(jid) in jobs_by_jid:
            status[wid] = jobs_by_jid[str(jid)].get("STATE", "UNKNOWN")
        elif jid and jid in saccts and saccts[jid].get('state', {}).get('current'):
            status[wid] = saccts[jid]['state']['current'][-1]
        else:
            status[wid] = "UNKNOWN"

    # Format for response
    wus = []
    for wid in sorted(configs.keys(), key=lambda x: int(x) if str(x).isdigit() else x):
        config = configs[wid]
        jid = config.get("jid", "")
        sacct = saccts.get(jid, {})
        metrics = last_metrics.get(wid, {})

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

        step = metrics.get("step")
        nsteps = config.get("nsteps")
        progress = f"{step / nsteps:.1%}" if step and nsteps else "n/a"

        wus.append({
            "wid": wid,
            "jid": jid,
            "restarts": sacct.get("restart_cnt", 0),
            "exit_code": sacct.get("exit_code", {}).get("return_code", {}).get("number", 0),
            "status": status.get(wid, "UNKNOWN"),
            "progress": progress,
            "step": step,
            "metrics": metrics,
            "config_args": sws_args,
            "name": config.get("name", ""),
            "qwait": qwait,
            "runtime": hms(elapsed),
            "workdir": str(config.get("workdir", "")),
            "launch_script": str(Path(config.get("workdir", "")) / f"../launch_{wid}.sh") if config.get("workdir") else "",
        })

    result = {
        "xid": xid,
        "wus": wus,
        "launch_command": (wd_path / "launchinfo.txt").read_text() if (wd_path / "launchinfo.txt").exists() else "",
    }
    log.info("GET /api/xid/%s - done: %d work units (%.2fs)", xid, len(wus), time.time() - t0)
    return result


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


def main():
    global PREFS_DIR, ACTIONS_ENABLED
    import argparse
    parser = argparse.ArgumentParser(description="sManager - Slurm job management web UI")
    parser.add_argument("--version", action="version", version=f"smanager {__version__}")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=2337)
    parser.add_argument("--prefs-dir", help=f"Directory for preferences files (default: /checkpoint/rigi/USER)")
    parser.add_argument("--no-actions", action="store_true", help="Disable all action endpoints (stop, resume, delete)")
    args = parser.parse_args()
    if args.prefs_dir:
        PREFS_DIR = Path(args.prefs_dir)
        log.info("Preferences directory: %s", PREFS_DIR)
    if args.no_actions:
        ACTIONS_ENABLED = False
        log.info("Actions are DISABLED (--no-actions flag)")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
