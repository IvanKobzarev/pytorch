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
import stat
import shutil
from datetime import datetime
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


# Probe for every LEGACY - BACKFILLED - REMOVE SOON branch. Appends a line to
# SMANAGER_LEGACY_LOG (default /checkpoint/rigi/bv2/smanager_legacy_used.log,
# shared across users) and emits a WARNING. Use this to verify a fallback is
# truly unreachable before deleting it. Small append writes are atomic under
# POSIX so concurrent appends from multiple smanager servers won't interleave.
LEGACY_LOG_PATH = Path(os.environ.get(
    "SMANAGER_LEGACY_LOG", "/checkpoint/rigi/bv2/smanager_legacy_used.log"))


def _legacy_used(tag, **ctx):
    ctx["_user"] = getuser()
    line = f"{datetime.now().isoformat(timespec='seconds')}\t{tag}\t{json.dumps(ctx, default=str)}\n"
    try:
        with open(LEGACY_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line)
    except OSError:
        pass
    log.warning("LEGACY_USED: %s %s", tag, ctx)


app = FastAPI(title="Slurm Manager API")

SCRIPT_DIR = Path(__file__).parent

# Use ThreadPoolExecutor for parallel I/O (NFS can be slow, so use many threads)
executor = ThreadPoolExecutor(max_workers=64)
fs_executor = ThreadPoolExecutor(max_workers=64)


def shutdown_handler(signum, frame):
    log.info("Received signal %s, shutting down...", signum)
    executor.shutdown(wait=False, cancel_futures=True)
    fs_executor.shutdown(wait=False, cancel_futures=True)
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

        # Skip compression for file downloads (they should stream directly)
        path = scope.get("path", "")
        if "/files/download/" in path:
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
_CLUSTER_GROUP = {"fair-sc": "rigi", "fair-sc-3": "rigi", "dm1": "fair_amaia_cw_explore"}
_CLUSTER_USERS = {"dm1": "pplx,qkv,zhai"}
_cluster = os.environ.get("SLURM_CLUSTER_NAME", "")
GROUP, USERS = _CLUSTER_GROUP.get(_cluster, "rigi"), _CLUSTER_USERS.get(_cluster, "")
BASEDIR = Path("/checkpoint/rigi/bv2/workdirs")
SRCDIR = Path("/checkpoint/rigi/bv2/srcdirs")
FBIDIR = Path("/checkpoint/rigi/fbi")
SLURM_OUT_DIR = Path("/checkpoint/rigi/bv2/slurm_out")
PREFS_DIR = Path(f"/checkpoint/rigi/{getuser()}")  # Set via --prefs-dir flag
ARCHIVE_DIR = Path("/checkpoint/rigi/bv2/workdirs-archive")  # Set via --archive-dir flag
NUM_RECENT = 50
ACTIONS_ENABLED = True  # Set via --no-actions flag
SLURM_NO_START_TIME = 0xFFFFFFFE

_xid_re = re.compile(r'\d{4,6}_\d{6}')


def run_cmd(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout.strip().split('\n')


def extract_xid(name):
    if firstmatch := _xid_re.search(name):
        return firstmatch.group()
    return None


def get_jobs(group=GROUP, users=USERS):
    widths = [20, 20, 20, 20, 20, 20, 40, 20, 20, 20, 20, 20, 20, 100]
    fmt = "JobId:20,Name:20,UserName:20,State:20,TimeUsed:20,NumCPUs:20,QOS:40,NumNodes:20,tres-per-node:20,RestartCnt:20,Reason:20,Priority:20,PriorityLong:20,Comment:100"
    lines = run_cmd(f"squeue -A {group}" + (f" -u {users}" if users else "") + f" -O {fmt}")
    offsets = [sum(widths[:i]) for i in range(len(widths))]
    jobs = [[j[offsets[i]:offsets[i]+widths[i]].strip() for i in range(len(widths))] for j in lines if j.strip()]
    if len(jobs) < 2:
        return [], []
    headers = jobs[0]
    # Handle duplicate header names (e.g., both Priority and PriorityLong may output as PRIORITY)
    seen = {}
    for i, h in enumerate(headers):
        if h in seen:
            headers[i] = h + "_LONG"
        seen[h] = True
    return headers, jobs[1:]


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


def dir_names(path):
    with os.scandir(path) as it:
        return [e.name for e in it if e.is_dir(follow_symlinks=False)]


def wid_from_workdir_name(name):
    if m := re.search(r'-(\d+)$', name):
        return int(m.group(1))
    return None


def state_from_exit_status(status):
    if not status:
        return None
    status = str(status).lower()
    if status.endswith(" (wip)"):
        status = status[:-6]
    if status == "done":
        return True
    if status == "stopped":
        return "CANCELLED"
    if status == "preempted":
        return "PREEMPTED"
    if status == "error":
        return False
    return status.upper()


def state_from_sacct(sacct):
    state = sacct.get('state', {}).get('current', [None])[-1]
    exit_code = sacct.get('exit_code', {}).get('return_code', {}).get('number', 0)
    if state == 'CANCELLED':
        return 'CANCELLED' if exit_code == 0 else 'CANCELLED_FAIL'
    return state or 'UNKNOWN'


def detail_status_from_exit_status(status):
    state = state_from_exit_status(status)
    if state is True:
        return "DONE"
    if state is False:
        return "FAILED"
    return state


def _extra_info(xid_info):
    """Get extra info for an XID.

    Args:
        xid_info: Tuple of (xid, info) or (xid, info, skip_config_loading)
            Config loading is only a fallback/backcompat path for legacy
            workdirs that cannot be mapped to WIDs from launch metadata/name.
    """
    if len(xid_info) == 3:
        xid, info, skip_config_loading = xid_info
    else:
        xid, info = xid_info
        skip_config_loading = False

    wd_path = BASEDIR / info["wd"]

    # Overlap independent NFS reads with the rest of the work.
    owner_fut = fs_executor.submit(wd_path.owner)
    launchinfo_fut = fs_executor.submit(lambda: (wd_path / 'launchinfo.txt').read_text())
    note_fut = fs_executor.submit(lambda: (wd_path / 'NOTE.md').read_text())

    launchids = load_launchids(wd_path)

    # Single scandir gathers both subdir names and (legacy) launch_*.sh files.
    wuwd_names = []
    launch_wids_from_files = set()
    with os.scandir(wd_path) as it:
        for e in it:
            name = e.name
            if e.is_dir(follow_symlinks=False):
                wuwd_names.append(name)
            elif (name.startswith("launch_") and name.endswith(".sh")
                  and e.is_file(follow_symlinks=False)
                  and (m := re.match(r'launch_(\d+)\.sh', name))):
                launch_wids_from_files.add(int(m.group(1)))
    # LEGACY - BACKFILLED - REMOVE SOON: every experiment now has launchids.json
    # (backfilled by bv2/tools/backfill_launchids_json). launch_wids_from_files
    # is unreachable in steady state.
    if launchids:
        launch_wids = {normalize_wid(wid) for wid in launchids}
    else:
        _legacy_used("extra_info.no_launchids", xid=xid)
        launch_wids = launch_wids_from_files
    info["total_wus"] = len(launch_wids)

    launch_jid_by_wid = {
        normalize_wid(wid): normalize_jid(meta.get("launchjid"))
        for wid, meta in launchids.items()
        if isinstance(meta, dict)
    }
    wid_by_launch_jid = {jid: wid for wid, jid in launch_jid_by_wid.items() if jid}
    workdir_names = []
    wid_counts = {}  # wid -> count (for duplicate detection)
    workdir_by_wid = {}
    workdir_done_by_wid = {}
    workdir_jid_by_wid = {}
    config_by_wid = {}
    info["wus"] = {}
    done_results = fs_executor.map(_load_done_only, [(wd_path, wuwd_name) for wuwd_name in wuwd_names])
    for wuwd_name, (_, wid, is_done, mtime) in zip(wuwd_names, done_results):
        wuwd = wd_path / wuwd_name
        if wid is None and not skip_config_loading:
            # LEGACY - BACKFILLED - REMOVE SOON: workdir names now always end
            # in "-{wid}" (backfilled by bv2/tools/backfill_wid_suffix; new
            # runs always produce the suffix). Config-loading recovery is
            # unreachable in steady state.
            _legacy_used("extra_info.no_wid_suffix", xid=xid, wuwd=wuwd_name)
            config = load_config(wuwd)
            wid = normalize_wid(config.get("wid", wuwd_name))
            config_by_wid[wid] = config
            if config.get("jid"):
                workdir_jid_by_wid[wid] = normalize_jid(config.get("jid"))
            try:
                mtime = (wuwd / "DONE").stat().st_mtime
                is_done = True
            except FileNotFoundError:
                pass
        if wid is None:
            continue
        # DONE mtime is the broad-scan signal for finish_time -- cheaper than
        # loading config.exit_status_at, and is the only available signal for
        # the inactive overview which skips config loading entirely.
        if is_done and mtime > info.get("finish_time", 0):
            info["finish_time"] = mtime
        workdir_names.append(wuwd_name)
        workdir_by_wid[wid] = wuwd_name
        wid_counts[wid] = wid_counts.get(wid, 0) + 1
        workdir_done_by_wid[wid] = workdir_done_by_wid.get(wid, False) or is_done

    for wid in launch_wids | set(workdir_done_by_wid):
        info["wus"][str(wid)] = workdir_done_by_wid.get(wid, False)

    try:
        info["config"] = next(re.finditer(r"bv2/config/(.*?) ", launchinfo_fut.result())).group(1)
    except (OSError, StopIteration):
        info["config"] = "?"

    active_jobs_by_wid = {}
    if "jobs" in info:
        active_by_wid = {}
        active_jid_by_wid = {}
        for job in info["jobs"]:
            wid = extract_wid(job.get("COMMENT", ""))
            jid = normalize_jid(job.get("JOBID"))
            if wid is None and jid:
                # LEGACY - BACKFILLED - REMOVE SOON: live jobs now always have
                # xid/wid in COMMENT (set by _launch.py; gaps patched by
                # bv2/tools/backfill_slurm_comments).
                _legacy_used("extra_info.no_comment", xid=xid, jid=jid)
                wid = wid_by_launch_jid.get(jid)
            if wid is not None and jid:
                active_jobs_by_wid.setdefault(wid, []).append(job)
                if wid not in active_jid_by_wid or jid > active_jid_by_wid[wid]:
                    active_by_wid[wid] = job.get("STATE", "UNKNOWN")
                    active_jid_by_wid[wid] = jid

        finished_by_wid = {}
        if not skip_config_loading:
            missing_config_wids = [
                wid for wid in set(workdir_done_by_wid) - set(active_by_wid)
                if wid in workdir_by_wid and wid not in config_by_wid
            ]
            config_results = fs_executor.map(
                _load_config_only,
                [(wd_path, workdir_by_wid[wid]) for wid in missing_config_wids],
            )
            for wid, (_, config) in zip(missing_config_wids, config_results):
                config_by_wid[wid] = config

            sacct_jids = {}
            for wid in (set(workdir_done_by_wid) - set(active_by_wid)):
                wuwd_name = workdir_by_wid.get(wid)
                if not wuwd_name:
                    continue
                config = config_by_wid.get(wid)
                state = state_from_exit_status(config.get("exit_status"))
                if state is not None:
                    finished_by_wid[wid] = state
                elif workdir_done_by_wid.get(wid):
                    # LEGACY - BACKFILLED - REMOVE SOON: exit_status backfilled
                    # via bv2/tools/backfill_exit_status; DONE-only path is
                    # unreachable in steady state.
                    _legacy_used("extra_info.done_no_exit_status", xid=xid, wid=wid)
                    finished_by_wid[wid] = True
                elif jid := normalize_jid(config.get("jid")):
                    # Live sacct fallback for in-flight WUs that have no DONE
                    # file and no exit_status -- typically jobs that were hard-
                    # killed mid-step or were restarted (train.py rewrites
                    # config.json on startup, wiping any previously-written
                    # exit_status). Backfill can't eliminate this path.
                    sacct_jids[jid] = wid

            # WUs in launchids that haven't created a workdir yet (just
            # submitted, queued, or cancelled-before-running). Without this
            # they fall into the "UNKNOWN" bucket in the overview even though
            # sacct knows their state. Same batched sacct call covers them.
            for wid in launch_wids - set(workdir_done_by_wid) - set(active_by_wid):
                if (jid := launch_jid_by_wid.get(wid)) is not None:
                    sacct_jids[jid] = wid

            for jid, sacct in load_sacct_many(sacct_jids).items():
                finished_by_wid[sacct_jids[jid]] = state_from_sacct(sacct)

        if active_by_wid or workdir_done_by_wid or launch_wids:
            states = Counter()
            for wid in launch_wids | set(workdir_done_by_wid) | set(active_by_wid):
                if wid in active_by_wid:
                    if (workdir_done_by_wid.get(wid) and active_by_wid[wid] == "RUNNING"
                            and active_jid_by_wid.get(wid) == (workdir_jid_by_wid.get(wid) or launch_jid_by_wid.get(wid))):
                        states["DONE_ISH"] += 1
                    else:
                        states[active_by_wid[wid]] += 1
                elif wid in finished_by_wid:
                    states[finished_by_wid[wid]] += 1
                elif wid in workdir_done_by_wid:
                    states[workdir_done_by_wid[wid]] += 1
                else:
                    states["UNKNOWN"] += 1
            info["effective_states"] = dict(states)

    info["name"] = extract_common_name(workdir_names, xid)
    warning_wids = {wid for wid, count in wid_counts.items() if count > 1}
    warning_wids.update(_active_duplicate_warnings(active_jobs_by_wid))
    info["warning_count"] = len(warning_wids)
    try:
        info["note"] = note_fut.result().strip()
    except FileNotFoundError:
        info["note"] = ""
    try:
        info["user"] = owner_fut.result()
    except Exception:
        info["user"] = "?"
    return xid, info


@app.get("/api/health")
def health():
    return {"status": "ok", "group": GROUP, "user": getuser(), "actions_enabled": ACTIONS_ENABLED}


def _read_xid_file(filename):
    try:
        text = (PREFS_DIR / filename).read_text()
    except FileNotFoundError:
        return []
    return [line.strip() for line in text.splitlines() if line.strip()]


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
        with os.scandir(BASEDIR) as it:
            for e in it:
                if xid in e.name and e.is_dir(follow_symlinks=False):
                    wd_path = Path(e.path)
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


@app.get("/fonts/{filename}")
def serve_font(filename: str):
    if not filename.endswith(".woff2"):
        raise HTTPException(status_code=404, detail="Not found")
    font_path = SCRIPT_DIR / filename
    if not font_path.exists():
        raise HTTPException(status_code=404, detail="Font not found")
    return Response(content=font_path.read_bytes(), media_type="font/woff2")


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
    jobs_by_xid = {}
    misc_jobs = []
    for row in job_rows:
        job = dict(zip(headers, row))
        name = job.get("NAME", "")
        if xid := extract_xid(name):
            if xid not in jobs_by_xid:
                jobs_by_xid[xid] = []
            jobs_by_xid[xid].append(job)
        else:
            misc_jobs.append(job)

    # Only resolve workdirs for XIDs that have active jobs. New launchers use
    # /workdirs/{xid}, so this avoids scanning the whole NFS directory.
    t2 = time.time()
    active_xids = list(jobs_by_xid.keys())
    is_dir_results = fs_executor.map(lambda xid: (BASEDIR / xid).is_dir(), active_xids)
    wd_by_xid = {xid: xid for xid, ok in zip(active_xids, is_dir_results) if ok}
    active_xids = set(active_xids)
    missing_xids = active_xids - set(wd_by_xid)
    if missing_xids:
        # LEGACY - BACKFILLED - REMOVE SOON: all workdirs now live exactly at
        # /workdirs/{xid} (bv2/tools/backfill_xid_dirname). The scan below is
        # unreachable in steady state; missing_xids without a hit just means
        # the workdir doesn't exist yet on this server.
        with os.scandir(BASEDIR) as it:
            for e in it:
                if (xid := extract_xid(e.name)) in missing_xids and e.is_dir(follow_symlinks=False):
                    _legacy_used("overview.missing_xids", xid=xid, found=e.name)
                    wd_by_xid[xid] = e.name
    log.info("  - workdir lookup took %.2fs (%d workdirs matched)", time.time() - t2, len(wd_by_xid))

    # Build hot xids
    hot_xids = {}
    for xid, wd in wd_by_xid.items():
        xid_jobs = jobs_by_xid.get(xid, [])
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
            info["total_gpus"] = gpus_per_job * info.get("effective_states", {}).get("RUNNING", 0)
            info["max_restarts"] = max(int(j.get("RESTART_COUNT", 0)) for j in xid_jobs)
        else:
            info["qos"] = ""
            info["users"] = ""
            info["gpus_per_job"] = 0
            info["total_gpus"] = 0
            info["max_restarts"] = 0
        # Strip server-internal fields from the response payload.
        info.pop("jobs", None)
        info.pop("states", None)

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
        misc = {"count": len(misc_jobs), "gpus": misc_gpus, "states": dict(misc_states), "jobs": misc_jobs}

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
    workdirs = dir_names(BASEDIR)
    wd_by_xid = {xid: wd for wd in workdirs if (xid := extract_xid(wd))}

    # Build list of all XIDs (client filters out hot ones)
    all_xids = {}
    for xid, wd in sorted(wd_by_xid.items(), reverse=True):
        all_xids[xid] = {"wd": wd}

    # Split into cold (first NUM_RECENT) and frozen (rest)
    all_list = list(all_xids.keys())
    cold_xids = {xid: all_xids[xid] for xid in all_list[:NUM_RECENT]}
    frozen_xids = {xid: all_xids[xid] for xid in all_list[NUM_RECENT:]}

    # Add extra info with threading. Config reads are disabled here because they
    # are only fallback/backcompat for legacy workdir names.
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
    try:
        return json.loads((wd_path / "config.json").read_text())
    except FileNotFoundError:
        return {}
    except Exception as e:
        log.debug("load_config failed for %s: %s", wd_path, e)
        return {}


def last_metric(wd_path, metric_name="train/loss"):
    # Try plattli format first
    if plattli.is_run(wd_path):
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

    # LEGACY - BACKFILLED - REMOVE SOON: every workdir is now plattli (old
    # metrics.jsonl files were converted in place). This fallback is dead.
    fname = wd_path / "metrics.jsonl"
    if not fname.exists():
        return {}
    _legacy_used("metric.jsonl_fallback", wd=str(wd_path))
    try:
        res = subprocess.run(["tail", "-n", "1", str(fname)], capture_output=True, text=True)
        if res.returncode == 0 and res.stdout.strip():
            return json.loads(res.stdout.strip())
    except:
        pass
    return {}


def load_sacct(jid):
    try:
        result = subprocess.run(["sacct", "--long", f"--jobs={jid}", "--json"], capture_output=True, text=True)
        if result.returncode != 0:
            log.debug("load_sacct nonzero for %s: rc=%s stderr=%s", jid, result.returncode, result.stderr.strip())
            return {}
        if not result.stdout.strip():
            log.debug("load_sacct empty stdout for %s: stderr=%s", jid, result.stderr.strip())
            return {}
        data = json.loads(result.stdout)
        if data.get('jobs'):
            return data['jobs'][0]
    except Exception as e:
        log.debug("load_sacct failed for %s: %s", jid, e)
    return {}


def load_sacct_many(jids):
    jids = sorted({normalize_jid(jid) for jid in jids if normalize_jid(jid)})
    if not jids:
        return {}
    try:
        result = subprocess.run(["sacct", "--long", f"--jobs={','.join(map(str, jids))}", "--json"], capture_output=True, text=True)
        if result.returncode != 0:
            log.debug("load_sacct_many nonzero for %s jobs: rc=%s stderr=%s", len(jids), result.returncode, result.stderr.strip())
            return {}
        if not result.stdout.strip():
            log.debug("load_sacct_many empty stdout for %s jobs: stderr=%s", len(jids), result.stderr.strip())
            return {}
        res = {}
        for job in json.loads(result.stdout).get('jobs', []):
            if jid := normalize_jid(job.get("job_id")):
                res[jid] = job
        return res
    except Exception as e:
        log.debug("load_sacct_many failed for %s jobs: %s", len(jids), e)
    return {}


def get_sacct_end_time(xid, user):
    """Get job end time from sacct for an XID."""
    try:
        result = run_cmd(f"sacct -n -X -o End -S now-14days -u {user} --name={xid}")
        max_time = 0
        for line in result:
            line = line.strip()
            if line and line != "Unknown":
                try:
                    dt = datetime.strptime(line, "%Y-%m-%dT%H:%M:%S")
                    max_time = max(max_time, dt.timestamp())
                except:
                    pass
        return max_time if max_time > 0 else None
    except:
        return None


def _load_config_only(args):
    wd_path, wuwd_name = args
    return wuwd_name, load_config(wd_path / wuwd_name)


def _load_done_only(args):
    wd_path, wuwd_name = args
    wid = wid_from_workdir_name(wuwd_name)
    if wid is None:
        return wuwd_name, None, False, None
    try:
        return wuwd_name, wid, True, (wd_path / wuwd_name / "DONE").stat().st_mtime
    except FileNotFoundError:
        return wuwd_name, wid, False, None


def _load_metric_only(args):
    wd_path, wuwd_name, metric_name = args
    return wuwd_name, last_metric(wd_path / wuwd_name, metric_name)


def load_launchids(wd_path):
    try:
        launchids = json.loads((wd_path / "launchids.json").read_text())
    except FileNotFoundError:
        return {}
    for info in launchids.values():
        if "launchjid" not in info and "jid" in info:
            info["launchjid"] = info["jid"]
    return launchids


def extract_wid(submit_line):
    if not submit_line:
        return None
    if m := re.search(r'\bwid:?=(\d+)', submit_line):
        return int(m.group(1))
    if m := re.search(r'launch_(\d+)\.sh', submit_line):
        return int(m.group(1))
    return None


def normalize_jid(jid):
    if jid is None:
        return None
    if isinstance(jid, int):
        return jid
    jid = str(jid).split('.', 1)[0]
    return int(jid) if jid.isdigit() else None


def normalize_wid(wid):
    normalized = normalize_jid(wid)
    return normalized if normalized is not None else wid


def extract_launch_info(launch_file):
    launch_line = ""
    name = ""
    for line in launch_file.read_text().splitlines():
        line = line.strip()
        if not line.startswith("sbatch"):
            continue
        launch_line = line
        for a in shlex.split(line):
            if m := re.match(r'name:=(.+)', a):
                name = m.group(1)
        break
    return {"name": name, "launch_line": launch_line}


def extract_sws_args(submit_line):
    if not submit_line:
        return []
    sws_args = []
    ignore = ["xid:=", "wid:=", "jid:=", "name:="]
    for a in shlex.split(submit_line):
        if "=" in a and not a.startswith("--") and not any(x in a for x in ignore):
            sws_args.append(a)
    return sws_args


def extract_exit_code(sacct):
    ret = sacct.get("exit_code", {}).get("return_code", {})
    if not ret.get("set"):
        return None
    return ret.get("number")


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


def format_sacct(sacct):
    time_info = sacct.get("time", {})
    start = time_info.get("start", 0)
    eligible = time_info.get("eligible", 0)
    state = sacct.get("state", {}).get("current", [])
    return {
        "restarts": sacct.get("restart_cnt", 0),
        "exit_code": extract_exit_code(sacct),
        "status": state[-1] if state else None,
        "qwait": "never" if start in [0, SLURM_NO_START_TIME] else hms(start - eligible),
        "runtime": hms(time_info.get("elapsed", 0)),
    }


def _find_xid_path(xid):
    """Find the workdir path for an XID."""
    wd_path = BASEDIR / xid
    if wd_path.exists():
        return wd_path
    # LEGACY - BACKFILLED - REMOVE SOON: all workdirs now live exactly at
    # /workdirs/{xid} (bv2/tools/backfill_xid_dirname). The scan below is
    # unreachable in steady state; the bare `return None` is the normal
    # multi-server "this server doesn't have that XID" path.
    t0 = time.time()
    with os.scandir(BASEDIR) as it:
        for e in it:
            if xid in e.name and e.is_dir(follow_symlinks=False):
                _legacy_used("find_xid_path.containment_scan", xid=xid, found=e.name)
                log.info("  - _find_xid_path fallback took %.2fs", time.time() - t0)
                return Path(e.path)
    log.info("  - _find_xid_path fallback took %.2fs (not found)", time.time() - t0)
    return None


def _load_current_xid_jobs(xid):
    xid_widths = [20, 20, 20, 20, 20, 20, 40, 20, 20, 20, 20, 100]
    xid_fmt = "JobId:20,Name:20,UserName:20,State:20,TimeUsed:20,NumCPUs:20,QOS:40,NumNodes:20,GRES:20,RestartCnt:20,Reason:20,Comment:100"
    lines = run_cmd(f"squeue -n {xid} -O {xid_fmt}")
    offsets = [sum(xid_widths[:i]) for i in range(len(xid_widths))]
    rows = [[j[offsets[i]:offsets[i]+xid_widths[i]].strip() for i in range(len(xid_widths))] for j in lines if j.strip()]
    if len(rows) < 2:
        return {}
    headers, job_rows = rows[0], rows[1:]
    return {row[0]: dict(zip(headers, row)) for row in job_rows}


def _active_duplicate_warnings(active_jobs_by_wid):
    warnings = {}
    for wid, jobs in active_jobs_by_wid.items():
        if len(jobs) <= 1:
            continue
        parts = []
        for job in sorted(jobs, key=lambda j: normalize_jid(j.get("JOBID")) or 0):
            jid = normalize_jid(job.get("JOBID"))
            state = job.get("STATE", "UNKNOWN")
            parts.append(f"{jid} {state}" if jid else state)
        warnings[wid] = [f"Multiple active Slurm jobs for this WID: {', '.join(parts)}"]
    return warnings


def _jobs_by_wid(jobs_by_jid, launchids):
    launch_wid_by_jid = {
        normalize_jid(meta.get("launchjid")): normalize_wid(wid)
        for wid, meta in launchids.items()
        if isinstance(meta, dict) and normalize_jid(meta.get("launchjid"))
    }
    jobs = {}
    active_jobs_by_wid = {}
    unknown = []
    for jid_str, job in jobs_by_jid.items():
        jid = normalize_jid(jid_str)
        wid = extract_wid(job.get("COMMENT", ""))
        if wid is None and jid:
            # LEGACY - BACKFILLED - REMOVE SOON: live jobs now always have
            # xid/wid in COMMENT (set by _launch.py; gaps patched by
            # bv2/tools/backfill_slurm_comments).
            _legacy_used("jobs_by_wid.no_comment", jid=jid)
            wid = launch_wid_by_jid.get(jid)
        if wid is None:
            # LEGACY - BACKFILLED - REMOVE SOON: the no-comment "??N" bucket
            # is unreachable in steady state; kept as a safety net so a
            # corrupted live job stays visible rather than vanishing.
            _legacy_used("jobs_by_wid.unknown_bucket", jid=jid)
            unknown.append(job)
            continue
        active_jobs_by_wid.setdefault(wid, []).append(job)
        old = jobs.get(wid)
        if old is None or (jid or 0) > (normalize_jid(old.get("JOBID")) or 0):
            jobs[wid] = job
    for i, job in enumerate(unknown):
        jobs[f"??{i}"] = job
    return jobs, _active_duplicate_warnings(active_jobs_by_wid)


def _workdirs_by_suffix(wd_path):
    workdirs = {}
    warnings = {}
    for wuwd_name in dir_names(wd_path):
        wid = wid_from_workdir_name(wuwd_name)
        if wid is None:
            continue
        if wid in workdirs:
            warnings.setdefault(wid, []).append(f"Duplicate workdir detected ({workdirs[wid]}, {wuwd_name})")
            try:
                if (wd_path / wuwd_name).stat().st_mtime <= (wd_path / workdirs[wid]).stat().st_mtime:
                    continue
            except OSError:
                continue
        workdirs[wid] = wuwd_name
    return workdirs, warnings


def _launch_args(meta):
    if not isinstance(meta, dict):
        return []
    return [*meta.get("overrides", []), *meta.get("args", [])]


def _nsteps_from_args(args):
    for arg in reversed(args):
        if not re.search(r'(^|[.])nsteps:?=', arg):
            continue
        value = re.split(r':?=', arg, 1)[1].strip('"\'')
        return int(value) if value.isdigit() else None
    return None


def _get_xid_info_from_launchids(xid, wd_path, launchids, t0):
    # Fire squeue + small NFS reads up front so they overlap with the scandir/config work.
    t1 = time.time()
    squeue_fut = fs_executor.submit(_load_current_xid_jobs, xid)
    note_fut = fs_executor.submit(lambda: (wd_path / "NOTE.md").read_text())
    launchinfo_fut = fs_executor.submit(lambda: (wd_path / "launchinfo.txt").read_text())

    t2 = time.time()
    workdirs, warnings = _workdirs_by_suffix(wd_path)
    log.info("  - mapped %d workdirs by suffix in %.2fs", len(workdirs), time.time() - t2)

    jobs_by_jid = squeue_fut.result()
    log.info("  - squeue took %.2fs", time.time() - t1)
    jobs_by_wid, active_warnings = _jobs_by_wid(jobs_by_jid, launchids)
    for wid, messages in active_warnings.items():
        warnings.setdefault(wid, []).extend(messages)
    launch_wids = {normalize_wid(wid) for wid in launchids}
    all_wids = launch_wids | set(jobs_by_wid)

    t3 = time.time()
    inactive_wids = [wid for wid in all_wids - set(jobs_by_wid) if wid in workdirs]
    config_results = fs_executor.map(_load_config_only, [(wd_path, workdirs[wid]) for wid in inactive_wids])
    configs = {wid: config for wid, (_, config) in zip(inactive_wids, config_results)}
    log.info("  - loaded %d inactive configs in %.2fs", len(configs), time.time() - t3)

    # Pre-stat DONE files in parallel for inactive WUs whose exit_status is missing —
    # avoids serializing N NFS roundtrips inside the wus loop below.
    done_by_wid = dict(zip(inactive_wids, fs_executor.map(
        lambda wid: (wd_path / workdirs[wid] / "DONE").is_file(),
        inactive_wids,
    )))

    wus = []
    for wid in sorted(all_wids, key=lambda x: (0, int(x)) if str(x).isdigit() else (1, str(x))):
        meta = launchids.get(str(wid), {})
        args = _launch_args(meta)
        job = jobs_by_wid.get(wid)
        wuwd_name = workdirs.get(wid)
        config = configs.get(wid, {})

        if job:
            jid = normalize_jid(job.get("JOBID"))
            status = job.get("STATE", "UNKNOWN")
            reason = job.get("REASON", "")
            restarts = int(job.get("RESTART_COUNT", 0) or 0)
            runtime = job.get("TIME") or job.get("TIME_USED") or "n/a"
        else:
            jid = normalize_jid(config.get("jid")) or normalize_jid(meta.get("launchjid") if isinstance(meta, dict) else None)
            status = detail_status_from_exit_status(config.get("exit_status"))
            if status is None:
                if done_by_wid.get(wid, False):
                    # LEGACY - BACKFILLED - REMOVE SOON: exit_status backfilled
                    # via bv2/tools/backfill_exit_status; DONE-only path is
                    # unreachable in steady state.
                    _legacy_used("xid_info.done_no_exit_status", xid=xid, wid=wid)
                    status = "DONE"
                else:
                    status = "UNKNOWN"
            reason = ""
            restarts = 0
            runtime = "n/a"

        wus.append({
            "wid": wid,
            "jid": jid,
            "restarts": restarts,
            "exit_code": None,
            "status": status,
            "reason": reason,
            "nsteps": config.get("nsteps") or _nsteps_from_args(args),
            "config_args": args,
            "name": config.get("name", meta.get("name", "") if isinstance(meta, dict) else ""),
            "qwait": "n/a",
            "runtime": runtime,
            "workdir": f"{wd_path.name}/{wuwd_name}" if wuwd_name else "",
            "launch_script": str(wd_path / f"launch_{wid}.sh"),
            "warnings": warnings.get(wid, []),
        })

    try:
        note = note_fut.result().strip()
    except FileNotFoundError:
        note = ""
    try:
        launch_command = launchinfo_fut.result()
    except FileNotFoundError:
        launch_command = ""
    result = {
        "xid": xid,
        "note": note,
        "name": extract_common_name(workdirs.values(), xid),
        "wus": wus,
        "launch_command": launch_command,
    }
    log.info("GET /api/xid/%s - done: %d launch-indexed work units (%.2fs)", xid, len(wus), time.time() - t0)
    return result


@app.get("/api/sacct")
def get_sacct_info(jids=""):
    """Get accounting info for known job IDs."""
    saccts = load_sacct_many(jid.strip() for jid in jids.split(","))
    return {str(jid): format_sacct(sacct) for jid, sacct in saccts.items()}


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

    launchids = load_launchids(wd_path)
    if launchids:
        return _get_xid_info_from_launchids(xid, wd_path, launchids, t0)

    # LEGACY - BACKFILLED - REMOVE SOON: full reconstruction for experiments
    # without launchids.json. Backfilled by bv2/tools/backfill_launchids_json;
    # this whole branch (and the comment-recovery fallback below) is dead in
    # steady state.
    _legacy_used("xid_info.no_launchids", xid=xid)

    # Fire squeue + small NFS reads up front so they overlap with the scandir/config work.
    t1 = time.time()
    squeue_fut = fs_executor.submit(_load_current_xid_jobs, xid)
    note_fut = fs_executor.submit(lambda: (wd_path / "NOTE.md").read_text())
    launchinfo_fut = fs_executor.submit(lambda: (wd_path / "launchinfo.txt").read_text())

    # Get workdirs
    t2 = time.time()
    workdirs = dir_names(wd_path)
    log.info("  - found %d workdirs (scandir took %.2fs)", len(workdirs), time.time() - t2)

    # Load configs in parallel
    t_cfg = time.time()
    config_results = list(fs_executor.map(_load_config_only, [(wd_path, wd) for wd in workdirs]))
    log.info("  - loaded configs in %.2fs", time.time() - t_cfg)

    jobs_by_jid = squeue_fut.result()
    log.info("  - squeue took %.2fs", time.time() - t1)

    with os.scandir(wd_path) as it:
        launch_names = sorted(e.name for e in it
                              if e.name.startswith("launch_") and e.name.endswith(".sh")
                              and e.is_file(follow_symlinks=False))
    launch_wids_names = [(int(m.group(1)), name) for name in launch_names
                         if (m := re.match(r'launch_(\d+)\.sh', name))]
    launch_futs = [(wid, fs_executor.submit(extract_launch_info, wd_path / name))
                   for wid, name in launch_wids_names]
    launches = {wid: f.result() for wid, f in launch_futs}

    configs = {}
    warnings = {}  # wid -> list of warning strings
    active_jobs_by_wid = {}
    for job in jobs_by_jid.values():
        if (wid := extract_wid(job.get("COMMENT", ""))) is not None:
            active_jobs_by_wid.setdefault(wid, []).append(job)
    for wid, messages in _active_duplicate_warnings(active_jobs_by_wid).items():
        warnings.setdefault(wid, []).extend(messages)

    for wuwd_name, config in config_results:
        wid = normalize_wid(config.get("wid", wuwd_name))
        new_jid = config.get("jid")
        config["_wuwd_name"] = wuwd_name  # Store actual workdir name for DONE check

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

        if wid in launches:
            config["_launch_line"] = launches[wid]["launch_line"]
            if not config.get("name"):
                config["name"] = launches[wid]["name"]
        configs[wid] = config

    # Get sacct info for all jids in one batched call.
    # (Launch-only WUs can't contribute jids here: this is the legacy path, which
    # only runs when launchids.json is absent, so launchids is always {}.)
    jids = set()
    for wid, config in configs.items():
        if "jid" in config:
            jids.add(config["jid"])
    jids.update(int(jid) for jid in jobs_by_jid.keys() if jid.isdigit())

    t2 = time.time()
    saccts = load_sacct_many(list(jids))
    log.info("  - loaded sacct in %.2fs", time.time() - t2)

    # Parallelize DONE stats — one NFS roundtrip per WU would otherwise serialize.
    t_done = time.time()
    done_wids = [wid for wid, config in configs.items() if config.get("_wuwd_name")]
    done_results = fs_executor.map(
        lambda wid: (wd_path / configs[wid]["_wuwd_name"] / "DONE").is_file(),
        done_wids,
    )
    done_by_wid = dict(zip(done_wids, done_results))
    log.info("  - DONE stats in %.2fs (%d WUs)", time.time() - t_done, len(done_wids))

    # Determine status for each wid
    status = {}
    for wid, config in configs.items():
        has_done = done_by_wid.get(wid, False)
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
        # LEGACY - BACKFILLED - REMOVE SOON: inside the launchids-missing
        # branch. Comments + launchids are both backfilled, so this
        # sacct.submit_line recovery is unreachable.
        wid = extract_wid(job_info.get("COMMENT", ""))
        if wid is None and jid and jid in saccts:
            _legacy_used("xid_info.sacct_submit_line", xid=xid, jid=jid)
            wid = extract_wid(saccts[jid].get("submit_line", ""))
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

    # Add WUs from launch files that aren't represented yet. Launchids is empty
    # in the legacy path, so we can't recover a jid here — UNKNOWN status.
    for wid, launch in sorted(launches.items()):
        if wid in configs:
            continue
        configs[wid] = {"jid": None, "name": launch["name"], "_launch_line": launch["launch_line"]}
        status[wid] = "UNKNOWN"

    # Format for response
    wus = []
    for wid in sorted(configs.keys(), key=lambda x: (0, int(x)) if str(x).isdigit() else (1, str(x))):
        config = configs[wid]
        jid = config.get("jid") or None
        sacct = saccts.get(jid, {})

        # Extract submit line args
        submit_line = sacct.get("submit_line", "") or config.get("_launch_line", "")
        sws_args = extract_sws_args(submit_line)

        time_info = sacct.get("time", {})
        elapsed = time_info.get("elapsed", 0)
        eligible = time_info.get("eligible", 0)
        start = time_info.get("start", 0)

        if start in [0, SLURM_NO_START_TIME]:
            qwait = "never"
        else:
            qwait = hms(start - eligible)

        wus.append({
            "wid": wid,
            "jid": jid,
            "restarts": sacct.get("restart_cnt", 0),
            "exit_code": extract_exit_code(sacct),
            "status": status.get(wid, "UNKNOWN"),
            "reason": jobs_by_jid.get(str(jid), {}).get("REASON", ""),
            "nsteps": config.get("nsteps"),
            "config_args": sws_args,
            "name": config.get("name", ""),
            "qwait": qwait,
            "runtime": hms(elapsed),
            "workdir": f"{wd_path.name}/{config['_wuwd_name']}" if "_wuwd_name" in config else "",
            "launch_script": str(wd_path / f"launch_{wid}.sh"),
            "warnings": warnings.get(wid, []),
        })

    try:
        note = note_fut.result().strip()
    except FileNotFoundError:
        note = ""
    try:
        launch_command = launchinfo_fut.result()
    except FileNotFoundError:
        launch_command = ""
    result = {
        "xid": xid,
        "note": note,
        "name": extract_common_name(workdirs, xid),
        "wus": wus,
        "launch_command": launch_command,
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

    launchids = load_launchids(wd_path)
    if launchids:
        workdirs, _ = _workdirs_by_suffix(wd_path)
        wd_names = list(workdirs.values())
        metric_futs = [fs_executor.submit(_load_metric_only, (wd_path, wd, metric)) for wd in wd_names]
        config_futs = [fs_executor.submit(_load_config_only, (wd_path, wd)) for wd in wd_names]
        metric_by_wuwd = dict(f.result() for f in metric_futs)
        config_by_wuwd = dict(f.result() for f in config_futs)
        result = {}
        for wid, wuwd_name in workdirs.items():
            result[wid] = metric_by_wuwd.get(wuwd_name, {})
            nsteps = config_by_wuwd.get(wuwd_name, {}).get("nsteps") or _nsteps_from_args(_launch_args(launchids.get(str(wid), {})))
            if nsteps:
                result[wid]["_nsteps"] = nsteps
        log.info("GET /api/xid/%s/metrics - done: %d launch-indexed WUs (%.2fs)", xid, len(result), time.time() - t0)
        return result

    # LEGACY - BACKFILLED - REMOVE SOON: config scan for experiments without
    # launchids.json. Backfilled by bv2/tools/backfill_launchids_json.
    _legacy_used("xid_metrics.no_launchids", xid=xid)

    # Get workdirs
    workdirs = dir_names(wd_path)

    # Load configs and metrics in parallel — submit both up front so they interleave I/O.
    config_futs = [fs_executor.submit(_load_config_only, (wd_path, wd)) for wd in workdirs]
    metric_futs = [fs_executor.submit(_load_metric_only, (wd_path, wd, metric)) for wd in workdirs]
    config_results = [f.result() for f in config_futs]
    metric_results = [f.result() for f in metric_futs]

    # Build wid -> metrics mapping (same duplicate resolution as main endpoint)
    metric_by_wuwd = {wuwd_name: metrics for wuwd_name, metrics in metric_results}
    config_by_wuwd = {wuwd_name: config for wuwd_name, config in config_results}
    wid_to_wuwd = {}
    for wuwd_name, config in config_results:
        wid = normalize_wid(config.get("wid", wuwd_name))
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
        if nsteps := config_by_wuwd.get(wuwd_name, {}).get("nsteps"):
            result[wid]["_nsteps"] = nsteps

    log.info("GET /api/xid/%s/metrics - done: %d WUs (%.2fs)", xid, len(result), time.time() - t0)
    return result


@app.get("/api/xid/{xid}/{wid}/config")
def get_wu_config(xid: str, wid: int):
    """Get full config.json for a specific work unit."""
    wd_path = BASEDIR / xid
    if not wd_path.exists():
        # LEGACY - BACKFILLED - REMOVE SOON: all workdirs now live exactly at
        # /workdirs/{xid} (bv2/tools/backfill_xid_dirname). The scan below is
        # unreachable in steady state; the bare 404 is the normal
        # "no such XID on this server" path.
        with os.scandir(BASEDIR) as it:
            for e in it:
                if xid in e.name and e.is_dir(follow_symlinks=False):
                    _legacy_used("wu_config.containment_scan", xid=xid, wid=wid, found=e.name)
                    wd_path = Path(e.path)
                    break
            else:
                raise HTTPException(status_code=404, detail=f"XID {xid} not found")

    wid_str = str(wid)
    suffix = f"-{wid_str}"
    with os.scandir(wd_path) as it:
        sub_dirs = [e.name for e in it if e.is_dir(follow_symlinks=False)]
    for name in sub_dirs:
        if not name.endswith(suffix):
            continue
        config_path = wd_path / name / "config.json"
        if config_path.exists():
            return Response(
                content=json.dumps(json.loads(config_path.read_text()), indent=2),
                media_type="application/json",
            )

    # LEGACY - BACKFILLED - REMOVE SOON: workdir names now always end in
    # "-{wid}" (bv2/tools/backfill_wid_suffix). The scan below is
    # unreachable in steady state; falling through to the 404 below is
    # the normal "no such WID on this server" path.
    for name in sub_dirs:
        config_path = wd_path / name / "config.json"
        if not config_path.exists():
            continue
        try:
            config = json.loads(config_path.read_text())
            if config.get("wid") == wid:
                _legacy_used("wu_config.config_scan_suffix", xid=xid, wid=wid, found=name)
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


def _build_tree(path, base):
    """Recursively build a file tree structure."""
    items = []
    try:
        with os.scandir(path) as it:
            entries = [(e, e.is_dir(follow_symlinks=False)) for e in it]
    except (PermissionError, OSError):
        return items
    entries.sort(key=lambda x: (not x[1], x[0].name.lower()))
    for entry, is_dir in entries:
        rel_path = os.path.relpath(entry.path, base)
        if is_dir:
            items.append({
                "name": entry.name,
                "path": rel_path,
                "type": "dir",
                "children": _build_tree(entry.path, base),
            })
        else:
            try:
                size = entry.stat(follow_symlinks=False).st_size
            except OSError:
                size = 0
            items.append({
                "name": entry.name,
                "path": rel_path,
                "type": "file",
                "size": size,
            })
    return items


@app.get("/api/xid/{xid}/code/tree")
def get_code_tree(xid: str):
    """Get the file tree for an XID's source directory."""
    src_path = _get_srcdir(xid)
    return {"xid": xid, "root_path": str(src_path), "tree": _build_tree(src_path, src_path)}


@app.get("/api/xid/{xid}/code/file/{file_path:path}")
def get_code_file(xid: str, file_path: str):
    """Get the content of a file from an XID's source directory."""
    src_path = _get_srcdir(xid)
    full_path = _safe_path(src_path, file_path)

    try:
        st = full_path.stat()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    if not stat.S_ISREG(st.st_mode):
        raise HTTPException(status_code=400, detail=f"Not a file: {file_path}")

    # Read file content (with size limit for safety)
    max_size = 10 * 1024 * 1024  # 10MB
    if st.st_size > max_size:
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


def _get_workdir(xid: str) -> Path:
    """Get and validate the workdir for an XID."""
    wd_path = _find_xid_path(xid)
    if not wd_path or not wd_path.exists():
        raise HTTPException(status_code=404, detail=f"Workdir not found for XID {xid}")
    return wd_path


def _is_text_file(path: Path) -> bool:
    """Check if a file is text (viewable) or binary (download-only)."""
    import mimetypes
    mime, _ = mimetypes.guess_type(str(path))
    if mime and mime.startswith('text/'):
        return True
    # Common text file extensions not always detected
    text_exts = {'.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf',
                 '.sh', '.bash', '.py', '.js', '.ts', '.md', '.rst', '.txt',
                 '.log', '.csv', '.xml', '.html', '.css', '.sql', '.env'}
    if path.suffix.lower() in text_exts:
        return True
    # Try reading first few bytes to check for binary content
    try:
        with open(path, 'rb') as f:
            chunk = f.read(8192)
            # Binary if contains null bytes
            if b'\x00' in chunk:
                return False
            # Try decoding as utf-8
            try:
                chunk.decode('utf-8')
                return True
            except UnicodeDecodeError:
                return False
    except:
        return False


def _build_files_tree(path, base, depth=-1, dir_sizes=False):
    """Build file tree with size info for workdir browsing.

    Args:
        path: Current directory to list
        base: Base path for computing relative paths
        depth: Max recursion depth. -1 for unlimited, 0 for no children, 1 for one level, etc.
        dir_sizes: Include recursive sizes for directories.
    """
    items = []
    try:
        with os.scandir(path) as it:
            entries = [(e, e.is_dir(follow_symlinks=False)) for e in it]
    except (PermissionError, OSError):
        return items
    entries.sort(key=lambda x: (not x[1], x[0].name.lower()))
    for entry, is_dir in entries:
        rel_path = os.path.relpath(entry.path, base)
        if is_dir:
            children = [] if depth == 0 else _build_files_tree(entry.path, base, depth - 1 if depth > 0 else -1, dir_sizes)
            item = {
                "name": entry.name,
                "path": rel_path,
                "type": "dir",
                "children": children
            }
            if dir_sizes:
                item["size"] = sum(child.get("size", 0) for child in children) if depth != 0 else _path_size(entry.path)
            items.append(item)
        else:
            try:
                size = entry.stat(follow_symlinks=False).st_size
            except OSError:
                size = 0
            items.append({
                "name": entry.name,
                "path": rel_path,
                "type": "file",
                "size": size
            })
    return items


def _path_size(path):
    try:
        st = os.lstat(path)
    except OSError:
        return 0
    if not stat.S_ISDIR(st.st_mode):
        return st.st_size
    total = 0
    try:
        with os.scandir(path) as it:
            for entry in it:
                try:
                    est = entry.stat(follow_symlinks=False)
                except OSError:
                    continue
                if stat.S_ISDIR(est.st_mode):
                    total += _path_size(entry.path)
                else:
                    total += est.st_size
    except OSError:
        pass
    return total


@app.get("/api/xid/{xid}/files/tree")
def get_files_tree(xid: str, depth: int = 1, dir_sizes=False):
    """Get file tree for XID's workdir. Use depth=1 for lazy loading."""
    wd_path = _get_workdir(xid)
    dir_sizes = str(dir_sizes).lower() in ("1", "true", "yes", "on")
    return {"xid": xid, "root_path": str(wd_path), "tree": _build_files_tree(wd_path, wd_path, depth, dir_sizes)}


@app.get("/api/xid/{xid}/files/tree/{wuname:path}")
def get_wu_files_tree(xid: str, wuname: str, depth: int = 1, dir_sizes=False):
    """Get file tree for a specific work-unit's directory. Use depth=1 for lazy loading."""
    wd_path = _get_workdir(xid)
    wu_path = _safe_path(wd_path, wuname)
    if not wu_path.exists() or not wu_path.is_dir():
        raise HTTPException(status_code=404, detail=f"Work-unit directory not found: {wuname}")
    dir_sizes = str(dir_sizes).lower() in ("1", "true", "yes", "on")
    return {"xid": xid, "wuname": wuname, "root_path": str(wu_path), "tree": _build_files_tree(wu_path, wu_path, depth, dir_sizes)}


@app.get("/api/xid/{xid}/files/size/{file_path:path}")
def get_files_size(xid, file_path):
    """Get recursive size for a file or directory in an XID's workdir."""
    wd_path = _get_workdir(xid)
    full_path = _safe_path(wd_path, file_path)
    if not full_path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {file_path}")
    return {"xid": xid, "path": file_path, "size": _path_size(full_path)}


@app.get("/api/xid/{xid}/files/content/{file_path:path}")
def get_files_content(xid: str, file_path: str):
    """Get file content - text for viewable files, error for binary."""
    wd_path = _get_workdir(xid)
    full_path = _safe_path(wd_path, file_path)
    try:
        st = full_path.stat()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    if not stat.S_ISREG(st.st_mode):
        raise HTTPException(status_code=400, detail=f"Not a file: {file_path}")
    # Size limit: 10MB for text display
    max_size = 10 * 1024 * 1024
    file_size = st.st_size
    if file_size > max_size:
        raise HTTPException(status_code=400, detail=f"File too large for inline display ({file_size // 1024 // 1024}MB). Use download instead.")
    if not _is_text_file(full_path):
        raise HTTPException(status_code=400, detail="Binary file. Use download instead.")
    try:
        content = full_path.read_text(errors="replace")
        return Response(content=content, media_type="text/plain; charset=utf-8")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read file: {e}")


@app.get("/api/xid/{xid}/files/download/{file_path:path}")
def download_file(xid: str, file_path: str):
    """Force download a file regardless of type."""
    from starlette.responses import StreamingResponse
    wd_path = _get_workdir(xid)
    full_path = _safe_path(wd_path, file_path)
    try:
        st = full_path.stat()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    if not stat.S_ISREG(st.st_mode):
        raise HTTPException(status_code=400, detail=f"Not a file: {file_path}")

    file_size = st.st_size

    def iter_file():
        chunk_size = 1024 * 1024  # 1MB chunks for better throughput
        with open(full_path, 'rb') as f:
            while chunk := f.read(chunk_size):
                yield chunk

    return StreamingResponse(
        iter_file(),
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{full_path.name}"',
            "Content-Length": str(file_size),
        }
    )


def _resolve_wu_dir(xid: str, wid) -> str:
    """Resolve WID to actual directory name within the XID workdir.

    Returns the directory name (not full path) or empty string for XID root.
    Priority:
    1. Folder ending in '-{wid}'
    2. Folder with exact work-unit name (from config.json wid match)
    3. Fallback to XID root
    """
    wd_path = _get_workdir(xid)
    wid_str = str(wid)
    suffix = f"-{wid_str}"

    with os.scandir(wd_path) as it:
        sub_dirs = [e.name for e in it if e.is_dir(follow_symlinks=False)]

    # First, use the normal launcher/train naming convention.
    for name in sub_dirs:
        if name.endswith(suffix):
            return name

    # LEGACY - BACKFILLED - REMOVE SOON: workdir names now always end in
    # "-{wid}" (bv2/tools/backfill_wid_suffix). The scan below is
    # unreachable in steady state; falling through to the XID-root return
    # is the normal "no such WID on this server" path.
    for name in sub_dirs:
        config_file = wd_path / name / "config.json"
        if config_file.exists():
            try:
                config = json.loads(config_file.read_text())
                if str(config.get("wid")) == wid_str:
                    _legacy_used("resolve_wu_dir.config_scan_suffix", xid=xid, wid=wid_str, found=name)
                    return name
            except:
                pass

    # Fallback to XID root
    return ""


@app.get("/files/{xid}/wu/{wid}")
def files_browser_wu_redirect(xid: str, wid: str):
    """Redirect to the correct workdir path for a work-unit."""
    from fastapi.responses import RedirectResponse
    dir_name = _resolve_wu_dir(xid, wid)
    if dir_name:
        return RedirectResponse(url=f"/files/{xid}/{dir_name}", status_code=302)
    return RedirectResponse(url=f"/files/{xid}", status_code=302)


@app.get("/files/{xid}")
@app.get("/files/{xid}/{file_path:path}")
def files_browser_page(xid: str, file_path: str = ""):
    """Serve the files browser page for an XID's workdir."""
    _get_workdir(xid)
    html = (SCRIPT_DIR / "files.html").read_text()
    html = html.replace("{{VERSION}}", __version__)
    return Response(content=html, media_type="text/html")


@app.get("/config/{xid}/{wid}")
def config_viewer_page(xid: str, wid: int):
    """Serve the config viewer page for a work unit."""
    html = (SCRIPT_DIR / "config.html").read_text()
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
    with os.scandir(SLURM_OUT_DIR) as it:
        user_dirs = [e.path for e in it if e.is_dir(follow_symlinks=False)]
    for user_dir in user_dirs:
        log_path = Path(user_dir) / log_filename
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
def archive_xid(xid: str, move: bool = True):
    """Archive an XID by removing checkpoints and optionally moving workdir to archive directory."""
    if not ACTIONS_ENABLED:
        raise HTTPException(status_code=403, detail="Actions are disabled (--no-actions)")
    if move and ARCHIVE_DIR is None:
        raise HTTPException(status_code=400, detail="Archive directory not configured (use --archive-dir)")
    if not xid or not _xid_re.fullmatch(xid):
        raise HTTPException(status_code=400, detail="Invalid XID format")
    log.info("POST /api/action/archive/%s (move=%s)", xid, move)
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
    if not move:
        return {"status": "ok", "xid": xid, "checkpoints_removed": True}
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
