#!/usr/bin/env python3
"""Slurm Manager Server - serves job info via REST API."""

import re
import json
import base64
import subprocess
import shlex
import logging
import time
import signal
import threading
import sys
import os
import stat
import shutil
import zipfile
import math
from email.parser import BytesParser
from email.policy import default as email_policy
from datetime import datetime
from getpass import getuser
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from collections import Counter
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import plattli
import zstandard

from fastapi import FastAPI, HTTPException, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
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
REPORT_ROOT_DIR = Path("/checkpoint/rigi/bv2/reports")  # Set via --report-root flag
NUM_RECENT = 50
ACTIONS_ENABLED = True  # Set via --no-actions flag
SLURM_NO_START_TIME = 0xFFFFFFFE

_xid_re = re.compile(r'\d{4,6}_\d{6}')
_report_id_re = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{0,127}')
_report_xid_re = re.compile(r'\b\d{6}_\d{6}\b')
_report_run_re = re.compile(r'\b(\d{6}_\d{6})/([^\s<>"\']+)')
_report_version_re = re.compile(r'v(\d+)\.json$')
_report_lock = threading.Lock()
REPORT_HTML_MAX_BYTES = 100 * 1024 * 1024
REPORT_SRC_MAX_BYTES = 32 * 1024 * 1024
REPORT_UPLOAD_MAX_BYTES = REPORT_HTML_MAX_BYTES + REPORT_SRC_MAX_BYTES + 1024 * 1024
REPORT_LOCK_WAIT_SECONDS = 30
REPORT_LOCK_STALE_SECONDS = 10 * 60


def run_cmd(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout.strip().split('\n')


SLURM_QUEUE_FIELDS = [
    ("job_id", "JobId", 24),
    ("name", "Name", 80),
    ("user", "UserName", 32),
    ("account", "Account", 32),
    ("partition", "Partition", 40),
    ("state", "State", 24),
    ("time_used", "TimeUsed", 20),
    ("time_limit", "TimeLimit", 20),
    ("num_cpus", "NumCPUs", 12),
    ("num_nodes", "NumNodes", 12),
    ("qos", "QOS", 32),
    ("tres_per_job", "tres-per-job", 48),
    ("tres_per_node", "tres-per-node", 48),
    ("gres", "GRES", 48),
    ("submit_time", "SubmitTime", 25),
    ("eligible_time", "EligibleTime", 25),
    ("start_time", "StartTime", 25),
    ("end_time", "EndTime", 25),
    ("reason", "Reason", 80),
    ("priority", "Priority", 20),
    ("priority_long", "PriorityLong", 24),
    ("restart_count", "RestartCnt", 12),
    ("comment", "Comment", 160),
]

SLURM_QUEUE_FALLBACK_FIELDS = [
    field for field in SLURM_QUEUE_FIELDS
    if field[0] not in {"tres_per_job", "submit_time", "eligible_time", "end_time", "time_limit"}
]


def _slurm_run(cmd, timeout=30):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout), None
    except FileNotFoundError:
        return None, f"{cmd[0]} not found"
    except subprocess.TimeoutExpired:
        return None, f"{cmd[0]} timed out after {timeout}s"


def _slurm_failure(result, fallback):
    if result is None:
        return fallback
    return result.stderr.strip() or result.stdout.strip() or fallback


def _slurm_run_or_raise(cmd, label, timeout=30):
    result, error = _slurm_run(cmd, timeout=timeout)
    if error:
        raise HTTPException(status_code=500, detail=f"{label} failed: {error}")
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=f"{label} failed: {_slurm_failure(result, 'unknown error')}")
    return result


def _query_value(value, default=""):
    value = default if value is None else value
    value = str(value).strip()
    return "" if value.lower() in {"", "*", "all", "none", "null"} else value


def _query_bool(value):
    return str(value).strip().lower() not in {"", "0", "false", "no", "off"}


def _slurm_int(value):
    value = str(value or "").strip()
    return int(value) if re.fullmatch(r"-?\d+", value) else None


def _fixed_width_dicts(text, fields):
    offsets = [sum(field[2] for field in fields[:i]) for i in range(len(fields))]
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        rows.append({
            key: line[offsets[i]:offsets[i]+width].strip()
            for i, (key, _field, width) in enumerate(fields)
        })
    return rows


def _squeue_format(fields):
    return ",".join(f"{field}:{width}" for _key, field, width in fields)


def _load_slurm_queue_rows(account=None, users=None, states=None, partition=None):
    errors = []
    for fields in (SLURM_QUEUE_FIELDS, SLURM_QUEUE_FALLBACK_FIELDS):
        cmd = ["squeue", "-h", "-O", _squeue_format(fields)]
        if account:
            cmd += ["-A", account]
        if users:
            cmd += ["-u", users]
        if states:
            cmd += ["-t", states]
        if partition:
            cmd += ["-p", partition]
        result, error = _slurm_run(cmd)
        if error:
            errors.append(error)
            continue
        if result.returncode == 0:
            return _fixed_width_dicts(result.stdout, fields)
        errors.append(_slurm_failure(result, "unknown error"))
    raise HTTPException(status_code=500, detail=f"squeue failed: {errors[-1] if errors else 'unknown error'}")


def _parse_gpu_count(value):
    if value is None:
        return 0
    if isinstance(value, dict):
        total = 0
        for key, item in value.items():
            if "gpu" in str(key).lower() and (n := _slurm_int(item)) is not None:
                total += n
            else:
                total += _parse_gpu_count(item)
        return total
    if isinstance(value, list):
        return sum(_parse_gpu_count(item) for item in value)

    text = str(value)
    total = 0
    matched = False
    for match in re.finditer(r"(?:^|[/,])gpu(?::[^:=,()]+)?[:=](\d+)", text, re.IGNORECASE):
        matched = True
        total += int(match.group(1))
    if matched:
        return total
    if "gpu" not in text.lower():
        return 0
    return 1


def _parse_slurm_api_time(value):
    return _parse_slurm_time(value, {"", "N/A", "Unknown", "None"})


def _format_slurm_queue_job(row, now):
    num_nodes = _slurm_int(row.get("num_nodes"))
    num_cpus = _slurm_int(row.get("num_cpus"))
    per_job_gpus = _parse_gpu_count(row.get("tres_per_job"))
    per_node_gpus = _parse_gpu_count(row.get("tres_per_node") or row.get("gres"))
    gpus = per_job_gpus or per_node_gpus * (num_nodes or 1)
    submit_ts = _parse_slurm_api_time(row.get("submit_time"))
    eligible_ts = _parse_slurm_api_time(row.get("eligible_time"))
    start_ts = _parse_slurm_api_time(row.get("start_time"))
    end_ts = _parse_slurm_api_time(row.get("end_time"))
    state = row.get("state", "").upper()
    job = {
        "job_id": normalize_jid(row.get("job_id")),
        "job_id_raw": row.get("job_id", ""),
        "name": row.get("name", ""),
        "xid": extract_xid(row.get("name", "")),
        "user": row.get("user", ""),
        "account": row.get("account", ""),
        "partition": row.get("partition", ""),
        "state": state,
        "qos": row.get("qos", ""),
        "reason": row.get("reason", ""),
        "priority": _slurm_int(row.get("priority")),
        "priority_long": _slurm_int(row.get("priority_long")) or _slurm_int(row.get("priority")),
        "restart_count": _slurm_int(row.get("restart_count")) or 0,
        "num_cpus": num_cpus,
        "num_nodes": num_nodes,
        "gpus": gpus,
        "gpus_per_node": per_node_gpus,
        "time_used": row.get("time_used", ""),
        "time_limit": row.get("time_limit", ""),
        "submit_time": row.get("submit_time", ""),
        "submit_time_ts": submit_ts,
        "eligible_time": row.get("eligible_time", ""),
        "eligible_time_ts": eligible_ts,
        "start_time": row.get("start_time", ""),
        "start_time_ts": start_ts,
        "end_time": row.get("end_time", ""),
        "end_time_ts": end_ts,
        "comment": row.get("comment", ""),
    }
    job["age_seconds"] = max(0, now - submit_ts) if submit_ts else None
    if state == "PENDING":
        job["eligible_age_seconds"] = max(0, now - eligible_ts) if eligible_ts else None
        job["wait_estimate_seconds"] = max(0, start_ts - now) if start_ts else None
    else:
        job["eligible_age_seconds"] = None
        job["wait_estimate_seconds"] = None
    return job


def _pending_sort_key(job):
    priority = job.get("priority_long")
    priority = priority if priority is not None else -1
    return (-priority, job.get("eligible_time_ts") or 10**18, job.get("submit_time_ts") or 10**18, job.get("job_id") or 10**18)


def _rank_pending_jobs(jobs):
    pending = sorted([job for job in jobs if job.get("state") == "PENDING"], key=_pending_sort_key)
    for i, job in enumerate(pending, 1):
        job["pending_rank"] = i

    by_partition = {}
    for job in pending:
        by_partition.setdefault(job.get("partition") or "(none)", []).append(job)
    for part_jobs in by_partition.values():
        for i, job in enumerate(sorted(part_jobs, key=_pending_sort_key), 1):
            job["partition_pending_rank"] = i
    return pending


def _empty_job_bucket():
    return {
        "jobs": 0,
        "running": 0,
        "pending": 0,
        "other": 0,
        "gpus_running": 0,
        "gpus_pending": 0,
        "gpus_total": 0,
        "states": {},
    }


def _add_job_to_bucket(bucket, job):
    state = job.get("state") or "UNKNOWN"
    gpus = job.get("gpus") or 0
    bucket["jobs"] += 1
    bucket["states"][state] = bucket["states"].get(state, 0) + 1
    bucket["gpus_total"] += gpus
    if state == "RUNNING":
        bucket["running"] += 1
        bucket["gpus_running"] += gpus
    elif state == "PENDING":
        bucket["pending"] += 1
        bucket["gpus_pending"] += gpus
    else:
        bucket["other"] += 1


def _group_job_summary(jobs, key):
    grouped = {}
    for job in jobs:
        value = job.get(key) or "(none)"
        if value not in grouped:
            grouped[value] = _empty_job_bucket()
        _add_job_to_bucket(grouped[value], job)
    return grouped


def _pending_estimate_summary(pending, now):
    known = [job for job in pending if job.get("start_time_ts")]
    if not known:
        return {"known": 0, "unknown": len(pending), "earliest_start_ts": None, "latest_start_ts": None}
    earliest = min(known, key=lambda job: job["start_time_ts"])
    latest = max(known, key=lambda job: job["start_time_ts"])
    return {
        "known": len(known),
        "unknown": len(pending) - len(known),
        "earliest_start_ts": earliest["start_time_ts"],
        "earliest_start_raw": earliest["start_time"],
        "earliest_job_id": earliest["job_id"],
        "earliest_reason": earliest["reason"],
        "latest_start_ts": latest["start_time_ts"],
        "latest_start_raw": latest["start_time"],
        "latest_job_id": latest["job_id"],
        "max_wait_estimate_seconds": max(0, latest["start_time_ts"] - now),
    }


def _split_slurm_labels(value):
    labels = []
    if isinstance(value, dict):
        for item in value.values():
            labels.extend(_split_slurm_labels(item))
        return labels
    if isinstance(value, list):
        for item in value:
            labels.extend(_split_slurm_labels(item))
        return labels
    for part in re.split(r"[,+\s]+", str(value or "")):
        part = part.strip("*~#").upper()
        if part:
            labels.append(part)
    return labels


def _node_state_bucket(labels):
    labels = set(labels)
    if labels & {"DOWN", "DRAIN", "DRAINING", "DRAINED", "FAIL", "FAILING", "MAINT", "POWER_DOWN", "POWERED_DOWN"}:
        return "unavailable"
    if labels & {"IDLE"}:
        return "idle"
    if labels & {"MIXED", "MIX"}:
        return "mixed"
    if labels & {"ALLOCATED", "ALLOC"}:
        return "allocated"
    if labels & {"RESERVED", "RESV"}:
        return "reserved"
    return "other"


def _node_can_accept_jobs(labels):
    labels = set(labels)
    return not labels & {"DOWN", "DRAIN", "DRAINING", "DRAINED", "FAIL", "FAILING", "MAINT", "POWER_DOWN", "POWERED_DOWN"}


def _as_slurm_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip("*") for item in value if str(item).strip("*")]
    return [item.strip("*") for item in str(value).split(",") if item.strip("*")]


def _dict_get_any(obj, keys, default=None):
    for key in keys:
        if key in obj:
            return obj[key]
    return default


def _format_slurm_node(node):
    name = _dict_get_any(node, ["name", "node", "hostname"], "")
    partitions = _as_slurm_list(_dict_get_any(node, ["partitions", "partition"], []))
    labels = _split_slurm_labels(_dict_get_any(node, ["state", "states"], ""))
    bucket = _node_state_bucket(labels)
    gres = _dict_get_any(node, ["gres", "gres_total"], "")
    gres_used = _dict_get_any(node, ["gres_used", "gres_used_total"], None)
    gpus = _parse_gpu_count(gres) or _parse_gpu_count(_dict_get_any(node, ["tres", "tres_fmt_str"], ""))
    gpus_used = _parse_gpu_count(gres_used) if gres_used is not None else None
    if gpus_used is None and bucket == "idle":
        gpus_used = 0
    if gpus_used is None:
        gpus_free = 0 if bucket == "allocated" else None
    else:
        gpus_free = max(0, gpus - gpus_used)
    available = _node_can_accept_jobs(labels) and gpus_free is not None and gpus_free > 0
    return {
        "name": str(name),
        "partitions": partitions,
        "state": "+".join(labels) if labels else "",
        "state_bucket": bucket,
        "cpus": _slurm_int(_dict_get_any(node, ["cpus", "cpus_total"])),
        "memory_mb": _slurm_int(_dict_get_any(node, ["real_memory", "memory"])),
        "gres": gres,
        "gres_used": gres_used,
        "gpus": gpus,
        "gpus_used": gpus_used,
        "gpus_free": gpus_free,
        "available": available,
    }


def _load_slurm_nodes_json():
    result, error = _slurm_run(["scontrol", "show", "nodes", "--json"], timeout=45)
    if error or result.returncode != 0 or not result.stdout.strip():
        return None
    data = json.loads(result.stdout)
    return [_format_slurm_node(node) for node in data.get("nodes", [])]


def _load_slurm_nodes_sinfo():
    result = _slurm_run_or_raise(["sinfo", "-h", "-N", "-o", "%N|%P|%T|%G|%c|%m"], "sinfo", timeout=30)
    nodes = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        name, partition, state, gres, cpus, memory = (line.split("|", 5) + [""] * 5)[:6]
        labels = _split_slurm_labels(state)
        bucket = _node_state_bucket(labels)
        gpus = _parse_gpu_count(gres)
        if bucket == "idle":
            gpus_used, gpus_free = 0, gpus
        elif bucket == "allocated":
            gpus_used, gpus_free = gpus, 0
        else:
            gpus_used, gpus_free = None, None
        node = nodes.setdefault(name, {
            "name": name,
            "partitions": [],
            "state": "+".join(labels) if labels else state,
            "state_bucket": bucket,
            "cpus": _slurm_int(cpus),
            "memory_mb": _slurm_int(memory),
            "gres": gres,
            "gres_used": None,
            "gpus": gpus,
            "gpus_used": gpus_used,
            "gpus_free": gpus_free,
            "available": _node_can_accept_jobs(labels) and gpus_free is not None and gpus_free > 0,
        })
        if partition.strip("*") and partition.strip("*") not in node["partitions"]:
            node["partitions"].append(partition.strip("*"))
    return list(nodes.values())


def _load_slurm_nodes():
    try:
        nodes = _load_slurm_nodes_json()
    except Exception as e:
        log.debug("scontrol node json failed, falling back to sinfo: %s", e)
        nodes = None
    return nodes if nodes is not None else _load_slurm_nodes_sinfo()


def _empty_node_bucket():
    return {
        "nodes": 0,
        "states": {},
        "gpus_total": 0,
        "gpus_used": 0,
        "gpus_free": 0,
        "gpus_available": 0,
        "gpu_used_known_nodes": 0,
        "gpu_free_known_nodes": 0,
    }


def _add_node_to_bucket(bucket, node):
    state = node.get("state_bucket") or "other"
    bucket["nodes"] += 1
    bucket["states"][state] = bucket["states"].get(state, 0) + 1
    bucket["gpus_total"] += node.get("gpus") or 0
    if node.get("gpus_used") is not None:
        bucket["gpus_used"] += node.get("gpus_used") or 0
        bucket["gpu_used_known_nodes"] += 1
    if node.get("gpus_free") is not None:
        bucket["gpus_free"] += node.get("gpus_free") or 0
        bucket["gpu_free_known_nodes"] += 1
    if node.get("available"):
        bucket["gpus_available"] += node.get("gpus_free") or 0


def _summarize_slurm_nodes(nodes):
    summary = _empty_node_bucket()
    summary["by_partition"] = {}
    for node in nodes:
        _add_node_to_bucket(summary, node)
        for partition in node.get("partitions") or ["(none)"]:
            if partition not in summary["by_partition"]:
                summary["by_partition"][partition] = _empty_node_bucket()
            _add_node_to_bucket(summary["by_partition"][partition], node)
    return summary


def _partition_options(queue_summary, nodes_summary):
    node_parts = (nodes_summary or {}).get("by_partition", {})
    partitions = set(queue_summary["by_partition"]) | set(node_parts)
    options = []
    for partition in sorted(partitions):
        jobs = queue_summary["by_partition"].get(partition, _empty_job_bucket())
        nodes = node_parts.get(partition, _empty_node_bucket())
        options.append({
            "partition": partition,
            "nodes": nodes.get("nodes", 0),
            "gpus_total": nodes.get("gpus_total", 0),
            "gpus_free": nodes.get("gpus_free", 0),
            "gpus_available": nodes.get("gpus_available", 0),
            "running_jobs": jobs.get("running", 0),
            "pending_jobs": jobs.get("pending", 0),
            "running_gpus": jobs.get("gpus_running", 0),
            "pending_gpus": jobs.get("gpus_pending", 0),
        })
    return sorted(options, key=lambda row: (-row["gpus_available"], row["pending_gpus"], row["pending_jobs"], row["partition"]))


def _summarize_slurm_queue(jobs, pending, nodes_summary, now):
    summary = _empty_job_bucket()
    for job in jobs:
        _add_job_to_bucket(summary, job)
    summary["by_user"] = _group_job_summary(jobs, "user")
    summary["by_account"] = _group_job_summary(jobs, "account")
    summary["by_partition"] = _group_job_summary(jobs, "partition")
    summary["by_qos"] = _group_job_summary(jobs, "qos")
    summary["pending_reasons"] = dict(Counter(job.get("reason") or "(none)" for job in pending))
    summary["pending_estimates"] = _pending_estimate_summary(pending, now)
    summary["partition_options"] = _partition_options(summary, nodes_summary)
    return summary


def extract_xid(name):
    if firstmatch := _xid_re.search(name):
        return firstmatch.group()
    return None


def get_jobs(group=GROUP, users=USERS):
    widths = [20, 20, 20, 20, 20, 20, 40, 20, 20, 20, 25, 30, 20, 20, 100]
    fmt = "JobId:20,Name:20,UserName:20,State:20,TimeUsed:20,NumCPUs:20,QOS:40,NumNodes:20,tres-per-node:20,RestartCnt:20,StartTime:25,Reason:30,Priority:20,PriorityLong:20,Comment:100"
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


def display_config_path(path):
    path = path.replace("\\", "/").removeprefix("./")
    for marker in ("bv2/configs/", "bv2/config/"):
        if marker in path:
            return path.split(marker, 1)[1]
    if "/x/" in path:
        return "x/" + path.split("/x/", 1)[1]
    return path or "?"


def extract_config_display(launch_command):
    if not launch_command:
        return "?"
    try:
        tokens = shlex.split(launch_command)
    except ValueError:
        tokens = launch_command.split()

    for i, token in enumerate(tokens):
        if token == "--config" and i + 1 < len(tokens):
            return display_config_path(tokens[i + 1])
        if token.startswith("--config="):
            return display_config_path(token.split("=", 1)[1])

    if len(tokens) > 2 and tokens[2].endswith(".py"):
        return display_config_path(tokens[2])
    if len(tokens) > 1 and Path(tokens[0]).name in {"launch.py", "launch_slurm"} and tokens[1].endswith(".py"):
        return display_config_path(tokens[1])
    if len(tokens) > 3 and tokens[1] == "-m" and tokens[3].endswith(".py"):
        return display_config_path(tokens[3])

    for token in tokens:
        if token.endswith(".py") and Path(token).name not in {"_launch.py", "launch.py", "train.py"}:
            return display_config_path(token)
    return "?"


def dir_names(path):
    with os.scandir(path) as it:
        return [e.name for e in it if e.is_dir(follow_symlinks=False)]


def wid_from_workdir_name(name):
    if m := re.search(r'-(\d+)$', name):
        return int(m.group(1))
    return None


def exit_status_is_wip(status):
    return str(status or "").lower().endswith(" (wip)")


def state_from_exit_status(status):
    if not status:
        return None
    status = str(status).lower()
    if exit_status_is_wip(status):
        return None
    if status == "done":
        return "DONE"
    if status == "stopped":
        return "CANCELLED"
    if status == "preempted":
        return "PREEMPTED"
    if status == "error":
        return "FAILED"
    return status.upper()


def state_from_sacct(sacct):
    if failed_step := _failed_sacct_json_step(sacct):
        return state_from_sacct_compact(
            _sacct_json_state(failed_step),
            _sacct_json_exit_code(failed_step),
        )
    state = _sacct_json_state(sacct)
    exit_code = _sacct_json_exit_code(sacct)
    return state_from_sacct_compact(state, exit_code)


def _sacct_return_code(exit_code):
    if isinstance(exit_code, dict):
        ret = exit_code.get("return_code", {}) or {}
        code = ret.get("number", 0) if ret.get("set") else 0
        return int(code) if str(code).isdigit() else 0
    code = str(exit_code or "").split(":", 1)[0]
    return int(code) if code.isdigit() else 0


def _sacct_state(state):
    parts = str(state or "").split()
    return parts[0].rstrip("+") if parts else ""


def state_from_sacct_compact(state, exit_code):
    state = _sacct_state(state)
    exit_code = _sacct_return_code(exit_code)
    if state == 'CANCELLED':
        return 'CANCELLED' if exit_code == 0 else 'CANCELLED_FAIL'
    if state == "COMPLETED" and exit_code:
        return "FAILED"
    return state or 'UNKNOWN'


def _failed_sacct_status(status):
    return status in {"FAILED", "TIMEOUT", "CANCELLED_FAIL", "NODE_FAIL", "OUT_OF_MEMORY", "BOOT_FAIL", "DEADLINE"}


def _sacct_step_name(job_id):
    parts = str(job_id or "").split(".", 1)
    return parts[1] if len(parts) > 1 else ""


def _failed_sacct_row(rows):
    # Slurm can show the allocation as CANCELLED 0:0 while the batch/srun step failed.
    for row in rows:
        step = _sacct_step_name(row[0])
        if step and step != "extern" and _failed_sacct_status(state_from_sacct_compact(row[1], row[2])):
            return row
    return None


def _state_from_sacct_rows(rows):
    if not rows:
        return "UNKNOWN"
    if row := _failed_sacct_row(rows):
        return state_from_sacct_compact(row[1], row[2])
    row = next((r for r in rows if not _sacct_step_name(r[0])), rows[0])
    return state_from_sacct_compact(row[1], row[2])


def _sacct_json_state(sacct):
    state = sacct.get("state")
    if isinstance(state, dict):
        current = state.get("current") or []
        return current[-1] if current else None
    if isinstance(state, list):
        return state[-1] if state else None
    return state


def _sacct_json_exit_code(sacct):
    return sacct.get("exit_code")


def _sacct_json_step_name(step):
    for key in ("name", "step", "step_id", "job_id"):
        value = step.get(key)
        if isinstance(value, dict):
            value = value.get("name") or value.get("id") or value.get("number")
        if value is not None:
            return _sacct_step_name(value) or str(value)
    return ""


def _failed_sacct_json_step(sacct):
    for step in sacct.get("steps") or []:
        name = _sacct_json_step_name(step)
        if name != "extern" and _failed_sacct_status(state_from_sacct_compact(
            _sacct_json_state(step),
            _sacct_json_exit_code(step),
        )):
            return step
    return None


def detail_status_from_exit_status(status):
    return state_from_exit_status(status)


def _extra_info(xid_info):
    """Get extra info for an XID.

    Args:
        xid_info: Tuple of (xid, info), (xid, info, skip_config_loading), or
            (xid, info, skip_config_loading, skip_active_done_loading).
            Inactive overview loads config.json for status and finish_time.
            Hot overview can skip config loading where squeue is already the
            authoritative current-state source, but inactive WUs still use
            config exit_status so clean preemptions do not look completed.
    """
    if len(xid_info) == 4:
        xid, info, skip_config_loading, _skip_active_done_loading = xid_info
    elif len(xid_info) == 3:
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
    active_jobs_by_wid = {}
    active_by_wid = {}
    active_jid_by_wid = {}
    if "jobs" in info:
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

    workdir_names = []
    wid_counts = {}  # wid -> count (for duplicate detection)
    workdir_by_wid = {}
    workdir_jid_by_wid = {}
    config_by_wid = {}
    info["wus"] = {}
    for wuwd_name in wuwd_names:
        wid = wid_from_workdir_name(wuwd_name)
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
        if wid is None:
            continue
        workdir_names.append(wuwd_name)
        workdir_by_wid[wid] = wuwd_name
        wid_counts[wid] = wid_counts.get(wid, 0) + 1

    try:
        info["config"] = extract_config_display(launchinfo_fut.result())
    except OSError:
        info["config"] = "?"

    finished_by_wid = {}
    workdir_wids = set(workdir_by_wid)
    launch_only_wids = launch_wids - workdir_wids - set(active_by_wid)
    inactive_workdir_wids = workdir_wids - set(active_by_wid)
    config_status_wids = inactive_workdir_wids
    missing_config_wids = [
        wid for wid in config_status_wids
        if wid in workdir_by_wid and wid not in config_by_wid
    ]
    config_results = fs_executor.map(
        _load_config_only,
        [(wd_path, workdir_by_wid[wid]) for wid in missing_config_wids],
    )
    for wid, (_, config) in zip(missing_config_wids, config_results):
        config_by_wid[wid] = config
        if config.get("jid"):
            workdir_jid_by_wid[wid] = normalize_jid(config.get("jid"))
    # WIP exit_status is only the signal handler's partial write; sacct owns
    # the finished state if the trainer never finalized config.json.
    ambiguous_sacct_jids = {}
    for wid in config_status_wids:
        config = config_by_wid.get(wid, {})
        exit_status = config.get("exit_status")
        state = state_from_exit_status(exit_status)
        if state is not None:
            finished_by_wid[wid] = state
        elif exit_status_is_wip(exit_status):
            finished_by_wid[wid] = "UNKNOWN"
            if jid := workdir_jid_by_wid.get(wid):
                ambiguous_sacct_jids[jid] = wid
        else:
            finished_by_wid[wid] = "FAILED"
        if config.get("exit_status_at"):
            try:
                info["finish_time"] = max(
                    info.get("finish_time", 0),
                    datetime.fromisoformat(config["exit_status_at"]).timestamp(),
                )
            except (TypeError, ValueError):
                pass

    for jid, sacct in load_sacct_many(ambiguous_sacct_jids).items():
        finished_by_wid[ambiguous_sacct_jids[jid]] = state_from_sacct(sacct)

    if "jobs" in info:
        if not skip_config_loading:
            sacct_jids = {}
            # WUs in launchids that haven't created a workdir yet (just
            # submitted, queued, or cancelled-before-running). Without this
            # they fall into the "UNKNOWN" bucket in the overview even though
            # sacct knows their state. Same batched sacct call covers them.
            for wid in launch_only_wids:
                if (jid := launch_jid_by_wid.get(wid)) is not None:
                    sacct_jids[jid] = wid

            for jid, sacct in load_sacct_many(sacct_jids).items():
                finished_by_wid[sacct_jids[jid]] = state_from_sacct(sacct)
        elif launch_only_wids:
            # Keep hot overview fast: return unresolved WUs as UNKNOWN and let
            # the browser resolve them via /api/sacct/states after first paint.
            info["pending_sacct_jids"] = {
                str(jid): str(wid)
                for wid in launch_only_wids
                if (jid := launch_jid_by_wid.get(wid)) is not None
            }

    if active_by_wid or workdir_wids or launch_wids:
        states = Counter()
        for wid in launch_wids | workdir_wids | set(active_by_wid):
            if wid in active_by_wid:
                if (finished_by_wid.get(wid) == "DONE" and active_by_wid[wid] == "RUNNING"
                        and active_jid_by_wid.get(wid) == (workdir_jid_by_wid.get(wid) or launch_jid_by_wid.get(wid))):
                    state = "DONE_ISH"
                else:
                    state = active_by_wid[wid]
            elif wid in finished_by_wid:
                state = finished_by_wid[wid]
            elif wid in workdir_wids:
                state = "UNKNOWN" if skip_config_loading else "FAILED"
            else:
                state = "UNKNOWN"
            info["wus"][str(wid)] = state
            states[state] += 1
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


class _ReportHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_title = False
        self.in_metadata = False
        self.title_parts = []
        self.metadata_parts = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag.lower() == "title":
            self.in_title = True
        elif tag.lower() == "script" and attrs.get("id") == "smanager-report-metadata":
            self.in_metadata = True

    def handle_endtag(self, tag):
        if tag.lower() == "title":
            self.in_title = False
        elif tag.lower() == "script":
            self.in_metadata = False

    def handle_data(self, data):
        if self.in_title:
            self.title_parts.append(data)
        if self.in_metadata:
            self.metadata_parts.append(data)


def _validate_report_id(report_id):
    if not _report_id_re.fullmatch(report_id):
        raise HTTPException(status_code=400, detail="Invalid report ID")
    return report_id


def _report_version(version):
    version = str(version)
    if not version.isdigit():
        raise HTTPException(status_code=400, detail="Invalid report version")
    return int(version)


def _report_dir(report_id):
    return REPORT_ROOT_DIR / _validate_report_id(report_id)


def _report_bytes_label(n):
    return f"{n // (1024 * 1024)}MB"


async def _read_limited_body(request, max_bytes, label):
    chunks = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"{label} too large (max {_report_bytes_label(max_bytes)})",
            )
        chunks.append(chunk)
    return b"".join(chunks)


@contextmanager
def _report_write_lock():
    REPORT_ROOT_DIR.mkdir(parents=True, exist_ok=True)
    lock_dir = REPORT_ROOT_DIR / ".write.lock"
    deadline = time.time() + REPORT_LOCK_WAIT_SECONDS
    with _report_lock:
        while True:
            try:
                lock_dir.mkdir()
                (lock_dir / "owner.json").write_text(json.dumps({
                    "pid": os.getpid(),
                    "user": getuser(),
                    "host": os.uname().nodename,
                    "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                }) + "\n", encoding="utf-8")
                break
            except FileExistsError:
                try:
                    if time.time() - lock_dir.stat().st_mtime > REPORT_LOCK_STALE_SECONDS:
                        shutil.rmtree(lock_dir, ignore_errors=True)
                        continue
                except FileNotFoundError:
                    continue
                if time.time() >= deadline:
                    raise HTTPException(
                        status_code=503,
                        detail="Report storage is locked by another smanager upload; retry in a few seconds",
                    )
                time.sleep(0.1)
        try:
            yield
        finally:
            shutil.rmtree(lock_dir, ignore_errors=True)


def _report_versions(report_id):
    path = _report_dir(report_id)
    if not path.exists():
        return []
    versions = []
    with os.scandir(path) as it:
        for e in it:
            if e.is_file(follow_symlinks=False) and (m := _report_version_re.fullmatch(e.name)):
                versions.append(int(m.group(1)))
    return sorted(versions)


def _read_report_metadata(report_id, version):
    path = _report_dir(report_id) / f"v{version}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Report version not found: {report_id}/v{version}")
    return json.loads(path.read_text())


def _latest_report_metadata(report_id):
    versions = _report_versions(report_id)
    if not versions:
        raise HTTPException(status_code=404, detail=f"Report not found: {report_id}")
    return _read_report_metadata(report_id, versions[-1])


def _atomic_write_text(path, text):
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_bytes(path, data):
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _write_report_metadata(metadata, update_latest=False):
    report_dir = _report_dir(metadata["report_id"])
    _atomic_write_text(report_dir / f"v{metadata['version']}.json", json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    if update_latest:
        _atomic_write_text(report_dir / "latest.json", json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def _run_from_parts(xid, name):
    name = str(name or "").rstrip(".,;:)]}")
    if not xid or not _report_xid_re.fullmatch(xid) or not name:
        return None
    run = {"xid": xid, "name": name, "raw": f"{xid}/{name}"}
    if m := re.search(r'-(\d+)$', name):
        run["wid"] = int(m.group(1))
    return run


def _normalize_report_runs(items):
    runs = {}
    for item in items or []:
        run = None
        if isinstance(item, str):
            if m := re.fullmatch(r'(\d{6}_\d{6})/([^\s<>"\']+)', item.strip()):
                run = _run_from_parts(m.group(1), m.group(2))
        elif isinstance(item, dict):
            raw = item.get("raw")
            if raw and (m := re.fullmatch(r'(\d{6}_\d{6})/([^\s<>"\']+)', str(raw).strip())):
                run = _run_from_parts(m.group(1), m.group(2))
            else:
                xid = str(item.get("xid") or "")
                name = str(item.get("name") or "")
                if not name and item.get("wid") is not None:
                    name = str(item.get("wid"))
                run = _run_from_parts(xid, name)
            if run and item.get("wid") is not None:
                wid = str(item.get("wid"))
                if wid.isdigit():
                    run["wid"] = int(wid)
        if run:
            runs[run["raw"]] = run
    return sorted(runs.values(), key=lambda r: r["raw"])


def _normalize_report_xids(items):
    return sorted({
        str(xid) for xid in items or []
        if isinstance(xid, str) and _report_xid_re.fullmatch(xid)
    })


def _extract_report_metadata(html_text):
    parser = _ReportHTMLParser()
    parser.feed(html_text)
    title = " ".join(" ".join(parser.title_parts).split())
    embedded_text = "".join(parser.metadata_parts).strip()
    extraction = {"source": "grep", "embedded_metadata": False}

    if embedded_text:
        try:
            embedded = json.loads(embedded_text)
            if not isinstance(embedded, dict):
                raise ValueError("metadata is not an object")
            runs = _normalize_report_runs(embedded.get("runs", []))
            xids = set(_normalize_report_xids(embedded.get("xids", [])))
            xids.update(run["xid"] for run in runs)
            return {
                "title": str(embedded.get("title") or title or "").strip(),
                "xids": sorted(xids),
                "runs": runs,
                "extraction": {"source": "embedded_metadata", "embedded_metadata": True},
            }
        except Exception as e:
            extraction["embedded_error"] = str(e)

    runs = _normalize_report_runs([f"{m.group(1)}/{m.group(2)}" for m in _report_run_re.finditer(html_text)])
    xids = set(_report_xid_re.findall(html_text))
    xids.update(run["xid"] for run in runs)
    return {"title": title, "xids": sorted(xids), "runs": runs, "extraction": extraction}


def _report_metadata(report_id, version, html_text):
    extracted = _extract_report_metadata(html_text)
    uploaded_at = datetime.now().astimezone()
    return {
        "schema": 1,
        "report_id": report_id,
        "version": version,
        "title": extracted["title"] or report_id,
        "uploader": getuser(),
        "uploaded_at": uploaded_at.isoformat(timespec="seconds"),
        "uploaded_ts": uploaded_at.timestamp(),
        "html_file": f"v{version}.html",
        "xids": extracted["xids"],
        "runs": extracted["runs"],
        "extraction": extracted["extraction"],
    }


def _all_report_metadata(include_versions=False):
    if not REPORT_ROOT_DIR.exists():
        return []
    reports = []
    with os.scandir(REPORT_ROOT_DIR) as it:
        names = sorted(e.name for e in it if e.is_dir(follow_symlinks=False) and _report_id_re.fullmatch(e.name))
    for report_id in names:
        versions = _report_versions(report_id)
        if not versions:
            continue
        for version in (versions if include_versions else [versions[-1]]):
            try:
                reports.append(_read_report_metadata(report_id, version))
            except Exception as e:
                log.warning("Failed to read report metadata %s/v%s: %s", report_id, version, e)
    return sorted(reports, key=lambda m: (m.get("uploaded_ts") or 0, m.get("report_id") or ""), reverse=True)


def _report_text(meta):
    parts = [meta.get("report_id", ""), meta.get("title", ""), meta.get("uploader", "")]
    parts.extend(meta.get("xids", []))
    for run in meta.get("runs", []):
        parts.extend([run.get("raw", ""), run.get("name", ""), str(run.get("wid", ""))])
    return " ".join(parts).lower()


def _report_matches_run(meta, run_filter, xid_filter=""):
    if not run_filter:
        return True
    run_filter = run_filter.strip().lower()
    if "/" in run_filter:
        xid_part, run_part = run_filter.split("/", 1)
        xid_filter = xid_filter or xid_part
        run_filter = run_part
    for run in meta.get("runs", []):
        if xid_filter and run.get("xid") != xid_filter:
            continue
        fields = [run.get("raw", ""), run.get("name", ""), str(run.get("wid", ""))]
        if any(run_filter == str(field).lower() or run_filter in str(field).lower() for field in fields):
            return True
    return False


def _truthy(value):
    return str(value).lower() in ("1", "true", "yes", "on")


def _report_matches(meta, q="", xid="", run=""):
    if q and not all(word in _report_text(meta) for word in q.lower().split()):
        return False
    if xid and xid not in meta.get("xids", []) and not any(r.get("xid") == xid for r in meta.get("runs", [])):
        return False
    return _report_matches_run(meta, run, xid)


def _report_html(report_id, version):
    path = _report_dir(report_id) / f"v{version}.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Report HTML not found: {report_id}/v{version}")
    return path.read_text(encoding="utf-8", errors="replace")


def _report_src_path(report_id, version):
    return _report_dir(report_id) / f"v{version}.src.zip"


def _validate_report_src(src_bytes):
    if not src_bytes:
        raise HTTPException(status_code=400, detail="Empty source upload")
    if len(src_bytes) > REPORT_SRC_MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Source upload too large (max {_report_bytes_label(REPORT_SRC_MAX_BYTES)})",
        )
    if not zipfile.is_zipfile(BytesIO(src_bytes)):
        raise HTTPException(status_code=400, detail="Source material must be a zip file")


def _source_material_metadata(src_bytes, version):
    uploaded_at = datetime.now().astimezone()
    return {
        "zip_file": f"v{version}.src.zip",
        "size": len(src_bytes),
        "uploader": getuser(),
        "uploaded_at": uploaded_at.isoformat(timespec="seconds"),
        "uploaded_ts": uploaded_at.timestamp(),
    }


def _latest_report_version(report_id):
    versions = _report_versions(report_id)
    if not versions:
        raise HTTPException(status_code=404, detail=f"Report not found: {report_id}")
    return versions[-1]


def _report_upload_parts(body, content_type):
    if not content_type.lower().startswith("multipart/form-data"):
        raise HTTPException(status_code=400, detail="Report upload must use multipart/form-data with an html part")

    msg = BytesParser(policy=email_policy).parsebytes(
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode() + body)
    if not msg.is_multipart():
        raise HTTPException(status_code=400, detail="Invalid multipart report upload")

    parts = {}
    filenames = {}
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        filename = part.get_filename()
        payload = part.get_payload(decode=True) or b""
        if name:
            parts[name] = payload
        if filename:
            filenames[filename] = payload

    html_bytes = parts.get("html") or parts.get("report") or parts.get("file") or filenames.get("report.html")
    src_bytes = parts.get("src") or parts.get("source") or parts.get("src_zip") or filenames.get("src.zip")
    if not html_bytes:
        raise HTTPException(status_code=400, detail="Multipart report upload needs an html part")
    return html_bytes, src_bytes


async def _upload_report_src(report_id, version, request):
    report_id = _validate_report_id(report_id)
    src_bytes = await _read_limited_body(request, REPORT_SRC_MAX_BYTES, "Source upload")
    _validate_report_src(src_bytes)

    with _report_write_lock():
        if version is None:
            version = _latest_report_version(report_id)
        else:
            version = _report_version(version)
        metadata = _read_report_metadata(report_id, version)
        _atomic_write_bytes(_report_src_path(report_id, version), src_bytes)
        metadata["source_material"] = _source_material_metadata(src_bytes, version)
        _write_report_metadata(metadata, update_latest=version == _latest_report_version(report_id))

    log.info("POST /api/reports/%s/v%s/src (%d bytes)", report_id, version, len(src_bytes))
    return metadata


def _download_report_src(report_id, version):
    report_id = _validate_report_id(report_id)
    if version is None:
        version = _latest_report_version(report_id)
    else:
        version = _report_version(version)
        _read_report_metadata(report_id, version)
    path = _report_src_path(report_id, version)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Source material not found for {report_id}/v{version}")
    return FileResponse(path, media_type="application/zip", filename="src.zip")


@app.post("/api/reports/{report_id}")
async def upload_report(report_id, request: Request):
    report_id = _validate_report_id(report_id)
    body = await _read_limited_body(request, REPORT_UPLOAD_MAX_BYTES, "Report upload")
    html_bytes, src_bytes = _report_upload_parts(body, request.headers.get("content-type", ""))
    if not html_bytes:
        raise HTTPException(status_code=400, detail="Empty report upload")
    if len(html_bytes) > REPORT_HTML_MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Report HTML too large (max {_report_bytes_label(REPORT_HTML_MAX_BYTES)})",
        )
    if src_bytes is not None:
        _validate_report_src(src_bytes)
    html_text = html_bytes.decode("utf-8", errors="replace")

    with _report_write_lock():
        report_dir = _report_dir(report_id)
        report_dir.mkdir(parents=True, exist_ok=True)
        version = (_report_versions(report_id) or [0])[-1] + 1
        metadata = _report_metadata(report_id, version, html_text)
        _atomic_write_text(report_dir / f"v{version}.html", html_text)
        if src_bytes is not None:
            _atomic_write_bytes(_report_src_path(report_id, version), src_bytes)
            metadata["source_material"] = _source_material_metadata(src_bytes, version)
        _write_report_metadata(metadata, update_latest=True)

    log.info("POST /api/reports/%s -> v%s", report_id, version)
    return metadata


@app.post("/api/reports/{report_id}/src")
async def upload_report_latest_src(report_id, request: Request):
    return await _upload_report_src(report_id, None, request)


@app.post("/api/reports/{report_id}/v{version}/src")
async def upload_report_version_src(report_id, version, request: Request):
    return await _upload_report_src(report_id, version, request)


@app.get("/api/reports/{report_id}/src")
def get_report_latest_src(report_id):
    return _download_report_src(report_id, None)


@app.get("/api/reports/{report_id}/v{version}/src")
def get_report_version_src(report_id, version):
    return _download_report_src(report_id, version)


@app.get("/api/reports")
def list_reports(q="", xid="", run="", include_versions=False):
    include_versions = _truthy(include_versions)
    reports = [m for m in _all_report_metadata(include_versions) if _report_matches(m, q, xid, run)]
    return {"reports": reports}


@app.get("/api/reports/by-xid/{xid}")
def reports_by_xid(xid, include_versions=False):
    if not _report_xid_re.fullmatch(xid):
        raise HTTPException(status_code=400, detail="Invalid XID format")
    include_versions = _truthy(include_versions)
    return {"reports": [m for m in _all_report_metadata(include_versions) if _report_matches(m, xid=xid)]}


@app.get("/api/reports/by-run/{xid}/{run}")
def reports_by_run(xid, run, include_versions=False):
    if not _report_xid_re.fullmatch(xid):
        raise HTTPException(status_code=400, detail="Invalid XID format")
    include_versions = _truthy(include_versions)
    return {"reports": [m for m in _all_report_metadata(include_versions) if _report_matches(m, xid=xid, run=run)]}


@app.get("/api/reports/{report_id}/versions")
def report_versions(report_id):
    versions = _report_versions(report_id)
    if not versions:
        raise HTTPException(status_code=404, detail=f"Report not found: {report_id}")
    return {"report_id": report_id, "versions": [_read_report_metadata(report_id, v) for v in versions]}


@app.get("/api/reports/{report_id}/v{version}")
def get_report_version(report_id, version):
    _validate_report_id(report_id)
    version = _report_version(version)
    return _read_report_metadata(report_id, version)


@app.get("/api/reports/{report_id}/v{version}/html")
def get_report_version_html(report_id, version):
    _validate_report_id(report_id)
    version = _report_version(version)
    return Response(content=_report_html(report_id, version), media_type="text/plain; charset=utf-8")


@app.get("/api/reports/{report_id}/html")
def get_report_html(report_id):
    metadata = _latest_report_metadata(report_id)
    return Response(content=_report_html(report_id, metadata["version"]), media_type="text/plain; charset=utf-8")


@app.get("/api/reports/{report_id}")
def get_report(report_id):
    return _latest_report_metadata(report_id)


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


@app.get("/reports/{report_id}")
@app.get("/reports/{report_id}/v{version}")
def report_viewer_page(report_id, version=None):
    _validate_report_id(report_id)
    if version is None:
        _latest_report_metadata(report_id)
    else:
        _read_report_metadata(report_id, _report_version(version))
    html = (SCRIPT_DIR / "report.html").read_text()
    return Response(content=html, media_type="text/html")


@app.get("/fonts/{filename}")
def serve_font(filename: str):
    if not filename.endswith(".woff2"):
        raise HTTPException(status_code=404, detail="Not found")
    font_path = SCRIPT_DIR / filename
    if not font_path.exists():
        raise HTTPException(status_code=404, detail="Font not found")
    return Response(content=font_path.read_bytes(), media_type="font/woff2")


@app.get("/api/slurm/nodes")
def get_slurm_nodes_api():
    """Get node and GPU availability from Slurm."""
    t0 = time.time()
    log.info("GET /api/slurm/nodes - fetching...")
    nodes = _load_slurm_nodes()
    summary = _summarize_slurm_nodes(nodes)
    log.info("GET /api/slurm/nodes - done: %d nodes (%.2fs)", len(nodes), time.time() - t0)
    return {
        "cluster": _cluster,
        "generated_at": int(time.time()),
        "nodes": nodes,
        "summary": summary,
    }


@app.get("/api/slurm/queue")
def get_slurm_queue(account=None, users=None, states=None, partition=None, include_nodes="1"):
    """Get Slurm queue jobs plus user/partition/QOS summaries."""
    t0 = time.time()
    now = int(time.time())
    account_filter = _query_value(account, GROUP)
    users_filter = _query_value(users, USERS)
    states_filter = _query_value(states)
    partition_filter = _query_value(partition)
    load_nodes = _query_bool(include_nodes)

    log.info(
        "GET /api/slurm/queue - account=%s users=%s states=%s partition=%s include_nodes=%s",
        account_filter or "all", users_filter or "all", states_filter or "all", partition_filter or "all", load_nodes,
    )
    nodes_fut = executor.submit(_load_slurm_nodes) if load_nodes else None
    rows = _load_slurm_queue_rows(
        account=account_filter,
        users=users_filter,
        states=states_filter,
        partition=partition_filter,
    )
    jobs = [_format_slurm_queue_job(row, now) for row in rows]
    pending = _rank_pending_jobs(jobs)
    nodes = nodes_fut.result() if nodes_fut else None
    nodes_summary = _summarize_slurm_nodes(nodes) if nodes is not None else None
    summary = _summarize_slurm_queue(jobs, pending, nodes_summary, now)
    log.info(
        "GET /api/slurm/queue - done: %d jobs, %d pending%s (%.2fs)",
        len(jobs), len(pending), f", {len(nodes)} nodes" if nodes is not None else "", time.time() - t0,
    )
    return {
        "cluster": _cluster,
        "generated_at": now,
        "filters": {
            "account": account_filter,
            "users": users_filter,
            "states": states_filter,
            "partition": partition_filter,
        },
        "jobs": jobs,
        "pending": pending,
        "summary": summary,
        "nodes": nodes,
        "nodes_summary": nodes_summary,
    }


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
    hot_items = [
        (xid, {"states": info["states"], "wd": info["wd"], "jobs": info["jobs"]}, True, True)
        for xid, info in hot_xids.items()
    ]
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
            pending_jobs = [j for j in xid_jobs if j.get("STATE") == "PENDING"]
            pending_estimates = []
            for j in pending_jobs:
                start_raw = _job_start_time_raw(j)
                if start_ts := _parse_slurm_start_time(start_raw):
                    pending_estimates.append((start_ts, start_raw, j.get("REASON", "")))
            if pending_jobs:
                if pending_estimates:
                    start_ts, start_raw, reason = min(pending_estimates)
                else:
                    start_ts, start_raw = None, ""
                    reason = Counter(j.get("REASON", "") for j in pending_jobs).most_common(1)[0][0]
                info["pending_start"] = {
                    "ts": start_ts,
                    "raw": start_raw,
                    "reason": reason,
                    "known": len(pending_estimates),
                    "total": len(pending_jobs),
                }
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

    # Add extra info with threading. Inactive WU status and finish_time come
    # from config.json, avoiding separate DONE stats.
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


ETA_MIN_POINTS = 100


def _finite_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _metric_rows(reader, metrics, name):
    return reader.rows(name) if name in metrics else 0


# Public plattli selectors fall back to full-array reads when hot rows exist.
# TODO: upstream a plattli last-value API and stop reaching into Reader internals.
def _metric_last_value(reader, metrics, name):
    if name not in metrics:
        return None
    spec = reader._metric_spec(name, allow_hot=True)
    count, last_step = reader._columnar_count_and_last_step(name, spec) if spec is not None else (0, None)
    _, hot_values = reader._hot_for_metric(name, last_step)
    if len(hot_values):
        return hot_values[-1]
    if not count:
        return None
    values = reader._read_value_slice(name, spec, count - 1, 1)
    return values[-1] if len(values) else None


def _metric_last_index(reader, metrics, name):
    if name not in metrics:
        return None
    spec = reader._metric_spec(name, allow_hot=True)
    count, last_step = reader._columnar_count_and_last_step(name, spec) if spec is not None else (0, None)
    hot_indices, _ = reader._hot_for_metric(name, last_step)
    if hot_indices.size:
        return int(hot_indices[-1])
    return int(last_step) if count and last_step is not None else None


def _metric_last_float(reader, metrics, name):
    return _finite_float(_metric_last_value(reader, metrics, name))


def _sum_float(values):
    if values is None or len(values) == 0:
        return None
    return _finite_float(values.sum())


def _metric_full_values(reader, metrics, name):
    if name not in metrics:
        return None
    values = reader.metric_values(name)
    return values if len(values) else None


def _runtime_eta_from_plattli(reader, metrics, include_eta=True, include_breakdown=False):
    result = {
        "min_points": ETA_MIN_POINTS,
        "counts": {"chrono/proctime": _metric_rows(reader, metrics, "chrono/proctime")},
    }

    proctime = _metric_last_float(reader, metrics, "chrono/proctime")
    if proctime is not None and proctime >= 0:
        result["runtime"] = proctime
    if not include_eta:
        return result

    result["counts"].update({
        "chrono/percent": _metric_rows(reader, metrics, "chrono/percent"),
    })

    missing = [name for name in ["chrono/proctime", "chrono/percent"] if not result["counts"][name]]
    if missing:
        result["eta_error"] = "missing " + ", ".join(missing)
        return result

    result["n_points"] = min(result["counts"][name] for name in ["chrono/proctime", "chrono/percent"])
    if result["n_points"] < ETA_MIN_POINTS:
        result["eta_error"] = f"too few measurements ({result['n_points']}/{ETA_MIN_POINTS})"
        return result

    percent = _metric_last_float(reader, metrics, "chrono/percent")
    if percent is None or percent <= 0 or percent >= 1:
        result["eta_error"] = f"invalid chrono/percent {percent}"
        return result
    if proctime is None or proctime <= 0:
        result["eta_error"] = f"invalid chrono/proctime {proctime}"
        return result

    scale = 1 / percent - 1
    eta_real = proctime * scale
    if not math.isfinite(eta_real):
        result["eta_error"] = "non-finite ETA"
        return result

    result.update({
        "percent": percent,
        "proctime": proctime,
        "eta_real": eta_real,
    })
    if not include_breakdown:
        return result

    result["counts"].update({
        "chrono/modeltime": _metric_rows(reader, metrics, "chrono/modeltime"),
        "chrono/evaltime": _metric_rows(reader, metrics, "chrono/evaltime"),
    })

    model_total = _sum_float(_metric_full_values(reader, metrics, "chrono/modeltime"))
    if model_total is None or model_total < 0:
        result["breakdown_error"] = f"invalid chrono/modeltime sum {model_total}"
        return result

    eta_model = model_total * scale
    if not math.isfinite(eta_model):
        result["breakdown_error"] = "non-finite model ETA"
        return result

    result.update({
        "model_total": model_total,
        "eta_model": eta_model,
    })

    eval_total = _sum_float(_metric_full_values(reader, metrics, "chrono/evaltime"))
    if eval_total is not None and eval_total >= 0:
        eta_eval = eval_total * scale
        eta_other = result["eta_real"] - eta_eval - result["eta_model"]
        if math.isfinite(eta_eval) and math.isfinite(eta_other):
            result["eval_total"] = eval_total
            result["eta_eval"] = eta_eval
            result["eta_other"] = eta_other
            result["eval_source"] = "chrono/evaltime"
        else:
            result["eval_source"] = "invalid"
    elif eval_total is not None:
        result["eval_source"] = "chrono/evaltime"
    else:
        result["eval_source"] = "missing"
    return result


def runtime_eta(wd_path, include_eta=True, include_breakdown=False):
    if not plattli.is_run(wd_path):
        return {}
    try:
        with plattli.Reader(wd_path) as r:
            metrics = set(r.metrics())
            return _runtime_eta_from_plattli(r, metrics, include_eta, include_breakdown)
    except Exception as e:
        log.debug("plattli ETA read failed for %s: %s", wd_path, e)
        return {}


def last_metric(wd_path, metric_name="train/loss", extra_metrics=(), include_eta=True):
    # Try plattli format first
    if plattli.is_run(wd_path):
        try:
            with plattli.Reader(wd_path) as r:
                result = {}
                metrics = r.metrics()
                metric_names = set(metrics)
                # Get step from first metric's last index
                if metrics:
                    step = _metric_last_index(r, metric_names, metrics[0])
                    if step is not None:
                        result["step"] = step
                # Only fetch the requested metrics
                for name in dict.fromkeys([metric_name, *extra_metrics]):
                    if name not in metric_names:
                        continue
                    v = _metric_last_value(r, metric_names, name)
                    if v is not None:
                        v = v.item() if hasattr(v, 'item') else v
                        # Sanitize non-finite float values (nan, inf, -inf) to strings
                        # JSON does not support these values and will raise ValueError.
                        # Using strings allows the UI to distinguish between "metric is nan/inf"
                        # vs "no metric data" (which shows as '-').
                        if isinstance(v, float) and not math.isfinite(v):
                            if math.isnan(v):
                                v = "nan"
                            elif v > 0:
                                v = "inf"
                            else:
                                v = "-inf"
                        result[name] = v
                result["_runtime_eta"] = _runtime_eta_from_plattli(r, metric_names, include_eta)
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


def _xid_start_time(xid):
    try:
        dt = datetime.strptime(xid, "%y%m%d_%H%M%S").replace(hour=0, minute=0, second=0)
        return dt.strftime("%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return None


def _load_sacct_compact(cmd):
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            log.debug("compact sacct nonzero: rc=%s stderr=%s", result.returncode, result.stderr.strip())
            return {}
        rows = {}
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            jid, state, exit_code = (line.split("|", 2) + ["", ""])[:3]
            if root_jid := normalize_jid(jid):
                rows.setdefault(root_jid, []).append((jid, state, exit_code))
        return {jid: _state_from_sacct_rows(job_rows) for jid, job_rows in rows.items()}
    except Exception as e:
        log.debug("compact sacct failed: %s", e)
    return {}


def load_sacct_states_xid(xid):
    cmd = ["sacct", "-n", "-P"]
    if start_time := _xid_start_time(xid):
        cmd += ["-S", start_time]
    cmd += ["--format=JobIDRaw,State,ExitCode", f"--name={xid}"]
    return _load_sacct_compact(cmd)


def load_sacct_states_many(jids, start_time=None):
    jids = sorted({normalize_jid(jid) for jid in jids if normalize_jid(jid)})
    if not jids:
        return {}
    cmd = ["sacct", "-n", "-P"]
    if start_time:
        cmd += ["-S", start_time]
    cmd += [
        f"--jobs={','.join(map(str, jids))}",
        "--format=JobIDRaw,State,ExitCode",
    ]
    return _load_sacct_compact(cmd)


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


def _load_metric_only(args):
    wd_path, wuwd_name, metric_name, extra_metrics, include_eta = args
    return wuwd_name, last_metric(wd_path / wuwd_name, metric_name, extra_metrics, include_eta)


def _load_eta_breakdown_only(args):
    wd_path, wuwd_name = args
    return wuwd_name, runtime_eta(wd_path / wuwd_name, include_breakdown=True)


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


def parse_wids_arg(wids):
    if wids is None:
        return None
    return {normalize_wid(wid.strip()) for wid in str(wids).split(",") if wid.strip()}


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
    if failed_step := _failed_sacct_json_step(sacct):
        return _sacct_return_code(_sacct_json_exit_code(failed_step))
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
    return {
        "restarts": sacct.get("restart_cnt", 0),
        "exit_code": extract_exit_code(sacct),
        "status": state_from_sacct(sacct),
        "qwait": "never" if start in [0, SLURM_NO_START_TIME] else hms(start - eligible),
        "runtime": hms(time_info.get("elapsed", 0)),
    }


def _parse_slurm_time(value, skip_values):
    if not value or value in skip_values:
        return None
    try:
        return int(datetime.strptime(value.split(".", 1)[0], "%Y-%m-%dT%H:%M:%S").timestamp())
    except (ValueError, TypeError):
        return None


def _parse_sacct_time(value):
    return _parse_slurm_time(value, {"Unknown", "None"})


def _parse_sacct_int(value):
    return int(value) if str(value).isdigit() else None


def _parse_slurm_start_time(value):
    return _parse_slurm_time(value, {"N/A"})


def _job_start_time_raw(job):
    return job.get("START_TIME") or job.get("STARTTIME") or job.get("START") or ""


def _parse_sacct_exit_code(value):
    if not value:
        return None
    code = value.split(":", 1)[0]
    return int(code) if code.isdigit() else None


def _nfs_safe_overwrite(path, text):
    with os.fdopen(os.open(path, os.O_RDWR | os.O_CREAT, 0o666), "r+", encoding="utf-8") as f:
        f.seek(0)
        f.write(text)
        f.truncate()
        f.flush()
        os.fsync(f.fileno())


def _update_metrics_plattli_config(cfg_path, config):
    zip_path = cfg_path.parent / "metrics.plattli"
    if not zip_path.is_file():
        return False
    tmp_path = zip_path.with_name(zip_path.name + ".tmp")
    config_text = json.dumps(config, ensure_ascii=False)
    wrote_config = False
    with zipfile.ZipFile(zip_path) as zin, zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_STORED) as zout:
        for info in zin.infolist():
            if info.filename == "config.json":
                if wrote_config:
                    continue
                zout.writestr(info, config_text)
                wrote_config = True
            else:
                zout.writestr(info, zin.read(info))
        if not wrote_config:
            zout.writestr("config.json", config_text)
    tmp_path.replace(zip_path)
    return True


def load_sacct_detail_compact(jids, start_time=None):
    jids = sorted({normalize_jid(jid) for jid in jids if normalize_jid(jid)})
    if not jids:
        return {}
    cmd = ["sacct", "-n", "-P"]
    if start_time:
        cmd += ["-S", start_time]
    try:
        base_cmd = cmd + [
            f"--jobs={','.join(map(str, jids))}",
        ]
        result = subprocess.run(
            base_cmd + ["--format=JobIDRaw,State,ExitCode,ElapsedRaw,Eligible,Start,Restarts"],
            capture_output=True, text=True,
        )
        has_restarts = True
        if result.returncode != 0:
            result = subprocess.run(
                base_cmd + ["--format=JobIDRaw,State,ExitCode,ElapsedRaw,Eligible,Start"],
                capture_output=True, text=True,
            )
            has_restarts = False
        if result.returncode != 0:
            log.debug("load_sacct_detail_compact nonzero for %s jobs: rc=%s stderr=%s",
                      len(jids), result.returncode, result.stderr.strip())
            return {}
        rows = {}
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            if has_restarts:
                jid, state, exit_code, elapsed, eligible, start, restarts = (line.split("|", 6) + [""] * 6)[:7]
            else:
                jid, state, exit_code, elapsed, eligible, start = (line.split("|", 5) + [""] * 5)[:6]
                restarts = 0
            root_jid = normalize_jid(jid)
            if not root_jid:
                continue
            rows.setdefault(root_jid, []).append({
                "job_id": jid,
                "state": state,
                "exit_code": exit_code,
                "elapsed": elapsed,
                "eligible": eligible,
                "start": start,
                "restarts": restarts,
            })
        detail = {}
        for jid, job_rows in rows.items():
            top = next((r for r in job_rows if not _sacct_step_name(r["job_id"])), job_rows[0])
            compact_rows = [(r["job_id"], r["state"], r["exit_code"]) for r in job_rows]
            failed = _failed_sacct_row(compact_rows)
            exit_code = failed[2] if failed else top["exit_code"]
            start_ts = _parse_sacct_time(top["start"])
            eligible_ts = _parse_sacct_time(top["eligible"])
            qwait = "never" if not start_ts else (hms(max(0, start_ts - eligible_ts)) if eligible_ts else "n/a")
            detail[jid] = {
                "restarts": _parse_sacct_int(top["restarts"]) or 0,
                "exit_code": _parse_sacct_exit_code(exit_code),
                "status": _state_from_sacct_rows(compact_rows),
                "qwait": qwait,
                "runtime": hms(_parse_sacct_int(top["elapsed"]) or 0),
            }
        return detail
    except Exception as e:
        log.debug("load_sacct_detail_compact failed for %s jobs: %s", len(jids), e)
    return {}


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
    xid_widths = [20, 20, 20, 20, 20, 20, 40, 20, 20, 20, 25, 30, 100]
    xid_fmt = "JobId:20,Name:20,UserName:20,State:20,TimeUsed:20,NumCPUs:20,QOS:40,NumNodes:20,GRES:20,RestartCnt:20,StartTime:25,Reason:30,Comment:100"
    lines = run_cmd(f"squeue -n {xid} -O {xid_fmt}")
    offsets = [sum(xid_widths[:i]) for i in range(len(xid_widths))]
    rows = [[j[offsets[i]:offsets[i]+xid_widths[i]].strip() for i in range(len(xid_widths))] for j in lines if j.strip()]
    if len(rows) < 2:
        return {}
    headers, job_rows = rows[0], rows[1:]
    return {row[0]: dict(zip(headers, row)) for row in job_rows}


def _load_current_job(jid):
    widths = [20, 20, 20, 100]
    fmt = "JobId:20,Name:20,State:20,Comment:100"
    result = subprocess.run(["squeue", "-j", str(jid), "-h", "-O", fmt], capture_output=True, text=True)
    if result.returncode != 0:
        log.debug("squeue -j %s failed: %s", jid, result.stderr.strip())
        return {}
    offsets = [sum(widths[:i]) for i in range(len(widths))]
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        values = [line[offsets[i]:offsets[i]+widths[i]].strip() for i in range(len(widths))]
        return dict(zip(["JOBID", "NAME", "STATE", "COMMENT"], values))
    return {}


def _mark_stopped_config(wd_path, wid):
    workdirs, _ = _workdirs_by_suffix(wd_path)
    wuwd_name = workdirs.get(wid)
    if not wuwd_name:
        return False
    cfg_path = wd_path / wuwd_name / "config.json"
    config = load_config(cfg_path.parent)
    if not config:
        return False
    if config.get("exit_status") == "stopped":
        return False
    config["exit_status"] = "stopped"
    config["exit_status_at"] = datetime.now().isoformat(timespec="seconds")
    _nfs_safe_overwrite(cfg_path, json.dumps(config, indent=0) + "\n")
    _update_metrics_plattli_config(cfg_path, config)
    return True


def _mark_stopped_for_jobs(xid, jobs_by_jid):
    wd_path = _find_xid_path(xid)
    if not wd_path:
        return []
    jobs_by_wid, _ = _jobs_by_wid(jobs_by_jid, load_launchids(wd_path))
    marked = []
    for wid, job in jobs_by_wid.items():
        if job.get("STATE") in {"RUNNING", "COMPLETING"}:
            continue
        if _mark_stopped_config(wd_path, wid):
            marked.append(wid)
    return marked


def _mark_stopped_for_job(job):
    if not job:
        return []
    xid = extract_xid(job.get("NAME", "")) or extract_xid(job.get("COMMENT", ""))
    jid = normalize_jid(job.get("JOBID"))
    if not xid or not jid:
        return []
    return _mark_stopped_for_jobs(xid, {str(jid): job})


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


RUN_QUERY_CONFIG_SKIP = {"jid", "wid", "xid", "name", "exit_status", "exit_status_at"}
RUN_QUERY_OPS = {"in", "exists", "ne", "gt", "gte", "lt", "lte"}
MISSING = object()


def _wid_key(wid):
    return (0, int(wid)) if str(wid).isdigit() else (1, str(wid))


def _xid_datetime(xid):
    return datetime.strptime(xid, "%y%m%d_%H%M%S")


def _scope_datetime(scope, key):
    value = scope.get(key)
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value)
        value = str(value)
        iso_value = value[:-1] + "+00:00" if value.endswith("Z") else value
        dt = _xid_datetime(value) if _xid_re.fullmatch(value) else datetime.fromisoformat(iso_value)
        return dt.astimezone().replace(tzinfo=None) if dt.tzinfo and dt.utcoffset() is not None else dt
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail=f"Invalid scope.{key}: {value}")


def _flatten_config(value, path="", out=None):
    out = out if out is not None else {}
    if isinstance(value, dict) and value:
        for k, v in value.items():
            p = f"{path}.{k}" if path else str(k)
            if p.split(".", 1)[0] not in RUN_QUERY_CONFIG_SKIP:
                _flatten_config(v, p, out)
    elif isinstance(value, list) and value:
        for i, v in enumerate(value):
            _flatten_config(v, f"{path}.{i}" if path else str(i), out)
    elif path:
        out[path] = value
    return out


def _config_value(config, path):
    value = config
    for part in path.split("."):
        if isinstance(value, dict):
            if part not in value:
                return MISSING
            value = value[part]
        elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        else:
            return MISSING
    return value


def _json_key(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _matches_predicate(value, pred):
    if isinstance(pred, dict) and any(k in pred for k in RUN_QUERY_OPS):
        if "exists" in pred and (value is not MISSING) != bool(pred["exists"]):
            return False
        if value is MISSING:
            return "exists" in pred and len(pred) == 1
        if "in" in pred and value not in pred["in"]:
            return False
        if "ne" in pred and value == pred["ne"]:
            return False
        for op, cmp in (
            ("gt", lambda a, b: a > b),
            ("gte", lambda a, b: a >= b),
            ("lt", lambda a, b: a < b),
            ("lte", lambda a, b: a <= b),
        ):
            if op not in pred:
                continue
            try:
                if not cmp(value, pred[op]):
                    return False
            except TypeError:
                return False
        return True
    return value is not MISSING and value == pred


def _run_status(run):
    raw = run["config"].get("exit_status")
    return state_from_exit_status(raw) or (str(raw).upper() if raw else "")


def _public_run(run):
    config = run["config"]
    return {
        "xid": run["xid"],
        "wid": run["wid"],
        "name": config.get("name", ""),
        "workdir": run["workdir"],
        "plattli_run_id": run["workdir"],
        "jid": normalize_jid(config.get("jid")),
        "exit_status": config.get("exit_status"),
        "status": _run_status(run),
    }


def _xid_run_configs(xid, wanted_wids=None):
    wd_path = _find_xid_path(xid)
    if not wd_path:
        return [], [{"xid": xid}]
    workdirs, _ = _workdirs_by_suffix(wd_path)
    if wanted_wids is None:
        launchids = load_launchids(wd_path)
        wids = sorted(set(workdirs) | {normalize_wid(wid) for wid in launchids}, key=_wid_key)
    else:
        wids = sorted({normalize_wid(wid) for wid in wanted_wids}, key=_wid_key)

    missing = [{"xid": xid, "wid": wid} for wid in wids if wid not in workdirs]
    wids = [wid for wid in wids if wid in workdirs]
    configs = fs_executor.map(_load_config_only, [(wd_path, workdirs[wid]) for wid in wids])
    runs = []
    for wid, (wuwd_name, config) in zip(wids, configs):
        if not config:
            missing.append({"xid": xid, "wid": wid})
            continue
        runs.append({
            "xid": xid,
            "wid": wid,
            "workdir": f"{wd_path.name}/{wuwd_name}",
            "config": config,
        })
    return runs, missing


def _runs_from_specs(specs):
    by_xid = {}
    order = []
    for spec in specs:
        if not isinstance(spec, dict):
            raise HTTPException(status_code=400, detail="runs entries must be objects")
        xid = str(spec.get("xid", ""))
        if not _xid_re.fullmatch(xid):
            raise HTTPException(status_code=400, detail=f"Invalid run xid: {xid}")
        if xid not in by_xid:
            by_xid[xid] = set()
            order.append(xid)
        if by_xid[xid] is None:
            continue
        if "wid" in spec:
            by_xid[xid].add(normalize_wid(spec["wid"]))
        else:
            by_xid[xid] = None

    runs, missing = [], []
    for xid in order:
        xid_runs, xid_missing = _xid_run_configs(xid, by_xid[xid])
        runs.extend(xid_runs)
        missing.extend(xid_missing)
    return runs, missing


def _xids_from_scope(scope):
    xids = [str(xid) for xid in scope.get("xids", [])]
    if not xids:
        xids = sorted({xid for wd in dir_names(BASEDIR) if (xid := extract_xid(wd))}, reverse=True)

    after = _scope_datetime(scope, "created_after")
    before = _scope_datetime(scope, "created_before")
    result = []
    for xid in xids:
        if not _xid_re.fullmatch(xid):
            continue
        dt = _xid_datetime(xid)
        if (after and dt < after) or (before and dt > before):
            continue
        result.append(xid)
    return result


def _launch_args(meta):
    if not isinstance(meta, dict):
        return []
    return [*meta.get("overrides", []), *meta.get("args", [])]


TRAIN_TARGET_METRICS = {
    "nsteps": "step",
    "nexamples": "chrono/examples_seen",
    "nmodeltokens": "chrono/model_tokens_seen",
    "ndatatokens": "chrono/data_tokens_seen",
    "nlosstokens": "chrono/loss_tokens_seen",
}


def _int_arg(value):
    value = str(value).strip().strip('"\'')
    return int(value.replace("_", "")) if re.fullmatch(r'\d[\d_]*', value) else None


def _target_from_args(args, name):
    for arg in reversed(args):
        if not re.search(rf'(^|[.]){name}:?=', arg):
            continue
        value = re.split(r':?=', arg, 1)[1].strip('"\'')
        return _int_arg(value)
    return None


def _train_target(config, args=()):
    targets = []
    for name, metric in TRAIN_TARGET_METRICS.items():
        value = config.get(name)
        if value is None:
            value = _target_from_args(args, name)
        elif isinstance(value, str):
            value = _int_arg(value)
        if value is not None:
            targets.append({"name": name, "value": value, "metric": metric})
    return targets[0] if len(targets) == 1 else None


def _nsteps_from_args(args):
    return _target_from_args(args, "nsteps")


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
            start_estimate_raw = _job_start_time_raw(job) if status == "PENDING" else ""
        else:
            jid = normalize_jid(config.get("jid")) or normalize_jid(meta.get("launchjid") if isinstance(meta, dict) else None)
            exit_status = config.get("exit_status")
            status = detail_status_from_exit_status(exit_status)
            if status is None:
                status = "UNKNOWN" if exit_status_is_wip(exit_status) else ("FAILED" if wuwd_name else "UNKNOWN")
            reason = ""
            restarts = 0
            runtime = "n/a"
            start_estimate_raw = ""

        target = _train_target(config, args)
        wus.append({
            "wid": wid,
            "jid": jid,
            "restarts": restarts,
            "exit_code": None,
            "status": status,
            "exit_status": config.get("exit_status"),
            "reason": reason,
            "nsteps": config.get("nsteps") or _nsteps_from_args(args),
            "target": target,
            "config_args": args,
            "name": config.get("name", meta.get("name", "") if isinstance(meta, dict) else ""),
            "qwait": "n/a",
            "qwait_estimate_ts": _parse_slurm_start_time(start_estimate_raw),
            "qwait_estimate_raw": start_estimate_raw,
            "qwait_estimate_reason": reason if status == "PENDING" else "",
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


@app.post("/api/sacct")
def get_sacct_info(payload=Body(...)):
    """Get accounting info for known job IDs."""
    t0 = time.time()
    if isinstance(payload, dict):
        jids = payload.get("jids", [])
        xid = payload.get("xid")
        start_time = _xid_start_time(xid) if xid and _xid_re.fullmatch(xid) else None
    else:
        jids = payload
        start_time = None
    detail = load_sacct_detail_compact(jids, start_time=start_time)
    log.info("POST /api/sacct - done: %d requested, %d resolved (%.2fs)",
             len(jids), len(detail), time.time() - t0)
    return {str(jid): info for jid, info in detail.items()}


@app.post("/api/sacct/states")
def get_sacct_states(payload=Body(...)):
    """Get canonical overview states for known job IDs."""
    t0 = time.time()
    if isinstance(payload, list):
        states = load_sacct_states_many(payload)
        log.info("POST /api/sacct/states - done: %d jids legacy payload (%.2fs)",
                 len(payload), time.time() - t0)
        return {str(jid): state for jid, state in states.items()}

    states = {}
    missing_by_xid = {}
    n_jids = 0
    t_xids = time.time()
    for xid, jids in payload.items():
        wanted = {normalize_jid(jid) for jid in jids}
        wanted.discard(None)
        n_jids += len(wanted)
        t_xid = time.time()
        xid_states = load_sacct_states_xid(xid) if _xid_re.fullmatch(xid) else {}
        xid_hits = 0
        xid_missing = set()
        for jid in wanted:
            if jid in xid_states:
                states[jid] = xid_states[jid]
                xid_hits += 1
            else:
                xid_missing.add(jid)
        if xid_missing:
            missing_by_xid[xid] = xid_missing
        log.debug(
            "  - sacct states %s: requested %d, name hits %d, fallback %d (%.2fs)%s",
            xid, len(wanted), xid_hits, len(xid_missing), time.time() - t_xid,
            f", misses {sorted(xid_missing)[:5]}" if xid_missing else "",
        )
    xid_time = time.time() - t_xids

    t_fallback = time.time()
    fallback_requested = 0
    fallback_resolved = 0
    for xid, missing_jids in missing_by_xid.items():
        t_xid = time.time()
        fallback_requested += len(missing_jids)
        start_time = _xid_start_time(xid) if _xid_re.fullmatch(xid) else None
        fallback_states = load_sacct_states_many(missing_jids, start_time=start_time)
        fallback_resolved += len(fallback_states)
        for jid, state in fallback_states.items():
            states[jid] = state
        log.debug(
            "  - sacct fallback %s: requested %d, resolved %d (%.2fs)",
            xid, len(missing_jids), len(fallback_states), time.time() - t_xid,
        )
    fallback_time = time.time() - t_fallback
    log.info(
        "POST /api/sacct/states - done: %d xids, %d jids, %d resolved, %d fallback requested, %d fallback resolved (xids %.2fs, fallback %.2fs, total %.2fs)",
        len(payload), n_jids, len(states), fallback_requested, fallback_resolved,
        xid_time, fallback_time, time.time() - t0,
    )
    return {str(jid): state for jid, state in states.items()}


@app.post("/api/runs/what_varies")
def runs_what_varies(payload=Body(...)):
    """Return flattened config leaves that vary over a set of XIDs/WIDs."""
    t0 = time.time()
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Expected object payload")
    specs = payload.get("runs", [])
    if not specs:
        raise HTTPException(status_code=400, detail="Expected non-empty runs list")

    runs, missing = _runs_from_specs(specs)
    flat_configs = [_flatten_config(run["config"]) for run in runs]
    config = {}
    for path in sorted({path for flat in flat_configs for path in flat}):
        counts = {}
        for flat in flat_configs:
            value = flat[path] if path in flat else MISSING
            key = "__missing__" if value is MISSING else _json_key(value)
            counts.setdefault(key, {"value": value, "count": 0})
            counts[key]["count"] += 1
        if len(counts) <= 1:
            continue
        values = []
        for item in counts.values():
            if item["value"] is MISSING:
                values.append({"missing": True, "count": item["count"]})
            else:
                values.append({"value": item["value"], "count": item["count"]})
        config[path] = values

    log.info("POST /api/runs/what_varies - done: %d runs, %d varying paths (%.2fs)",
             len(runs), len(config), time.time() - t0)
    return {
        "runs_total": len(runs) + len(missing),
        "runs_used": len(runs),
        "missing_runs": missing,
        "config": config,
    }


@app.post("/api/runs/query")
def runs_query(payload=Body(...)):
    """Find runs on this smanager backend by config predicates."""
    t0 = time.time()
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Expected object payload")

    scope = payload.get("scope") or {}
    if not isinstance(scope, dict):
        raise HTTPException(status_code=400, detail="Expected object scope")
    query = payload.get("config_where") or {}
    if not isinstance(query, dict):
        raise HTTPException(status_code=400, detail="Expected object config_where")

    limit = scope.get("limit")
    if limit is not None:
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail=f"Invalid scope.limit: {limit}")
        if limit < 1:
            raise HTTPException(status_code=400, detail="scope.limit must be positive")
    wanted_statuses = {str(status).lower() for status in payload.get("status", [])}

    matched, missing = [], []
    scanned = 0
    source_batches = [_runs_from_specs(scope["runs"])] if "runs" in scope else (
        _xid_run_configs(xid) for xid in _xids_from_scope(scope)
    )
    for source_runs, source_missing in source_batches:
        missing.extend(source_missing)
        for run in source_runs:
            scanned += 1
            statuses = {str(run["config"].get("exit_status", "")).lower(), _run_status(run).lower()}
            if wanted_statuses and not (statuses & wanted_statuses):
                continue
            if all(_matches_predicate(_config_value(run["config"], path), pred) for path, pred in query.items()):
                matched.append(_public_run(run))
                if limit is not None and len(matched) >= limit:
                    break
        if limit is not None and len(matched) >= limit:
            break

    log.info("POST /api/runs/query - done: %d matched, %d scanned (%.2fs)",
             len(matched), scanned, time.time() - t0)
    return {
        "runs": matched,
        "runs_scanned": scanned,
        "runs_matched": len(matched),
        "missing_runs": missing,
    }


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
        config["_wuwd_name"] = wuwd_name

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

    # Determine status for each wid
    status = {}
    for wid, config in configs.items():
        jid = normalize_jid(config.get("jid"))
        slurm_state = jobs_by_jid.get(str(jid), {}).get("STATE") if jid else None
        exit_status = config.get("exit_status")
        config_state = detail_status_from_exit_status(exit_status)

        if jid and str(jid) in jobs_by_jid:
            status[wid] = "DONE_ISH" if config_state == "DONE" and slurm_state == "RUNNING" else slurm_state or "UNKNOWN"
        elif config_state is not None:
            status[wid] = config_state
        elif jid and jid in saccts:
            status[wid] = state_from_sacct(saccts[jid])
        elif exit_status_is_wip(exit_status):
            status[wid] = "UNKNOWN"
        else:
            status[wid] = "FAILED"

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
        jid = normalize_jid(config.get("jid")) or None
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

        target = _train_target(config, sws_args)
        wus.append({
            "wid": wid,
            "jid": jid,
            "restarts": sacct.get("restart_cnt", 0),
            "exit_code": extract_exit_code(sacct),
            "status": status.get(wid, "UNKNOWN"),
            "exit_status": config.get("exit_status"),
            "reason": jobs_by_jid.get(str(jid), {}).get("REASON", ""),
            "nsteps": config.get("nsteps"),
            "target": target,
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
def get_xid_metrics(xid: str, metric: str = "train/loss", eta_wids=None):
    """Get metrics for all WUs of an XID (separate from main xid-detail for lazy loading)."""
    t0 = time.time()
    log.info("GET /api/xid/%s/metrics (metric=%s)", xid, metric)
    wd_path = _find_xid_path(xid)
    if not wd_path:
        raise HTTPException(status_code=404, detail=f"XID {xid} not found")
    eta_wids = parse_wids_arg(eta_wids)

    launchids = load_launchids(wd_path)
    if launchids:
        workdirs, _ = _workdirs_by_suffix(wd_path)
        config_futs = [fs_executor.submit(_load_config_only, (wd_path, wd)) for wd in workdirs.values()]
        config_by_wuwd = dict(f.result() for f in config_futs)
        targets = {}
        metric_futs = []
        for wid, wuwd_name in workdirs.items():
            target = _train_target(config_by_wuwd.get(wuwd_name, {}), _launch_args(launchids.get(str(wid), {})))
            targets[wuwd_name] = target
            extra_metrics = (target["metric"],) if target and target["metric"] != "step" else ()
            metric_futs.append(fs_executor.submit(_load_metric_only, (
                wd_path, wuwd_name, metric, extra_metrics, eta_wids is None or wid in eta_wids)))
        metric_by_wuwd = dict(f.result() for f in metric_futs)
        result = {}
        for wid, wuwd_name in workdirs.items():
            result[wid] = metric_by_wuwd.get(wuwd_name, {})
            if target := targets.get(wuwd_name):
                result[wid]["_target_name"] = target["name"]
                result[wid]["_target_value"] = target["value"]
                result[wid]["_target_metric"] = target["metric"]
                if target["name"] == "nsteps":
                    result[wid]["_nsteps"] = target["value"]
        log.info("GET /api/xid/%s/metrics - done: %d launch-indexed WUs (%.2fs)", xid, len(result), time.time() - t0)
        return result

    # LEGACY - BACKFILLED - REMOVE SOON: config scan for experiments without
    # launchids.json. Backfilled by bv2/tools/backfill_launchids_json.
    _legacy_used("xid_metrics.no_launchids", xid=xid)

    # Get workdirs
    workdirs = dir_names(wd_path)

    # Load configs first so we know which progress metric to include.
    config_futs = [fs_executor.submit(_load_config_only, (wd_path, wd)) for wd in workdirs]
    config_results = [f.result() for f in config_futs]
    targets = {wuwd_name: _train_target(config) for wuwd_name, config in config_results}
    metric_futs = []
    for wuwd_name, config in config_results:
        target = targets[wuwd_name]
        extra_metrics = (target["metric"],) if target and target["metric"] != "step" else ()
        wid = normalize_wid(config.get("wid", wuwd_name))
        metric_futs.append(fs_executor.submit(_load_metric_only, (
            wd_path, wuwd_name, metric, extra_metrics, eta_wids is None or wid in eta_wids)))
    metric_results = [f.result() for f in metric_futs]

    # Build wid -> metrics mapping (same duplicate resolution as main endpoint)
    metric_by_wuwd = {wuwd_name: metrics for wuwd_name, metrics in metric_results}
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
        if target := targets.get(wuwd_name):
            result[wid]["_target_name"] = target["name"]
            result[wid]["_target_value"] = target["value"]
            result[wid]["_target_metric"] = target["metric"]
            if target["name"] == "nsteps":
                result[wid]["_nsteps"] = target["value"]

    log.info("GET /api/xid/%s/metrics - done: %d WUs (%.2fs)", xid, len(result), time.time() - t0)
    return result


@app.get("/api/xid/{xid}/eta_breakdown")
def get_xid_eta_breakdown(xid, eta_wids=""):
    t0 = time.time()
    log.info("GET /api/xid/%s/eta_breakdown", xid)
    wd_path = _find_xid_path(xid)
    if not wd_path:
        raise HTTPException(status_code=404, detail=f"XID {xid} not found")

    eta_wids = parse_wids_arg(eta_wids) or set()
    if not eta_wids:
        return {}

    workdirs, _ = _workdirs_by_suffix(wd_path)
    futs = [
        (wid, fs_executor.submit(_load_eta_breakdown_only, (wd_path, wuwd_name)))
        for wid, wuwd_name in workdirs.items()
        if wid in eta_wids
    ]
    result = {wid: fut.result()[1] for wid, fut in futs}
    log.info("GET /api/xid/%s/eta_breakdown - done: %d WUs (%.2fs)", xid, len(result), time.time() - t0)
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
    job = _load_current_job(jid)
    result = subprocess.run(["scancel", str(jid)], capture_output=True, text=True)
    if result.returncode != 0:
        log.error("scancel failed: %s", result.stderr)
        raise HTTPException(status_code=500, detail=f"scancel failed: {result.stderr}")
    marked = _mark_stopped_for_job(job)
    log.info("scancel %s succeeded; marked %d non-running WU(s) stopped", jid, len(marked))
    return {"status": "ok", "jid": jid, "marked_stopped": marked}


@app.post("/api/action/stop_xid/{xid}")
def stop_xid(xid: str):
    """Stop all jobs for an XID using scancel -n."""
    if not ACTIONS_ENABLED:
        raise HTTPException(status_code=403, detail="Actions are disabled (--no-actions)")
    log.info("POST /api/action/stop_xid/%s", xid)
    jobs = _load_current_xid_jobs(xid)
    result = subprocess.run(["scancel", "-n", xid], capture_output=True, text=True)
    if result.returncode != 0:
        log.error("scancel -n %s failed: %s", xid, result.stderr)
        raise HTTPException(status_code=500, detail=f"scancel failed: {result.stderr}")
    marked = _mark_stopped_for_jobs(xid, jobs)
    log.info("scancel -n %s succeeded; marked %d non-running WU(s) stopped", xid, len(marked))
    return {"status": "ok", "xid": xid, "marked_stopped": marked}


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


_remote_launch_tmp_re = re.compile(r"^/tmp/remote_launch_[A-Za-z0-9]+$")


def _remote_launch_tmp(src_text):
    src = Path(src_text).resolve()
    if src.name != "src" or not _remote_launch_tmp_re.fullmatch(str(src.parent)):
        return None
    return src.parent


def _cleanup_launch_tmp(tmp):
    tmp = Path(tmp).resolve()
    if not _remote_launch_tmp_re.fullmatch(str(tmp)):
        raise RuntimeError(f"Refusing to delete launch tmp {tmp}")
    try:
        shutil.rmtree(tmp)
    except FileNotFoundError:
        pass


def _validate_launch_src(src_text, kind):
    tmp = _remote_launch_tmp(src_text)
    if tmp is None:
        raise HTTPException(status_code=400, detail="Launch source must be /tmp/remote_launch_*/src")
    src = tmp / "src"
    script = src / f"bv2/tools/launch_{kind}"
    if not script.is_file():
        raise HTTPException(status_code=400, detail=f"Launch source does not contain bv2/tools/launch_{kind}")
    script.chmod(0o755)
    return src


_launch_procs = {}
_launch_lock = threading.Lock()


def _signal_launch(proc, sig):
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        pass


def _stream_launch(tmp, src, kind, argv):
    proc = None
    key = str(tmp)
    try:
        with _launch_lock:
            proc = subprocess.Popen(
                [f"bv2/tools/launch_{kind}", *argv],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=src, bufsize=0, start_new_session=True)
            _launch_procs[key] = proc
        while chunk := proc.stdout.read(4096):
            yield chunk
        ret = proc.wait()
        if ret:
            code = 128 + abs(ret) if ret < 0 else ret
            log.error("launch_%s failed with return code %s", kind, ret)
            yield f"\nRemote launch failed with return code {code}\n".encode()
        else:
            log.info("launch_%s succeeded", kind)
    finally:
        if proc:
            with _launch_lock:
                if _launch_procs.get(key) is proc:
                    del _launch_procs[key]
            if proc.poll() is None:
                _signal_launch(proc, signal.SIGINT)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    _signal_launch(proc, signal.SIGKILL)
                    proc.wait()
        _cleanup_launch_tmp(tmp)


async def _launch(request, kind):
    if not ACTIONS_ENABLED:
        raise HTTPException(status_code=403, detail="Actions are disabled (--no-actions)")
    if request.headers.get("origin"):
        raise HTTPException(status_code=403, detail="Launch actions are CLI-only")

    encoded_args = request.headers.get("x-launch-args-b64", "")
    if not encoded_args:
        raise HTTPException(status_code=400, detail="Missing x-launch-args-b64 header")
    encoded_src = request.headers.get("x-launch-src-b64", "")
    if not encoded_src:
        raise HTTPException(status_code=400, detail="Missing x-launch-src-b64 header")
    try:
        argv = [base64.b64decode(arg, validate=True).decode("utf-8") for arg in encoded_args.split(",") if arg]
        src_text = base64.b64decode(encoded_src, validate=True).decode("utf-8")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid launch request: {e}")
    if not argv:
        raise HTTPException(status_code=400, detail="No launch arguments provided")

    log.info("POST /api/action/launch_%s argv=%s", kind, shlex.join(argv))
    try:
        src = _validate_launch_src(src_text, kind)
    except Exception:
        if tmp := _remote_launch_tmp(src_text):
            _cleanup_launch_tmp(tmp)
        raise

    return StreamingResponse(_stream_launch(src.parent, src, kind, argv), media_type="text/plain")


@app.post("/api/action/launch_interrupt")
async def launch_interrupt(request: Request):
    if not ACTIONS_ENABLED:
        raise HTTPException(status_code=403, detail="Actions are disabled (--no-actions)")
    if request.headers.get("origin"):
        raise HTTPException(status_code=403, detail="Launch actions are CLI-only")
    try:
        src_text = base64.b64decode(request.headers.get("x-launch-src-b64", ""), validate=True).decode("utf-8")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid launch request: {e}")
    tmp = _remote_launch_tmp(src_text)
    if tmp is None:
        raise HTTPException(status_code=400, detail="Launch source must be /tmp/remote_launch_*/src")
    with _launch_lock:
        proc = _launch_procs.get(str(tmp))
    if proc and proc.poll() is None:
        log.info("POST /api/action/launch_interrupt tmp=%s", tmp)
        _signal_launch(proc, signal.SIGINT)
    else:
        _cleanup_launch_tmp(tmp)
    return {"status": "ok"}


@app.post("/api/action/launch_slurm")
async def launch_slurm(request: Request):
    """Launch a new Slurm experiment from an existing source directory."""
    return await _launch(request, "slurm")


@app.post("/api/action/launch_local")
async def launch_local(request: Request):
    """Launch a local GPU run from an existing source directory."""
    return await _launch(request, "local")


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
    global PREFS_DIR, ACTIONS_ENABLED, ARCHIVE_DIR, REPORT_ROOT_DIR
    import argparse
    parser = argparse.ArgumentParser(description="sManager - Slurm job management web UI")
    parser.add_argument("--version", action="version", version=f"smanager {__version__}")
    parser.add_argument("--port", type=int, default=2337)
    parser.add_argument("--prefs-dir", help=f"Directory for preferences files (default: /checkpoint/rigi/USER)")
    parser.add_argument("--no-actions", action="store_true", help="Disable all action endpoints (stop, resume, delete)")
    parser.add_argument("--archive-dir", help="Directory where archived XIDs are moved to")
    parser.add_argument("--report-root", help="Directory where uploaded reports are stored")
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
    if args.report_root:
        REPORT_ROOT_DIR = Path(args.report_root)
        log.info("Report root directory: %s", REPORT_ROOT_DIR)
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
