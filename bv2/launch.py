#!/usr/bin/env python
"""
Example call flexing all features (while being realistic):

python bv2/launch.py bv2/config/code.py --qos h200_lowest --gpus-per-node 2 nsteps=100 'name:=f"code-test-{c.xid}-{c.wid}"'

Here's an example of defining a sweep in a config file.
The important part is to return a collection of argument sequences.


def sweep():
    for lr in [1e-4, 3e-5, 1e-5]:
        for wd in [1.0, 0.1, 0.0]:
            for nsteps in [10_000, 30_000, 100_000]:
                yield f"lr={lr}", f"wd={wd * lr}", f"nsteps={nsteps}"


This is a valid alternative:


sweep = lambda: [
    [f"lr={lr}", f"wd={wd * lr}", f"nsteps={nsteps}"]
    for lr in [1e-4, 3e-5, 1e-5]
    for wd in [1.0, 0.1, 0.0]
    for nsteps in [10_000, 30_000, 100_000]
]

If there is no sweep function in the config, it just launches the single job.
"""

import os
import re
import subprocess
import sys
import time
from datetime import datetime
from runpy import run_path


# ANSI escape codes
RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
BLUE = '\033[34m'
BOLD = '\033[1m'
RESET = '\033[0m'
LIGHT = '\033[90m'


if __name__ == "__main__":
    # First, get the sweep function out of the config file.
    conf_file = sys.argv[1]
    assert conf_file.endswith(".py"), "First argument of sweep needs to be config file."

    # The default sweep function returns a single run, with empty arg overrides:
    sweep_fn = run_path(conf_file).get("sweep", lambda: [[]])

    # Second, separate the slurm arguments from the (optional) sws override args.
    # If a lone "--" is provided, slurm args are to its left, and sws to its right.
    # Otherwise, we try to distinguish them on a best effort: slurm args are
    # --foo=bar or --foo bar, whereas sws ones are foo=bar or foo:=bar, so we can
    # distinguish them by the combination of "--" and "=".
    if "--" in sys.argv:
        slurm_args = sys.argv[2:sys.argv.index("--")]
        sws_args = sys.argv[sys.argv.index("--") + 1:]
    else:
        _is_sws = lambda a: "=" in a and not "--" in a
        slurm_args = [a for a in sys.argv[2:] if not _is_sws(a)]
        sws_args = [a for a in sys.argv[2:] if _is_sws(a)]

    xid = datetime.now().strftime('%m%d_%H%M%S')

    # Construct the common part of the launch command:
    slurm = ["sbatch", *slurm_args, "--job-name", xid, "bv2/tools/launch_fair_srun"]
    torch = ["-m", "bv2.train", "--config", conf_file]

    all_jobs = list(sweep_fn())
    njobs = len(all_jobs)

    c = GREEN if njobs <= 4 else YELLOW if njobs <= 16 else RED
    print(f"About to launch experiment {BLUE}{xid}{RESET} with {c}{BOLD}{njobs}{RESET} jobs...", flush=True)

    # Now, we actually need to copy the whole source-code folder to a folder with XID in its name.
    # The reason is that slurm doesn't checkpoint the code at launch-time. If a job from this sweep
    # later gets pre-empted and resumed, it will run whatever is in the code folder at that point,
    # which might already be very different as we continue working on the code while sweeps run!
    code_dst = f"/checkpoint/rigi/bv2/srcdirs/{xid}"
    excludes = [f"--exclude={p}" for p in (".git/", "__pycache__/")]
    print(f"Copying the code from pwd to {BLUE}{code_dst}{RESET} ...", flush=True)
    subprocess.run(["rsync", "-az", "--mkpath", "--info=progress2", *excludes, "./", code_dst], check=True)
    os.chdir(code_dst)  # This does change dir for all subsequent calls, such as slurm ones.
    for i in range(5):
        print(f"\rDone! Giving you {5-i} more seconds of grace period...", flush=True, end="")
        time.sleep(1)
    print("Let's gooooo!")

    try:
        for wid, work_unit_args in enumerate(all_jobs):
            log_xwid = f"{LIGHT}xid {RESET}{xid}{LIGHT} | wid {RESET}{wid:{len(str(njobs))}d}{LIGHT}"
            log_args = "args: " + ", ".join(f"{RESET}{BOLD}{arg}{RESET}{LIGHT}" for arg in work_unit_args) + LIGHT
            if sws_args:
                log_over = "overrides: " + ", ".join(f"{RESET}{BOLD}{arg}{RESET}{LIGHT}" for arg in sws_args) + LIGHT
                log_args = f"{log_over} | {log_args}"

            print(f"{log_xwid} | {log_args}", end="", flush=True)

            ret = subprocess.run(
                [*slurm, *torch, *work_unit_args, f"xid:=\"{xid}\"", f"wid:={wid}", *sws_args],
                capture_output=True, text=True, shell=False)

            if ret.returncode == 0:
                if match := re.search(r"Submitted batch job (\d+)", ret.stdout):
                    print(f"\r{log_xwid} | job {RESET}{match.group(1)}{LIGHT} | {log_args}", flush=True)
                else:
                    print("No JobID found in STDOUT??:", flush=True)
                    print(ret.stdout)
            else:
                print(f" | {RED}{BOLD}Error!{RESET} Return code: {ret.returncode}")
                print(f"==={RED}{BOLD}STDERR{RESET}===")
                print(ret.stderr)
                print(f"==={BLUE}{BOLD}STDOUT{RESET}===")
                print(ret.stdout)
    except KeyboardInterrupt:
        print(f"\n{RED}{BOLD}Launch interrupted. See command below to kill launched jobs.{RESET}")

    print(f"{RESET}To kill all these jobs: {BLUE}scancel -n {xid}{RESET}")
    print(f"To see status of all these jobs (triple-click to select line):\n{BLUE}squeue -n {xid}{RESET} -O JobId:7,Name:20,UserName:5,State:10,TimeUsed:9,NumCPUs:5,QOS:9,NumNodes:6,GRES:14,RestartCnt:4,Reason")
