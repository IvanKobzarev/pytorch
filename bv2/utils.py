import atexit
import hashlib
import json
import os
import re
import signal
import sys
import warnings
from contextlib import ContextDecorator
from datetime import datetime
from functools import cache, wraps
from itertools import count as icount
from threading import Thread, local
from time import perf_counter
from types import FunctionType

import numpy as np
import torch
import torch.distributed as distr

# ANSI escape codes, but not in logfiles.
RED = '\033[31m' if sys.stdout.isatty() else ''
GREEN = '\033[32m' if sys.stdout.isatty() else ''
YELLOW = '\033[33m' if sys.stdout.isatty() else ''
BLUE = '\033[34m' if sys.stdout.isatty() else ''
BOLD = '\033[1m' if sys.stdout.isatty() else ''
RESET = '\033[0m' if sys.stdout.isatty() else ''
LIGHT = '\033[90m' if sys.stdout.isatty() else ''


def count(start, *, end=None, step=1):
    if end is None:
        yield from icount(start, step)
    else:
        yield from range(start, end, step)


def thread_local_cache(f):
    """Like @functools.cache, but has a separate cache per thread."""
    tls = local()
    kwd_mark = object()

    @wraps(f)
    def g(*args, **kwargs):
        cache = getattr(tls, "cache", None)
        if cache is None:
            tls.cache = cache = {}
        # The `kwd_mark` separator avoid cache key collision corner-case.
        k = args if not kwargs else args + (kwd_mark, *kwargs.items())
        if k not in cache:
            cache[k] = f(*args, **kwargs)
        return cache[k]

    def cache_clear():
        if hasattr(tls, "cache"):
            tls.cache.clear()

    g.cache_clear = cache_clear
    return g


#  _   _       _ _
# | | | |_ __ (_) |_ ___
# | | | | '_ \| | __/ __|
# | |_| | | | | | |_\__ \
#  \___/|_| |_|_|\__|___/
#


UNIT_SUFFIXES = {
    "steps": "nsteps",
    "examples": "nexamples",
    "modeltoks": "nmodeltokens",
    "datatoks": "ndatatokens",
    "losstoks": "nlosstokens",
    "modeltokens": "nmodeltokens",
    "datatokens": "ndatatokens",
    "losstokens": "nlosstokens",
}


def schedule(c, prefix, name="schedule", default=None, required=True, none_disables=False, delay_first=False):
    fields = [f"{prefix}{suffix}" for suffix in UNIT_SUFFIXES]
    present = [field for field in fields if field in c]
    values = [
        (f"{prefix}{suffix}", unit, c.get(f"{prefix}{suffix}"))
        for suffix, unit in UNIT_SUFFIXES.items()
        if c.get(f"{prefix}{suffix}") is not None
    ]
    if not values:
        if none_disables and present:
            values = []
        elif default is not None:
            values = [(f"{prefix}steps", UNIT_SUFFIXES["steps"], default)]

    if len(values) != 1:
        if not values and not required:
            return None
        got = ", ".join(field for field, _, _ in values) or "none"
        raise ValueError(f"{name} needs exactly one unit field; got {got}")

    field, unit, spec = values[0]
    if not isinstance(spec, int) or spec <= 0:
        raise ValueError(f"{field} must be a positive integer frequency, got {spec}")
    return {"field": field, "unit": unit, "spec": spec, "delay_first": delay_first}


def crossed(spec, before, after):
    """Check whether going from `before` to `after` crosses any point defined in `spec`,
       an integer meaning a frequency (eg 500 means "every 500").
    """
    if not isinstance(spec, int) or spec <= 0:
        raise ValueError(f"schedule frequency must be a positive integer, got {spec}")
    return before // spec < after // spec


class TrainingProgress:
    def __init__(self, step=0, examples_seen=0, data_tokens_seen=0, model_tokens_seen=0, loss_tokens_seen=0):
        self.initial = self.before = self.after = {
            "nsteps": step,
            "nexamples": examples_seen,
            "nmodeltokens": model_tokens_seen,
            "ndatatokens": data_tokens_seen,
            "nlosstokens": loss_tokens_seen,
        }

    @classmethod
    def from_dicts(cls, before, after):
        progress = cls()
        progress.initial = progress.before = before
        progress.after = after
        return progress

    def advance(self, examples=0, data_tokens=0, model_tokens=0, loss_tokens=0):
        self.before = self.after
        self.after = {
            "nsteps": self.before["nsteps"] + 1,
            "nexamples": self.before["nexamples"] + examples,
            "nmodeltokens": self.before["nmodeltokens"] + model_tokens,
            "ndatatokens": self.before["ndatatokens"] + data_tokens,
            "nlosstokens": self.before["nlosstokens"] + loss_tokens,
        }
        return self

    def __getitem__(self, unit):
        return self.after[unit]

    def crossed(self, schedule):
        if not schedule:
            return False
        did_cross = crossed(schedule["spec"], self.before[schedule["unit"]], self.after[schedule["unit"]])
        if not schedule["delay_first"]:
            return did_cross
        # Let step 1 finish before running long side work, then replay that event at step 2.
        if self.after["nsteps"] == 1:
            return False
        if self.after["nsteps"] == 2 and crossed(schedule["spec"], self.initial[schedule["unit"]], self.before[schedule["unit"]]):
            return True
        return did_cross

    def reached(self, schedule):
        return self[schedule["unit"]] >= schedule["spec"]

#    ____
#   / ___|___  _ __ ___  _ __ ___  ___
#  | |   / _ \| '_ ` _ \| '_ ` _ \/ __|
#  | |__| (_) | | | | | | | | | | \__ \
#   \____\___/|_| |_| |_|_| |_| |_|___/


def global_gpu_barrier(device):
    # all_reduce is a collective (=barrier) on the NCCL stream, which PyTorch
    # auto-syncs with the current stream. .item() then forces CPU-GPU sync.
    _t = torch.zeros(1, device=device)
    distr.all_reduce(_t, op=distr.ReduceOp.SUM)
    _t.item()
    # The alternative is: torch.cuda.synchronize() ; distr.barrier()


def all_reduce_scalars(*scalars, op=distr.ReduceOp.SUM):
    device = next((x.device for x in scalars if isinstance(x, torch.Tensor) and x.is_cuda), None)
    device = device or next((x.device for x in scalars if isinstance(x, torch.Tensor)), None)
    scalars = torch.stack([torch.as_tensor(x, device=device) for x in scalars])  # Also clones, so none gets overwritten.
    distr.all_reduce(scalars, op=op)
    return tuple(s.item() for s in scalars)


@cache
def gloo_group():
    return distr.new_group(backend="gloo")


def all_gather(tensor, world_size=None):
    world_size = world_size or distr.get_world_size()
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    distr.all_gather(tensor_list, tensor)
    return tensor_list


def all_gather_object(obj, world_size=None):
    world_size = world_size or distr.get_world_size()
    all_objs = [None] * world_size
    distr.all_gather_object(all_objs, obj, group=gloo_group())
    return all_objs


def gather_object_to(rank, obj, world_size=None, my_rank=None):
    world_size = world_size or distr.get_world_size()
    my_rank = my_rank if my_rank is not None else distr.get_rank()
    all_objs = [None] * world_size if my_rank == rank else None
    distr.gather_object(obj, all_objs, dst=rank, group=gloo_group())
    return all_objs if my_rank == rank else None


def printR(s, *, stamp=True, my_rank=None, world_size=None, **kw):
    """Collective print: every rank contributes one line, rank0 prints them."""
    my_rank = distr.get_rank() if my_rank is None else my_rank
    if stamp:
        t = datetime.now().time().isoformat(timespec="milliseconds")
        s = f"[{my_rank} {t}] {s}"
    if ss := gather_object_to(0, s, world_size=world_size, my_rank=my_rank):
        print(*ss, sep='\n', **kw)


def sum_to(rank, *, world_size=None, my_rank=None, **objs):
    if all_objs := gather_object_to(rank, objs, world_size=world_size, my_rank=my_rank):
        return {k: sum(o[k] for o in all_objs) for k in objs}


def broadcast_object_from(rank, obj, world_size=None, my_rank=None):
    world_size = world_size or distr.get_world_size()
    my_rank = my_rank if my_rank is not None else distr.get_rank()
    distr.barrier()  # Not really sure why we need the barrier, but it fails without.
    objlist = [obj] if my_rank == rank else [None]
    distr.broadcast_object_list(objlist, src=rank, group=gloo_group())
    return objlist[0]


#                                            _   _
#  _ __  _ __ ___        ___ _ __ ___  _ __ | |_(_) ___  _ __  ___
# | '_ \| '__/ _ \_____ / _ \ '_ ` _ \| '_ \| __| |/ _ \| '_ \/ __|
# | |_) | | |  __/_____|  __/ | | | | | |_) | |_| | (_) | | | \__ \
# | .__/|_|  \___|      \___|_| |_| |_| .__/ \__|_|\___/|_| |_|___/
# |_|                                 |_|
#


_ABOUT_TO_GET_KILLED = False


def install_exit_handler(config_path=None):
    global _ABOUT_TO_GET_KILLED
    _ABOUT_TO_GET_KILLED = False
    exit_status = None
    had_exception = False

    def write_exit_status(status):
        nonlocal exit_status
        exit_status = status
        if config_path:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            config["exit_status"] = status
            config["exit_status_at"] = datetime.now().isoformat(timespec="seconds")
            nfs_safe_overwrite(config_path, json.dumps(config, indent=0) + "\n")
        if status == "done":
            remove_exit_handler()

    def handler(signum, frame):
        global _ABOUT_TO_GET_KILLED
        _ABOUT_TO_GET_KILLED = _ABOUT_TO_GET_KILLED or perf_counter()
        if config_path and exit_status is None:
            write_exit_status(("preempted" if signum == signal.SIGUSR2 else "stopped") + " (wip)")
        print(f"Got termination signal {signum}, checkpointing and quitting ASAP! ({_ABOUT_TO_GET_KILLED})")

    def finalize_exit_status():
        if exit_status is None:
            write_exit_status("error" if had_exception else "done")
        elif exit_status.endswith(" (wip)"):
            write_exit_status("error" if had_exception else exit_status[:-6])

    old_excepthook = sys.excepthook
    signals = (signal.SIGUSR2, signal.SIGTERM)
    old_signal_handlers = {s: signal.getsignal(s) for s in signals}

    def excepthook(*args):
        nonlocal had_exception
        had_exception = True
        old_excepthook(*args)

    def remove_exit_handler():
        for s, old_handler in old_signal_handlers.items():
            signal.signal(s, old_handler)
        sys.excepthook = old_excepthook
        if config_path:
            atexit.unregister(finalize_exit_status)

    for s in signals:
        signal.signal(s, handler)
    sys.excepthook = excepthook
    if config_path:
        atexit.register(finalize_exit_status)
    return write_exit_status


def about_to_get_killed():
    return _ABOUT_TO_GET_KILLED


#   ____                 _                   _   _                 _
#  |  _ \ __ _ _ __   __| | ___  _ __ ___   | \ | |_   _ _ __ ___ | |__   ___ _ __ ___
#  | |_) / _` | '_ \ / _` |/ _ \| '_ ` _ \  |  \| | | | | '_ ` _ \| '_ \ / _ \ '__/ __|
#  |  _ < (_| | | | | (_| | (_) | | | | | | | |\  | |_| | | | | | | |_) |  __/ |  \__ \
#  |_| \_\__,_|_| |_|\__,_|\___/|_| |_| |_| |_| \_|\__,_|_| |_| |_|_.__/ \___|_|  |___/
#


def shash(s, nbytes=8, signed=False):
    digest = hashlib.blake2b(s.encode("utf-8"), digest_size=nbytes).digest()
    return int.from_bytes(digest, signed=signed)


def _to_s_for_seeds(x):  # It's faster if this function is outer, not inner to `def seeds`
    if isinstance(x, (list, tuple)):
        # Combining seeds by str.join then hash is much faster than hash and then merge. I tested.
        # Use "string terminator" codepoint, insanely unlikely to be used and very short (2bytes)
        return '\u009c'.join(map(_to_s_for_seeds, x))
    if isinstance(x, str):
        return x
    if isinstance(x, (int, np.integer)):
        return str(x)
    raise ValueError(f"Seed leaves can only be str or ints, got: {x} ({type(x)})")


@torch.compiler.assume_constant_result
def seeds(*seedz, nbytes=8, signed=False):
    return shash(_to_s_for_seeds(seedz), nbytes=nbytes, signed=signed)


def rng(*seedz):
    return np.random.default_rng(seeds(*seedz))  # noqa:TID251


def rng_torch(*seedz, device="cpu"):  # Same default device as PyTorch API.
    return torch.Generator(device=device).manual_seed(seeds(*seedz))  # noqa:TID251


_can_torch = {
    np.float32, np.float64, np.float16,
    np.int8, np.int16, np.int32, np.int64,
    np.uint8, np.bool_, np.complex64, np.complex128,
}

def to_gpu(seq, device):
    """Move a dict of numpy/torch/BlockMask values to device.

    Handles BlockMask closure tensors and _dynamo_dynamic_indices preservation."""
    from torch.nn.attention.flex_attention import BlockMask

    def maybe_to_gpu(x):
        if isinstance(x, dict):
            return {k: maybe_to_gpu(v) for k, v in x.items()}
        if isinstance(x, BlockMask):
            from flexmaskli import blockmask_to_gpu
            return blockmask_to_gpu(x, device)
        if isinstance(x, np.ndarray) and any(x.dtype == t for t in _can_torch):
            x = torch.from_numpy(x)
        if isinstance(x, torch.Tensor):
            return x.pin_memory().to(device=device, non_blocking=True)
        return x

    return {k: maybe_to_gpu(v) for k, v in seq.items()}


#              _             _          __  __
#  _   _  __ _| |_   _   ___| |_ _   _ / _|/ _|
# | | | |/ _` | | | | | / __| __| | | | |_| |_
# | |_| | (_| | | |_| | \__ \ |_| |_| |  _|  _|
#  \__,_|\__, |_|\__, | |___/\__|\__,_|_| |_|
#        |___/   |___/


def install_torch_trace(rank, workdir):
    # Hacky way of always-enabling TORCH_TRACE from now on. No run overhead, only compile.
    if rank == 0:  # Big jobs have >700MB per rank, so do rank0 only.
        h = torch._logging._internal.LOG_TRACE_HANDLER
        h.root_dir = os.path.join(workdir, "torch_trace")

        # Now we also monkey-patch the handler to change file permission after it's created,
        # because by default it's o600 which doesn't even inherit parent folder's g+rw.
        if not hasattr(h, "_group_perm_patch"):
            old_emit = h.emit

            def emit(record):
                old_emit(record)
                if h.stream is not None and h.stream.name != getattr(h, "_group_perm_last", None):
                    os.chmod(h.stream.name, 0o660)
                    h._group_perm_last = h.stream.name

            h.emit = emit
            h._group_perm_patch = True
            h._group_perm_last = None


# Can be used both as function annotator, and as with-context.
class suppress_warnings(ContextDecorator):
    def __init__(self, message, category=Warning, regex=False):
        self.category = category
        self.pattern = message if regex else f".*{re.escape(message)}.*"
        self._ctx = None  # will hold the catch_warnings context manager
    def __enter__(self):
        # Isolate warning filter changes to this scope
        self._ctx = warnings.catch_warnings()
        self._ctx.__enter__()
        warnings.filterwarnings("ignore", message=self.pattern, category=self.category)
        return self
    def __exit__(self, exc_type, exc, tb):
        # Restore previous warnings state
        return self._ctx.__exit__(exc_type, exc, tb)


_FILTER_STDERR_INSTALLED = False


def filter_stderr(*prefixes):
    """Redirect stderr through a background thread that drops lines matching any prefix."""
    global _FILTER_STDERR_INSTALLED
    if _FILTER_STDERR_INSTALLED:
        return
    _FILTER_STDERR_INSTALLED = True

    r_fd, pipe_w_fd = os.pipe()
    restore_fd = os.dup(2)
    os.dup2(pipe_w_fd, 2)
    os.close(pipe_w_fd)
    sys.stderr = os.fdopen(2, "w", buffering=1, closefd=False)  # Line-buffered!
    r = os.fdopen(r_fd, "r", errors="replace")
    w = os.fdopen(os.dup(restore_fd), "w", errors="replace")

    def _run():
        for line in r:
            if any(line.startswith(p) for p in prefixes):
                continue
            w.write(line)
            w.flush()
        r.close()
        w.close()

    t = Thread(target=_run, daemon=True)
    t.start()

    def _flush():
        sys.stderr.flush()
        os.dup2(restore_fd, 2)
        sys.stderr = os.fdopen(2, "w", buffering=1, closefd=False)
        t.join(timeout=2)
        os.close(restore_fd)

    atexit.register(_flush)


def clone_function(f, name_suffix=""):
    """Return a copy of `f` so that it has a separate torch.compile cache."""
    g = FunctionType(
        f.__code__.replace(),
        f.__globals__,
        f.__name__ + name_suffix,
        argdefs=f.__defaults__,
        closure=f.__closure__
    )
    g.__kwdefaults__ = f.__kwdefaults__
    # g.__dict__.update(f.__dict__)  # Ignore attributes; torch dynamo adds some.
    g.__annotations__ = getattr(f, "__annotations__", {}).copy()
    g.__doc__ = f.__doc__
    g.__module__ = f.__module__
    g.__qualname__ = f.__qualname__
    return g


def nfs_safe_overwrite(path, text):
    # There's a bug in cw's NFS for some combination of doing tmp-write+rename and then later
    # doing "w+" from a different machine, causes the file to be corrupted. This works around it.
    # I hate this just as much as you.
    with os.fdopen(os.open(path, os.O_RDWR | os.O_CREAT, 0o666), "r+", encoding="utf-8") as f:
        f.seek(0)
        f.write(text)
        f.truncate()
        f.flush()
        os.fsync(f.fileno())
