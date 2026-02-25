import hashlib
import re
import signal
import sys
import warnings
from contextlib import ContextDecorator
from functools import cache
from itertools import count as icount
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


def count(start, *, end=None, step=1):
    if end is None:
        yield from icount(start, step)
    else:
        yield from range(start, end, step)


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
    scalars = torch.stack([torch.as_tensor(x) for x in scalars])  # Also clones, so none gets overwritten.
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


def install_preemption_handler(signals=(signal.SIGTERM,)):
    def handler(signum, frame):
        global _ABOUT_TO_GET_KILLED
        _ABOUT_TO_GET_KILLED = perf_counter()
        print(f"[{distr.get_rank()}] Got termination signal {signum}, checkpointing and quitting ASAP! ({_ABOUT_TO_GET_KILLED})")
    for s in signals:
        signal.signal(s, handler)


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
