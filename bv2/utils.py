import hashlib
import warnings
from functools import cache, update_wrapper

import numpy as np
import torch
import torch.distributed as distr


def suppress_warnings(message, category=Warning):
    def decorator(func):
        def wrapper(*args, **kwargs):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=message, category=category)
                return func(*args, **kwargs)
        update_wrapper(wrapper, func)
        return wrapper

    return decorator

#    ____
#   / ___|___  _ __ ___  _ __ ___  ___
#  | |   / _ \| '_ ` _ \| '_ ` _ \/ __|
#  | |__| (_) | | | | | | | | | | \__ \
#   \____\___/|_| |_| |_|_| |_| |_|___/


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
