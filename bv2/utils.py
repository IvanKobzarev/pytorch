import hashlib
from functools import cache

import numpy as np
import torch
import torch.distributed as distr


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


def hash64(s):
    digest = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, signed=False)


def rng(seeds):
    def to_nat(x):
        if isinstance(x, str):
            return hash64(x)
        return x  # Anything else bad, numpy rng will raise a clear exception.

    return np.random.default_rng([to_nat(s) for s in seeds])
