import torch
import torch.distributed as distr


def all_gather(tensor, world_size=None):
    world_size = world_size or distr.get_world_size()
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    distr.all_gather(tensor_list, tensor)
    return tensor_list


def all_gather_object(obj, world_size=None):
    world_size = world_size or distr.get_world_size()
    all_objs = [None] * world_size
    distr.all_gather_object(all_objs, obj)
    return all_objs


def gather_object_to(rank, obj, world_size=None, my_rank=None):
    world_size = world_size or distr.get_world_size()
    my_rank = my_rank if my_rank is not None else distr.get_rank()
    all_objs = [None] * world_size if my_rank == rank else None
    distr.gather_object(obj, all_objs, dst=rank)
    return all_objs if my_rank == rank else None


def broadcast_object_from(rank, obj, world_size=None, my_rank=None):
    world_size = world_size or distr.get_world_size()
    my_rank = my_rank if my_rank is not None else distr.get_rank()
    distr.barrier()  # Not really sure why we need the barrier, but it fails without.
    objlist = [obj] if my_rank == rank else [None]
    distr.broadcast_object_list(objlist, src=rank)
    return objlist[0]
