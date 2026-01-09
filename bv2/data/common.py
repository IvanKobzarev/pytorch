from functools import cache

import bagz
import numpy as np

import bv2.data.dpack as d
import bv2.utils as u
from bv2.data.pp import unpatchify


def random_exids(seed, n=None, start_offset=0, rank=0, world_size=1):
    if rank_n := n:  # Turn global `n` into `n` for this rank
        extra = rank < (n % world_size)
        rank_n = n // world_size + extra
    for i in u.count(start=start_offset, end=rank_n):  # Infinite if rank_n is None btw.
        yield {"exid": u.rng(seed, rank, i).integers(2**32)}, {"start_offset": i + 1}


def sharded_iota_exids(n, seed, epochs=None, start_epoch=0, start_offset=0, rank=0, world_size=1):
    split_size = n / world_size
    start = round(rank * split_size)
    end = round((rank + 1) * split_size)

    for ep in u.count(start=start_epoch, end=epochs):
        exids = u.rng(seed, ep, rank).permutation(np.arange(start, end))

        # For each exid, yielding (kwargs for make_example, kwargs for self next step)
        for i, exid in enumerate(map(int, exids[start_offset:-1])):
            yield {"exid": exid, "epoch": ep}, {"start_epoch": ep, "start_offset": start_offset + i + 1}
        yield {"exid": exids[-1], "epoch": ep}, {"start_epoch": ep + 1, "start_offset": 0}
        start_offset = 0


@cache
def get_bagz_reader(fspec, cache_limits=True):
    # NOTE1: See this file for the full definition of `fspec`:
    # https://github.com/google-deepmind/bagz/blob/main/src/file/file_system/shard_spec.h

    # NOTE2: The bagz reader (on posix) always opens all files from the spec using mmap.
    # If we wanted to only open a subset, we'd have to pass only that subset, but also
    # maybe first open all to get the total dataset length. Let's think about that only
    # when number of open files on a machine become an issue, which might be never?

    return bagz.Reader(fspec, bagz.Reader.Options(
        limits_storage=bagz.LimitsStorage.IN_MEMORY if cache_limits else bagz.LimitsStorage.ON_DISK,
        max_parallelism=1,  # We do our own prefetch, and don't read ranges.
    ))  # fmt: skip


def vis_image_text_unpack(tokens, *, ph, pw):
    txt, _, mask = d.unpack_as_text(tokens)
    txt = txt.numpy()[mask.numpy()]

    if not any(tokens[..., -1] == d.MOD_IMG):
        return txt, []

    patches, positions, _, mask = d.unpack_as_image(tokens, ph, pw)
    patches, positions = patches.numpy()[mask.numpy()], positions.numpy()[mask.numpy()]

    # Supports multiple images.
    curr_patches, curr_positions, images = [], [], []
    for pos, patch in zip(positions, patches):
        curr_patches.append(patch)
        curr_positions.append(pos)

        if len(curr_patches) == pos[2] * pos[3]:
            images.append(unpatchify(np.array(curr_patches), np.array(curr_positions)))
            curr_patches, curr_positions = [], []

    return txt, images


def vis_image_text_wandb(data, tiktoken, *, ph, pw):
    import wandb  # Local import to not pollute tests with silly warnings.
    table = wandb.Table(["id", "text", "images"])

    tokens = data["tokens"].cpu()
    iseq = data["iseq"].cpu()

    for _id in range(iseq.max() + 1):
        txt, images = vis_image_text_unpack(tokens[iseq == _id], ph=ph, pw=pw)
        txt = tiktoken.decode(txt)

        wandb_images = [wandb.Image(img) for img in images] or None
        table.add_data(_id, txt, wandb_images)

    return table
