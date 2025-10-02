from functools import cache

import bagz
import numpy as np

import bv2.data.dpack as d  # isort: skip
from bv2.data.pp import unpatchify  # isort: skip


def infinite_random_exids(seed, epoch=0, rank=0, world_size=1, epoch_size=1024):
    # Infinite data. It's implemented as an infinite number of epochs, for two reasons:
    # 1. evaluators run for one epoch, so `epoch_size` is eval set size.
    # 2. for checkpointing: epoch boundary allows "fast-forward jump" upon resuming.
    rng = np.random.default_rng([seed, epoch, rank])
    extra = rank < (epoch_size % world_size)
    num_examples = epoch_size // world_size + extra  # For this rank.
    return (rng.integers(2**32) for _ in range(num_examples))


def sharded_iota_exids(n, seed, epoch=0, rank=0, world_size=1):
    split_size = n / world_size
    start = round(rank * split_size)
    end = round((rank + 1) * split_size)

    rng = np.random.default_rng([seed, epoch])
    return rng.permutation(np.arange(start, end))


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
    # decode text
    txt, _, mask = d.unpack_as_text(tokens)
    txt, mask = txt.numpy(), mask.numpy()
    txt = txt[mask]

    # decode image
    patches, positions, _, mask = d.unpack_as_image(tokens, ph, pw)
    patches, positions, mask = patches.numpy(), positions.numpy(), mask.numpy()
    patches, positions = patches[mask], positions[mask]
    image = unpatchify(patches, positions)

    return txt, image


def vis_image_text_wandb(data, tiktoken, *, ph, pw):
    import wandb  # Local import to not pollute tests with silly warnings.
    table = wandb.Table(["id", "text", "image"])

    tokens = data["tokens"].cpu()
    iseq = data["iseq"].cpu()

    for _id in range(iseq.max() + 1):
        txt, image = vis_image_text_unpack(tokens[iseq == _id], ph=ph, pw=pw)
        txt = tiktoken.decode(txt)
        table.add_data(_id, txt, wandb.Image(image))

    return table
