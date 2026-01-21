from functools import cache

import bagz
import numpy as np

import bv2.data.dpack as d
import bv2.utils as u
from bv2.data.pp import unpatchify


def iota_exids(n=None, start_offset=0, rank=0, world_size=1):
    if rank_n := n:
        extra = rank < (n % world_size)
        rank_n = n // world_size + extra
    for i in u.count(start=start_offset, end=rank_n):  # Infinite if rank_n is None btw.
        yield {"exid": i * world_size + rank}, {"start_offset": i + 1}


def random_exids(seed, n=None, start_offset=0, rank=0, world_size=1):
    # NOTE: For this one, it's a bit unclear whether the iter seed (this one) should yield different exids,
    #       or just the same exids but in different order. Good thing it doesn't matter for our current uses.
    for exid, state_after in iota_exids(n=n, start_offset=start_offset, rank=rank, world_size=world_size):
        yield {"exid": u.rng(seed, exid["exid"]).integers(2**32).item()}, state_after


def shuffled_iota_exids(n, seed, epochs=None, start_epoch=0, start_offset=0, rank=0, world_size=1):
    assert n is not None, "Doesn't make sense for infinite, just use `random_exids`!"

    split_size = n / world_size
    start = round(rank * split_size)
    end = round((rank + 1) * split_size)

    for ep in u.count(start=start_epoch, end=epochs):
        exids = u.rng(seed, ep, rank).permutation(np.arange(start, end)).tolist()

        # For each exid, yielding (kwargs for make_example, kwargs for self next step)
        for i, exid in enumerate(exids[start_offset:-1]):
            yield {"exid": exid, "epoch": ep}, {"start_epoch": ep, "start_offset": start_offset + i + 1}
        yield {"exid": exids[-1], "epoch": ep}, {"start_epoch": ep + 1, "start_offset": 0}
        start_offset = 0


def cycle_qas(qas, epoch, seed):
    # `qas` is a {ID: ("q", ["a", "a", ...]), ...} or similar.

    # An example can have multiple Q/A pairs, and each Q can have multiple A's.
    # For many datasets with multiple Q's, the order is structured, for example
    # first all questions of one type, then all of another, etc. Training in that
    # order is a bad idea, and so we randomize the order we cycle through, both
    # questions and answers for any given question, on a per-exid basis.

    num_qs = len(qas)
    q_cycle, q_idx = divmod(epoch, num_qs)
    if seed is not None and num_qs > 1:
        q_idx = u.rng(seed, q_cycle).permutation(num_qs)[q_idx]
    qid = list(qas)[q_idx]
    question, answers = qas[qid]

    num_as = len(answers)
    if seed is not None and num_as > 1:
        a_idx = u.rng(seed, q_idx).permutation(num_as)[q_cycle % num_as]
    else:
        a_idx = q_cycle % num_as
    answer = answers[a_idx]
    return qid, question, answer


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
