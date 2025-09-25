"""This file tests roughly balanced reading of sharded data on multi-process."""

import json
from collections import Counter

import bagz

from bv2.data.common import get_bagz_reader, sharded_iota_exids


class Dataset:
    def __init__(self, fspec, cache_limits=True):
        self.reader = get_bagz_reader(fspec, cache_limits=cache_limits)

    def make_exids(self, *a, **kw):
        return sharded_iota_exids(len(self.reader), *a, **kw)

    def make_example(self, exid, epoch):
        return {"data": json.loads(self.reader[exid]), "ep": epoch, "id": exid}


def write_shard(examples, ishard, fname):
    with bagz.Writer(fname.format(ishard=ishard)) as writer:
        for ex in examples:
            writer.write(json.dumps({"example": ex, "shard": ishard}))


# Very even shards.
def write_balanced_dataset(dir="/tmp/"):
    write_shard(["a", "b", "c"], ishard=0, fname=dir + "balanced-{ishard:05d}-of-00005")
    write_shard(["d", "e", "f"], ishard=1, fname=dir + "balanced-{ishard:05d}-of-00005")
    write_shard(["g", "h", "i"], ishard=2, fname=dir + "balanced-{ishard:05d}-of-00005")
    write_shard(["j", "k", "l"], ishard=3, fname=dir + "balanced-{ishard:05d}-of-00005")
    write_shard(["m", "n", "o"], ishard=4, fname=dir + "balanced-{ishard:05d}-of-00005")


# Very uneven shards.
def write_imbalanced_dataset(dir="/tmp/"):
    # fmt:off
    write_shard(["a"]                                   , ishard=0, fname=dir + "imbalanced-{ishard:05d}-of-00005")
    write_shard(["b", "c", "d", "e", "f", "g", "h", "i"], ishard=1, fname=dir + "imbalanced-{ishard:05d}-of-00005")
    write_shard([]                                      , ishard=2, fname=dir + "imbalanced-{ishard:05d}-of-00005")
    write_shard(["j", "k", "l", "m"]                    , ishard=3, fname=dir + "imbalanced-{ishard:05d}-of-00005")
    write_shard(["n", "o"]                              , ishard=4, fname=dir + "imbalanced-{ishard:05d}-of-00005")
    # fmt:on


def test_balanced(dir="/tmp/"):
    write_balanced_dataset(dir)
    ds = Dataset(fspec=dir + "balanced@*")

    exids = list(ds.make_exids(seed=42, epoch=0, rank=0, world_size=2))
    assert Counter(exids) == Counter([0, 1, 2, 3, 4, 5, 6, 7])
    exids = list(ds.make_exids(seed=42, epoch=0, rank=1, world_size=2))
    assert Counter(exids) == Counter([8, 9, 10, 11, 12, 13, 14])

    assert ds.make_example(exid=0, epoch=0) == {"data": {"example": "a", "shard": 0}, "ep": 0, "id": 0}
    assert ds.make_example(exid=1, epoch=0) == {"data": {"example": "b", "shard": 0}, "ep": 0, "id": 1}
    assert ds.make_example(exid=2, epoch=0) == {"data": {"example": "c", "shard": 0}, "ep": 0, "id": 2}
    assert ds.make_example(exid=3, epoch=0) == {"data": {"example": "d", "shard": 1}, "ep": 0, "id": 3}
    assert ds.make_example(exid=4, epoch=0) == {"data": {"example": "e", "shard": 1}, "ep": 0, "id": 4}
    assert ds.make_example(exid=5, epoch=0) == {"data": {"example": "f", "shard": 1}, "ep": 0, "id": 5}
    assert ds.make_example(exid=6, epoch=0) == {"data": {"example": "g", "shard": 2}, "ep": 0, "id": 6}
    assert ds.make_example(exid=7, epoch=0) == {"data": {"example": "h", "shard": 2}, "ep": 0, "id": 7}
    assert ds.make_example(exid=8, epoch=0) == {"data": {"example": "i", "shard": 2}, "ep": 0, "id": 8}
    assert ds.make_example(exid=9, epoch=0) == {"data": {"example": "j", "shard": 3}, "ep": 0, "id": 9}
    assert ds.make_example(exid=10, epoch=0) == {"data": {"example": "k", "shard": 3}, "ep": 0, "id": 10}
    assert ds.make_example(exid=11, epoch=0) == {"data": {"example": "l", "shard": 3}, "ep": 0, "id": 11}
    assert ds.make_example(exid=12, epoch=0) == {"data": {"example": "m", "shard": 4}, "ep": 0, "id": 12}
    assert ds.make_example(exid=13, epoch=0) == {"data": {"example": "n", "shard": 4}, "ep": 0, "id": 13}
    assert ds.make_example(exid=14, epoch=0) == {"data": {"example": "o", "shard": 4}, "ep": 0, "id": 14}


def test_imbalanced(dir="/tmp/"):
    write_imbalanced_dataset(dir)
    ds = Dataset(fspec=dir + "imbalanced@*")

    exids = list(ds.make_exids(seed=42, epoch=0, rank=0, world_size=2))
    assert Counter(exids) == Counter([0, 1, 2, 3, 4, 5, 6, 7])
    exids = list(ds.make_exids(seed=42, epoch=0, rank=1, world_size=2))
    assert Counter(exids) == Counter([8, 9, 10, 11, 12, 13, 14])

    assert ds.make_example(exid=0, epoch=0) == {"data": {"example": "a", "shard": 0}, "ep": 0, "id": 0}
    assert ds.make_example(exid=1, epoch=0) == {"data": {"example": "b", "shard": 1}, "ep": 0, "id": 1}
    assert ds.make_example(exid=2, epoch=0) == {"data": {"example": "c", "shard": 1}, "ep": 0, "id": 2}
    assert ds.make_example(exid=3, epoch=0) == {"data": {"example": "d", "shard": 1}, "ep": 0, "id": 3}
    assert ds.make_example(exid=4, epoch=0) == {"data": {"example": "e", "shard": 1}, "ep": 0, "id": 4}
    assert ds.make_example(exid=5, epoch=0) == {"data": {"example": "f", "shard": 1}, "ep": 0, "id": 5}
    assert ds.make_example(exid=6, epoch=0) == {"data": {"example": "g", "shard": 1}, "ep": 0, "id": 6}
    assert ds.make_example(exid=7, epoch=0) == {"data": {"example": "h", "shard": 1}, "ep": 0, "id": 7}
    assert ds.make_example(exid=8, epoch=0) == {"data": {"example": "i", "shard": 1}, "ep": 0, "id": 8}
    assert ds.make_example(exid=9, epoch=0) == {"data": {"example": "j", "shard": 3}, "ep": 0, "id": 9}
    assert ds.make_example(exid=10, epoch=0) == {"data": {"example": "k", "shard": 3}, "ep": 0, "id": 10}
    assert ds.make_example(exid=11, epoch=0) == {"data": {"example": "l", "shard": 3}, "ep": 0, "id": 11}
    assert ds.make_example(exid=12, epoch=0) == {"data": {"example": "m", "shard": 3}, "ep": 0, "id": 12}
    assert ds.make_example(exid=13, epoch=0) == {"data": {"example": "n", "shard": 4}, "ep": 0, "id": 13}
    assert ds.make_example(exid=14, epoch=0) == {"data": {"example": "o", "shard": 4}, "ep": 0, "id": 14}


if __name__ == "__main__":
    test_balanced("/tmp/")
    test_imbalanced("/tmp/")
