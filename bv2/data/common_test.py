import json
from functools import partial

from bv2.data.common import cycle_qas, iota_exids, shuffled_iota_exids


def assert_same_exids(actual, ref):
    canon = partial(json.dumps, sort_keys=True)
    assert sorted(map(canon, actual)) == sorted(map(canon, ref))


def test_iota_exids():
    # Test that they are consistent...
    ref = [exid for exid, _ in iota_exids(n=11)]

    # ...across world sizes:
    r0 = [exid for exid, _ in iota_exids(n=11, rank=0, world_size=2)]
    r1 = [exid for exid, _ in iota_exids(n=11, rank=1, world_size=2)]
    assert_same_exids(r0 + r1, ref)

    # ...and across restarts from any single point, including past the end.
    full = list(iota_exids(n=11))
    for i in range(1, len(full) + 1):
        beg = full[:i]
        end = list(iota_exids(n=11, **beg[-1][1]))
        assert_same_exids([exid for exid, _ in beg + end], ref)
    assert not end  # And for the last one, the end should be empty, too.

    # ...and both at the same time!
    full0 = list(iota_exids(n=11, rank=0, world_size=2))
    full1 = list(iota_exids(n=11, rank=1, world_size=2))
    for i in range(1, max(len(full0), len(full1)) + 1):
        beg0, beg1 = full0[:i], full1[:i]
        end0 = list(iota_exids(n=11, rank=0, world_size=2, **beg0[-1][1]))
        end1 = list(iota_exids(n=11, rank=1, world_size=2, **beg1[-1][1]))
        assert_same_exids([exid for exid, _ in beg0 + beg1 + end0 + end1], ref)


def test_shuffled_iota_exids():
    # Test that they are consistent... across world sizes and restarts.
    ref = [exid for exid, _ in shuffled_iota_exids(seed=0, n=11, epochs=1)]

    # ...across world sizes:
    r0 = [exid for exid, _ in shuffled_iota_exids(seed=0, n=11, epochs=1, rank=0, world_size=2)]
    r1 = [exid for exid, _ in shuffled_iota_exids(seed=0, n=11, epochs=1, rank=1, world_size=2)]
    assert_same_exids(r0 + r1, ref)

    # ...and across restarts from any single point, including past the end.
    full = list(shuffled_iota_exids(seed=0, n=11, epochs=1))
    for i in range(1, len(full) + 1):
        beg = full[:i]
        end = list(shuffled_iota_exids(seed=0, n=11, epochs=1, **beg[-1][1]))
        assert_same_exids([exid for exid, _ in beg + end], ref)
    assert not end  # And for the last one, the end should be empty, too.

    # ...and the same again in the multi-epoch case...
    solo = [exid for exid, _ in shuffled_iota_exids(seed=0, n=11, epochs=2)]
    assert_same_exids(solo, [{**exid, "epoch": ep} for exid in ref for ep in [0, 1]])

    # ...across world sizes:
    r0 = [exid for exid, _ in shuffled_iota_exids(seed=0, n=11, epochs=2, rank=0, world_size=2)]
    r1 = [exid for exid, _ in shuffled_iota_exids(seed=0, n=11, epochs=2, rank=1, world_size=2)]
    assert_same_exids(r0 + r1, solo)

    # ...and across restarts from any single point, including past the end.
    full = list(shuffled_iota_exids(seed=0, n=11, epochs=2))
    for i in range(1, len(full) + 1):
        beg = full[:i]
        end = list(shuffled_iota_exids(seed=0, n=11, epochs=2, **beg[-1][1]))
        assert_same_exids([exid for exid, _ in beg + end], solo)
    assert not end  # And for the last one, the end should be empty, too.

    # ...and finally all together, larger world and restart everywhere:
    r0 = list(shuffled_iota_exids(seed=0, n=11, epochs=2, rank=0, world_size=2))
    r1 = list(shuffled_iota_exids(seed=0, n=11, epochs=2, rank=1, world_size=2))
    for i in range(max(len(r0), len(r1)) + 1):
        beg0, beg1 = r0[:i], r1[:i]
        end0 = list(shuffled_iota_exids(seed=0, n=11, epochs=2, rank=0, world_size=2, **(beg0[-1][1] if beg0 else {})))
        end1 = list(shuffled_iota_exids(seed=0, n=11, epochs=2, rank=1, world_size=2, **(beg1[-1][1] if beg1 else {})))
        assert_same_exids([exid for exid, _ in beg0 + beg1 + end0 + end1], solo)
    assert not end


def test_cycle_qas():
    qas = {
        "q1": ("q", ["a"]),
        "q2": ("p", ["a", "b"]),
        "q3": ("Q", ["A", "B", "C", "D"]),
    }

    noseed_seen = []
    seed1_seen = []
    seed2_seen = []
    for epoch in range(12):
        noseed_seen.append(cycle_qas(qas, epoch, seed=None))
        seed1_seen.append(cycle_qas(qas, epoch, seed="hahaha"))
        seed2_seen.append(cycle_qas(qas, epoch, seed="lollol"))
    expected = [
        ("q1", "q", "a"), ("q2", "p", "a"), ("q3", "Q", "A"),
        ("q1", "q", "a"), ("q2", "p", "b"), ("q3", "Q", "B"),
        ("q1", "q", "a"), ("q2", "p", "a"), ("q3", "Q", "C"),
        ("q1", "q", "a"), ("q2", "p", "b"), ("q3", "Q", "D"),
    ]

    # Visits should match in all cases:
    assert noseed_seen == expected
    assert set(seed1_seen) == set(expected)
    assert set(seed2_seen) == set(expected)

    # But order of visits should be different in all cases:
    assert seed1_seen != noseed_seen
    assert seed2_seen != noseed_seen
    assert seed1_seen != seed2_seen


if __name__ == "__main__":
    test_iota_exids()
    test_shuffled_iota_exids()
    test_cycle_qas()
    print("All tests passed!")
