import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor

from fscache import Cached, _has_working_odirect


class FakeReader:
    """Minimal reader mock: list of bytes."""
    def __init__(self, items):
        self._items = items
    def __getitem__(self, idx):
        return self._items[idx]
    def __len__(self):
        return len(self._items)


def _make_cache(items, cache_dir):
    return Cached(FakeReader(items), f"test:{len(items)}", cache_dir=cache_dir)


def test_cached_basic():
    d = tempfile.mkdtemp()
    try:
        items = [os.urandom(i * 100 + 50) for i in range(10)]
        c = _make_cache(items, d)
        for i, item in enumerate(items):
            assert c[i] == item
        for i, item in enumerate(items):
            assert c[i] == item
    finally:
        shutil.rmtree(d)


def test_cached_len_and_attr():
    d = tempfile.mkdtemp()
    try:
        items = [b"hello", b"world"]
        c = _make_cache(items, d)
        assert len(c) == 2
        assert c._items == items
    finally:
        shutil.rmtree(d)


def test_cached_large_item():
    d = tempfile.mkdtemp()
    try:
        big = os.urandom(2 * 1024 * 1024 + 7)
        c = _make_cache([big], d)
        assert c[0] == big
        assert c[0] == big
    finally:
        shutil.rmtree(d)


def test_cached_persistence():
    d = tempfile.mkdtemp()
    try:
        items = [os.urandom(500) for _ in range(5)]
        c1 = Cached(FakeReader(items), "persist:5", cache_dir=d)
        for i in range(5):
            c1[i]

        bad_reader = FakeReader([b"wrong"] * 5)
        c2 = Cached(bad_reader, "persist:5", cache_dir=d)
        for i in range(5):
            assert c2[i] == items[i]
    finally:
        shutil.rmtree(d)


def test_cached_forced_mmap_fallback():
    d = tempfile.mkdtemp()
    import fscache
    old_has_odirect = _has_working_odirect
    _has_working_odirect.cache_clear()
    fscache._has_working_odirect = lambda cache_dir: False
    try:
        items = [os.urandom(i * 43 + 33) for i in range(20)]
        c = Cached(FakeReader(items), "mmap:20", cache_dir=d)
        assert c._mode == "mmap"
        for i, item in enumerate(items):
            assert c[i] == item
        bad = Cached(FakeReader([b"wrong"] * len(items)), "mmap:20", cache_dir=d)
        assert bad._mode == "mmap"
        for i, item in enumerate(items):
            assert bad[i] == item
    finally:
        fscache._has_working_odirect = old_has_odirect
        old_has_odirect.cache_clear()
        shutil.rmtree(d)


def test_cached_threaded():
    d = tempfile.mkdtemp()
    try:
        items = [os.urandom(i * 37 + 100) for i in range(50)]
        c = _make_cache(items, d)

        def read_item(idx):
            return c[idx]

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(read_item, range(50)))
        for i, item in enumerate(items):
            assert results[i] == item

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(read_item, range(50)))
        for i, item in enumerate(items):
            assert results[i] == item
    finally:
        shutil.rmtree(d)


def test_cached_threaded_overlapping():
    d = tempfile.mkdtemp()
    try:
        items = [os.urandom(i * 17 + 200) for i in range(8)]
        c = _make_cache(items, d)

        def read_item(idx):
            return c[idx]

        jobs = [(i * 13) % len(items) for i in range(4000)]
        with ThreadPoolExecutor(max_workers=24) as pool:
            results = list(pool.map(read_item, jobs))
        for i, idx in enumerate(jobs):
            assert results[i] == items[idx]

        with ThreadPoolExecutor(max_workers=24) as pool:
            results = list(pool.map(read_item, jobs))
        for i, idx in enumerate(jobs):
            assert results[i] == items[idx]
    finally:
        shutil.rmtree(d)


def test_cached_wipe_existing_clears_idx():
    d = tempfile.mkdtemp()
    try:
        c1 = Cached(FakeReader([b"a" * 101, b"b" * 222]), "wipe:2", cache_dir=d, wipe_existing=True)
        _ = c1[0]
        _ = c1[1]

        c2 = Cached(FakeReader([b"x" * 101, b"y" * 222]), "wipe:2", cache_dir=d, wipe_existing=True)
        assert c2[1] == b"y" * 222
        assert c2[0] == b"x" * 101
    finally:
        shutil.rmtree(d)


if __name__ == "__main__":
    test_cached_basic()
    test_cached_len_and_attr()
    test_cached_large_item()
    test_cached_persistence()
    test_cached_forced_mmap_fallback()
    test_cached_threaded()
    test_cached_threaded_overlapping()
    test_cached_wipe_existing_clears_idx()
    print("All tests passed!")
