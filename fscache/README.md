# fscache

`fscache` is a tiny local-disk cache wrapper for random-access readers:
it stores fetched items in `.dat/.idx` files so future reads come from cache,
and it is designed to be shared by threads within one process.
Cache filenames include PID, so each process gets an independent cache.
On Linux it probes `O_DIRECT` at construction and uses it when available;
otherwise it falls back to `mmap` with random-access and `dontneed` hints.
Current implementation is Linux-only (no macOS support).

```python
from fscache import Cached

reader = SomeRandomAccessReader()  # supports __len__ and __getitem__ returning bytes-like
cache = Cached(reader, fspec="my_dataset:v1")

x = cache[123]  # first read fills cache
y = cache[123]  # subsequent read is served from cache
```

Note: indices are expected to be non-negative (`idx >= 0`).

Run tests:

```bash
python -m fscache.test_cached
```
