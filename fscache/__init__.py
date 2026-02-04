import ctypes
import hashlib
import mmap
import os
import struct
import threading
from functools import cache

_libc = ctypes.CDLL("libc.so.6", use_errno=True)
_libc.posix_memalign.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_size_t]
_libc.posix_memalign.restype = ctypes.c_int
_libc.write.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
_libc.write.restype = ctypes.c_ssize_t
_libc.pread.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_longlong]
_libc.pread.restype = ctypes.c_ssize_t
_libc.memset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
_libc.free.argtypes = [ctypes.c_void_p]
_ALIGN = 4096
_MADV_RANDOM = getattr(mmap, "MADV_RANDOM", None)
_MADV_DONTNEED = getattr(mmap, "MADV_DONTNEED", None)
_tls = threading.local()


def _aligned_alloc(size):
    ptr = ctypes.c_void_p()
    ret = _libc.posix_memalign(ctypes.byref(ptr), _ALIGN, max(size, _ALIGN))
    if ret != 0 or not ptr.value:
        raise MemoryError(f"posix_memalign failed: {ret}")
    return ptr


@cache
def _has_working_odirect(cache_dir):
    d = getattr(os, "O_DIRECT", 0)
    if not d:
        return False
    path = f"{cache_dir}/.odirect_probe.{os.getpid()}.{threading.get_ident()}"
    try:
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC | d, 0o666)
    except OSError:
        return False
    buf = _aligned_alloc(_ALIGN)
    try:
        _libc.memset(buf, 0, _ALIGN)
        return _libc.write(fd, buf, _ALIGN) == _ALIGN and _libc.pread(fd, buf, _ALIGN, 0) == _ALIGN
    finally:
        os.close(fd)
        _libc.free(buf)
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


class Cached:
    """Caches reader items on local NVMe (/tmp), process-local by PID."""
    _E = struct.Struct('<qq')  # (offset, length); length=0 means uncached.
    _Q = struct.Struct('<q')

    def __init__(self, obj, fspec, cache_dir="/tmp/sackli_cache", wipe_existing=False):
        self._obj = obj
        self._wlock = threading.Lock()
        self._klocks = [threading.Lock() for _ in range(256)]
        n = len(obj)
        # PID-scoped key is intentionally good-enough here; PID reuse collisions are accepted.
        h = hashlib.md5(f"{fspec}:{n}:{os.getpid()}".encode()).hexdigest()[:16]
        os.makedirs(cache_dir, exist_ok=True)
        idx_path = f"{cache_dir}/{h}.idx"
        dat_path = f"{cache_dir}/{h}.dat"
        idx_size = n * self._E.size
        fd = os.open(idx_path, os.O_RDWR | os.O_CREAT, 0o666)
        try:
            if wipe_existing:
                os.ftruncate(fd, 0)
            if os.fstat(fd).st_size != idx_size:
                os.ftruncate(fd, idx_size)
            self._idx = mmap.mmap(fd, idx_size)
            if _MADV_RANDOM is not None and hasattr(self._idx, "madvise"):
                self._idx.madvise(_MADV_RANDOM)  # Random access pattern; disable readahead.
        finally:
            os.close(fd)
        wflags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | (os.O_TRUNC if wipe_existing else 0)
        if _has_working_odirect(cache_dir):
            try:
                wfd = os.open(dat_path, wflags | os.O_DIRECT, 0o666)
                try:
                    rfd = os.open(dat_path, os.O_RDONLY | os.O_DIRECT, 0o666)
                except OSError:
                    os.close(wfd)
                    raise
                self._mode = "direct"
            except OSError:
                self._mode = "mmap"
                wfd = os.open(dat_path, wflags, 0o666)
                rfd = os.open(dat_path, os.O_RDONLY, 0o666)
        else:
            self._mode = "mmap"
            wfd = os.open(dat_path, wflags, 0o666)
            rfd = os.open(dat_path, os.O_RDONLY, 0o666)
        self._wfd = wfd
        self._rfd = rfd
        # Intended process-lifetime singleton; cleanup currently relies on process teardown.
        # Pre-allocate aligned write buffer (1MB, reused under _wlock).
        self._wbuf_cap = 1 << 20
        self._wbuf = _aligned_alloc(self._wbuf_cap)

    def __getitem__(self, idx):
        idx = int(idx)
        assert idx >= 0, "negative indices are unsupported"
        with self._klocks[idx & 255]:
            E = self._E
            pos = idx * E.size
            length = self._Q.unpack(self._idx[pos + 8:pos + 16])[0]
            if length > 0:
                offset = self._Q.unpack(self._idx[pos:pos + 8])[0]
                if offset >= 0 and not (offset & (_ALIGN - 1)):
                    padded = (length + _ALIGN - 1) & ~(_ALIGN - 1)
                    if offset + padded <= os.fstat(self._rfd).st_size:
                        if self._mode == "direct":
                            cap = getattr(_tls, 'cap', 0)
                            if cap < padded:
                                if cap: _libc.free(_tls.buf)
                                _tls.buf = _aligned_alloc(padded)
                                _tls.cap = padded
                            n = _libc.pread(self._rfd, _tls.buf, padded, offset)
                            if n == padded:
                                return ctypes.string_at(_tls.buf, length)
                        else:
                            m = mmap.mmap(self._rfd, padded, access=mmap.ACCESS_READ, offset=offset)
                            if _MADV_RANDOM is not None and hasattr(m, "madvise"):
                                m.madvise(_MADV_RANDOM)
                            out = m[:length]
                            if _MADV_DONTNEED is not None and hasattr(m, "madvise"):
                                m.madvise(_MADV_DONTNEED)
                            m.close()
                            return out
            v = bytes(self._obj[idx])
            padded = (len(v) + _ALIGN - 1) & ~(_ALIGN - 1)
            with self._wlock:
                if padded > self._wbuf_cap:
                    _libc.free(self._wbuf)
                    self._wbuf_cap = padded
                    self._wbuf = _aligned_alloc(padded)
                ctypes.memmove(self._wbuf, v, len(v))
                if padded > len(v):
                    _libc.memset(ctypes.c_void_p(self._wbuf.value + len(v)), 0, padded - len(v))
                n = _libc.write(self._wfd, self._wbuf, padded)
                if n == padded:
                    offset = os.lseek(self._wfd, 0, os.SEEK_CUR) - padded
            if n == padded:
                self._idx[pos:pos + 8] = self._Q.pack(offset)  # Write offset first.
                self._idx[pos + 8:pos + 16] = self._Q.pack(len(v))  # Publish length last.
            return v

    def __getattr__(self, name):
        return getattr(self._obj, name)

    def __len__(self):
        return len(self._obj)
