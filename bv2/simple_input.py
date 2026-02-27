from multiprocessing.pool import ThreadPool
from queue import Full, Queue
from threading import Event, Thread

import numpy as np


def prefetch(it, n=1):
    """Wrap an iterator to prefetch next elements in a background thread."""
    if not n:  # Separate non-parallel codepath for ease of pdb'ing:
        yield from it
        return

    _DONE = object()
    q = Queue(maxsize=n)
    stop = Event()
    def feeder():
        try:
            for item in it:
                while not stop.is_set():
                    try:
                        q.put(item, timeout=0.1)
                        break
                    except Full:
                        pass
                if stop.is_set():
                    return
        except BaseException as e:
            if not stop.is_set():
                q.put(e)
        if not stop.is_set():
            q.put(_DONE)

    t = Thread(target=feeder, daemon=True)
    t.start()
    try:
        for item in iter(q.get, _DONE):
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
        t.join(timeout=5)
        while not q.empty():
            try:
                q.get_nowait()
            except Exception:
                break


def pmap(it, fn, n_prefetch=16, n_threads=16):
    """Like map(fn, it) but fn runs in a thread pool, results yielded in order."""
    if not n_threads:  # Separate non-parallel codepath for ease of pdb'ing:
        for x in it:
            yield fn(x)
        return

    # NOTE: always following FIFO order, not first-ready, so we're deterministic.
    q = Queue(maxsize=n_prefetch)
    stop = Event()
    def feeder(pool):
        try:
            for x in it:
                f = pool.apply_async(fn, (x,))
                while not stop.is_set():
                    try:
                        q.put(f, timeout=0.1)
                        break
                    except Full:
                        pass
                if stop.is_set():
                    return
        except BaseException as e:
            if not stop.is_set():
                q.put(e)
        if not stop.is_set():
            q.put(None)

    with ThreadPool(n_threads) as pool:
        t = Thread(target=feeder, args=(pool,), daemon=True)
        t.start()
        try:
            while (f := q.get()) is not None:
                if isinstance(f, BaseException):
                    raise f
                yield f.get()
        finally:
            stop.set()
            t.join(timeout=5)
            while not q.empty():
                try:
                    q.get_nowait()
                except Exception:
                    break


def iter_packed_examples(
    example_generator,
    max_seqlen,
    *,
    leader="toki",
    debugid="id",
    dont_repeat=("id", "ndatatoks", "src"),
    keep_last_only=("state_after",)
):
    def not_too_long(ex):  # pyre-ignore[53]
        if len(ex[leader]) <= max_seqlen:
            return True
        else:
            print(f"Dropping too long example {ex[debugid]} because {len(ex[leader])} > {max_seqlen}")
            return False

    good_example_generator = (ex for ex in example_generator if not_too_long(ex))
    dont_repeat = set(dont_repeat)

    def seq_from(exs):  # NOTE: This also works when `exs` is empty.
        seq = {}
        seq["ntok"] = [len(ex[leader]) for ex in exs]
        seq["iseq"] = np.repeat(np.arange(len(exs)), seq["ntok"])

        # Concat all arrays, potentially repeat all non-arrays.
        for k in set().union(*exs) - {"iseq"}:
            # TODO: What if `k` does not exist in some example? Currently, we raise,
            #       but conceivably we could also treat as non-array and use `None`?
            if any(isinstance(ex[k], np.ndarray) for ex in exs):
                seq[k] = np.concatenate([ex[k] for ex in exs])
            elif k in keep_last_only:
                seq[k] = exs[-1][k]
            elif k in dont_repeat:
                seq[k] = [ex[k] for ex in exs]
            else:  # Repeat for each token in each example
                seq[k] = []
                for ex in exs:  # This is much faster than the sum(, []) one-liner
                    seq[k].extend([ex[k]] * len(ex[leader]))
        return seq

    seq_exs = []
    for ex in good_example_generator:
        # Packing next one would be too much -> yield current and start new using next.
        if sum(len(e[leader]) for e in seq_exs) + len(ex[leader]) > max_seqlen:
            # Pack the sequence of examples into an actual sequence:
            yield seq_from(seq_exs)
            seq_exs = []

        seq_exs.append(ex)

    yield seq_from(seq_exs)  # Let's not forget about the last sequence!


def to_len(seq, to_len, *, pad_values=0, allow_cut=False):
    def _rpad_dim0(a, npad, padval):
        return np.pad(a, [(0, npad)] + [(0, 0)] * (a.ndim - 1), constant_values=padval)

    padded = {}
    for k, v in seq.items():
        # This is the API to skip padding on some keys:
        if isinstance(pad_values, dict) and k not in pad_values:
            padded[k] = v
            continue

        # Padding is requested. Pad numpy arrays as such, otherwise assume it's lists.
        padval = pad_values[k] if isinstance(pad_values, dict) else pad_values
        npad = to_len - len(v)
        if npad > 0:
            if isinstance(v, np.ndarray):
                padded[k] = _rpad_dim0(v, npad, padval)
            else:
                padded[k] = v + [padval] * npad
        elif npad < 0:
            if not allow_cut:
                raise ValueError("Cutting inputs with `to_len` is now allowed by default. Set `allow_cut=True` to enable.")
            padded[k] = v[:to_len]
        else:
            padded[k] = v

    return padded
