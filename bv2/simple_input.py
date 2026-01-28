from itertools import islice
from multiprocessing.pool import ThreadPool

import numpy as np


def parallel_prefetch(seedgen, workfn, n_parallel=16):
    # Separate non-parallel codepath for ease of pdb'ing:
    if not n_parallel:
        for seed in seedgen:
            yield workfn(seed)
        return

    with ThreadPool(n_parallel) as pool:
        # Prefill:
        futures = [pool.apply_async(workfn, (seed,)) for seed in islice(seedgen, n_parallel)]

        # Keep consuming one and filling up next one, until we're all out of jobs.
        # NOTE: always following FIFO order, not first-ready, so we're deterministic.
        while len(futures):
            f = futures.pop(0)

            try:
                futures.append(pool.apply_async(workfn, (next(seedgen),)))
            except StopIteration:
                pass

            yield f.get()


def iter_packed_examples(
    example_generator,
    max_seqlen,
    *,
    leader="tokens",
    debugid="id",
    dont_repeat=("id", "state_after", "src"),
):
    def not_too_long(ex):  # pyre-ignore[53]
        if len(ex[leader]) <= max_seqlen:
            return True
        else:
            print(f"Dropping too long example {ex[debugid]} because {len(ex[leader])} > {max_seqlen}")
            return False

    good_example_generator = (ex for ex in example_generator if not_too_long(ex))

    def tolist_maybe_repeat(ex, k):
        return [ex[k]] * (len(ex[leader]) if k not in dont_repeat else 1)

    def seq_from(exs):  # NOTE: This also works when `exs` is empty.
        seq = {}
        seq["lens"] = [len(ex[leader]) for ex in exs]
        seq["iseq"] = np.repeat(np.arange(len(exs)), seq["lens"])

        # Concat all arrays, potentially repeat all non-arrays.
        for k in set().union(*exs) - {"lens", "iseq"}:
            # TODO: What if `k` does not exist in some example? Currently, we raise,
            #       but conceivably we could also treat as non-array and use `None`?
            if any(isinstance(ex[k], np.ndarray) for ex in exs):
                seq[k] = np.concatenate([ex[k] for ex in exs])
            else:
                seq[k] = []
                for ex in exs:  # This is much faster than the sum(, []) one-liner
                    seq[k].extend(tolist_maybe_repeat(ex, k))
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
