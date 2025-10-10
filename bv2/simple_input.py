from itertools import islice
from multiprocessing import Pool

import numpy as np


def parallel_prefetch(seedgen, workfn, n_parallel=16):
    # Separate non-parallel codepath for ease of pdb'ing:
    if not n_parallel:
        for seed in seedgen:
            yield workfn(seed)
        return

    with Pool(n_parallel) as pool:
        # Prefill:
        futures = [
            pool.apply_async(workfn, (seed,)) for seed in islice(seedgen, n_parallel)
        ]

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
    """Numpy arrays get packed, anything else gets repeated. Add `iseq` counter."""

    def not_too_long(ex):  # pyre-ignore[53]
        if len(ex[leader]) <= max_seqlen:
            return True
        else:
            print(
                f"Dropping too long example {ex[debugid]} because "
                f"{len(ex[leader])} > {max_seqlen}"
            )
            return False

    good_example_generator = (ex for ex in example_generator if not_too_long(ex))

    def tolist_maybe_repeat(ex, k):
        return [ex[k]] * (len(ex[leader]) if k not in dont_repeat else 1)

    def start_from(ex):  # pyre-ignore[53]
        return {
            "iseq": np.zeros(len(ex[leader]), np.int64),
            "lens": [len(ex[leader])],
        } | {
            k: v.copy() if isinstance(v, np.ndarray) else tolist_maybe_repeat(ex, k)
            for k, v in ex.items()
        }

    # Start with the first example (that's not too long)
    seq = start_from(next(good_example_generator))

    for ex in good_example_generator:
        # Packing next one would be too much -> yield seq and start new using next.
        if len(seq[leader]) + len(ex[leader]) > max_seqlen:
            yield seq
            seq = start_from(ex)
            continue

        # Not yet full, and next fits: concat them on arrays, repeat non-arrays (eg ID).
        for k in seq:
            if k == "iseq":  # Special-case counting the sequence number.
                seq[k] = np.r_[seq[k], np.full(len(ex[leader]), seq[k][-1] + 1)]
            elif k == "lens":  # Special-case tracking sequence lengths.
                seq[k].append(len(ex[leader]))
            elif isinstance(seq[k], np.ndarray):
                seq[k] = np.r_[seq[k], ex[k]]
            else:
                seq[k].extend(tolist_maybe_repeat(ex, k))
    yield seq  # Let's not forget about the last sequence!


def pad_seq(seq, to_length, *, pad_values=0):
    def _rpad_dim0(a, npad, padval):
        if npad == 0:
            return a
        return np.pad(a, [(0, npad)] + [(0, 0)] * (a.ndim - 1), constant_values=padval)

    padded = {}
    for k, v in seq.items():
        # This is the API to skip padding on some keys:
        if isinstance(pad_values, dict) and k not in pad_values:
            padded[k] = v
            continue

        # Padding is requested. Pad numpy arrays as such, otherwise assume it's lists.
        padval = pad_values[k] if isinstance(pad_values, dict) else pad_values
        if isinstance(v, np.ndarray):
            padded[k] = _rpad_dim0(v, to_length - v.shape[0], padval)
        else:
            padded[k] = v + [padval] * (to_length - len(v))

    return padded
