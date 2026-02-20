"""Important simplifying assumptions made for now:
- we assume the prompt ends at the last token with zero `lowe`.
- we hard-code epoch=0 for the dataset
- we only decode text modality
"""

import copy
import functools

import numpy as np
import torch
from flexlimaskli import make_batchmask_cpu

import bv2.data.dpack as dpack
import bv2.utils as u
from bv2.simple_input import pmap, to_len


def _make_ex(exid_and_state_after, ds, max_prefix, max_decode):
    exid, state_after = exid_and_state_after
    ex = ds.make_example(**exid)
    attn_keys = [k for k in ex.keys() if k.startswith("attn_regions")]

    # Prompt ends at the last unsupervised token, which is one after the last lowe-0 token.
    prompt_idx = np.where(ex["lowe"] == 0)[0][-1] + 1
    ex.pop("lowe")  # not needed anymore
    ex.pop("toko")  # Never needed.

    if prompt_idx >= max_prefix:
        print(f"[{__file__}] Dropping a too long example: {ex["id"]=}.")
        return None

    # The token we decode is the one after the prompt's last token:
    ex["decode_idx"] = prompt_idx + 1
    for k in ("toki", *attn_keys):
        ex[k] = ex[k][:prompt_idx + 1]

    return to_len(ex, max_prefix + max_decode,
                  pad_values={"toki": 0, **{k: 0 for k in attn_keys}})


def decode_batch(predict_fn, batch, *, decode_idx, rng,
                 T, eos, device, max_prefix, max_decode):

    # Get the last txt token position for each sequence for positional embeddings.
    _, txtpos, mask = dpack.unpack_as_text(torch.from_numpy(batch["toki"]))
    next_token_pos = (txtpos * mask).max(dim=1).values.numpy() + 1

    batch_size = len(batch["toki"])
    ntoks = max_prefix + max_decode

    # Create flexmasks on CPU using numba (batched), then move everything to GPU.
    attn_keys = [k for k in batch if k.startswith("attn_regions")]
    flex_masks = {k: make_batchmask_cpu(ntoks, batch[k], BLOCK_SIZE=128) for k in attn_keys}

    batch = u.to_gpu(batch, device)
    flex_masks = u.to_gpu(flex_masks, device)
    tokens = batch["toki"]

    reached_eos = np.zeros(len(decode_idx), dtype=np.bool_)
    for step in range(max_decode):

        # Finish if all reach eos.
        if all(u.all_gather_object(all(reached_eos))):
            break

        logits_tok_idx = torch.from_numpy(decode_idx - 1).to(device)
        logits, _ = predict_fn(tokens, None, flex_masks, None, torch.zeros(tokens.shape[:2], dtype=torch.int64), mode="logits", logits_tok_idx=logits_tok_idx)

        # TODO: add support for T=0
        probs = torch.softmax(logits / T, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1, generator=rng)[:, 0]
        next_token_id = next_token_id.cpu().numpy()

        next_token = np.zeros((next_token_id.shape[0], tokens.shape[-1]), dtype=np.uint8)
        dpack.pack_text(next_token_id, positions=next_token_pos, out=next_token)

        # Do not update tokens if eos was reached. It is convenient,
        # because later we can filter out padded tokens that come after EOS.
        # Tokens are zero-padded, and modality idx 0 is "nothing".
        m = ~reached_eos
        tokens[np.arange(batch_size)[m], decode_idx[m]] = torch.from_numpy(next_token[m]).to(device)

        reached_eos = reached_eos | (next_token_id == eos)

        next_token_pos += 1
        decode_idx += 1

    return tokens


def decoding_iterator(predict_fn, ds, *, max_prefix, max_decode, device, batch_size,
                      seed=0, T=1.0, omit_eos=False, rank=0, world_size=1, eagerness=24):
    """Infinite iterator over data that runs decoding (marks padded examples by a boolean output).

    Performs batching under the hood, but yields flat sequence of examples.
    """

    exid_gen = ds.make_exids(seed=seed, rank=rank, world_size=world_size)
    make_ex = functools.partial(_make_ex, ds=ds, max_prefix=max_prefix, max_decode=max_decode)
    ex_iter = pmap(exid_gen, make_ex, n_prefetch=eagerness, n_threads=eagerness)

    # Since examples can be filtered out, we iterate until we get a valid example
    dummy_ex = next(ex for ex in map(make_ex, ds.make_exids(seed=seed)) if ex is not None)

    def _batched_iter():
        """Yields a tuple of (batch, done_indicator). Batch is always padded to `batch_size`."""
        _batch_fn = lambda exs: {k: np.concatenate([[ex[k]] for ex in exs], axis=0) for k in exs[0]}

        exs = []
        for ex in ex_iter:
            if ex is None:
                continue
            exs.append(ex)

            if len(exs) == batch_size:
                yield _batch_fn(exs), [False] * batch_size
                exs = []

        if exs:
            remainder = batch_size - len(exs)
            yield _batch_fn(exs + [dummy_ex] * remainder), [False] * len(exs) + [True] * remainder

        dummy_batch = _batch_fn([dummy_ex] * batch_size)
        while True:
            yield copy.deepcopy(dummy_batch), [True] * batch_size

    for i, (batch, done) in enumerate(_batched_iter()):
        if all(u.all_gather_object(np.all(done))):
            break

        ids = batch.pop("id")

        # Start decoding from the first non-zero weight.
        decode_idx = batch["decode_idx"]
        prefix_lens = np.array(decode_idx) # make copy for later

        tokens = decode_batch(
            predict_fn, batch,
            decode_idx=decode_idx,
            rng=u.rng_torch("decode", i, rank, device=device),  # TODO: make one per example-id?
            T=T,
            eos=ds.tt.eos,
            device=device,
            max_prefix=max_prefix,
            max_decode=max_decode)

        suffix, _, mask_suffix = dpack.unpack_as_text(tokens)
        suffix_token_ids = [s[start:][m[start:]].cpu().numpy() for s, m, start in zip(suffix, mask_suffix, prefix_lens)]
        if omit_eos:
            suffix_token_ids = [(s[:-1] if s[-1] == ds.tt.eos else s) for s in suffix_token_ids]

        yield from ({"id": id_,
                     "packed_prefix_torch": tok[:prefix_len],  # Nice to have for debugging
                     "suffix": suffix,
                     "done": done}
            for id_, tok, prefix_len, suffix, done in zip(ids, tokens, prefix_lens, suffix_token_ids, done))
