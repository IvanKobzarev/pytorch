import copy
import io
import json
from functools import partial

import numpy as np
import torch
from torch.nn.attention.flex_attention import create_block_mask

import bv2.data.dpack as dpack
import bv2.utils as u
from bv2.data.common import vis_image_text_wandb
from bv2.simple_input import parallel_prefetch, to_len


def _make_ex(_id, ds, max_prefix, max_decode):
    ex = ds.make_example(_id, epoch=0)
    attn_keys = [k for k in ex.keys() if k.startswith("attn_regions")]

    # Prompt ends at the last unsupervised token.
    prompt_idx = np.where(ex["loss_weights"] == 0)[0][-1]
    assert prompt_idx + 1 < len(ex["loss_weights"]), "Last token of an example has to be supervised"

    if prompt_idx >= max_prefix:
        print(f"[decode_to_file] Dropping a too long example: {int(ex["id"])=}.")
        return None

    for k in ["tokens", "loss_weights"] + attn_keys:
        ex[k] = ex[k][:prompt_idx + 1]

    # We use loss_weights==-1 to mark padding
    return to_len(ex, max_prefix + max_decode,
                  pad_values={"loss_weights": -1, "tokens": 0, **{k: 0 for k in attn_keys}})


def run(predict_fn, ds, iter_args, max_decode, T=1.0, ps=16):

    max_prefix = iter_args["max_prefix"]

    # TODO: figure out where to compile (including `partial` compile downstream)
    cbm = torch.compile(create_block_mask)

    exid_gen = ds.make_exids(
        epoch=0, seed=iter_args.get("seed", 0), rank=iter_args["rank"], world_size=iter_args["world_size"])

    make_ex = partial(_make_ex, ds=ds, max_prefix=max_prefix, max_decode=max_decode)
    ex_iter = parallel_prefetch(iter(exid_gen), make_ex)

    # Since examples can be filtered out, we iterate until we get a valid example
    dummy_ex = next(ex for ex in (make_ex(_id) for _id in ds.make_exids(epoch=0, seed=0)) if ex is not None)

    def _batched_iter():
        """Yields a tuple of (batch, done_indicator)."""
        _batch_fn = lambda exs: {k: np.concatenate([ex[k][None] for ex in exs], axis=0) for k in exs[0]}

        exs = []
        for ex in ex_iter:
            if ex is None:
                continue
            exs.append(ex)

            if len(exs) == iter_args["batch_size"]:
                yield _batch_fn(exs), False
                exs = []

        if exs:
            yield _batch_fn(exs), False

        dummy_batch = _batch_fn([dummy_ex])
        while True:
            yield copy.deepcopy(dummy_batch), True

    device = iter_args["device"]

    # TODO: make rng for each example based on the example id.
    rng = torch.Generator(device=device).manual_seed(iter_args["rank"])

    all_preds, wandb_table = {}, None
    for batch, eval_done in _batched_iter():

        if all(u.all_gather_object(eval_done)):
            break

        # Get the last txt token position for each sequence for positional embeddings.
        _, txtpos, mask = dpack.unpack_as_text(torch.from_numpy(batch["tokens"]))
        txtpos = (txtpos * mask).max(dim=1).values.numpy() + 1

        ids = batch.pop("id")
        batch = {k: torch.from_numpy(v).to(device) for k, v in batch.items()}
        tokens, loss_weights = batch["tokens"], batch["loss_weights"]
        attn_regions = {k: v for k, v in batch.items() if k.startswith("attn_regions")}
        batch_size = len(loss_weights)

        # Prediction is causal, no need for padding.
        def mask_mod(b, h, q_idx, kv_idx, mask_key):
            causal = q_idx >= kv_idx
            dense_region = (attn_regions[mask_key][b][q_idx] > 0) & (attn_regions[mask_key][b][kv_idx] > 0)
            same_region = attn_regions[mask_key][b][q_idx] == attn_regions[mask_key][b][kv_idx]
            return (causal | (same_region & dense_region))

        mask = {}
        for k in attn_regions:
            mask[k] = cbm(partial(mask_mod, mask_key=k),
                          Q_LEN=max_prefix + max_decode, KV_LEN=max_prefix + max_decode,
                          B=batch_size, H=None, device=device)

        decode_idx = (loss_weights.cpu().numpy() == -1).argmax(axis=1)
        reached_eos = np.zeros(len(decode_idx), dtype=np.bool)
        for i in range(max_decode):

            # Finish if all reach eos.
            if all(u.all_gather_object(all(reached_eos))):
                break

            logits = predict_fn(tokens, mask, loss_weights, torch.zeros_like(loss_weights), mode="logits")
            logits = logits[torch.arange(batch_size), decode_idx - 1]
            probs = torch.softmax(logits / T, dim=-1)
            next_txt_tok = torch.multinomial(probs, num_samples=1, generator=rng)[:, 0]

            next_txt_tok = next_txt_tok.cpu().numpy()

            next_tok = np.zeros((next_txt_tok.shape[0], tokens.shape[-1]), dtype=np.uint8)
            dpack.pack_text(next_txt_tok, positions=txtpos, out=next_tok)

            tokens[np.arange(batch_size)[~reached_eos], decode_idx[~reached_eos]] = (
                torch.from_numpy(next_tok[~reached_eos]).to(device)
            )

            reached_eos = reached_eos | (next_txt_tok == ds.tt.eos)
            txtpos += ~reached_eos
            decode_idx += ~reached_eos

        # visualize the first batch to wandb
        if not all_preds and iter_args["rank"] == 0:
            toks_list, iseq_list = [], []
            txt_len = decode_idx - (~reached_eos)
            for i in range(batch_size):
                toks_list.append(tokens[i, :txt_len[i]+1])
                iseq_list.append(torch.full((txt_len[i]+1,), i, dtype=torch.int64))
            d = {"tokens": torch.concat(toks_list, dim=0), "iseq": torch.concat(iseq_list, dim=0)}
            wandb_table = vis_image_text_wandb(d, ds.tt, ph=ps, pw=ps)

        toks, _, mask = dpack.unpack_as_text(tokens)
        id_text = [(id_, ds.tt.decode(t[m].cpu().numpy()))
                   for id_, t, m in zip(ids, toks, mask)]
        if id_text_done_list := u.gather_object_to(rank=0, obj=(id_text, eval_done)):
            id_text = sum([id_txt for id_txt, done in id_text_done_list if not done], [])
            for id_, text in id_text:
                all_preds[int(id_)] = text

    buf = io.BytesIO()
    buf.write(json.dumps(all_preds, ensure_ascii=False, indent=2).encode('utf-8'))
    return {"predictions.json": buf, "vis": wandb_table}
