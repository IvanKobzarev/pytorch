from functools import partial

import numpy as np
import torch
import torch.distributed as distr
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.attention.flex_attention import AuxRequest, flex_attention
from torch.utils.checkpoint import checkpoint

import bv2.data.dpack as dpack  # usort: skip

# 1: Compiled flex_attention is necessary to checkpoint the attention block, see:
# https://github.com/pytorch/pytorch/issues/147879#issuecomment-3041193259
# 2: max-autotune-no-cudagraphs takes a long time, but did 1039ms->1028ms on 4k seqlen.
# cflex_attention = torch.compile(flex_attention, mode="max-autotune-no-cudagraphs")
cflex_attention = torch.compile(flex_attention, dynamic=False, fullgraph=True)


class Attention(nn.Module):
    def __init__(self, dim, head_dim, kv_reduce=1):
        super().__init__()
        assert dim % head_dim == 0, f"Bad {dim=}/{head_dim=}"
        assert (dim // head_dim) % kv_reduce == 0, f"Bad {kv_reduce=} for {dim=} and {head_dim=}"  # fmt: skip
        self.dim = dim
        self.head_dim = head_dim
        self.n_q_heads = dim // head_dim
        self.n_kv_heads = self.n_q_heads // kv_reduce
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, self.n_kv_heads * head_dim, bias=False)
        self.v = nn.Linear(dim, self.n_kv_heads * head_dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)

    def forward(self, x, flex_mask):
        batch_dims = x.size()[:-2]
        q = rearrange(self.q(x), "... T (Q D) -> (...) Q T D", Q=self.n_q_heads)
        k = rearrange(self.k(x), "... T (K D) -> (...) K T D", K=self.n_kv_heads)
        v = rearrange(self.v(x), "... T (V D) -> (...) V T D", V=self.n_kv_heads)

        # fmt: off
        x, aux = cflex_attention(
            q, k, v,
            block_mask=flex_mask,  # NB: scaled by 1/sqrt if scale=None, the default
            enable_gqa=self.n_q_heads != self.n_kv_heads,
            return_aux=AuxRequest(max_scores=True),
        )
        # fmt: on
        o = self.o(rearrange(x, "... Q T D -> ... T (Q D)"))
        return o.reshape(*batch_dims, *o.shape[-2:]), {"max_logit": aux.max_scores.max()}

    def init_weights(self, rng):
        # TODO: more careful: qk such that dot-var is 1, and care about o.
        nn.init.trunc_normal_(self.q.weight, mean=0.0, std=0.02, generator=rng)
        nn.init.trunc_normal_(self.k.weight, mean=0.0, std=0.02, generator=rng)
        nn.init.trunc_normal_(self.v.weight, mean=0.0, std=0.02, generator=rng)
        nn.init.trunc_normal_(self.o.weight, mean=0.0, std=1 / np.sqrt(self.dim), generator=rng)  # fmt: skip


class MLP(nn.Module):
    def __init__(self, dim, grow=4):
        super().__init__()
        self.dim = dim
        self.grow = grow
        self.l1 = nn.Linear(dim, int(grow * dim))
        self.l2 = nn.Linear(int(grow * dim), dim)

    def forward(self, x):
        x = self.l1(x)
        x = F.gelu(x, approximate="tanh")
        x = self.l2(x)
        return x, {}

    def init_weights(self, rng):
        nn.init.trunc_normal_(self.l1.weight, mean=0.0, std=1/np.sqrt(self.dim * self.grow / 2), generator=rng)  # fmt: skip
        nn.init.trunc_normal_(self.l2.weight, mean=0.0, std=1/np.sqrt(self.dim * self.grow / 2), generator=rng)  # fmt: skip
        nn.init.zeros_(self.l1.bias)
        nn.init.zeros_(self.l2.bias)


class Block(nn.Module):
    def __init__(self, dim, head_dim=128, grow=4, kv_reduce=4, remat=True):
        super().__init__()
        self.att_ln = nn.LayerNorm(dim)  # TODO: better ln parametrization
        self.mlp_ln = nn.LayerNorm(dim)
        self.att = Attention(dim, head_dim, kv_reduce)
        self.mlp = MLP(dim, grow)
        self.remat = remat

    def forward(self, x, flex_mask):
        def att_ln_fn(y, fm):
            return self.att(self.att_ln(y), fm)

        def mlp_ln_fn(y):
            return self.mlp(self.mlp_ln(y))

        if self.remat:
            att_ln_fn = partial(checkpoint, att_ln_fn, use_reentrant=False)
            mlp_ln_fn = partial(checkpoint, mlp_ln_fn, use_reentrant=False)

        extras = {}
        y, extras["attn"] = att_ln_fn(x, flex_mask)
        x = x + y
        z, extras["mlp"] = mlp_ln_fn(x)
        x = x + z
        return x, extras

    def init_weights(self, rng):
        self.att.init_weights(rng)
        self.mlp.init_weights(rng)
        self.att_ln.reset_parameters()  # scale: 1.0, bias: 0.0
        self.mlp_ln.reset_parameters()


class TxtEmbedding(nn.Module):
    def __init__(self, dim, vocab, posemb=True):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.dim = dim
        self.pe_scale = (
            nn.Parameter(torch.empty((), dtype=torch.float32)) if posemb else None
        )

    def forward(self, data):
        tokens, positions, mask = dpack.unpack_as_text(data)
        x = self.emb(tokens)

        if self.pe_scale is not None:
            # Compute and add absolute position embedding.
            # NOTE: we could probably precompute and keep `freqs` in a buffer!
            ifreqs = torch.arange(0, self.dim, 2, dtype=torch.float32, device=x.device)
            ifreqs /= self.dim
            freqs = 10000.0 ** (-ifreqs)

            # fmt: off
            # Note: checked that pre-allocating and in-place-ing doesn't make things better.
            x[..., 0::2] += torch.sin(positions[..., :, None] * freqs[None, :]) * self.pe_scale
            x[..., 1::2] += torch.cos(positions[..., :, None] * freqs[None, :]) * self.pe_scale
            # fmt: on

        return x * mask[..., None], {}

    def init_weights(self, rng):
        nn.init.trunc_normal_(self.emb.weight, 0, 1 / self.dim, generator=rng)
        if self.pe_scale is not None:
            # The /0.7071 gives std=1.0 then /dim to get same as above emb.weight, see https://fburl.com/anp/djuqezov
            # But then, the / 0.03 is from a sweep, see https://meta.wandb.io/axl/bv2/reports/Random-nouns-posemb-ln-vs-scalar--Vmlldzo0MDI5
            nn.init.constant_(self.pe_scale, 1 / 0.7071 / 0.03 / self.dim)


class TxtUnembedding(nn.Module):
    def __init__(self, dim, vocab, chunksz=None, init_std=0.0):
        super().__init__()
        # TODO: Should we move the pre-head LN to the unembeddings, maybe?
        self.head = nn.Linear(dim, vocab, bias=True)
        self.chunksz = chunksz
        self.init_std = init_std

    def _process_chunk(self, x, targets, loss_weights, global_total_loss_weights, mode):
        logits = self.head(x)
        pred = logits.argmax(dim=-1)

        # We need to flatten/unflatten batch_dims because of torch's cross-entropy API.
        toklosses = F.cross_entropy(
            logits.to(torch.float32).reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="none",
        )
        toklosses = toklosses.reshape(*targets.shape)

        toklosses = toklosses * (loss_weights > 0)
        lsum = (toklosses * loss_weights).sum()
        loss = lsum / global_total_loss_weights
        if mode == "loss and bwd":
            loss.backward()

        return loss.detach(), toklosses.detach(), pred.detach()

    def forward(self, x, targets, loss_weights, seqids, mode, logits_tok_idx=None):
        assert mode in ("loss and bwd", "loss", "logits"), f"Invalid mode {mode}"

        if mode == "logits":
            if logits_tok_idx is not None:
                assert x.ndim == 3, "Only works with 1D batch dimension."
                batch_indices = torch.arange(x.shape[0], device=x.device)
                return self.head(x[batch_indices, logits_tok_idx, :]), {}
            else:
                return self.head(x), {}

        targets, _, mask = dpack.unpack_as_text(targets)
        x_detached = x.detach().requires_grad_() if mode == "loss and bwd" else x

        seqlen = x_detached.shape[-2]

        total_loss = 0
        total_pplx = 0
        total_correct = 0
        predictions = torch.empty_like(targets)
        tok_losses = torch.empty_like(targets, dtype=torch.float32)
        loss_weights = loss_weights * mask

        # Sum of loss weights across all tokens and devices:
        global_total_loss_weights = loss_weights.sum()
        distr.all_reduce(global_total_loss_weights, op=distr.ReduceOp.SUM)
        global_total_loss_weights = torch.clamp(global_total_loss_weights, min=1.0)

        # How many tokens get a loss, across all devices:
        global_total_loss_toks = (loss_weights > 0).sum()
        distr.all_reduce(global_total_loss_toks, op=distr.ReduceOp.SUM)

        # NOTE: This is the case because of our choice to do static compiles without recompiles.
        # In principle we could relax it and compile two variants, or leave chunk dim dynamic.
        assert seqlen % self.chunksz == 0, f"{seqlen=} has to be chunkable by {self.chunksz=}"
        for start in range(0, seqlen, self.chunksz):
            end = start + self.chunksz

            chunk_x = x_detached[..., start:end, :]
            chunk_targets = targets[..., start:end]
            chunk_loss_weights = loss_weights[..., start:end]

            loss, tok_losses_chunk, pred = self._process_chunk(
                chunk_x, chunk_targets, chunk_loss_weights, global_total_loss_weights, mode)

            total_loss += loss
            total_pplx += tok_losses_chunk.sum()
            predictions[..., start:end] = pred
            tok_losses[..., start:end] = tok_losses_chunk
            total_correct += ((pred == chunk_targets) * (chunk_loss_weights > 0)).sum()

        if mode == "loss and bwd":  # Yes, this graph-breaks. It's ok.
            x.backward(x_detached.grad)

        extras = {
            "pplx": total_pplx,
            "ncorrect": total_correct,
            "predictions": predictions,
            "tok_losses": tok_losses,
            "global_total_loss_toks": global_total_loss_toks,
        }

        return total_loss, extras

    def init_weights(self, rng=None):
        if self.init_std > 0.0:
            nn.init.trunc_normal_(self.head.weight, 0.0, self.init_std, generator=rng)  # fmt: skip
        else:
            nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)


class PosEmbSinCos2D(nn.Module):
    """From big_vision, follows MoCo-v3 logic."""

    def __init__(self, dim):
        super().__init__()
        assert dim % 4 == 0, "Only works with multiple of 4 dimension"
        self.dim = dim

    def forward(self, positions):
        i, j = positions[..., 0], positions[..., 1]

        ifreqs = torch.arange(0, self.dim, 4, dtype=torch.float32, device=i.device)
        ifreqs /= self.dim - 1
        freqs = 10_000.0 ** (-ifreqs)
        # NOTE: could register_buffer `freqs`, but how do I handle devices??
        #       Looks not worth it though, it's only 2ms (out of >1s).

        # fmt: off
        posemb = torch.concatenate([
            torch.sin(i[..., :, None] * freqs[None, :]),
            torch.cos(i[..., :, None] * freqs[None, :]),
            torch.sin(j[..., :, None] * freqs[None, :]),
            torch.cos(j[..., :, None] * freqs[None, :]),
        ], dim=-1)
        # fmt: on
        return posemb


class ImgEmbedding(nn.Module):
    def __init__(self, dim, ph=16, pw=16, c=3, posemb=True, tiptoi=0):
        super().__init__()
        self.proj = nn.Linear((ph * pw * c) + tiptoi, dim, bias=False)
        self.ps = (ph, pw, c)

        self.ln = nn.LayerNorm(dim)

        self.ape = PosEmbSinCos2D(dim) if posemb else None
        self.ape_ln = nn.LayerNorm(dim) if posemb else None

        self.tiptoi = tiptoi

    def forward(self, data):
        # If there's no image packed into the data, skip this whole thing!
        if data.shape[-1] < dpack.nbytes_image_with_extras(
            *self.ps, tiptoi=self.tiptoi
        ):
            return 0, {}

        patches, positions, sincos, mask = dpack.unpack_as_image(
            data, *self.ps, tiptoi=self.tiptoi, keep_flat=True
        )

        # At this point, patches are raw uint8 pixels. Normalize to [-1, 1]
        patches = patches.to(self.proj.weight.dtype) / 127.5 - 1.0
        if self.tiptoi:
            patches = torch.concatenate([patches, sincos.to(patches.dtype)], dim=-1)

        x = self.ln(self.proj(patches))

        if self.ape is not None:
            x += self.ape_ln(self.ape(positions).to(x.dtype))

        return x * mask[..., None], {}  # Set non-patch token embeddings back to 0.

    def init_weights(self, rng):
        # This is Kaiming fan-in, preserves var in fwd. Not sure if best, but reasonable
        nn.init.trunc_normal_(self.proj.weight, mean=0.0, std=1/np.sqrt(np.prod(self.ps)), generator=rng)  # fmt: skip
        if self.ape is not None:
            self.ape_ln.reset_parameters()
        self.ln.reset_parameters()


class RegEmbedding(nn.Module):
    def __init__(self, dim, nreg=0, mod_id=dpack.MOD_REG):
        super().__init__()
        if nreg:
            self.emb = nn.Embedding(nreg, dim)
            self.dim = dim
        self.nreg = nreg
        self.mod_id = mod_id

    def forward(self, data):
        if self.nreg:
            regs, mask = dpack.unpack_as_reg(data, self.mod_id)
            return self.emb(regs) * mask[..., None], {}
        else:
            return 0, {}

    def init_weights(self, rng, init_zeros=False):
        if self.nreg:
            if init_zeros:
                nn.init.zeros_(self.emb.weight)
            else:
                nn.init.trunc_normal_(self.emb.weight, 0, 1 / self.dim, generator=rng)


class SimpleTransformer(nn.Module):
    def __init__(self, dim, depth, vocab,
                 stages="single", *,
                 glope=None,
                 txt_unemb={}, txt={}, img={}, reg={}, sep={}, **block_kw):  # fmt:skip
        super().__init__()
        self.img_emb = ImgEmbedding(dim=dim, **img)
        self.reg_emb = RegEmbedding(dim=dim, **reg)
        self.sep_emb = RegEmbedding(dim=dim, **{"mod_id": dpack.MOD_SEP, **sep})
        self.txt_emb = TxtEmbedding(dim=dim, vocab=vocab, **txt)
        self.txt_unemb = TxtUnembedding(dim=dim, vocab=vocab, **txt_unemb)
        self.blocks = nn.ModuleList([Block(dim, **block_kw) for _ in range(depth)])
        self.ln = nn.LayerNorm(dim)

        self.glope = nn.Embedding(glope, dim) if glope else None

        if stages == "single":
            # By default, all blocks use the same default attention/flex mask:
            self.flex_masks = ["attn_regions"] * depth
        elif stages == "half":
            # Or use mask 1 for the first blocks and mask 2 for the remaining ones.
            self.flex_masks = ["attn_regions"] * (depth // 2) + ["attn_regions2"] * (
                (depth + 1) // 2
            )
        else:
            raise ValueError(f"Not implemented model staging {stages}")
            # Or alternating, or global-local, or ...

        # Just to avoid silly mistakes making `zip` skip layers in `forward`.
        assert len(self.flex_masks) == depth

    def forward(self, tokens, flex_masks, loss_weights, seqids, mode, **mode_kw):
        assert mode in ("loss and bwd", "loss", "logits"), f"Invalid mode {mode}"

        extras = {}
        xtxt, extras["txt_emb"] = self.txt_emb(tokens)
        xreg, extras["reg_emb"] = self.reg_emb(tokens)
        xsep, extras["sep_emb"] = self.sep_emb(tokens)
        ximg, extras["img_emb"] = checkpoint(self.img_emb, tokens, use_reentrant=False)

        # Embeddings of non-relevant tokens are 0
        x = xtxt + xreg + ximg + xsep  # ...so the addition really just combines them!

        if self.glope is not None:
            x += self.glope(_seqids_to_pos(seqids))

        for i, (blk, fm_name) in enumerate(zip(self.blocks, self.flex_masks)):
            x, extras.setdefault("blk", {})[i] = blk(x, flex_masks[fm_name])
        x = self.ln(x)

        # We do the slicing here (and waste 1 token fwd pass) so we don't need to
        # adjust the `flex_mask` above according to slicing. Simplifies code overall.
        loss, extras_txt_unemb = self.txt_unemb(
            x[..., :-1, :],
            tokens[..., 1:, :],
            loss_weights[..., 1:] if loss_weights is not None else None,
            seqids[..., 1:],
            mode,
            **mode_kw,
        )
        extras.update(extras_txt_unemb)

        return loss, extras

    def init_weights(self, rng):
        self.img_emb.init_weights(rng)
        self.reg_emb.init_weights(rng)
        self.sep_emb.init_weights(rng, init_zeros=True)
        self.txt_emb.init_weights(rng)
        self.txt_unemb.init_weights(rng)
        self.ln.reset_parameters()  # TODO: better parametrization

        if self.glope is not None:
            # Start from unopinionated embeddings
            nn.init.zeros_(self.glope.weight)

        for block in self.blocks:
            block.init_weights(rng)


def _seqids_to_pos(x):
    """Converts seqids to intra-sequence positions.

    Example:
     x       [0, 0, 0, 1, 1, 2, 2, 2, 2, 3]
     returns [0, 1, 2, 0, 1, 0, 1, 2, 3, 0]

    """
    i = torch.arange(x.numel(), device=x.device)
    s = torch.zeros_like(i, dtype=torch.bool)
    s[0] = True
    s[1:] = x[1:] != x[:-1]
    return i - torch.cummax(i * s, 0).values
