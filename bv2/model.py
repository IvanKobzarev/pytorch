import numpy as np
import torch
import torch.distributed as distr
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.distributed import _functional_collectives as funcol
from torch.nn.attention.flex_attention import AuxRequest, flex_attention
from torch.utils.checkpoint import checkpoint

import bv2.data.dpack as dpack  # usort: skip
import bv2.graph_trainer_utils.adapter as gt  # usort: skip


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), weight=self.gamma.abs())

    def init_weights(self):
        nn.init.ones_(self.gamma)


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
        nn.init.trunc_normal_(self.q.weight, mean=0.0, std=1/np.sqrt(self.dim), generator=rng)
        nn.init.trunc_normal_(self.k.weight, mean=0.0, std=1/np.sqrt(self.dim), generator=rng)
        nn.init.trunc_normal_(self.v.weight, mean=0.0, std=1/np.sqrt(self.dim), generator=rng)
        nn.init.trunc_normal_(self.o.weight, mean=0.0, std=1/np.sqrt(self.dim), generator=rng)


class MLP(nn.Module):
    def __init__(self, dim, grow=4):
        super().__init__()
        self.dim = dim
        self.grow = grow
        self.l1 = nn.Linear(dim, int(grow * dim), bias=False)
        self.l2 = nn.Linear(int(grow * dim), dim, bias=False)

    def forward(self, x):
        x = self.l1(x)
        x = F.gelu(x, approximate="tanh")
        x = self.l2(x)
        return x, {}

    def init_weights(self, rng):
        nn.init.trunc_normal_(self.l1.weight, mean=0.0, std=1/np.sqrt(self.dim / 2), generator=rng)  # Kaiming fan-in, /2 for GELU
        nn.init.trunc_normal_(self.l2.weight, mean=0.0, std=1/np.sqrt(self.dim * self.grow), generator=rng)  # Kaiming fan-in


class Block(nn.Module):
    def __init__(self, dim, head_dim=128, grow=4, kv_reduce=4, remat=True):
        super().__init__()
        self.att_ln = RMSNorm(dim)
        self.mlp_ln = RMSNorm(dim)
        self.att = Attention(dim, head_dim, kv_reduce)
        self.mlp = MLP(dim, grow)
        self.remat = remat

    def forward(self, x, flex_mask):
        if self.remat:
            return checkpoint(self._block_fn, x, flex_mask, use_reentrant=False)
        return self._block_fn(x, flex_mask)

    def _block_fn(self, x, flex_mask):
        extras = {}
        y, extras["attn"] = self.att(self.att_ln(x), flex_mask)
        x = x + y
        z, extras["mlp"] = self.mlp(self.mlp_ln(x))
        x = x + z
        return x, extras

    def init_weights(self, rng):
        self.att.init_weights(rng)
        self.mlp.init_weights(rng)
        self.att_ln.init_weights()
        self.mlp_ln.init_weights()


class TxtEmbedding(nn.Module):
    def __init__(self, dim, vocab, posemb=True):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.dim = dim
        self.ln = RMSNorm(dim)
        self.ape_ln = RMSNorm(dim) if posemb else None

    def forward(self, data):
        tokens, positions, mask = dpack.unpack_as_text(data)
        x = self.ln(self.emb(tokens))

        if self.ape_ln is not None:
            ifreqs = torch.arange(0, self.dim, 2, dtype=torch.float32, device=x.device)
            freqs = 10000.0 ** (-ifreqs / self.dim)

            posemb = torch.empty_like(x)
            posemb[..., 0::2] = torch.sin(positions[..., :, None] * freqs[None, :])
            posemb[..., 1::2] = torch.cos(positions[..., :, None] * freqs[None, :])
            x = x + self.ape_ln(posemb)

        return x * mask[..., None], {}

    def init_weights(self, rng):
        nn.init.trunc_normal_(self.emb.weight, 0, 1 / np.sqrt(self.dim), generator=rng)  # 1/sqrt(dim) for dim-independent norm
        self.ln.init_weights()
        if self.ape_ln is not None:
            self.ape_ln.init_weights()


class TxtUnembedding(nn.Module):
    def __init__(self, dim, vocab, chunksz=None):
        super().__init__()
        self.head = nn.Linear(dim, vocab, bias=False)
        self.head_bias = nn.Parameter(torch.zeros(vocab))
        self.gamma_head = nn.Parameter(torch.ones(vocab))
        self.vocab = vocab
        self.chunksz = chunksz

    def _norm_logits(self, x):
        return F.rms_norm(x, (x.shape[-1],), weight=self.gamma_head.abs())

    def _loss_from_logits(self, logits, targets, loss_weights, global_total_loss_toks):
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
        loss = lsum / global_total_loss_toks
        return loss, toklosses, pred

    def _process_chunk(self, x, targets, loss_weights, global_total_loss_toks, mode):
        logits = self._norm_logits(self.head(x)) + self.head_bias
        loss, toklosses, pred = self._loss_from_logits(
            logits, targets, loss_weights, global_total_loss_toks
        )
        if mode == "loss and bwd":
            loss.backward()

        return loss.detach(), toklosses.detach(), pred.detach()

    def _process_chunk_gt(self, x, targets, loss_weights, global_total_loss_toks):
        head_weight = self.head.weight
        head_bias = self.head_bias
        gamma_head = self.gamma_head
        grad_inputs = (x, head_weight, head_bias, gamma_head)

        head_out = F.linear(x, head_weight)
        logits = F.rms_norm(
            head_out, (head_out.shape[-1],), weight=gamma_head.abs()
        ) + head_bias
        loss, toklosses, pred = self._loss_from_logits(
            logits, targets, loss_weights, global_total_loss_toks
        )
        loss_out = loss.detach()
        toklosses = toklosses.detach()
        pred = pred.detach()
        grads = torch.autograd.grad(loss, grad_inputs)
        return loss_out, toklosses, pred, grads, grad_inputs[1:]

    def forward(
        self,
        x,
        targets,
        loss_weights,
        seqids,
        mode,
        logits_tok_idx=None,
        graph_trainer=False,
    ):
        assert mode in ("loss and bwd", "loss", "logits"), f"Invalid mode {mode}"

        if mode == "logits":
            if logits_tok_idx is not None:
                assert x.ndim == 3, "Only works with 1D batch dimension."
                batch_indices = torch.arange(x.shape[0], device=x.device)
                return self._norm_logits(self.head(x[batch_indices, logits_tok_idx, :])) + self.head_bias, {}
            else:
                return self._norm_logits(self.head(x)) + self.head_bias, {}

        targets, _, mask = dpack.unpack_as_text(targets)
        x_detached = x.detach().requires_grad_() if mode == "loss and bwd" else x

        graph_trainer = mode == "loss and bwd" and graph_trainer
        # GraphTrainer captures fwd+bwd as one graph; use autograd.grad so each
        # chunk writes a small input grad into one full-seqlen accumulator.
        grad_acc = x_detached.detach() if graph_trainer else None

        seqlen = x_detached.shape[-2]

        total_loss = torch.zeros((), dtype=torch.float32, device=x_detached.device)
        total_pplx = torch.zeros((), dtype=torch.float32, device=x_detached.device)
        total_correct = torch.zeros((), dtype=torch.int64, device=x_detached.device)
        predictions = torch.empty_like(targets)
        tok_losses = torch.empty_like(targets, dtype=torch.float32)
        loss_weights = loss_weights * mask

        # How many tokens get a loss, across all devices.
        # We normalize by count(lowe > 0) so that lowe magnitude is meaningful for weighting.
        global_total_loss_toks = (loss_weights > 0).sum()
        global_total_loss_toks = funcol.wait_tensor(
            funcol.all_reduce(global_total_loss_toks, "sum", distr.group.WORLD)
        )
        global_total_loss_toks = torch.clamp(global_total_loss_toks, min=1.0)

        # NOTE: This is the case because of our choice to do static compiles without recompiles.
        # In principle we could relax it and compile two variants, or leave chunk dim dynamic.
        chunksz = self.chunksz or seqlen
        assert seqlen % chunksz == 0, f"{seqlen=} has to be chunkable by {chunksz=}"
        for start in range(0, seqlen, chunksz):
            end = start + chunksz

            chunk_x = x_detached[..., start:end, :]
            chunk_targets = targets[..., start:end]
            chunk_loss_weights = loss_weights[..., start:end]
            region_name = f"txt_unemb_chunk_{start // chunksz}"

            if graph_trainer:
                with gt.subgraph(region_name, unshard_outside=True):
                    loss, tok_losses_chunk, pred, grads, grad_params = self._process_chunk_gt(
                        chunk_x,
                        chunk_targets,
                        chunk_loss_weights,
                        global_total_loss_toks,
                    )
                    # Each chunk returns its input grad plus the head param grads;
                    # accumulate them as loss.backward would.
                    chunk_x_grad, *param_grads = grads
                    hidden_grad = chunk_x_grad.detach()
                    accum_grads = tuple(grad.detach() for grad in param_grads)
                    torch.autograd.backward(
                        grad_params,
                        accum_grads,
                    )
                grad_acc[..., start:end, :].copy_(hidden_grad)
            else:
                loss, tok_losses_chunk, pred = self._process_chunk(
                    chunk_x,
                    chunk_targets,
                    chunk_loss_weights,
                    global_total_loss_toks,
                    mode,
                )

            chunk_loss_sum = tok_losses_chunk.sum()
            total_loss = total_loss + chunk_loss_sum
            total_pplx = total_pplx + chunk_loss_sum
            predictions[..., start:end] = pred
            tok_losses[..., start:end] = tok_losses_chunk
            total_correct = total_correct + ((pred == chunk_targets) * (chunk_loss_weights > 0)).sum()

        total_loss = total_loss / global_total_loss_toks

        if mode == "loss and bwd":  # Yes, this graph-breaks. It's ok.
            x.backward(grad_acc if graph_trainer else x_detached.grad)

        extras = {
            "pplx": total_pplx,
            "ncorrect": total_correct,
            "predictions": predictions,
            "tok_losses": tok_losses,
            "global_total_loss_toks": global_total_loss_toks,
        }

        return total_loss, extras

    def init_weights(self, rng=None):
        nn.init.trunc_normal_(self.head.weight, 0.0, 1/np.sqrt(self.head.in_features), generator=rng)  # fmt: skip
        nn.init.zeros_(self.head_bias)
        nn.init.constant_(self.gamma_head, 0.01)


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

        self.ln = RMSNorm(dim)

        self.ape = PosEmbSinCos2D(dim) if posemb else None
        self.ape_ln = RMSNorm(dim) if posemb else None

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
        # This is Kaiming fan-in, preserves var in fwd.
        nn.init.trunc_normal_(self.proj.weight, mean=0.0, std=1/np.sqrt(np.prod(self.ps)), generator=rng)  # fmt: skip
        if self.ape is not None:
            self.ape_ln.init_weights()
        self.ln.init_weights()


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
                nn.init.trunc_normal_(self.emb.weight, 0, 1 / np.sqrt(self.dim), generator=rng)  # 1/sqrt(dim) for dim-independent norm


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
        self.ln = RMSNorm(dim)

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

    def forward(self, toki, toko, flex_masks, loss_weights, seqids, mode, **mode_kw):
        assert mode in ("loss and bwd", "loss", "logits"), f"Invalid mode {mode}"

        extras = {}
        xtxt, extras["txt_emb"] = self.txt_emb(toki)
        xreg, extras["reg_emb"] = self.reg_emb(toki)
        xsep, extras["sep_emb"] = self.sep_emb(toki)
        ximg, extras["img_emb"] = checkpoint(self.img_emb, toki, use_reentrant=False)

        # Embeddings of non-relevant tokens are 0
        x = xtxt + xreg + ximg + xsep  # ...so the addition really just combines them!

        if self.glope is not None:
            x += self.glope(_seqids_to_pos(seqids))

        for i, (blk, fm_name) in enumerate(zip(self.blocks, self.flex_masks)):
            x, extras.setdefault("blk", {})[i] = blk(x, flex_masks[fm_name])
        x = self.ln(x)

        # We do the slicing here (and waste 1 token fwd pass) so we don't need to
        # adjust the `flex_mask` above according to slicing. Simplifies code overall.
        loss, extras_txt_unemb = self.txt_unemb(x, toko, loss_weights, seqids, mode, **mode_kw)
        extras.update(extras_txt_unemb)
        return loss, extras

    def init_weights(self, rng):
        self.img_emb.init_weights(rng)
        self.reg_emb.init_weights(rng)
        self.sep_emb.init_weights(rng, init_zeros=True)
        self.txt_emb.init_weights(rng)
        self.txt_unemb.init_weights(rng)
        self.ln.init_weights()

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
