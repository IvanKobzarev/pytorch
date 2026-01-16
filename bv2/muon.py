import numpy as np
import torch
from torch.distributed.tensor import DTensor, Replicate


@torch.no_grad()
def ns_ortho(G, *, steps=5, eps=1e-7, a=3.4445, b=-4.7750, c=2.0315):
    orig_dtype = G.dtype
    X = G.to(dtype=torch.bfloat16)
    X = X / X.norm(dim=(-2, -1), keepdim=True).clamp_min(eps)

    if G.shape[-2] > G.shape[-1]:
        X = X.transpose(-2, -1)

    for _ in range(steps):
        A = X @ X.transpose(-2, -1)
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if G.shape[-2] > G.shape[-1]:
        X = X.transpose(-2, -1)

    return X.to(orig_dtype)


class Muon(torch.optim.Optimizer):

    def __init__(
        self,
        params,
        *,
        lr,
        muon_momentum=0.95,
        muon_nesterov=True,
        ns_steps=5,
        ns_eps=1e-7,
        beta1=0.9,
        beta2=0.99,
        eps=1e-8):

        defaults = dict(
            use_muon=True,
            lr=lr,
            muon_momentum=muon_momentum,
            muon_nesterov=muon_nesterov,
            ns_steps=ns_steps,
            ns_eps=ns_eps,
            adam_beta1=beta1,
            adam_beta2=beta2,
            adam_eps=eps,
        )

        return super().__init__(params, defaults)

    def init_state(self):
        for group in self.param_groups:
            for p in (p for p in group["params"] if p.requires_grad):
                if group["use_muon"]:
                    self.state[p]["momentum"] = torch.zeros_like(p, dtype=torch.float32)
                else:
                    self.state[p]["step"] = torch.tensor(0)
                    self.state[p]["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                    self.state[p]["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is None  # not supported

        for group in self.param_groups:
            lr = group["lr"]

            for p in (p for p in group["params"] if p.grad is not None):
                if group["use_muon"]:
                    assert p.ndim == 2

                    assert "momentum" in self.state[p], "Did you forget to run `init_state()`?"
                    m = self.state[p]["momentum"]

                    # Follows https://github.com/KellerJordan/Muon/blob/master/muon.py
                    m.lerp_(p.grad, 1 - group["muon_momentum"])
                    G = p.grad.lerp(m, group["muon_momentum"]) if group["muon_nesterov"] else m

                    G = G.to(torch.bfloat16)
                    G_full = G.redistribute(
                        placements=(Replicate(),) * p.device_mesh.ndim,
                        forward_dtype=torch.bfloat16,
                    ).to_local()

                    GO = ns_ortho(G_full, steps=group["ns_steps"], eps=group["ns_eps"])

                    # Scaling rule from https://arxiv.org/abs/2502.16982.
                    GO = GO * 0.2 * np.sqrt(max(GO.shape))

                    GO_dt = DTensor.from_local(
                        GO,
                        device_mesh=p.device_mesh,
                        placements=(Replicate(),) * p.device_mesh.ndim,
                    ).redistribute(placements=p.placements)

                    p.add_(GO_dt.to(p.dtype), alpha=-lr)
                else:
                    # Fallback to adam
                    state = self.state[p]
                    assert "step" in state, "Did you forget to run `init_state()`?"
                    state["step"] += 1
                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    exp_avg.mul_(group["adam_beta1"]).add_(p.grad, alpha=1.0 - group["adam_beta1"])
                    exp_avg_sq.mul_(group["adam_beta2"]).addcmul_(p.grad, p.grad, value=1.0 - group["adam_beta2"])
                    bias_c1 = 1.0 - group["adam_beta1"] ** state["step"]
                    bias_c2 = 1.0 - group["adam_beta2"] ** state["step"]
                    step_size = lr / bias_c1
                    denom = (exp_avg_sq / bias_c2).sqrt().add_(group["adam_eps"])
                    update = exp_avg / denom
                    p.add_(update.to(p.dtype), alpha=-step_size)

        return None

