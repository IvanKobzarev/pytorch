# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools

import torch
from torchtitan.experiments.graph_trainer.common_utils import (
    maybe_register_blockmask_pytree_node,
)
from torchtitan.experiments.graph_trainer.inductor_passes import (
    full_inductor_compilation_pass,
)
from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    minimal_fx_tracer,
    run_traced,
)
from torchtitan.experiments.graph_trainer.passes import apply_graph_passes
from torchtitan.experiments.graph_trainer.remove_noop_passes import (
    eliminate_dead_code_pass,
)
from torchtitan.experiments.graph_trainer.subgraph_regions import (
    apply_subgraph_region_annotations_pass,
    subgraph as torchtitan_subgraph,
)

_FUSED_FW_BW_GRAD_ACCUM_ROLE = "fw_bw_grad_accum"


def subgraph(name, role=None, unshard_outside=False):
    if unshard_outside and role is None:
        role = _FUSED_FW_BW_GRAD_ACCUM_ROLE
    return torchtitan_subgraph(name, role=role, unshard_outside=unshard_outside)


def _graph_passes(_traced_result):
    passes = [eliminate_dead_code_pass]
    passes.append(
        functools.partial(
            apply_subgraph_region_annotations_pass,
            min_cut_rematerialization=True,
        )
    )
    passes.append(full_inductor_compilation_pass)
    return passes


def make_train_step_dispatcher(model, optim, decay_param_names):
    def _make_fx_fwd_bwd_step(weight_decay, *a):
        optim.zero_grad(set_to_none=True)
        loss, extras = model(
            *a,
            mode="loss and bwd",
            graph_trainer=True,
        )
        optim.step()
        if weight_decay is not None:
            with torch.no_grad():
                # minimal_fx_tracer reparametrizes module state, so mutate the
                # traced graph inputs rather than original Parameter objects.
                params = dict(model.named_parameters(remove_duplicate=False))
                for name in decay_param_names:
                    params[name].mul_(1.0 - weight_decay)
        return loss.detach(), extras

    return _MinimalFxDispatcher(
        _make_fx_fwd_bwd_step, module=model, optimizer=optim
    )


class _MinimalFxDispatcher:
    def __init__(self, fn, *, module=None, optimizer=None):
        self.fn = fn
        self.module = module
        self.optimizer = optimizer
        self.traced = None

    def __call__(self, *args, **kwargs):
        maybe_register_blockmask_pytree_node()
        if self.traced is None:
            traced = minimal_fx_tracer(
                self.fn, module=self.module, optimizer=self.optimizer
            )(*args, **kwargs)
            traced.gm = apply_graph_passes(
                traced.gm, traced.example_inputs, _graph_passes(traced)
            )
            self.traced = traced
        return run_traced(self.traced, module=self.module, optimizer=self.optimizer)(
            *args, **kwargs
        )
