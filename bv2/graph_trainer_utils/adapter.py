# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
import logging
import operator

import torch
from torch._functorch.partitioners import get_default_op_list
from torch.utils.checkpoint import CheckpointPolicy
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
from torchtitan.experiments.graph_trainer.region_cse import (
    extract_region_cse_into_prologue_pass,
)
from torchtitan.experiments.graph_trainer.selective_activation_remat import (
    selective_activation_remat_pass,
)
from torchtitan.experiments.graph_trainer.subgraph_regions import (
    SUBGRAPH_REGION,
    apply_subgraph_region_annotations_pass,
    subgraph as torchtitan_subgraph,
)

log = logging.getLogger(__name__)

_RECOMPUTE = {
    CheckpointPolicy.PREFER_RECOMPUTE,
    CheckpointPolicy.MUST_RECOMPUTE,
}
_RESHARD_AFTER_FORWARD_PASSTHROUGH_TARGETS = {
    operator.getitem,
    torch.ops._c10d_functional.wait_tensor.default,
    torch.ops.aten._to_copy.default,
    torch.ops.prims.convert_element_type.default,
}
_ALL_GATHER = torch.ops._c10d_functional.all_gather_into_tensor.default
_EMBEDDING = torch.ops.aten.embedding.default
_RIGI_FULL_INDUCTOR_CONFIGS = {
    "allow_buffer_reuse": False,
    "peak_aware_fusion": True,
    "size_threshold_for_succ_based_strategy": 1,
    "eager_numerics.division_rounding": False,
    "aten_distributed_optimizations.insert_overlap_deps": False,
    "aten_distributed_optimizations.enable_simple_overlap": False,
}


def subgraph(name, role=None):
    return torchtitan_subgraph(name, role=role)


def reshard_after_forward_pass(gm, example_inputs=None):
    """Reshard-after-forward for root-graph FSDP all-gathers.

    make_fx traces one fused fwd+bwd graph, so each parameter is all-gathered
    once in the forward and the unsharded buffer stays live until its backward
    consumer -- e.g. the input-embedding weight co-lives with the unembedding
    weight at the loss-head peak (the "double overlapping buffers").

    Tagging the forward all-gather -- and the view/getitem/wait/_to_copy chain up
    to its consumer -- PREFER_RECOMPUTE makes ``selective_activation_remat_pass``
    recompute the unshard just before its backward consumer: the forward copy
    frees after the forward use and the weight is re-gathered in backward.
    PREFER_RECOMPUTE also exempts the collective from ``force_save_collectives``
    (``must_recompute()`` becomes true).

    Views MUST be crossed (``op_types.is_view``): DTensor emits ``view_as`` /
    ``slice`` between the collective and its consumer, and if the tag stops at a
    view the remat walk never reaches the collective and the pass silently
    no-ops -- the log line below is the guardrail against that regression.

    Loss-head-chunk all-gathers (``SUBGRAPH_REGION`` nodes) are skipped -- that
    fused fwd+bwd region needs the weight at both ends, so the per-region
    min-cut owns them. Every other (block + embedding) param is resharded.
    """
    op_types = get_default_op_list()
    tagged_ag = 0
    tagged_passthrough = 0
    tagged_embedding = 0
    saved_ag = 0
    for node in gm.graph.nodes:
        if node.op != "call_function" or node.meta.get("autograd_backward", False):
            continue
        custom = node.meta.get("custom") or {}
        if node.target == _ALL_GATHER and SUBGRAPH_REGION not in custom:
            if node.meta.get("recompute") == CheckpointPolicy.MUST_SAVE:
                saved_ag += 1
                continue
            node.meta["recompute"] = CheckpointPolicy.PREFER_RECOMPUTE
            tagged_ag += 1
        elif (
            node.target in _RESHARD_AFTER_FORWARD_PASSTHROUGH_TARGETS
            or op_types.is_view(node)
        ) and any(
            inp.meta.get("recompute") in _RECOMPUTE for inp in node.all_input_nodes
        ):
            node.meta["recompute"] = CheckpointPolicy.PREFER_RECOMPUTE
            tagged_passthrough += 1
        elif (
            node.target == _EMBEDDING
            and SUBGRAPH_REGION not in custom
            and node.args[0].meta.get("recompute") in _RECOMPUTE
        ):
            # Embedding backward needs the embedding output. Recomputing this cheap
            # consumer lets the full vocab all-gather die after forward use.
            node.meta["recompute"] = CheckpointPolicy.PREFER_RECOMPUTE
            tagged_embedding += 1
    log.info(
        "reshard_after_forward_pass: tagged %d root all-gathers + %d passthrough "
        "+ %d embedding nodes PREFER_RECOMPUTE; preserved %d root all-gathers "
        "MUST_SAVE",
        tagged_ag,
        tagged_passthrough,
        tagged_embedding,
        saved_ag,
    )
    if tagged_ag == 0:
        log.warning(
            "reshard_after_forward_pass: tagged 0 root all-gathers -- reshard is a "
            "no-op (check SUBGRAPH_REGION scoping / graph structure)"
        )
    return gm


def _graph_passes(_traced_result):
    # Reshard-after-forward is the default GraphTrainer behavior: tag root-graph
    # FSDP all-gathers PREFER_RECOMPUTE, then materialize that recompute before
    # annotating/inlining loss-head subgraph regions.
    passes = [
        eliminate_dead_code_pass,
        # Share repeated per-chunk unshards before outlining regions.
        extract_region_cse_into_prologue_pass,
        reshard_after_forward_pass,
        selective_activation_remat_pass,
        functools.partial(
            apply_subgraph_region_annotations_pass,
            min_cut_rematerialization=True,
        ),
    ]
    passes.append(
        functools.partial(
            full_inductor_compilation_pass,
            inductor_configs=_RIGI_FULL_INDUCTOR_CONFIGS,
        )
    )
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
