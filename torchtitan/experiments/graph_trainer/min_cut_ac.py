# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Min-cut activation checkpointing.

The memory policy runs first and sets the activation-checkpointing contract:

  - MUST_SAVE / MUST_RECOMPUTE are hard constraints unless the user explicitly
    enabled the relax pass before min-cut.
  - PREFER_SAVE / PREFER_RECOMPUTE are soft save candidates. They are recomputed
    unless min-cut upgrades them to MUST_SAVE.

selective_activation_remat runs last to materialize the choice (duplicate the
recompute ops into backward). This pass normally edits only
``node.meta["recompute"]`` tags. Optional canonical fp32 ``aten._log_softmax``
decomposition before min-cut is an explicit experiment: compile's AOT
partitioner sees the row-stat form, so GraphTrainer can expose the same ``amax``
and ``log`` save candidates while keeping full softmax intermediates
recompute-only.

``ac_relax_relaxable_must_saves`` is an explicit opt-in pre-pass: it downgrades
eligible MUST_SAVE activations to PREFER_SAVE before min-cut, so min-cut may
replace rigid saves with a budgeted save set. When invoked through
``min_cut_ac_pass``, the peak budget stays anchored to the memory-policy floor
before relaxation. It is off by default.

``ac_allow_allowed_saves`` is an explicit opt-in experiment pre-pass: it marks
every eligible non-MUST forward activation as PREFER_SAVE so ``save_scope=all``
can search a broader soft-save pool. Storage ownership and cutability are still
checked later by min-cut. ``ac_allow_unsaveable_recomputes`` is the narrower
variant for deterministic but unsaveable boundaries such as pre-decomposition
softmax outputs: it makes them recomputable without widening ordinary save
candidates. Both are off by default.

    memory policy (floor)  ->  min_cut_ac_pass  ->  selective_activation_remat
      MUST_SAVE /               upgrade selected     materialize: only
      MUST_RECOMPUTE /          PREFER_* to          MUST_SAVE stays saved
      PREFER_*                  MUST_SAVE            (tags only)

When ``ac_min_cut_enabled`` is true, the pass profiles the current policy first:
``reference_peak = peak(reference tags after normal memory policy and mandatory
safety normalization)``. Optional relax/allow pre-passes widen the candidate pool
after that reference is measured. It then computes the min-cut frontier and
greedily saves candidates using peak-progress ranking when over target while
satisfying ``reference_peak + ac_min_cut_max_peak_increase_gb``. The default
budget is 0GB, so enabling min-cut does not allow modelled peak regression.
Negative budgets require a lower peak than the reference policy; positive budgets
spend extra peak for less recompute.

``ac_min_cut_save_scope`` controls the candidate pool: ``min_cut`` searches only
the frontier, while ``all`` processes the sorted frontier first, then sorted
remaining eligible PREFER_* activations. Functional collectives are never save
candidates. They may still disappear from the recompute closure when min-cut
saves a downstream activation, matching the compile partitioner semantics.
Greedy tie-breaks use deterministic graph order so every rank selects the
identical set.

``ac_min_cut_memory_estimator`` controls candidate peak checks:
  - approximate (default): use a memory curve built from Inductor's
    GraphAliasTracker plus remat-closure chain modelling for ordering and
    peak-progress ranking. Candidates that pass the memory-curve check are
    exact-checked with build_memory_profile on a throwaway remat before they
    are committed.
  - exact: slow debugging mode for approximate; tentatively save each candidate,
    remat, and measure build_memory_profile before keeping it without
    memory-curve pruning.

The exact remat peak (build_memory_profile on a throwaway remat) is a static-graph
upper bound that holds the FSDP all-gathered params, so the budget is a *relative*
knob, not the absolute GPU peak.
"""

from __future__ import annotations

import hashlib
import json
import math
import operator
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TypeAlias

import torch
import torch.fx as fx
from torch._functorch.partitioners import _size_of, get_default_op_list
from torch._inductor.fx_passes.overlap_scheduling import estimate_roofline_runtime_ms
from torch._logging import trace_structured
from torch.fx.node import map_arg
from torch.utils.checkpoint import CheckpointPolicy

from torchtitan.experiments.graph_trainer.common_utils import (
    _is_backward_node,
    _MODULE_FQN,
)
from torchtitan.experiments.graph_trainer.memory_utils import (
    _apply_delta,
    _build_memory_curve,
    _candidate_memory_curve_delta,
    _cuda_output_storage_nbytes_by_node,
    _memory_curve_peak,
    _memory_profile_after_remat,
    _MemoryCurve,
    _NO_TAG,
    _peak_after_remat,
    _recompute_nodes_closure_and_uses,
    _RECOMPUTE_POLICIES,
    _restore_recompute_tags,
)
from torchtitan.tools.logging import logger


_SAVE_POLICIES = (CheckpointPolicy.MUST_SAVE,)
_PREFER_POLICIES = (
    CheckpointPolicy.PREFER_SAVE,
    CheckpointPolicy.PREFER_RECOMPUTE,
)
_HARD_POLICIES = (
    CheckpointPolicy.MUST_SAVE,
    CheckpointPolicy.MUST_RECOMPUTE,
    CheckpointPolicy.MUST_CPU_OFFLOAD,
)
_GB = 1e9
_COST_EPS = 1e-12
_SaveScope: TypeAlias = Literal["min_cut", "all"]
_SelectionMode: TypeAlias = Literal["budgeted", "compile_like", "chain_cut"]
_MemoryEstimator: TypeAlias = Literal["approximate", "exact"]

_MEMORY_ESTIMATORS: tuple[_MemoryEstimator, ...] = ("approximate", "exact")
_SAVE_SCOPES: tuple[_SaveScope, ...] = ("min_cut", "all")
_SELECTION_MODES: tuple[_SelectionMode, ...] = (
    "budgeted",
    "compile_like",
    "chain_cut",
)
_PEAK_PROGRESS_TOLERANCE_BYTES = 1 << 20
_PEAK_PROGRESS_RANKING_MIN_EXCESS_RATIO = 0.01
_MIN_NONPOSITIVE_BUDGET_BROAD_ONLY_CANDIDATES = 0
_LOG_SAMPLE_LIMIT = 8
_MIN_CUT_SOFTMAX_DECOMP = "graph_trainer_min_cut_softmax_decomp"
_SOFTMAX_ANCHOR_MAX_CHAIN_NODES = 16


def _gb_value(value: float | int) -> float | str:
    if math.isinf(value):
        return "inf"
    if math.isnan(value):
        return "nan"
    return round(float(value) / _GB, 6)


def _log_payload(event: str, payload: dict[str, object]) -> None:
    logger.info(
        "min_cut_ac %s: %s",
        event,
        json.dumps(payload, sort_keys=True),
    )
    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": f"graph_trainer_min_cut_ac_{event}",
            "encoding": "json",
        },
        payload_fn=lambda: json.dumps(payload, sort_keys=True),
        expect_trace_id=False,
    )


def _policy_name(policy: object) -> str:
    if isinstance(policy, CheckpointPolicy):
        return policy.name
    if policy is _NO_TAG:
        return "NO_TAG"
    return str(policy)


def _module_fqn(node: fx.Node) -> str:
    custom = node.meta.get("custom")
    if isinstance(custom, dict):
        return str(custom.get(_MODULE_FQN, ""))
    return ""


def _node_shape(node: fx.Node) -> list[str] | str:
    val = node.meta.get("val")
    if isinstance(val, torch.Tensor):
        return [str(dim) for dim in val.shape]
    if isinstance(val, (tuple, list)):
        return "nested"
    if val is None:
        return "missing"
    return type(val).__name__


def _node_dtype(node: fx.Node) -> str:
    val = node.meta.get("val")
    if isinstance(val, torch.Tensor):
        return str(val.dtype).replace("torch.", "")
    if isinstance(val, (tuple, list)):
        dtypes = {
            str(item.dtype).replace("torch.", "")
            for item in val
            if isinstance(item, torch.Tensor)
        }
        return ",".join(sorted(dtypes)) if dtypes else "nested"
    return type(val).__name__ if val is not None else "missing"


def _target_name(node: fx.Node) -> str:
    return str(node.target)


def _node_bucket(node: fx.Node) -> str:
    target = _target_name(node).lower()
    fqn = _module_fqn(node).lower()
    if _is_backward_node(node):
        return "backward"
    if _is_collective(node):
        return "collective"
    if any(term in fqn for term in ("loss", "unemb", "lm_head", "output")):
        return "output_or_loss"
    if "flex_attention" in target or "attention" in fqn:
        return "attention"
    if any(term in target for term in ("mm", "matmul", "addmm", "linear")):
        return "matmul"
    if any(term in target for term in ("sum", "mean", "amax", "prod", "var")):
        return "reduction"
    if "norm" in target or "norm" in fqn:
        return "norm"
    if any(
        term in target
        for term in (
            "view",
            "reshape",
            "slice",
            "select",
            "transpose",
            "permute",
            "expand",
            "getitem",
            "detach",
            "as_strided",
        )
    ):
        return "view_or_alias"
    if any(term in target for term in ("copy", "clone", "_to_copy", "to.dtype")):
        return "copy_or_dtype"
    if any(
        term in target
        for term in (
            "add",
            "sub",
            "mul",
            "div",
            "relu",
            "silu",
            "gelu",
            "sigmoid",
            "tanh",
            "sin",
            "cos",
        )
    ):
        return "residual_or_pointwise"
    return "other"


def _node_log_payload(node: fx.Node) -> dict[str, object]:
    num_bytes = _node_log_byte_size(node)
    return {
        "name": node.name,
        "target": str(node.target),
        "module_fqn": _module_fqn(node),
        "bucket": _node_bucket(node),
        "policy": _policy_name(node.meta.get("recompute", _NO_TAG)),
        "shape": _node_shape(node),
        "dtype": _node_dtype(node),
        "bytes": num_bytes,
        "gb": _gb_value(num_bytes),
    }


def _ordered_nodes(
    gm: fx.GraphModule, nodes: set[fx.Node] | list[fx.Node]
) -> list[fx.Node]:
    order = {node: index for index, node in enumerate(gm.graph.nodes)}
    return sorted(nodes, key=lambda node: order.get(node, math.inf))


def _node_digest(gm: fx.GraphModule, nodes: set[fx.Node] | list[fx.Node]) -> str:
    h = hashlib.sha256()
    for node in _ordered_nodes(gm, nodes):
        h.update(node.name.encode())
        h.update(b"\0")
        h.update(_target_name(node).encode())
        h.update(b"\0")
        h.update(_module_fqn(node).encode())
        h.update(b"\0")
        h.update(str(_node_shape(node)).encode())
        h.update(b"\0")
        h.update(str(_node_log_byte_size(node)).encode())
        h.update(b"\n")
    return h.hexdigest()[:12]


def _node_names(gm: fx.GraphModule, nodes: set[fx.Node] | list[fx.Node]) -> list[str]:
    return [node.name for node in _ordered_nodes(gm, nodes)]


def _node_log_payloads(
    gm: fx.GraphModule, nodes: set[fx.Node] | list[fx.Node]
) -> list[dict[str, object]]:
    return [_node_log_payload(node) for node in _ordered_nodes(gm, nodes)]


def _bucket_counts(nodes: set[fx.Node] | list[fx.Node]) -> dict[str, int]:
    return dict(Counter(_node_bucket(node) for node in nodes))


def _module_counts(
    nodes: set[fx.Node] | list[fx.Node], limit: int = 12
) -> dict[str, int]:
    counts = Counter(_module_fqn(node) or "<none>" for node in nodes)
    return dict(counts.most_common(limit))


# --------------------------------------------------------------------------- #
# Per-node cost helpers
# --------------------------------------------------------------------------- #
def _is_nondeterministic(node: fx.Node) -> bool:
    return (
        hasattr(node.target, "tags")
        and torch.Tag.nondeterministic_seeded in node.target.tags
    )


def _node_byte_size(node: fx.Node) -> int:
    if "val" in node.meta:
        return _size_of(node)
    return 0


def _node_log_byte_size(node: fx.Node) -> int:
    try:
        return _node_byte_size(node)
    except Exception:
        return 0


def _is_collective(node: fx.Node) -> bool:
    """True for functional collectives and their wait_tensor."""
    return node.op == "call_function" and "c10d_functional" in str(node.target)


def _is_mutable(node: fx.Node) -> bool:
    schema = getattr(node.target, "_schema", None)
    return bool(schema is not None and schema.is_mutable)


def _is_pre_decomp_softmax_boundary(node: fx.Node) -> bool:
    return node.target in {
        torch.ops.aten._log_softmax.default,
        torch.ops.aten._softmax.default,
    }


def _fake_mode_for_node(node: fx.Node):
    for input_node in node.all_input_nodes:
        val = input_node.meta.get("val")
        fake_mode = getattr(val, "fake_mode", None)
        if fake_mode is not None:
            return fake_mode
    return None


def _set_inserted_softmax_meta(
    node: fx.Node,
    source: fx.Node,
    recompute: CheckpointPolicy | None,
) -> None:
    meta = dict(source.meta)
    meta.pop("eager_input_vals", None)
    meta.pop("tensor_meta", None)
    meta.pop("recompute", None)
    meta[_MIN_CUT_SOFTMAX_DECOMP] = True
    if recompute is not None:
        meta["recompute"] = recompute
    node.meta = meta

    fake_mode = _fake_mode_for_node(node)
    if fake_mode is None:
        return

    def _arg_val(arg):
        return arg.meta["val"] if isinstance(arg, fx.Node) else arg

    args = map_arg(node.args, _arg_val)
    kwargs = map_arg(node.kwargs, _arg_val)
    with fake_mode:
        node.meta["val"] = node.target(*args, **kwargs)


def _mark_softmax_detach_aliases_must_recompute(node: fx.Node) -> None:
    stack = list(node.users)
    seen: set[fx.Node] = set()
    while stack:
        user = stack.pop()
        if user in seen:
            continue
        seen.add(user)
        if user.target != torch.ops.aten.detach.default:
            continue
        user.meta["recompute"] = CheckpointPolicy.MUST_RECOMPUTE
        stack.extend(user.users)


def _decompose_log_softmax_for_min_cut(gm: fx.GraphModule) -> int:
    op_types = get_default_op_list()
    cuda_storage_bytes = _cuda_output_storage_nbytes_by_node(gm)
    decomposed = 0
    for node in list(gm.graph.nodes):
        if (
            node.op != "call_function"
            or node.target != torch.ops.aten._log_softmax.default
        ):
            continue
        if len(node.args) < 3:
            continue
        x, dim, half_to_float = node.args[:3]
        if not isinstance(x, fx.Node) or not isinstance(dim, int) or half_to_float:
            continue
        x_val = x.meta.get("val")
        if not isinstance(x_val, torch.Tensor) or x_val.dtype != torch.float32:
            continue

        anchor = _pre_decomp_softmax_save_anchor(node, op_types, cuda_storage_bytes)
        if anchor is not None:
            chain = _pre_decomp_softmax_recompute_chain(node, anchor, op_types)
            if chain is not None:
                if anchor.meta.get("recompute") != CheckpointPolicy.MUST_SAVE:
                    anchor.meta["recompute"] = CheckpointPolicy.MUST_SAVE
                for chain_node in chain:
                    chain_node.meta["recompute"] = CheckpointPolicy.MUST_RECOMPUTE

        dim_list = [dim]
        with gm.graph.inserting_before(node):
            amax = gm.graph.call_function(
                torch.ops.aten.amax.default, args=(x, dim_list, True)
            )
            _set_inserted_softmax_meta(amax, node, CheckpointPolicy.MUST_SAVE)

            shifted = gm.graph.call_function(torch.ops.aten.sub.Tensor, args=(x, amax))
            _set_inserted_softmax_meta(shifted, node, CheckpointPolicy.MUST_RECOMPUTE)

            exp = gm.graph.call_function(torch.ops.aten.exp.default, args=(shifted,))
            _set_inserted_softmax_meta(exp, node, CheckpointPolicy.MUST_RECOMPUTE)

            sum_exp = gm.graph.call_function(
                torch.ops.aten.sum.dim_IntList, args=(exp, dim_list, True)
            )
            _set_inserted_softmax_meta(sum_exp, node, CheckpointPolicy.MUST_RECOMPUTE)

            log_sum_exp = gm.graph.call_function(
                torch.ops.aten.log.default, args=(sum_exp,)
            )
            _set_inserted_softmax_meta(log_sum_exp, node, CheckpointPolicy.MUST_SAVE)

            result = gm.graph.call_function(
                torch.ops.aten.sub.Tensor, args=(shifted, log_sum_exp)
            )
            _set_inserted_softmax_meta(result, node, CheckpointPolicy.MUST_RECOMPUTE)

        node.replace_all_uses_with(result)
        _mark_softmax_detach_aliases_must_recompute(result)
        gm.graph.erase_node(node)
        decomposed += 1

    if decomposed:
        gm.graph.lint()
        gm.recompile()
        _log_payload(
            "softmax_decomp_summary",
            {
                "decomposed": decomposed,
                "policy": "save_amax_log_recompute_full_intermediates",
            },
        )
    return decomposed


def _is_pre_decomp_softmax_alias(node: fx.Node) -> bool:
    if node.target != torch.ops.aten.detach.default:
        return False
    input_node = node.args[0] if node.args else None
    while (
        isinstance(input_node, fx.Node)
        and input_node.target == torch.ops.aten.detach.default
    ):
        input_node = input_node.args[0] if input_node.args else None
    return isinstance(input_node, fx.Node) and _is_pre_decomp_softmax_boundary(
        input_node
    )


def _pre_decomp_softmax_origin(node: fx.Node) -> fx.Node | None:
    if _is_pre_decomp_softmax_boundary(node):
        return node
    if node.target != torch.ops.aten.detach.default:
        return None
    input_node = node.args[0] if node.args else None
    while (
        isinstance(input_node, fx.Node)
        and input_node.target == torch.ops.aten.detach.default
    ):
        input_node = input_node.args[0] if input_node.args else None
    if isinstance(input_node, fx.Node) and _is_pre_decomp_softmax_boundary(input_node):
        return input_node
    return None


def _pre_decomp_softmax_save_anchor(
    node: fx.Node,
    op_types,
    cuda_storage_bytes: dict[fx.Node, int],
) -> fx.Node | None:
    anchor, _ = _pre_decomp_softmax_anchor_and_chain(node, op_types, cuda_storage_bytes)
    return anchor


def _has_cuda_tensor_value(node: fx.Node) -> bool:
    stack = [node.meta.get("val")]
    while stack:
        val = stack.pop()
        if isinstance(val, torch.Tensor):
            if val.device.type != "cpu":
                return True
        elif isinstance(val, (tuple, list)):
            stack.extend(val)
    return False


def _softmax_anchor_fallback(
    node: fx.Node,
    op_types,
    cuda_storage_bytes: dict[fx.Node, int],
) -> tuple[fx.Node | None, list[fx.Node] | None]:
    softmax = _pre_decomp_softmax_origin(node)
    if softmax is None or not softmax.args:
        return None, None
    anchor = softmax.args[0]
    chain: list[fx.Node] = []
    seen: set[fx.Node] = set()
    while isinstance(anchor, fx.Node) and anchor not in seen:
        seen.add(anchor)
        if (
            anchor.target == torch.ops.aten._to_copy.default
            or anchor.target == torch.ops.aten.detach.default
            or op_types.is_view(anchor)
        ):
            next_anchor = anchor.args[0] if anchor.args else None
            if isinstance(next_anchor, fx.Node):
                chain.append(anchor)
                anchor = next_anchor
                continue
        break
    if isinstance(anchor, fx.Node) and math.isfinite(
        _node_weight(anchor, op_types, cuda_storage_bytes)
    ):
        return anchor, chain
    return None, None


def _softmax_chain_predecessor(node: fx.Node) -> fx.Node | None:
    if node.target is operator.getitem:
        input_node = node.args[0] if node.args else None
        return input_node if isinstance(input_node, fx.Node) else None
    if node.target == torch.ops.aten._fused_rms_norm.default:
        input_node = node.args[0] if node.args else None
        return input_node if isinstance(input_node, fx.Node) else None

    input_nodes = [
        input_node
        for input_node in node.all_input_nodes
        if _has_cuda_tensor_value(input_node)
    ]
    if not input_nodes:
        return None
    return max(input_nodes, key=_node_log_byte_size)


def _can_recompute_pre_softmax_node(node: fx.Node, op_types) -> bool:
    if node.op != "call_function" or _is_backward_node(node):
        return False
    policy = node.meta.get("recompute")
    if policy in _HARD_POLICIES and policy != CheckpointPolicy.MUST_RECOMPUTE:
        return False
    if not _has_cuda_tensor_value(node):
        return False
    if _is_collective(node):
        return False
    if _must_not_recompute(node, op_types):
        return False
    if (
        node.target == torch.ops.aten._to_copy.default
        or node.target == torch.ops.aten.detach.default
        or node.target is operator.getitem
        or op_types.is_view(node)
        or node.target == torch.ops.aten._fused_rms_norm.default
    ):
        return True
    return op_types.is_recomputable(node) and not op_types.is_compute_intensive(node)


def _pre_decomp_softmax_anchor_and_chain(
    node: fx.Node,
    op_types,
    cuda_storage_bytes: dict[fx.Node, int],
) -> tuple[fx.Node | None, list[fx.Node] | None]:
    softmax = _pre_decomp_softmax_origin(node)
    if softmax is None or not softmax.args:
        return None, None

    input_node = softmax.args[0]
    current = input_node
    chain: list[fx.Node] = []
    seen: set[fx.Node] = set()
    while isinstance(current, fx.Node) and current not in seen:
        seen.add(current)
        if op_types.is_compute_intensive(current) and math.isfinite(
            _node_weight(current, op_types, cuda_storage_bytes)
        ):
            return current, chain
        if len(chain) >= _SOFTMAX_ANCHOR_MAX_CHAIN_NODES:
            break
        if not _can_recompute_pre_softmax_node(current, op_types):
            break
        next_node = _softmax_chain_predecessor(current)
        if not isinstance(next_node, fx.Node):
            break
        chain.append(current)
        current = next_node

    return _softmax_anchor_fallback(node, op_types, cuda_storage_bytes)


def _pre_decomp_softmax_recompute_chain(
    node: fx.Node,
    anchor: fx.Node,
    op_types,
) -> list[fx.Node] | None:
    owning_module = node.graph.owning_module
    if owning_module is not None:
        cuda_storage_bytes = _cuda_output_storage_nbytes_by_node(owning_module)
        found_anchor, chain = _pre_decomp_softmax_anchor_and_chain(
            node, op_types, cuda_storage_bytes
        )
        if found_anchor is anchor:
            return chain

    softmax = _pre_decomp_softmax_origin(node)
    if softmax is None or not softmax.args:
        return None
    input_node = softmax.args[0]
    chain: list[fx.Node] = []
    seen: set[fx.Node] = set()
    while isinstance(input_node, fx.Node) and input_node not in seen:
        if input_node is anchor:
            return chain
        seen.add(input_node)
        if not (
            input_node.target == torch.ops.aten._to_copy.default
            or input_node.target == torch.ops.aten.detach.default
            or op_types.is_view(input_node)
        ):
            return None
        next_input = input_node.args[0] if input_node.args else None
        if not isinstance(next_input, fx.Node):
            return None
        chain.append(input_node)
        input_node = next_input
    return None


def _can_recompute_softmax_chain_node(node: fx.Node, op_types) -> bool:
    if node.op != "call_function" or _is_backward_node(node):
        return False
    policy = node.meta.get("recompute")
    if policy in _HARD_POLICIES and policy != CheckpointPolicy.MUST_RECOMPUTE:
        return False
    if not _has_cuda_tensor_value(node):
        return False
    if _is_collective(node):
        return False
    return not _must_not_recompute(node, op_types)


def _is_unsaveable_alias(node: fx.Node, op_types) -> bool:
    if node.target is operator.getitem:
        return False
    if node.target == torch.ops.aten.detach.default:
        input_node = node.args[0] if node.args else None
        return not (
            isinstance(input_node, fx.Node) and input_node.target is operator.getitem
        )
    return op_types.is_view(node) or node.target == torch.ops.aten.detach.default


def _must_not_recompute(node: fx.Node, op_types) -> bool:
    """Intrinsic remat bans independent of the current policy tag."""
    return op_types.is_random(node) or _is_nondeterministic(node) or _is_mutable(node)


def _must_not_save(node: fx.Node, op_types) -> bool:
    """Hard save bans excluding storage ownership.

    Candidate weight is based on CUDA output storage, including projected
    outputs of multi-output ops. Plain view/detach aliases are still rejected;
    they can be cut at their producer instead.
    """
    if "val" not in node.meta:
        return True
    val = node.meta["val"]
    if not isinstance(val, torch.Tensor):
        return True
    if val.device.type == "cpu":
        return True
    if _is_mutable(node):
        return True
    if _is_pre_decomp_softmax_boundary(node):
        return True
    if _is_unsaveable_alias(node, op_types):
        return True
    if node.meta.get("recompute") == CheckpointPolicy.MUST_RECOMPUTE:
        return True
    return _is_collective(node)


def _allowed_save_candidate(node: fx.Node, op_types) -> bool:
    return not _must_not_save(node, op_types) and not _must_not_recompute(
        node, op_types
    )


def _allowed_recompute_candidate(node: fx.Node, op_types) -> bool:
    if node.op != "call_function" or _is_backward_node(node):
        return False
    if node.meta.get("recompute") in _HARD_POLICIES:
        return False
    if "val" not in node.meta:
        return False
    val = node.meta["val"]
    if not isinstance(val, torch.Tensor) or val.device.type == "cpu":
        return False
    if _is_collective(node):
        return False
    if not (
        _is_pre_decomp_softmax_boundary(node) or _is_pre_decomp_softmax_alias(node)
    ):
        return False
    return not _must_not_recompute(node, op_types)


def _node_weight(
    node: fx.Node,
    op_types,
    cuda_storage_bytes: dict[fx.Node, int],
) -> float:
    # Capacity of the node's in->out edge in the flow network: the bytes the
    # min-cut "pays" to save this activation. Min-cut minimizes total saved
    # bytes, so smaller (cheaper-to-save) tensors are preferred as cut points.
    if node.op == "placeholder":
        return 0.0  # params/inputs are always resident -- saving them is free
    if not _allowed_save_candidate(node, op_types):
        return math.inf
    if node not in cuda_storage_bytes:
        return math.inf
    return float(cuda_storage_bytes[node])


def _candidate_reject_reason(
    node: fx.Node,
    op_types,
    cuda_storage_bytes: dict[fx.Node, int],
) -> str:
    if node.op != "call_function":
        return "not_call_function"
    if _is_backward_node(node):
        return "backward_node"
    policy = node.meta.get("recompute")
    if policy not in _PREFER_POLICIES:
        return f"policy_{_policy_name(policy)}"
    if "val" not in node.meta:
        return "missing_val"
    val = node.meta["val"]
    if not isinstance(val, torch.Tensor):
        return "non_tensor"
    if val.device.type == "cpu":
        return "cpu_tensor"
    if _is_mutable(node):
        return "mutable"
    if _is_pre_decomp_softmax_boundary(node):
        return "pre_decomp_softmax_boundary"
    if _is_unsaveable_alias(node, op_types):
        return "view_or_alias"
    if _is_collective(node):
        return "collective"
    if op_types.is_random(node):
        return "random"
    if _is_nondeterministic(node):
        return "nondeterministic"
    if node not in cuda_storage_bytes:
        return "no_cuda_output_storage"
    return "eligible"


def _candidate_eligibility_payload(gm: fx.GraphModule) -> dict[str, object]:
    op_types = get_default_op_list()
    cuda_storage_bytes = _cuda_output_storage_nbytes_by_node(gm)
    forward_nodes: list[fx.Node] = []
    backward_nodes = 0
    policy_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    rejected_sample: list[dict[str, object]] = []
    eligible_nodes: list[fx.Node] = []
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        if _is_backward_node(node):
            backward_nodes += 1
            continue
        forward_nodes.append(node)
        policy_counts[_policy_name(node.meta.get("recompute", _NO_TAG))] += 1
        reason = _candidate_reject_reason(node, op_types, cuda_storage_bytes)
        reason_counts[reason] += 1
        if reason == "eligible":
            eligible_nodes.append(node)
        elif (
            node.meta.get("recompute") in _PREFER_POLICIES
            and len(rejected_sample) < _LOG_SAMPLE_LIMIT
        ):
            rejected_sample.append(
                {
                    "reason": reason,
                    "node": _node_log_payload(node),
                }
            )
    eligible_by_size = sorted(
        eligible_nodes,
        key=lambda node: (_node_byte_size(node), node.name),
        reverse=True,
    )
    return {
        "forward_call_functions": len(forward_nodes),
        "backward_call_functions": backward_nodes,
        "cuda_output_storage_nodes": len(cuda_storage_bytes),
        "policy_counts": dict(policy_counts),
        "forward_bucket_counts": _bucket_counts(forward_nodes),
        "candidate_reason_counts": dict(reason_counts),
        "eligible_candidates": len(eligible_nodes),
        "eligible_bytes": sum(_node_byte_size(node) for node in eligible_nodes),
        "eligible_gb": _gb_value(sum(_node_byte_size(node) for node in eligible_nodes)),
        "eligible_bucket_counts": _bucket_counts(eligible_nodes),
        "eligible_module_counts": _module_counts(eligible_nodes),
        "eligible_digest": _node_digest(gm, eligible_nodes),
        "eligible_sample": [
            _node_log_payload(node) for node in eligible_by_size[:_LOG_SAMPLE_LIMIT]
        ],
        "rejected_sample": rejected_sample,
    }


def _graph_identity_payload(
    gm: fx.GraphModule,
    mandatory_recompute_nodes: set[fx.Node] | None = None,
) -> dict[str, object]:
    if mandatory_recompute_nodes is None:
        mandatory_recompute_nodes = set()
    h = hashlib.sha256()
    policy_counts: Counter[str] = Counter()
    bucket_counts: Counter[str] = Counter()
    placeholders: list[dict[str, object]] = []
    call_functions = 0
    backward_call_functions = 0
    collective_nodes: list[fx.Node] = []
    for node in gm.graph.nodes:
        h.update(node.op.encode())
        h.update(b"\0")
        h.update(node.name.encode())
        h.update(b"\0")
        h.update(_target_name(node).encode())
        h.update(b"\0")
        h.update(str(_node_shape(node)).encode())
        h.update(b"\0")
        if node.op == "placeholder":
            placeholders.append(_node_log_payload(node))
        elif node.op == "call_function":
            call_functions += 1
            if _is_backward_node(node):
                backward_call_functions += 1
            else:
                policy_counts[_policy_name(node.meta.get("recompute", _NO_TAG))] += 1
                bucket_counts[_node_bucket(node)] += 1
                if _is_collective(node):
                    collective_nodes.append(node)
    return {
        "graph_digest": h.hexdigest()[:12],
        "nodes": len(list(gm.graph.nodes)),
        "call_functions": call_functions,
        "forward_call_functions": call_functions - backward_call_functions,
        "backward_call_functions": backward_call_functions,
        "placeholder_count": len(placeholders),
        "placeholder_sample": placeholders[:_LOG_SAMPLE_LIMIT],
        "policy_counts": dict(policy_counts),
        "bucket_counts": dict(bucket_counts),
        "has_collective": bool(collective_nodes),
        "collective_count": len(collective_nodes),
        "collective_sample": [
            _node_log_payload(node) for node in collective_nodes[:_LOG_SAMPLE_LIMIT]
        ],
        "mandatory_recompute_nodes": len(mandatory_recompute_nodes),
        "mandatory_recompute_digest": _node_digest(gm, mandatory_recompute_nodes),
    }


def _set_mandatory_must_saves(gm: fx.GraphModule) -> None:
    op_types = get_default_op_list()
    forced_hard_saves = 0
    forced_nodes: list[fx.Node] = []
    for node in gm.graph.nodes:
        if node.op != "call_function" or _is_backward_node(node):
            continue
        policy = node.meta.get("recompute")
        if policy not in _RECOMPUTE_POLICIES:
            continue
        if not _must_not_recompute(node, op_types):
            continue
        if policy == CheckpointPolicy.MUST_RECOMPUTE:
            raise RuntimeError(
                "min_cut_ac cannot rematerialize RNG/nondeterministic/mutable node "
                f"{node.name!r} because it is tagged MUST_RECOMPUTE"
            )
        node.meta["recompute"] = CheckpointPolicy.MUST_SAVE
        forced_hard_saves += 1
        forced_nodes.append(node)
    if forced_hard_saves:
        logger.info(
            "min_cut_ac: forced %d RNG/nondeterministic/mutable recompute node(s) "
            "to MUST_SAVE",
            forced_hard_saves,
        )
        _log_payload(
            "mandatory_safety_summary",
            {
                "forced": forced_hard_saves,
                "forced_digest": _node_digest(gm, forced_nodes),
                "forced_bucket_counts": _bucket_counts(forced_nodes),
                "forced_sample": [
                    _node_log_payload(node) for node in forced_nodes[:_LOG_SAMPLE_LIMIT]
                ],
            },
        )


def _mandatory_recompute_nodes_for_planning(gm: fx.GraphModule) -> set[fx.Node]:
    return set()


def ac_relax_relaxable_must_saves(
    gm: fx.GraphModule, example_inputs=None
) -> fx.GraphModule:
    """Downgrade eligible MUST_SAVE tags to PREFER_SAVE.

    This is an opt-in experiment pass. It is independent of the memory policy
    that produced the tags: any finite, safe-to-save forward activation currently
    tagged MUST_SAVE becomes a save candidate. Unsafe or uncuttable saves stay
    hard so min-cut AC can try a different min-cut without
    recomputing RNG, collectives, views, or non-tensor values.
    """
    op_types = get_default_op_list()
    cuda_storage_bytes = _cuda_output_storage_nbytes_by_node(gm)
    relaxed = 0
    relaxed_nodes: list[fx.Node] = []
    for node in gm.graph.nodes:
        if (
            node.op != "call_function"
            or _is_backward_node(node)
            or node.meta.get("recompute") != CheckpointPolicy.MUST_SAVE
        ):
            continue
        if not math.isfinite(_node_weight(node, op_types, cuda_storage_bytes)):
            continue
        node.meta["recompute"] = CheckpointPolicy.PREFER_SAVE
        relaxed += 1
        relaxed_nodes.append(node)
    if relaxed:
        logger.info(
            "min_cut_ac: relaxed %d eligible MUST_SAVE activation(s) to PREFER_SAVE",
            relaxed,
        )
    _log_payload(
        "relax_summary",
        {
            "relaxed": relaxed,
            "relaxed_digest": _node_digest(gm, relaxed_nodes),
            "relaxed_bytes": sum(_node_byte_size(node) for node in relaxed_nodes),
            "relaxed_gb": _gb_value(
                sum(_node_byte_size(node) for node in relaxed_nodes)
            ),
            "relaxed_bucket_counts": _bucket_counts(relaxed_nodes),
            "relaxed_module_counts": _module_counts(relaxed_nodes),
            "relaxed_sample": [
                _node_log_payload(node) for node in relaxed_nodes[:_LOG_SAMPLE_LIMIT]
            ],
        },
    )
    gm.recompile()
    return gm


def _allow_candidates(
    gm: fx.GraphModule,
    *,
    allow_save_candidates: bool,
    event: str,
) -> fx.GraphModule:
    op_types = get_default_op_list()
    cuda_storage_bytes = _cuda_output_storage_nbytes_by_node(gm)
    allowed = 0
    allowed_nodes: list[fx.Node] = []
    recompute_allowed = 0
    recompute_allowed_nodes: list[fx.Node] = []
    recompute_anchor_nodes: list[fx.Node] = []
    recompute_anchor_set: set[fx.Node] = set()
    recompute_chain_nodes: list[fx.Node] = []
    recompute_chain_set: set[fx.Node] = set()
    for node in gm.graph.nodes:
        if node.op != "call_function" or _is_backward_node(node):
            continue
        policy = node.meta.get("recompute")
        if policy in _HARD_POLICIES:
            continue
        if not math.isfinite(_node_weight(node, op_types, cuda_storage_bytes)):
            if (
                _allowed_recompute_candidate(node, op_types)
                and policy != CheckpointPolicy.PREFER_RECOMPUTE
            ):
                anchor = _pre_decomp_softmax_save_anchor(
                    node, op_types, cuda_storage_bytes
                )
                if anchor is None:
                    continue
                chain = _pre_decomp_softmax_recompute_chain(node, anchor, op_types)
                if chain is None or not all(
                    _can_recompute_softmax_chain_node(chain_node, op_types)
                    for chain_node in chain
                ):
                    continue
                anchor_policy = anchor.meta.get("recompute")
                if anchor_policy in (
                    CheckpointPolicy.MUST_RECOMPUTE,
                    CheckpointPolicy.MUST_SAVE,
                ):
                    if anchor_policy != CheckpointPolicy.MUST_SAVE:
                        continue
                else:
                    anchor.meta["recompute"] = CheckpointPolicy.MUST_SAVE
                    if anchor not in recompute_anchor_set:
                        recompute_anchor_set.add(anchor)
                        recompute_anchor_nodes.append(anchor)
                for chain_node in chain:
                    if (
                        chain_node.meta.get("recompute")
                        != CheckpointPolicy.MUST_RECOMPUTE
                    ):
                        chain_node.meta["recompute"] = CheckpointPolicy.MUST_RECOMPUTE
                        if chain_node not in recompute_chain_set:
                            recompute_chain_set.add(chain_node)
                            recompute_chain_nodes.append(chain_node)
                node.meta["recompute"] = CheckpointPolicy.PREFER_RECOMPUTE
                recompute_allowed += 1
                recompute_allowed_nodes.append(node)
            continue
        if allow_save_candidates and policy != CheckpointPolicy.PREFER_SAVE:
            node.meta["recompute"] = CheckpointPolicy.PREFER_SAVE
            allowed += 1
            allowed_nodes.append(node)
    if allowed:
        logger.info(
            "min_cut_ac: marked %d allowed activation(s) as PREFER_SAVE",
            allowed,
        )
    if recompute_allowed:
        logger.info(
            "min_cut_ac: marked %d safe unsaveable activation(s) as PREFER_RECOMPUTE",
            recompute_allowed,
        )
    _log_payload(
        event,
        {
            "allowed": allowed,
            "allowed_digest": _node_digest(gm, allowed_nodes),
            "allowed_bytes": sum(_node_byte_size(node) for node in allowed_nodes),
            "allowed_gb": _gb_value(
                sum(_node_byte_size(node) for node in allowed_nodes)
            ),
            "allowed_bucket_counts": _bucket_counts(allowed_nodes),
            "allowed_module_counts": _module_counts(allowed_nodes),
            "allowed_sample": [
                _node_log_payload(node) for node in allowed_nodes[:_LOG_SAMPLE_LIMIT]
            ],
            "recompute_allowed": recompute_allowed,
            "recompute_allowed_digest": _node_digest(gm, recompute_allowed_nodes),
            "recompute_allowed_bytes": sum(
                _node_byte_size(node) for node in recompute_allowed_nodes
            ),
            "recompute_allowed_gb": _gb_value(
                sum(_node_byte_size(node) for node in recompute_allowed_nodes)
            ),
            "recompute_allowed_bucket_counts": _bucket_counts(recompute_allowed_nodes),
            "recompute_allowed_module_counts": _module_counts(recompute_allowed_nodes),
            "recompute_allowed_sample": [
                _node_log_payload(node)
                for node in recompute_allowed_nodes[:_LOG_SAMPLE_LIMIT]
            ],
            "recompute_anchors": len(recompute_anchor_nodes),
            "recompute_anchor_digest": _node_digest(gm, recompute_anchor_nodes),
            "recompute_anchor_bytes": sum(
                _node_byte_size(node) for node in recompute_anchor_nodes
            ),
            "recompute_anchor_gb": _gb_value(
                sum(_node_byte_size(node) for node in recompute_anchor_nodes)
            ),
            "recompute_anchor_bucket_counts": _bucket_counts(recompute_anchor_nodes),
            "recompute_anchor_module_counts": _module_counts(recompute_anchor_nodes),
            "recompute_anchor_sample": [
                _node_log_payload(node)
                for node in recompute_anchor_nodes[:_LOG_SAMPLE_LIMIT]
            ],
            "recompute_chain_nodes": len(recompute_chain_nodes),
            "recompute_chain_digest": _node_digest(gm, recompute_chain_nodes),
            "recompute_chain_bytes": sum(
                _node_byte_size(node) for node in recompute_chain_nodes
            ),
            "recompute_chain_gb": _gb_value(
                sum(_node_byte_size(node) for node in recompute_chain_nodes)
            ),
            "recompute_chain_bucket_counts": _bucket_counts(recompute_chain_nodes),
            "recompute_chain_module_counts": _module_counts(recompute_chain_nodes),
            "recompute_chain_sample": [
                _node_log_payload(node)
                for node in recompute_chain_nodes[:_LOG_SAMPLE_LIMIT]
            ],
        },
    )
    gm.recompile()
    return gm


def ac_allow_allowed_saves(gm: fx.GraphModule, example_inputs=None) -> fx.GraphModule:
    """Mark safe non-MUST forward activations as soft save/recompute candidates."""
    return _allow_candidates(
        gm,
        allow_save_candidates=True,
        event="allow_summary",
    )


def ac_allow_unsaveable_recomputes(
    gm: fx.GraphModule, example_inputs=None
) -> fx.GraphModule:
    """Mark deterministic unsaveable boundaries as recompute candidates only."""
    return _allow_candidates(
        gm,
        allow_save_candidates=False,
        event="allow_unsaveable_recompute_summary",
    )


def default_runtime_estimator(node: fx.Node) -> float:
    """Per-node recompute cost in *milliseconds* via inductor's roofline estimator
    (``max`` of FLOP-compute-bound and memory-bandwidth-bound) -- the same
    estimator the inductor overlap scheduler uses, so AC and overlap scheduling
    agree on cost.

    Returns milliseconds; never a byte size (bytes are the knapsack's separate
    memory axis and would dwarf any ms value, corrupting the
    recompute-cost-per-byte ranking). Hard-fails if a node cannot be costed --
    we surface the error rather than silently treating the node as free.
    """
    return float(estimate_roofline_runtime_ms(node))


# --------------------------------------------------------------------------- #
# Min-cut flow network
# --------------------------------------------------------------------------- #
def _build_flow_network(
    fwd_nodes,
    placeholders,
    bwd_node_set,
    bwd_consumed,
    op_types,
    cuda_storage_bytes,
    dont_ban=None,
    mandatory_recompute_nodes=None,
):
    """Build flow network. Nodes already tagged MUST_SAVE are banned from
    recompute (SOURCE side). All other forward nodes are candidates."""
    import networkx as nx

    if dont_ban is None:
        dont_ban = set()
    if mandatory_recompute_nodes is None:
        mandatory_recompute_nodes = set()
    nx_graph = nx.DiGraph()
    nx_graph.add_node("source")
    nx_graph.add_node("sink")

    for node in placeholders:
        nx_graph.add_edge(
            "source",
            f"{node.name}_in",
            capacity=math.inf,
            reason="placeholder is always resident",
        )
        nx_graph.add_edge(
            f"{node.name}_in",
            f"{node.name}_out",
            capacity=0.0,
            reason="placeholder save is free",
        )
        for user in node.users:
            if user.op == "call_function" and not _is_backward_node(user):
                if (
                    user.meta.get("recompute") == CheckpointPolicy.MUST_RECOMPUTE
                    or user in mandatory_recompute_nodes
                ):
                    continue
                nx_graph.add_edge(
                    f"{node.name}_out",
                    f"{user.name}_in",
                    capacity=math.inf,
                    reason="data dependency",
                )
            elif user in bwd_node_set:
                nx_graph.add_edge(
                    f"{node.name}_out",
                    "sink",
                    capacity=math.inf,
                    reason="placeholder value used by backward",
                )

    for node in fwd_nodes:
        policy = node.meta.get("recompute")
        is_already_saved = policy in _SAVE_POLICIES
        is_hard_banned = _must_not_recompute(node, op_types)
        is_mandatory_recompute = (
            policy == CheckpointPolicy.MUST_RECOMPUTE
            or node in mandatory_recompute_nodes
        )
        if is_already_saved:
            weight = math.inf
            weight_reason = "already saved by memory policy"
        else:
            weight = _node_weight(node, op_types, cuda_storage_bytes)
            weight_reason = (
                "save candidate"
                if math.isfinite(weight)
                else f"cannot save: {_candidate_reject_reason(node, op_types, cuda_storage_bytes)}"
            )
        nx_graph.add_edge(
            f"{node.name}_in",
            f"{node.name}_out",
            capacity=weight,
            reason=weight_reason,
        )

        is_pinned_source = is_already_saved or is_hard_banned
        if is_pinned_source and node not in dont_ban:
            reason = (
                "already saved by memory policy"
                if is_already_saved
                else "cannot recompute: random/nondeterministic/mutable"
            )
            nx_graph.add_edge(
                "source",
                f"{node.name}_in",
                capacity=math.inf,
                reason=reason,
            )

        for user in node.users:
            if user.op == "call_function" and not _is_backward_node(user):
                if is_already_saved and (
                    user.meta.get("recompute") == CheckpointPolicy.MUST_RECOMPUTE
                    or user in mandatory_recompute_nodes
                ):
                    continue
                nx_graph.add_edge(
                    f"{node.name}_out",
                    f"{user.name}_in",
                    capacity=math.inf,
                    reason="data dependency",
                )

        if is_mandatory_recompute:
            reason = (
                "must recompute: checkpoint policy"
                if policy == CheckpointPolicy.MUST_RECOMPUTE
                else "must recompute: collective planning constraint"
            )
            nx_graph.add_edge(
                f"{node.name}_in",
                "sink",
                capacity=math.inf,
                reason=reason,
            )
        elif node in bwd_consumed and not is_pinned_source:
            nx_graph.add_edge(
                f"{node.name}_out",
                "sink",
                capacity=math.inf,
                reason="activation used by backward",
            )

    return nx_graph


def _find_infinite_capacity_path(nx_graph) -> list[tuple[str, str, str]] | None:
    visited = {"source"}
    queue = deque([("source", [])])

    while queue:
        node, edge_path = queue.popleft()
        for neighbor in nx_graph.successors(node):
            if neighbor in visited:
                continue
            edge_data = nx_graph[node][neighbor]
            if edge_data.get("capacity") != math.inf:
                continue
            reason = str(edge_data.get("reason", "unknown"))
            new_path = [*edge_path, (node, neighbor, reason)]
            if neighbor == "sink":
                return new_path
            visited.add(neighbor)
            queue.append((neighbor, new_path))
    return None


def _flow_graph_reason_counts(nx_graph) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for _, _, data in nx_graph.edges(data=True):
        if data.get("capacity") == math.inf:
            counts[str(data.get("reason", "unknown"))] += 1
    return dict(counts)


def _base_flow_node_name(name: str) -> str:
    for suffix in ("_in", "_out"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _flow_path_payload(
    gm: fx.GraphModule,
    name_to_node: dict[str, fx.Node],
    nx_graph,
) -> dict[str, object]:
    inf_path = _find_infinite_capacity_path(nx_graph)
    if inf_path is None:
        return {
            "infinite_path_found": False,
            "infinite_edge_reason_counts": _flow_graph_reason_counts(nx_graph),
        }

    path_node_names = []
    seen = set()
    for source, target, _ in inf_path:
        for graph_name in (source, target):
            base = _base_flow_node_name(graph_name)
            if base in ("source", "sink") or base in seen:
                continue
            seen.add(base)
            path_node_names.append(base)
    path_nodes = [
        name_to_node[name] for name in path_node_names if name in name_to_node
    ]
    return {
        "infinite_path_found": True,
        "infinite_path": [
            {"from": source, "to": target, "reason": reason}
            for source, target, reason in inf_path
        ],
        "infinite_path_nodes": [
            _node_log_payload(node) for node in _ordered_nodes(gm, path_nodes)
        ],
        "infinite_edge_reason_counts": _flow_graph_reason_counts(nx_graph),
    }


def _run_min_cut(nx_graph) -> set[str] | None:
    """Run min-cut. Returns saved node names or None on failure."""
    import networkx as nx

    try:
        _, (reachable, non_reachable) = nx.minimum_cut(nx_graph, "source", "sink")
    except nx.NetworkXUnbounded:
        return None

    saved_names: set[str] = set()
    for u in reachable:
        for v in nx_graph[u]:
            if v in non_reachable and u.endswith("_in") and v.endswith("_out"):
                saved_names.add(u[:-3])
    return saved_names


def _min_cut(
    gm: fx.GraphModule,
    mandatory_recompute_nodes: set[fx.Node] | None = None,
) -> set[fx.Node]:
    """Return the byte-minimal set of forward activations
    whose saving breaks every recompute chain at its cheapest link, given the
    current MUST_SAVE set. Does NOT tag -- the caller decides which cut nodes
    to save (budget-gated).

    Self-contained: builds the flow network from the current tags and runs the
    cut. Returns an empty set if the cut is unbounded -- a degenerate network with
    an all-infinite-capacity SOURCE->SINK path (e.g. a value reaching backward only
    through view ops, which are uncuttable), so no finite save set can break it;
    the frontier is then empty and the floor stands.
    """
    op_types = get_default_op_list()
    fwd_nodes: list[fx.Node] = []
    placeholders: list[fx.Node] = []
    bwd_node_set: set[fx.Node] = set()
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            placeholders.append(node)
        elif node.op == "call_function":
            if _is_backward_node(node):
                bwd_node_set.add(node)
            else:
                fwd_nodes.append(node)
    bwd_consumed = {n for n in fwd_nodes if any(u in bwd_node_set for u in n.users)}
    name_to_node = {n.name: n for n in fwd_nodes}
    already_saved = {n for n in fwd_nodes if n.meta.get("recompute") in _SAVE_POLICIES}
    cuda_storage_bytes = _cuda_output_storage_nbytes_by_node(gm)

    nx_graph = _build_flow_network(
        fwd_nodes,
        placeholders,
        bwd_node_set,
        bwd_consumed,
        op_types,
        cuda_storage_bytes,
        mandatory_recompute_nodes=mandatory_recompute_nodes,
    )
    saved_names = _run_min_cut(nx_graph)
    if saved_names is None:
        logger.warning("min_cut_ac: unbounded cut, frontier empty")
        _log_payload(
            "unbounded_cut",
            {
                "graph": _graph_identity_payload(gm, mandatory_recompute_nodes),
                "already_saved": len(already_saved),
                "already_saved_digest": _node_digest(gm, already_saved),
                "bwd_consumed": len(bwd_consumed),
                "bwd_consumed_digest": _node_digest(gm, bwd_consumed),
                "cuda_output_storage_nodes": len(cuda_storage_bytes),
                "candidate_eligibility": _candidate_eligibility_payload(gm),
                **_flow_path_payload(gm, name_to_node, nx_graph),
            },
        )
        return set()
    frontier = {
        name_to_node[n]
        for n in saved_names
        if n in name_to_node and name_to_node[n] not in already_saved
    }
    invalid = sorted(
        (node.name for node in frontier if node not in cuda_storage_bytes),
    )
    if invalid:
        raise RuntimeError(
            "min_cut_ac internal error: min-cut selected non-storage "
            f"node(s): {invalid}"
        )
    return frontier


def _validate_memory_estimator(memory_estimator: _MemoryEstimator) -> None:
    if memory_estimator not in _MEMORY_ESTIMATORS:
        raise ValueError(f"unknown ac_min_cut_memory_estimator: {memory_estimator!r}")
    if memory_estimator == "approximate":
        logger.info(
            "min_cut_ac: approximate peak analysis uses the memory curve "
            "with per-candidate exact checks and final remat/profile validation"
        )


def _validate_save_scope(save_scope: _SaveScope) -> None:
    if save_scope not in _SAVE_SCOPES:
        raise ValueError(f"unknown ac_min_cut_save_scope: {save_scope!r}")


def _validate_selection_mode(selection_mode: _SelectionMode) -> None:
    if selection_mode not in _SELECTION_MODES:
        raise ValueError(f"unknown ac_min_cut_selection_mode: {selection_mode!r}")


def _target_peak_from_budget(
    baseline_peak: int,
    peak_budget_gb: float,
) -> float:
    """Return the absolute byte target for a relative GB budget."""
    if math.isinf(peak_budget_gb):
        return math.inf
    if math.isnan(peak_budget_gb):
        raise ValueError("ac_min_cut_max_peak_increase_gb must not be NaN")
    target_peak = baseline_peak + peak_budget_gb * _GB
    if target_peak < 0:
        raise ValueError(
            "ac_min_cut_max_peak_increase_gb makes the target peak negative: "
            f"baseline={baseline_peak / _GB:.2f}GB, budget={peak_budget_gb:.2f}GB"
        )
    return target_peak


# --------------------------------------------------------------------------- #
# Candidate scoring and tag transitions
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _RankedSaveCandidate:
    node: fx.Node
    cost_per_byte: float
    removed_recompute_cost: float
    saved_bytes: int
    marginal_peak_tax: float = 0.0
    runtime_per_peak_byte: float = 0.0
    modeled_peak_drop: float = 0.0
    modeled_peak_drop_per_byte: float = 0.0


@dataclass
class _MemoryCurveSelectionState:
    memory_curve: _MemoryCurve
    selected_delta: list[float]


@dataclass
class _BudgetedSaveSelectionResult:
    peak: int
    kept: list[fx.Node]
    rejected_peak: int
    pruned_memory_curve: int


@dataclass(frozen=True)
class _CandidatePhasePlan:
    phases: tuple[set[fx.Node], ...]
    frontier: set[fx.Node]
    broad: set[fx.Node]
    frontier_count: int
    candidate_count: int
    broad_count: int
    broad_unfiltered_count: int
    broad_pruned_by_cap: int
    broad_skipped_empty_frontier: bool

    @property
    def is_empty_frontier_broad_only(self) -> bool:
        return self.frontier_count == 0 and self.broad_count > 0


@dataclass(frozen=True)
class _BudgetedOptimizationResult:
    baseline_peak: int
    final_peak: int
    target_peak: float
    target_met: bool
    filled_best_effort: bool
    restored_reference: bool


@dataclass(frozen=True)
class _TentativeSaveResult:
    accepted: bool
    profile: object


def _profile_tentative_save(
    gm: fx.GraphModule,
    node: fx.Node,
    profile_fn: Callable[[], object],
    accept_fn: Callable[[object], bool],
) -> _TentativeSaveResult:
    """Save one candidate, exact-profile it, and keep it only when accepted."""
    old = node.meta.get("recompute", _NO_TAG)
    node.meta["recompute"] = CheckpointPolicy.MUST_SAVE
    profile = profile_fn()
    if accept_fn(profile):
        return _TentativeSaveResult(accepted=True, profile=profile)
    _restore_recompute_tags(gm, {node: old})
    return _TentativeSaveResult(accepted=False, profile=profile)


def _candidate_phases(*phases: set[fx.Node]) -> tuple[set[fx.Node], ...]:
    """Deduplicate ordered candidate phases while preserving phase priority."""
    ordered_phases: list[set[fx.Node]] = []
    seen: set[fx.Node] = set()
    for phase in phases:
        unique_phase = phase - seen
        if unique_phase:
            ordered_phases.append(unique_phase)
            seen.update(unique_phase)
    return tuple(ordered_phases)


def _candidate_phase_nodes(candidate_phases: tuple[set[fx.Node], ...]) -> set[fx.Node]:
    nodes: set[fx.Node] = set()
    for phase in candidate_phases:
        nodes.update(phase)
    return nodes


def _removes_mandatory_recompute(
    removed_recompute: set[fx.Node],
    mandatory_recompute_nodes: set[fx.Node],
) -> bool:
    """Reject saves that would eliminate hard MUST_RECOMPUTE work."""
    return any(
        n.meta.get("recompute") == CheckpointPolicy.MUST_RECOMPUTE
        or n in mandatory_recompute_nodes
        for n in removed_recompute
    )


def _closure_cost(
    nodes: set[fx.Node],
    runtime_estimator: Callable[[fx.Node], float],
    cost_cache: dict[fx.Node, float],
) -> float:
    total = 0.0
    for node in sorted(nodes, key=lambda n: n.name):
        if node not in cost_cache:
            cost_cache[node] = float(runtime_estimator(node))
        total += cost_cache[node]
    return total


def _format_missed_peak_budget_message(
    *,
    goal: str,
    baseline_peak: int,
    reference_peak: int | None,
    target_peak: float,
    peak_budget_gb: float,
    final_peak: int,
    baseline_cost: float | None = None,
    final_cost: float | None = None,
) -> str:
    message = (
        f"min_cut_ac {goal}: missed target peak; "
        f"baseline peak={baseline_peak / _GB:.2f}GB, "
    )
    if reference_peak is not None and reference_peak != baseline_peak:
        message += f"reference peak={reference_peak / _GB:.2f}GB, "
    message += (
        f"target({peak_budget_gb:+.2f})={target_peak / _GB:.2f}GB, "
        f"candidate peak={final_peak / _GB:.2f}GB"
    )
    if baseline_cost is not None and final_cost is not None:
        message += f", recompute {baseline_cost:.4f}ms -> {final_cost:.4f}ms"
    return message


def _save_candidates(gm: fx.GraphModule) -> set[fx.Node]:
    """Return eligible PREFER_* activations min-cut may upgrade to MUST_SAVE."""
    op_types = get_default_op_list()
    cuda_storage_bytes = _cuda_output_storage_nbytes_by_node(gm)
    return {
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and not _is_backward_node(n)
        and n.meta.get("recompute") in _PREFER_POLICIES
        and math.isfinite(_node_weight(n, op_types, cuda_storage_bytes))
    }


def _snapshot_recompute_tags(gm: fx.GraphModule) -> dict[fx.Node, object]:
    return {n: n.meta.get("recompute", _NO_TAG) for n in gm.graph.nodes}


def _set_recompute_policy(
    nodes: set[fx.Node] | list[fx.Node],
    policy: CheckpointPolicy,
) -> None:
    for node in nodes:
        node.meta["recompute"] = policy


def _mark_must_save(nodes: set[fx.Node] | list[fx.Node]) -> None:
    _set_recompute_policy(nodes, CheckpointPolicy.MUST_SAVE)


def _ranked_candidate_sort_key(
    candidate: _RankedSaveCandidate,
    order: dict[fx.Node, int],
    *,
    prefer_peak_progress: bool,
) -> tuple[float, ...]:
    if prefer_peak_progress:
        return (
            candidate.modeled_peak_drop,
            candidate.modeled_peak_drop_per_byte,
            candidate.removed_recompute_cost,
            candidate.runtime_per_peak_byte,
            candidate.cost_per_byte,
            -order[candidate.node],
        )
    return (
        candidate.cost_per_byte,
        candidate.removed_recompute_cost,
        -candidate.marginal_peak_tax,
        candidate.runtime_per_peak_byte,
        -order[candidate.node],
    )


def _candidate_metrics(
    gm: fx.GraphModule,
    node: fx.Node,
    current_closure: set[fx.Node],
    selected_saves: set[fx.Node],
    mandatory_recompute_nodes: set[fx.Node],
    runtime_estimator: Callable[[fx.Node], float],
    cost_cache: dict[fx.Node, float],
    memory_curve_state: _MemoryCurveSelectionState | None,
) -> _RankedSaveCandidate | None:
    next_closure, _ = _recompute_nodes_closure_and_uses(
        gm,
        selected_saves | {node},
        extra_recomputed=mandatory_recompute_nodes,
    )
    removed = current_closure - next_closure
    if _removes_mandatory_recompute(removed, mandatory_recompute_nodes):
        return None
    removed_recompute_cost = _closure_cost(removed, runtime_estimator, cost_cache)
    if removed_recompute_cost <= _COST_EPS:
        return None

    saved_bytes = _node_byte_size(node)
    modeled_peak_drop = 0.0
    modeled_peak_drop_per_byte = 0.0
    marginal_peak_tax = float(saved_bytes)
    if memory_curve_state is not None:
        local_delta, saved_bytes = _candidate_memory_curve_delta(
            memory_curve_state.memory_curve, node, removed
        )
        selected_delta = list(memory_curve_state.selected_delta)
        _apply_delta(selected_delta, local_delta)
        modeled_peak = _memory_curve_peak(
            memory_curve_state.memory_curve,
            selected_delta,
        )
        current_modeled_peak = _memory_curve_peak(
            memory_curve_state.memory_curve, memory_curve_state.selected_delta
        )
        modeled_peak_drop = max(0.0, current_modeled_peak - modeled_peak)
        marginal_peak_tax = max(0.0, modeled_peak - current_modeled_peak)
        modeled_peak_drop_per_byte = modeled_peak_drop / max(saved_bytes, 1)

    return _RankedSaveCandidate(
        node=node,
        cost_per_byte=removed_recompute_cost / max(saved_bytes, 1),
        removed_recompute_cost=removed_recompute_cost,
        saved_bytes=saved_bytes,
        marginal_peak_tax=marginal_peak_tax,
        runtime_per_peak_byte=removed_recompute_cost / max(marginal_peak_tax, 1.0),
        modeled_peak_drop=modeled_peak_drop,
        modeled_peak_drop_per_byte=modeled_peak_drop_per_byte,
    )


def _rank_save_candidates_by_peak_aware_value(
    gm: fx.GraphModule,
    candidates: set[fx.Node],
    current_closure: set[fx.Node],
    selected_saves: set[fx.Node],
    mandatory_recompute_nodes: set[fx.Node],
    runtime_estimator: Callable[[fx.Node], float],
    cost_cache: dict[fx.Node, float],
    memory_curve_state: _MemoryCurveSelectionState | None = None,
    prefer_peak_progress: bool = False,
) -> list[_RankedSaveCandidate]:
    """Rank candidates by the current objective.

    Once the plan fits the peak target, saved-byte efficiency remains the runtime
    objective. While the plan is over budget, prioritize modeled peak progress.
    """
    order = {n: i for i, n in enumerate(gm.graph.nodes)}
    ranked_candidates: list[_RankedSaveCandidate] = []
    for node in candidates:
        if node not in current_closure:
            continue
        candidate = _candidate_metrics(
            gm,
            node,
            current_closure,
            selected_saves,
            mandatory_recompute_nodes,
            runtime_estimator,
            cost_cache,
            memory_curve_state if prefer_peak_progress else None,
        )
        if candidate is not None:
            ranked_candidates.append(candidate)

    ranked_candidates.sort(
        key=lambda p: _ranked_candidate_sort_key(
            p, order, prefer_peak_progress=prefer_peak_progress
        ),
        reverse=True,
    )
    return ranked_candidates


def _rank_save_candidates_by_recompute_cost_per_byte(*args, **kwargs):
    return _rank_save_candidates_by_peak_aware_value(*args, **kwargs)


def _rank_rejection_payload(
    gm: fx.GraphModule,
    candidates: set[fx.Node],
    current_closure: set[fx.Node],
    selected_saves: set[fx.Node],
    mandatory_recompute_nodes: set[fx.Node],
    runtime_estimator: Callable[[fx.Node], float],
    cost_cache: dict[fx.Node, float],
) -> dict[str, object]:
    reason_counts: Counter[str] = Counter()
    sample: list[dict[str, object]] = []
    for node in _ordered_nodes(gm, list(candidates)):
        if node not in current_closure:
            reason = "not_in_current_recompute_closure"
            removed: set[fx.Node] = set()
            removed_cost = 0.0
        else:
            next_closure, _ = _recompute_nodes_closure_and_uses(
                gm,
                selected_saves | {node},
                extra_recomputed=mandatory_recompute_nodes,
            )
            removed = current_closure - next_closure
            if not removed:
                reason = "no_recompute_closure_removed"
                removed_cost = 0.0
            elif _removes_mandatory_recompute(removed, mandatory_recompute_nodes):
                reason = "would_remove_mandatory_recompute"
                removed_cost = 0.0
            else:
                removed_cost = _closure_cost(removed, runtime_estimator, cost_cache)
                reason = (
                    "zero_removed_recompute_cost"
                    if removed_cost <= _COST_EPS
                    else "rankable"
                )
        reason_counts[reason] += 1
        if reason != "rankable" and len(sample) < _LOG_SAMPLE_LIMIT:
            sample.append(
                {
                    "reason": reason,
                    "node": _node_log_payload(node),
                    "removed_nodes": len(removed),
                    "removed_recompute_cost_ms": round(removed_cost, 6),
                }
            )
    return {
        "reason_counts": dict(reason_counts),
        "sample": sample,
    }


def _candidate_peak_fits_or_makes_progress(
    *,
    candidate_peak: int,
    current_peak: int,
    target_peak: float,
) -> bool:
    """Accept candidates that fit, or materially reduce an over-budget peak."""
    if candidate_peak <= target_peak:
        return True
    if current_peak > target_peak:
        return candidate_peak <= current_peak - _PEAK_PROGRESS_TOLERANCE_BYTES
    return False


def _should_rank_by_peak_progress(
    *,
    current_peak: int,
    target_peak: float,
) -> bool:
    if current_peak <= target_peak:
        return False
    excess = current_peak - target_peak
    min_excess = target_peak * _PEAK_PROGRESS_RANKING_MIN_EXCESS_RATIO
    if target_peak >= _GB:
        min_excess = max(min_excess, _PEAK_PROGRESS_TOLERANCE_BYTES)
    return excess > min_excess


def _new_memory_curve_selection_state(
    gm: fx.GraphModule,
    *,
    exact_peak: int,
    target_peak: float,
) -> _MemoryCurveSelectionState:
    memory_curve = _build_memory_curve(
        gm,
        exact_current_peak=exact_peak,
        exact_target_peak=target_peak,
    )
    return _MemoryCurveSelectionState(
        memory_curve=memory_curve,
        selected_delta=[0.0] * len(memory_curve.curve),
    )


def _memory_curve_state_payload(
    memory_curve_state: _MemoryCurveSelectionState | None,
) -> dict[str, object]:
    if memory_curve_state is None:
        return {}
    curve = memory_curve_state.memory_curve.curve
    selected_delta = memory_curve_state.selected_delta
    modeled_values = [
        value + selected_delta[index] for index, value in enumerate(curve)
    ]
    if not modeled_values:
        return {"memory_curve_points": 0}
    peak_index, peak_value = max(
        enumerate(modeled_values),
        key=lambda item: item[1],
    )
    return {
        "memory_curve_points": len(modeled_values),
        "modeled_curve_peak_gb": _gb_value(peak_value),
        "modeled_curve_peak_index": peak_index,
        "selected_delta_at_peak_gb": _gb_value(selected_delta[peak_index]),
    }


def _budgeted_save_candidate_phases(
    gm: fx.GraphModule,
    save_scope: _SaveScope,
    mandatory_recompute_nodes: set[fx.Node],
    min_broad_candidate_bytes: int | None = None,
    max_broad_candidate_bytes: int | None = None,
    min_broad_without_frontier_candidates: int = 0,
) -> _CandidatePhasePlan:
    """Return min-cut candidates first, then broad candidates for all-scope."""
    candidates = _save_candidates(gm)
    frontier = _min_cut(gm, mandatory_recompute_nodes) & candidates
    broad_candidates = candidates - frontier
    broad_unfiltered_count = len(broad_candidates)
    broad_pruned_by_cap = 0
    broad_skipped_empty_frontier = False
    if save_scope == "min_cut":
        return _CandidatePhasePlan(
            phases=_candidate_phases(frontier),
            frontier=frontier,
            broad=set(),
            frontier_count=len(frontier),
            candidate_count=len(candidates),
            broad_count=0,
            broad_unfiltered_count=broad_unfiltered_count,
            broad_pruned_by_cap=0,
            broad_skipped_empty_frontier=False,
        )
    if not frontier and len(candidates) < min_broad_without_frontier_candidates:
        broad_skipped_empty_frontier = bool(broad_candidates)
        if broad_candidates:
            logger.info(
                "min_cut_ac: skipped %d broad candidate(s) because the "
                "min-cut frontier is empty and the candidate pool is too "
                "small for non-positive-budget broad-only saves (%d < %d)",
                len(broad_candidates),
                len(candidates),
                min_broad_without_frontier_candidates,
            )
        broad_candidates = set()
    if max_broad_candidate_bytes is not None:
        unfiltered_n = len(broad_candidates)
        broad_candidates = {
            n
            for n in broad_candidates
            if _node_byte_size(n) <= max_broad_candidate_bytes
        }
        broad_pruned_by_cap = unfiltered_n - len(broad_candidates)
        if broad_pruned_by_cap:
            logger.info(
                "min_cut_ac: pruned %d broad candidate(s) above collective "
                "graph cap %.2fGB",
                broad_pruned_by_cap,
                max_broad_candidate_bytes / _GB,
            )
    if min_broad_candidate_bytes is not None:
        unfiltered_n = len(broad_candidates)
        broad_candidates = {
            n
            for n in broad_candidates
            if _node_byte_size(n) >= min_broad_candidate_bytes
        }
        pruned = unfiltered_n - len(broad_candidates)
        broad_pruned_by_cap += pruned
        if pruned:
            logger.info(
                "min_cut_ac: pruned %d broad candidate(s) below graph cap %.2fGB",
                pruned,
                min_broad_candidate_bytes / _GB,
            )
    return _CandidatePhasePlan(
        phases=_candidate_phases(frontier, broad_candidates),
        frontier=frontier,
        broad=broad_candidates,
        frontier_count=len(frontier),
        candidate_count=len(candidates),
        broad_count=len(broad_candidates),
        broad_unfiltered_count=broad_unfiltered_count,
        broad_pruned_by_cap=broad_pruned_by_cap,
        broad_skipped_empty_frontier=broad_skipped_empty_frontier,
    )


def _select_saves_under_peak_budget(
    gm: fx.GraphModule,
    example_inputs: tuple | None,
    *,
    candidate_phases: tuple[set[fx.Node], ...],
    current_peak: int,
    target_peak: float,
    memory_estimator: _MemoryEstimator,
    runtime_estimator: Callable[[fx.Node], float],
    mandatory_recompute_nodes: set[fx.Node] | None = None,
) -> _BudgetedSaveSelectionResult:
    """Greedily add valuable saves while exact checks enforce the peak target."""
    if mandatory_recompute_nodes is None:
        mandatory_recompute_nodes = set()
    exact_peak = current_peak
    memory_curve_state = (
        _new_memory_curve_selection_state(
            gm,
            exact_peak=exact_peak,
            target_peak=target_peak,
        )
        if memory_estimator == "approximate"
        else None
    )

    kept: list[fx.Node] = []
    selected_saves: set[fx.Node] = set()
    rejected_peak = 0
    pruned_memory_curve = 0
    cost_cache: dict[fx.Node, float] = {}
    skipped_stale_not_in_closure = 0
    skipped_no_removed_closure = 0
    skipped_mandatory_removal = 0
    memory_curve_rebuilds = 0
    current_closure, _ = _recompute_nodes_closure_and_uses(
        gm,
        extra_recomputed=mandatory_recompute_nodes,
    )
    initial_phase_nodes = _candidate_phase_nodes(candidate_phases)
    initial_rankable = initial_phase_nodes & current_closure
    initial_closure_cost = _closure_cost(current_closure, runtime_estimator, cost_cache)
    phase_sizes = [len(phase) for phase in candidate_phases]
    _log_payload(
        "selection_start",
        {
            "memory_estimator": memory_estimator,
            "current_peak_gb": _gb_value(current_peak),
            "target_peak_gb": _gb_value(target_peak),
            "phase_sizes": phase_sizes,
            "candidate_nodes": len(initial_phase_nodes),
            "closure_nodes": len(current_closure),
            "rankable_nodes": len(initial_rankable),
            "mandatory_recompute_nodes": len(mandatory_recompute_nodes),
            "closure_cost_ms": round(initial_closure_cost, 6),
            "candidate_digest": _node_digest(gm, initial_phase_nodes),
            "candidate_bucket_counts": _bucket_counts(initial_phase_nodes),
            "candidate_module_counts": _module_counts(initial_phase_nodes),
            **_memory_curve_state_payload(memory_curve_state),
        },
    )
    rank_rounds = 0
    total_ranked = 0
    exact_fallback_attempted = 0
    exact_fallback_accepted = 0
    exact_fallback_rejected = 0
    rank_sample: list[dict[str, object]] = []
    curve_prune_sample: list[dict[str, object]] = []
    exact_fallback_accepted_sample: list[dict[str, object]] = []
    exact_fallback_rejected_sample: list[dict[str, object]] = []
    exact_reject_sample: list[dict[str, object]] = []
    kept_sample: list[dict[str, object]] = []
    empty_rank_phases: list[dict[str, object]] = []

    for phase_index, candidate_phase in enumerate(candidate_phases):
        pending_candidates = set(candidate_phase)
        ranked_candidates: list[_RankedSaveCandidate] = []
        while True:
            if not ranked_candidates:
                rank_input = pending_candidates & current_closure
                prefer_peak_progress = memory_estimator == "approximate" and (
                    _should_rank_by_peak_progress(
                        current_peak=exact_peak,
                        target_peak=target_peak,
                    )
                )
                ranked_candidates = _rank_save_candidates_by_peak_aware_value(
                    gm,
                    rank_input,
                    current_closure,
                    selected_saves,
                    mandatory_recompute_nodes,
                    runtime_estimator,
                    cost_cache,
                    memory_curve_state=memory_curve_state,
                    prefer_peak_progress=prefer_peak_progress,
                )
                if ranked_candidates:
                    rank_rounds += 1
                    total_ranked += len(ranked_candidates)
                if not rank_sample and ranked_candidates:
                    rank_sample = [
                        {
                            "node": _node_log_payload(candidate.node),
                            "removed_recompute_cost_ms": round(
                                candidate.removed_recompute_cost, 6
                            ),
                            "cost_per_byte": candidate.cost_per_byte,
                            "saved_gb": _gb_value(candidate.saved_bytes),
                            "marginal_peak_tax_gb": _gb_value(
                                candidate.marginal_peak_tax
                            ),
                            "runtime_per_peak_byte": candidate.runtime_per_peak_byte,
                            "modeled_peak_drop_gb": _gb_value(
                                candidate.modeled_peak_drop
                            ),
                            "modeled_peak_drop_per_byte": (
                                candidate.modeled_peak_drop_per_byte
                            ),
                        }
                        for candidate in ranked_candidates[:8]
                    ]
                if not ranked_candidates:
                    if len(empty_rank_phases) < _LOG_SAMPLE_LIMIT:
                        empty_rank_phases.append(
                            {
                                "phase": phase_index,
                                "pending_candidates": len(pending_candidates),
                                "rank_input": len(rank_input),
                                "closure_nodes": len(current_closure),
                                "rank_rejections": _rank_rejection_payload(
                                    gm,
                                    rank_input,
                                    current_closure,
                                    selected_saves,
                                    mandatory_recompute_nodes,
                                    runtime_estimator,
                                    cost_cache,
                                ),
                            }
                        )
                    break

            ranked_candidate = ranked_candidates.pop(0)
            node = ranked_candidate.node
            pending_candidates.discard(node)
            if node not in current_closure:
                skipped_stale_not_in_closure += 1
                continue

            next_closure, _ = _recompute_nodes_closure_and_uses(
                gm,
                selected_saves | {node},
                extra_recomputed=mandatory_recompute_nodes,
            )
            removed = current_closure - next_closure
            if not removed:
                skipped_no_removed_closure += 1
                continue
            if _removes_mandatory_recompute(removed, mandatory_recompute_nodes):
                skipped_mandatory_removal += 1
                continue

            local_delta: dict[int, float] | None = None
            result: _TentativeSaveResult | None = None
            rebuild_memory_curve = False
            modeled_current_peak: float | None = None
            modeled_candidate_peak: float | None = None
            if memory_estimator == "approximate":
                if memory_curve_state is None:
                    raise RuntimeError(
                        "missing memory-curve state for approximate mode"
                    )
                local_delta, _ = _candidate_memory_curve_delta(
                    memory_curve_state.memory_curve, node, removed
                )
                candidate_delta = list(memory_curve_state.selected_delta)
                _apply_delta(candidate_delta, local_delta)
                modeled_candidate_peak = _memory_curve_peak(
                    memory_curve_state.memory_curve,
                    candidate_delta,
                )
                modeled_current_peak = _memory_curve_peak(
                    memory_curve_state.memory_curve,
                    memory_curve_state.selected_delta,
                )
                if not _candidate_peak_fits_or_makes_progress(
                    candidate_peak=int(modeled_candidate_peak),
                    current_peak=int(modeled_current_peak),
                    target_peak=target_peak,
                ):
                    exact_fallback_attempted += 1
                    result = _profile_tentative_save(
                        gm,
                        node,
                        lambda: _peak_after_remat(gm, example_inputs),
                        lambda peak: _candidate_peak_fits_or_makes_progress(
                            candidate_peak=peak,
                            current_peak=exact_peak,
                            target_peak=target_peak,
                        ),
                    )
                    rebuild_memory_curve = result.accepted
                    if result.accepted:
                        exact_fallback_accepted += 1
                        if len(exact_fallback_accepted_sample) < _LOG_SAMPLE_LIMIT:
                            exact_fallback_accepted_sample.append(
                                {
                                    "node": _node_log_payload(node),
                                    "modeled_current_peak_gb": _gb_value(
                                        modeled_current_peak
                                    ),
                                    "modeled_candidate_peak_gb": _gb_value(
                                        modeled_candidate_peak
                                    ),
                                    "exact_current_peak_gb": _gb_value(exact_peak),
                                    "exact_candidate_peak_gb": _gb_value(
                                        result.profile
                                    ),
                                    "target_peak_gb": _gb_value(target_peak),
                                    "removed_recompute_cost_ms": round(
                                        ranked_candidate.removed_recompute_cost,
                                        6,
                                    ),
                                    "removed_nodes": len(removed),
                                }
                            )
                    else:
                        exact_fallback_rejected += 1
                        if len(exact_fallback_rejected_sample) < _LOG_SAMPLE_LIMIT:
                            exact_fallback_rejected_sample.append(
                                {
                                    "node": _node_log_payload(node),
                                    "modeled_current_peak_gb": _gb_value(
                                        modeled_current_peak
                                    ),
                                    "modeled_candidate_peak_gb": _gb_value(
                                        modeled_candidate_peak
                                    ),
                                    "exact_current_peak_gb": _gb_value(exact_peak),
                                    "exact_candidate_peak_gb": _gb_value(
                                        result.profile
                                    ),
                                    "target_peak_gb": _gb_value(target_peak),
                                    "removed_recompute_cost_ms": round(
                                        ranked_candidate.removed_recompute_cost,
                                        6,
                                    ),
                                    "removed_nodes": len(removed),
                                }
                            )
                    if result is None or not result.accepted:
                        pruned_memory_curve += 1
                        if len(curve_prune_sample) < _LOG_SAMPLE_LIMIT:
                            curve_prune_sample.append(
                                {
                                    "node": _node_log_payload(node),
                                    "modeled_current_peak_gb": _gb_value(
                                        modeled_current_peak
                                    ),
                                    "modeled_candidate_peak_gb": _gb_value(
                                        modeled_candidate_peak
                                    ),
                                    "target_peak_gb": _gb_value(target_peak),
                                    "removed_recompute_cost_ms": round(
                                        ranked_candidate.removed_recompute_cost,
                                        6,
                                    ),
                                    "removed_nodes": len(removed),
                                    "exact_checked": result is not None,
                                }
                            )
                        continue
            elif memory_estimator != "exact":
                raise RuntimeError(
                    f"unknown ac_min_cut_memory_estimator: {memory_estimator!r}"
                )

            if result is None:
                result = _profile_tentative_save(
                    gm,
                    node,
                    lambda: _peak_after_remat(gm, example_inputs),
                    lambda peak: _candidate_peak_fits_or_makes_progress(
                        candidate_peak=peak,
                        current_peak=exact_peak,
                        target_peak=target_peak,
                    ),
                )
            new_peak = result.profile
            if not result.accepted:
                rejected_peak += 1
                if len(exact_reject_sample) < _LOG_SAMPLE_LIMIT:
                    exact_reject_sample.append(
                        {
                            "node": _node_log_payload(node),
                            "exact_current_peak_gb": _gb_value(exact_peak),
                            "exact_candidate_peak_gb": _gb_value(new_peak),
                            "target_peak_gb": _gb_value(target_peak),
                            "removed_recompute_cost_ms": round(
                                ranked_candidate.removed_recompute_cost,
                                6,
                            ),
                            "removed_nodes": len(removed),
                        }
                    )
                continue

            kept.append(node)
            selected_saves.add(node)
            current_closure = next_closure
            previous_exact_peak = exact_peak
            exact_peak = new_peak
            if len(kept_sample) < _LOG_SAMPLE_LIMIT:
                kept_sample.append(
                    {
                        "node": _node_log_payload(node),
                        "exact_peak_before_gb": _gb_value(previous_exact_peak),
                        "exact_peak_after_gb": _gb_value(exact_peak),
                        "modeled_current_peak_gb": None
                        if modeled_current_peak is None
                        else _gb_value(modeled_current_peak),
                        "modeled_candidate_peak_gb": None
                        if modeled_candidate_peak is None
                        else _gb_value(modeled_candidate_peak),
                        "removed_recompute_cost_ms": round(
                            ranked_candidate.removed_recompute_cost,
                            6,
                        ),
                        "removed_nodes": len(removed),
                    }
                )
            ranked_candidates = []
            if memory_estimator == "approximate":
                if memory_curve_state is None or local_delta is None:
                    raise RuntimeError(
                        "missing memory-curve state for approximate mode"
                    )
                if rebuild_memory_curve:
                    memory_curve_state = _new_memory_curve_selection_state(
                        gm,
                        exact_peak=exact_peak,
                        target_peak=target_peak,
                    )
                    memory_curve_rebuilds += 1
                else:
                    _apply_delta(memory_curve_state.selected_delta, local_delta)

    _log_payload(
        "selection_summary",
        {
            "memory_estimator": memory_estimator,
            "target_peak_gb": _gb_value(target_peak),
            "final_peak_gb": _gb_value(exact_peak),
            "kept": len(kept),
            "kept_bytes": sum(_node_byte_size(node) for node in kept),
            "kept_gb": _gb_value(sum(_node_byte_size(node) for node in kept)),
            "rejected_peak": rejected_peak,
            "pruned_memory_curve": pruned_memory_curve,
            "skipped_stale_not_in_closure": skipped_stale_not_in_closure,
            "skipped_no_removed_closure": skipped_no_removed_closure,
            "skipped_mandatory_removal": skipped_mandatory_removal,
            "rank_rounds": rank_rounds,
            "total_ranked": total_ranked,
            "memory_curve_rebuilds": memory_curve_rebuilds,
            "exact_fallback_attempted": exact_fallback_attempted,
            "exact_fallback_accepted": exact_fallback_accepted,
            "exact_fallback_rejected": exact_fallback_rejected,
            "kept_digest": _node_digest(gm, kept),
            "kept_names": _node_names(gm, kept),
            "kept_bucket_counts": _bucket_counts(kept),
            "kept_module_counts": _module_counts(kept),
            **_memory_curve_state_payload(memory_curve_state),
            "empty_rank_phases": empty_rank_phases,
            "rank_sample": rank_sample,
            "curve_prune_sample": curve_prune_sample,
            "exact_fallback_accepted_sample": exact_fallback_accepted_sample,
            "exact_fallback_rejected_sample": exact_fallback_rejected_sample,
            "exact_reject_sample": exact_reject_sample,
            "kept_sample": kept_sample,
        },
    )

    return _BudgetedSaveSelectionResult(
        peak=exact_peak,
        kept=kept,
        rejected_peak=rejected_peak,
        pruned_memory_curve=pruned_memory_curve,
    )


def _optimize_under_peak_budget(
    gm: fx.GraphModule,
    example_inputs: tuple | None,
    max_peak_increase_gb: float | None,
    memory_estimator: _MemoryEstimator,
    save_scope: _SaveScope,
    runtime_estimator: Callable[[fx.Node], float],
    reference_peak: int | None = None,
    fallback_tags: dict[fx.Node, object] | None = None,
    min_broad_candidate_bytes: int | None = None,
    max_broad_candidate_bytes: int | None = None,
) -> _BudgetedOptimizationResult:
    """Upgrade the most valuable soft save candidates under a peak budget."""
    _validate_memory_estimator(memory_estimator)
    _validate_save_scope(save_scope)

    baseline_peak, baseline_cost = _memory_profile_after_remat(
        gm, example_inputs, runtime_estimator
    )
    budget_reference_peak = baseline_peak if reference_peak is None else reference_peak
    peak_budget_gb = math.inf if max_peak_increase_gb is None else max_peak_increase_gb
    target_peak = _target_peak_from_budget(budget_reference_peak, peak_budget_gb)
    mandatory_recompute_nodes = _mandatory_recompute_nodes_for_planning(gm)
    candidate_plan = _budgeted_save_candidate_phases(
        gm,
        save_scope,
        mandatory_recompute_nodes,
        min_broad_candidate_bytes=min_broad_candidate_bytes,
        max_broad_candidate_bytes=max_broad_candidate_bytes,
        min_broad_without_frontier_candidates=(
            _MIN_NONPOSITIVE_BUDGET_BROAD_ONLY_CANDIDATES if peak_budget_gb <= 0 else 0
        ),
    )
    candidate_phases = candidate_plan.phases
    candidates = _candidate_phase_nodes(candidate_phases)
    _log_payload(
        "budgeted_plan",
        {
            "memory_estimator": memory_estimator,
            "save_scope": save_scope,
            "peak_budget_gb": _gb_value(peak_budget_gb * _GB),
            "reference_peak_gb": _gb_value(budget_reference_peak),
            "baseline_peak_gb": _gb_value(baseline_peak),
            "target_peak_gb": _gb_value(target_peak),
            "baseline_recompute_cost_ms": round(baseline_cost, 6),
            "candidate_count": candidate_plan.candidate_count,
            "frontier_count": candidate_plan.frontier_count,
            "broad_count": candidate_plan.broad_count,
            "broad_unfiltered_count": candidate_plan.broad_unfiltered_count,
            "broad_pruned_by_cap": candidate_plan.broad_pruned_by_cap,
            "broad_skipped_empty_frontier": (
                candidate_plan.broad_skipped_empty_frontier
            ),
            "phase_sizes": [len(phase) for phase in candidate_phases],
            "selected_candidate_count": len(candidates),
            "selected_candidate_digest": _node_digest(gm, candidates),
            "selected_candidate_bucket_counts": _bucket_counts(candidates),
            "selected_candidate_module_counts": _module_counts(candidates),
            "frontier_digest": _node_digest(gm, candidate_plan.frontier),
            "frontier_bucket_counts": _bucket_counts(candidate_plan.frontier),
            "frontier_sample": [
                _node_log_payload(node)
                for node in _ordered_nodes(gm, list(candidate_plan.frontier))[
                    :_LOG_SAMPLE_LIMIT
                ]
            ],
            "broad_digest": _node_digest(gm, candidate_plan.broad),
            "broad_bucket_counts": _bucket_counts(candidate_plan.broad),
            "broad_sample": [
                _node_log_payload(node)
                for node in _ordered_nodes(gm, list(candidate_plan.broad))[
                    :_LOG_SAMPLE_LIMIT
                ]
            ],
            "candidate_eligibility": _candidate_eligibility_payload(gm),
            "mandatory_recompute_nodes": len(mandatory_recompute_nodes),
            "max_broad_candidate_gb": None
            if max_broad_candidate_bytes is None
            else _gb_value(max_broad_candidate_bytes),
            "min_broad_candidate_gb": None
            if min_broad_candidate_bytes is None
            else _gb_value(min_broad_candidate_bytes),
            "min_broad_without_frontier_candidates": (
                _MIN_NONPOSITIVE_BUDGET_BROAD_ONLY_CANDIDATES
                if peak_budget_gb <= 0
                else 0
            ),
        },
    )

    result = _select_saves_under_peak_budget(
        gm,
        example_inputs,
        candidate_phases=candidate_phases,
        current_peak=baseline_peak,
        target_peak=target_peak,
        memory_estimator=memory_estimator,
        mandatory_recompute_nodes=mandatory_recompute_nodes,
        runtime_estimator=runtime_estimator,
    )

    final_peak, final_cost = _memory_profile_after_remat(
        gm, example_inputs, runtime_estimator
    )
    target_met = final_peak <= target_peak
    filled_best_effort = False
    should_fill_under_reference = (
        not target_met
        and target_peak < budget_reference_peak
        and final_peak <= budget_reference_peak
    )
    if should_fill_under_reference:
        fill_target_peak = budget_reference_peak
        logger.info(
            "min_cut_ac budgeted: first-pass target %.2fGB ended at %.2fGB; "
            "filling remaining saves under %.2fGB",
            target_peak / _GB,
            final_peak / _GB,
            fill_target_peak / _GB,
        )
        fill_candidate_plan = _budgeted_save_candidate_phases(
            gm,
            save_scope,
            mandatory_recompute_nodes,
            min_broad_candidate_bytes=min_broad_candidate_bytes,
            max_broad_candidate_bytes=max_broad_candidate_bytes,
            min_broad_without_frontier_candidates=(
                _MIN_NONPOSITIVE_BUDGET_BROAD_ONLY_CANDIDATES
                if peak_budget_gb <= 0
                else 0
            ),
        )
        _log_payload(
            "budgeted_fill_plan",
            {
                "memory_estimator": memory_estimator,
                "save_scope": save_scope,
                "fill_target_peak_gb": _gb_value(fill_target_peak),
                "candidate_count": fill_candidate_plan.candidate_count,
                "frontier_count": fill_candidate_plan.frontier_count,
                "broad_count": fill_candidate_plan.broad_count,
                "broad_unfiltered_count": (fill_candidate_plan.broad_unfiltered_count),
                "broad_pruned_by_cap": fill_candidate_plan.broad_pruned_by_cap,
                "broad_skipped_empty_frontier": (
                    fill_candidate_plan.broad_skipped_empty_frontier
                ),
                "phase_sizes": [len(phase) for phase in fill_candidate_plan.phases],
                "selected_candidate_digest": _node_digest(
                    gm,
                    _candidate_phase_nodes(fill_candidate_plan.phases),
                ),
                "frontier_digest": _node_digest(gm, fill_candidate_plan.frontier),
                "broad_digest": _node_digest(gm, fill_candidate_plan.broad),
                "mandatory_recompute_nodes": len(mandatory_recompute_nodes),
            },
        )
        fill_result = _select_saves_under_peak_budget(
            gm,
            example_inputs,
            candidate_phases=fill_candidate_plan.phases,
            current_peak=final_peak,
            target_peak=fill_target_peak,
            memory_estimator=memory_estimator,
            mandatory_recompute_nodes=mandatory_recompute_nodes,
            runtime_estimator=runtime_estimator,
        )
        result = _BudgetedSaveSelectionResult(
            peak=fill_result.peak,
            kept=[*result.kept, *fill_result.kept],
            rejected_peak=result.rejected_peak + fill_result.rejected_peak,
            pruned_memory_curve=(
                result.pruned_memory_curve + fill_result.pruned_memory_curve
            ),
        )
        final_peak, final_cost = _memory_profile_after_remat(
            gm, example_inputs, runtime_estimator
        )
        target_met = final_peak <= target_peak
        filled_best_effort = bool(fill_result.kept)
    restored_reference = False
    if (
        not target_met
        and fallback_tags is not None
        and final_peak > budget_reference_peak
    ):
        logger.info(
            "min_cut_ac budgeted: restoring reference tags because best-effort "
            "peak %.2fGB is worse than reference %.2fGB",
            final_peak / _GB,
            budget_reference_peak / _GB,
        )
        _restore_recompute_tags(gm, fallback_tags)
        final_peak, final_cost = _memory_profile_after_remat(
            gm, example_inputs, runtime_estimator
        )
        target_met = final_peak <= target_peak
        restored_reference = True

    if not target_met:
        logger.info(
            "%s; %s",
            _format_missed_peak_budget_message(
                goal="budgeted",
                baseline_peak=baseline_peak,
                reference_peak=budget_reference_peak,
                target_peak=target_peak,
                peak_budget_gb=peak_budget_gb,
                final_peak=final_peak,
                baseline_cost=baseline_cost,
                final_cost=final_cost,
            ),
            "restored reference tags"
            if restored_reference
            else "kept best-effort tags",
        )

    logger.info(
        "min_cut_ac budgeted[memory_estimator=%s, save_scope=%s]: "
        "reference=%.2fGB baseline=%.2fGB "
        "target(%+.2f)=%.2fGB final=%.2fGB target_met=%s; "
        "recompute %.4fms -> %.4fms; "
        "pool %d, kept %d, rejected_peak %d, pruned_memory_curve %d, "
        "filled_best_effort=%s, restored_reference=%s",
        memory_estimator,
        save_scope,
        budget_reference_peak / _GB,
        baseline_peak / _GB,
        peak_budget_gb,
        target_peak / _GB,
        final_peak / _GB,
        target_met,
        baseline_cost,
        final_cost,
        len(candidates),
        len(result.kept),
        result.rejected_peak,
        result.pruned_memory_curve,
        filled_best_effort,
        restored_reference,
    )
    _log_payload(
        "budgeted_summary",
        {
            "memory_estimator": memory_estimator,
            "save_scope": save_scope,
            "reference_peak_gb": _gb_value(budget_reference_peak),
            "baseline_peak_gb": _gb_value(baseline_peak),
            "target_peak_gb": _gb_value(target_peak),
            "final_peak_gb": _gb_value(final_peak),
            "peak_budget_gb": _gb_value(peak_budget_gb * _GB),
            "target_met": target_met,
            "baseline_recompute_cost_ms": round(baseline_cost, 6),
            "final_recompute_cost_ms": round(final_cost, 6),
            "candidate_count": len(candidates),
            "kept": len(result.kept),
            "kept_bytes": sum(_node_byte_size(node) for node in result.kept),
            "kept_gb": _gb_value(sum(_node_byte_size(node) for node in result.kept)),
            "kept_digest": _node_digest(gm, result.kept),
            "kept_names": _node_names(gm, result.kept),
            "kept_bucket_counts": _bucket_counts(result.kept),
            "kept_module_counts": _module_counts(result.kept),
            "rejected_peak": result.rejected_peak,
            "pruned_memory_curve": result.pruned_memory_curve,
            "filled_best_effort": filled_best_effort,
            "restored_reference": restored_reference,
        },
    )
    return _BudgetedOptimizationResult(
        baseline_peak=baseline_peak,
        final_peak=final_peak,
        target_peak=target_peak,
        target_met=target_met,
        filled_best_effort=filled_best_effort,
        restored_reference=restored_reference,
    )


def _apply_compile_like_min_cut(
    gm: fx.GraphModule,
    example_inputs: tuple | None,
    runtime_estimator: Callable[[fx.Node], float],
) -> _BudgetedOptimizationResult:
    """Save only the byte-minimal frontier and recompute the remaining closure."""
    baseline_peak, baseline_cost = _memory_profile_after_remat(
        gm, example_inputs, runtime_estimator
    )
    candidates = _save_candidates(gm)
    frontier = _min_cut(gm)
    kept = frontier & candidates
    _mark_must_save(kept)
    final_peak, final_cost = _memory_profile_after_remat(
        gm, example_inputs, runtime_estimator
    )
    logger.info(
        "min_cut_ac compile_like: baseline=%.2fGB final=%.2fGB; "
        "recompute %.4fms -> %.4fms; candidates=%d frontier=%d kept=%d",
        baseline_peak / _GB,
        final_peak / _GB,
        baseline_cost,
        final_cost,
        len(candidates),
        len(frontier),
        len(kept),
    )
    _log_payload(
        "compile_like_summary",
        {
            "baseline_peak_gb": _gb_value(baseline_peak),
            "final_peak_gb": _gb_value(final_peak),
            "baseline_recompute_cost_ms": round(baseline_cost, 6),
            "final_recompute_cost_ms": round(final_cost, 6),
            "candidate_count": len(candidates),
            "candidate_digest": _node_digest(gm, candidates),
            "candidate_bucket_counts": _bucket_counts(candidates),
            "frontier_count": len(frontier),
            "frontier_digest": _node_digest(gm, frontier),
            "frontier_bucket_counts": _bucket_counts(frontier),
            "frontier_sample": [
                _node_log_payload(node)
                for node in _ordered_nodes(gm, list(frontier))[:_LOG_SAMPLE_LIMIT]
            ],
            "kept": len(kept),
            "kept_bytes": sum(_node_byte_size(node) for node in kept),
            "kept_gb": _gb_value(sum(_node_byte_size(node) for node in kept)),
            "kept_digest": _node_digest(gm, kept),
            "kept_names": _node_names(gm, kept),
            "kept_bucket_counts": _bucket_counts(kept),
            "kept_module_counts": _module_counts(kept),
        },
    )
    return _BudgetedOptimizationResult(
        baseline_peak=baseline_peak,
        final_peak=final_peak,
        target_peak=baseline_peak,
        target_met=final_peak <= baseline_peak,
        filled_best_effort=False,
        restored_reference=False,
    )


def _log_current_remat_profile(
    gm: fx.GraphModule,
    example_inputs: tuple | None,
    runtime_estimator: Callable[[fx.Node], float],
    event: str,
) -> None:
    peak, recompute_cost = _memory_profile_after_remat(
        gm, example_inputs, runtime_estimator
    )
    logger.info(
        "min_cut_ac %s: remat peak=%.2fGB recompute=%.4fms",
        event,
        peak / _GB,
        recompute_cost,
    )
    _log_payload(
        event,
        {
            "remat_peak_gb": _gb_value(peak),
            "recompute_cost_ms": round(recompute_cost, 6),
            "graph": _graph_identity_payload(gm),
            "candidate_eligibility": _candidate_eligibility_payload(gm),
        },
    )


def _log_save_summary(gm: fx.GraphModule) -> None:
    """Log the final forward-activation save/recompute split."""
    saved_n = recompute_n = 0
    saved_bytes = 0.0
    saved_nodes: list[fx.Node] = []
    recompute_nodes: list[fx.Node] = []
    for n in gm.graph.nodes:
        if n.op != "call_function" or _is_backward_node(n):
            continue
        policy = n.meta.get("recompute")
        if policy in _SAVE_POLICIES:
            saved_n += 1
            saved_bytes += _node_byte_size(n)
            saved_nodes.append(n)
        elif policy in _RECOMPUTE_POLICIES:
            recompute_n += 1
            recompute_nodes.append(n)
    logger.info(
        "min_cut_ac: %d forward activations saved (%.2f GB), %d recompute",
        saved_n,
        saved_bytes / _GB,
        recompute_n,
    )
    _log_payload(
        "save_summary",
        {
            "saved": saved_n,
            "saved_bytes": int(saved_bytes),
            "saved_gb": _gb_value(saved_bytes),
            "recompute": recompute_n,
            "saved_digest": _node_digest(gm, saved_nodes),
            "saved_names": _node_names(gm, saved_nodes),
            "saved_values": _node_log_payloads(gm, saved_nodes),
            "saved_bucket_counts": _bucket_counts(saved_nodes),
            "saved_module_counts": _module_counts(saved_nodes),
            "recompute_digest": _node_digest(gm, recompute_nodes),
            "recompute_bucket_counts": _bucket_counts(recompute_nodes),
            "recompute_module_counts": _module_counts(recompute_nodes),
        },
    )


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def min_cut_ac_pass(
    gm: fx.GraphModule,
    example_inputs: tuple | None = None,
    *,
    max_peak_increase_gb: float | None = 0.0,
    selection_mode: _SelectionMode = "budgeted",
    memory_estimator: _MemoryEstimator = "approximate",
    save_scope: _SaveScope = "min_cut",
    min_broad_candidate_gb: float | None = None,
    max_broad_candidate_gb: float | None = None,
    decompose_log_softmax: bool = False,
    relax_relaxable_must_saves: bool = False,
    allow_allowed_saves: bool = False,
    allow_unsaveable_recomputes: bool = False,
    runtime_estimator: Callable[[fx.Node], float] | None = None,
) -> fx.GraphModule:
    """Refine the memory policy's tags under a relative peak budget.

    Sets recompute tags only; selective_activation_remat materializes downstream.

    Args:
        max_peak_increase_gb: peak budget in GB relative to the pre-min-cut
            reference policy. None means no hard peak requirement. Negative
            values require peak reduction; 0 means no peak regression.
        selection_mode: "budgeted" uses GraphTrainer's peak-budgeted greedy
            refinement. "compile_like" saves only the byte-minimal min-cut
            frontier and recomputes the rest of the widened closure. "chain_cut"
            only rematerializes deterministic unsaveable softmax chains.
        memory_estimator: "approximate" (memory curve with
            per-candidate exact peak checks), or "exact" (per-candidate remat).
        save_scope: "min_cut" or "all".
        min_broad_candidate_gb: optional size floor for non-frontier candidates
            when save_scope="all"; the min-cut frontier is never capped.
        max_broad_candidate_gb: optional size cap for non-frontier candidates
            when save_scope="all"; the min-cut frontier is never capped.
        decompose_log_softmax: opt-in experiment that exposes log-softmax
            internals as save candidates.
        relax_relaxable_must_saves: downgrade eligible MUST_SAVE activations to
            PREFER_SAVE before min-cut. The peak budget is still relative to the
            pre-relax memory-policy reference.
        allow_allowed_saves: mark every eligible non-MUST forward activation as
            PREFER_SAVE before min-cut. The peak budget is still relative to the
            pre-allow memory-policy reference.
        allow_unsaveable_recomputes: mark deterministic unsaveable forward
            boundaries as PREFER_RECOMPUTE before min-cut, without broadening the
            ordinary save-candidate pool.
        runtime_estimator: node -> recompute cost (ms). Defaults to inductor's
            roofline estimator.
    """
    runtime_estimator = runtime_estimator or default_runtime_estimator
    _validate_selection_mode(selection_mode)

    if decompose_log_softmax:
        _decompose_log_softmax_for_min_cut(gm)
    _set_mandatory_must_saves(gm)
    pre_widen_tags = _snapshot_recompute_tags(gm)
    min_broad_candidate_bytes = (
        None if min_broad_candidate_gb is None else int(min_broad_candidate_gb * _GB)
    )
    max_broad_candidate_bytes = (
        None if max_broad_candidate_gb is None else int(max_broad_candidate_gb * _GB)
    )
    _log_payload(
        "pass_start",
        {
            "max_peak_increase_gb": max_peak_increase_gb,
            "selection_mode": selection_mode,
            "memory_estimator": memory_estimator,
            "save_scope": save_scope,
            "decompose_log_softmax": decompose_log_softmax,
            "relax_relaxable_must_saves": relax_relaxable_must_saves,
            "allow_allowed_saves": allow_allowed_saves,
            "allow_unsaveable_recomputes": allow_unsaveable_recomputes,
            "requested_min_broad_candidate_gb": min_broad_candidate_gb,
            "requested_max_broad_candidate_gb": max_broad_candidate_gb,
            "min_broad_candidate_gb": None
            if min_broad_candidate_bytes is None
            else _gb_value(min_broad_candidate_bytes),
            "max_broad_candidate_gb": None
            if max_broad_candidate_bytes is None
            else _gb_value(max_broad_candidate_bytes),
            "graph": _graph_identity_payload(gm),
            "candidate_eligibility": _candidate_eligibility_payload(gm),
        },
    )
    compile_like = selection_mode == "compile_like"
    chain_cut = selection_mode == "chain_cut"
    reference_peak = None
    widened = compile_like or chain_cut or (
        relax_relaxable_must_saves or allow_allowed_saves or allow_unsaveable_recomputes
    )
    if (
        widened
        and (max_peak_increase_gb is not None)
        and not compile_like
        and not chain_cut
    ):
        reference_peak, _ = _memory_profile_after_remat(
            gm, example_inputs, runtime_estimator
        )
        _log_payload(
            "reference_profile",
            {
                "reference_peak_gb": _gb_value(reference_peak),
                "reason": "pre_relax_or_allow_budget_anchor",
            },
        )
    if compile_like or relax_relaxable_must_saves:
        ac_relax_relaxable_must_saves(gm, example_inputs)
    if compile_like or allow_allowed_saves:
        ac_allow_allowed_saves(gm, example_inputs)
    if compile_like or chain_cut or allow_unsaveable_recomputes:
        ac_allow_unsaveable_recomputes(gm, example_inputs)
    if widened:
        _log_payload(
            "post_widen_summary",
            {
                "graph": _graph_identity_payload(gm),
                "candidate_eligibility": _candidate_eligibility_payload(gm),
            },
        )
    if compile_like:
        _apply_compile_like_min_cut(gm, example_inputs, runtime_estimator)
    elif chain_cut:
        _log_current_remat_profile(
            gm, example_inputs, runtime_estimator, "chain_cut_summary"
        )
    else:
        _optimize_under_peak_budget(
            gm,
            example_inputs,
            max_peak_increase_gb,
            memory_estimator,
            save_scope,
            runtime_estimator,
            reference_peak=reference_peak,
            fallback_tags=pre_widen_tags if widened else None,
            min_broad_candidate_bytes=min_broad_candidate_bytes
            if save_scope == "all"
            else None,
            max_broad_candidate_bytes=max_broad_candidate_bytes
            if save_scope == "all"
            else None,
        )

    _log_save_summary(gm)
    gm.recompile()
    return gm
