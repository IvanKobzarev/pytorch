# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Memory profiling helpers for GraphTrainer activation remat.

Keep this module thin and aligned with PyTorch Inductor's memory utilities. The
authoritative peak calculation is ``torch._inductor.fx_passes.memory_estimator``:
``GraphAliasTracker`` provides storage aliasing and ``build_memory_profile``
computes the allocated-memory curve. The helpers here only add GraphTrainer
specific pieces around activation-remat tags and the memory-curve approximation
used by min-cut AC. If the PyTorch utilities grow equivalent APIs, prefer
delegating to them instead of expanding this module.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass

import torch.fx as fx
from torch._functorch._activation_checkpointing.remat_using_tags_for_fwd_loss_bwd_graph_pass import (  # noqa: E501
    remat_using_tags_for_fwd_loss_bwd_graph as _torch_remat_using_tags_for_fwd_loss_bwd_graph,  # noqa: E501
)
from torch._inductor.fx_passes.memory_estimator import (
    build_memory_profile,
    GraphAliasTracker,
)
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.experimental.symbolic_shapes import optimization_hint
from torch.fx.node import map_arg
from torch.utils.checkpoint import CheckpointPolicy

from torchtitan.experiments.graph_trainer.common_utils import _is_backward_node


_RECOMPUTE_POLICIES = (
    CheckpointPolicy.PREFER_SAVE,
    CheckpointPolicy.PREFER_RECOMPUTE,
    CheckpointPolicy.MUST_RECOMPUTE,
)
_NO_TAG = object()


def _is_releasable(n: fx.Node) -> bool:
    # On the joint graph, placeholders are params/buffers/inputs (stable-address
    # state) -- treat as live for the whole graph (conservative).
    return n.op != "placeholder"


def _restore_recompute_tags(gm: fx.GraphModule, snapshot: dict) -> None:
    """Restore recompute tags from a snapshot taken before a tentative step."""
    for n, v in snapshot.items():
        if v is _NO_TAG:
            n.meta.pop("recompute", None)
        else:
            n.meta["recompute"] = v


def _normalize_soft_tags_for_remat(gm: fx.GraphModule) -> None:
    """Materialize only concrete MUST_SAVE tags as saved activations."""
    for node in gm.graph.nodes:
        if node.meta.get("recompute") == CheckpointPolicy.PREFER_SAVE:
            node.meta["recompute"] = CheckpointPolicy.PREFER_RECOMPUTE


def _is_recomputed_node(node: fx.Node) -> bool:
    return (
        node.meta.get("ac_recomputed") is True
        or node.name.endswith("_recomputed")
        or "_recomputed_r" in node.name
    )


def _refresh_fake_tensor_meta(gm: fx.GraphModule) -> None:
    """Repropagate recomputed-node metadata after remat node_copy.

    ``fx.Graph.node_copy`` shallow-copies ``node.meta``, so each recomputed
    duplicate starts with the original forward node's ``meta["val"]``. That is
    wrong for memory profiling: compute duplicates need fresh storage, while view
    duplicates should alias their recomputed parent. Re-running each duplicate op
    on already-refreshed fake inputs fixes the storage identity before
    ``build_memory_profile`` sees the graph.
    """
    fake_mode = next(
        (
            n.meta["val"].fake_mode
            for n in gm.graph.nodes
            if isinstance(n.meta.get("val"), FakeTensor)
        ),
        None,
    )
    assert (
        fake_mode is not None
    ), "GraphTrainer remat memory profiling requires FakeTensor metadata"

    redone: dict[fx.Node, object] = {}

    def _in_val(n: fx.Node) -> object:
        return redone[n] if n in redone else n.meta.get("val")

    for node in gm.graph.nodes:
        if not _is_recomputed_node(node) or "val" not in node.meta:
            continue
        args = map_arg(node.args, _in_val)
        kwargs = map_arg(node.kwargs, _in_val)
        with fake_mode:
            val = node.target(*args, **kwargs)
        node.meta["val"] = val
        redone[node] = val


def _clone_graph_module_for_profile(gm: fx.GraphModule) -> fx.GraphModule:
    cloned = fx.GraphModule(gm, copy.deepcopy(gm.graph))
    cloned.meta = copy.copy(getattr(gm, "meta", {}))
    return cloned


def remat_using_tags_for_fwd_loss_bwd_graph(gm: fx.GraphModule) -> fx.GraphModule:
    """GraphTrainer wrapper around PyTorch's tag-driven remat pass.

    The upstream pass returns a rematerialized graph with shallow-copied
    ``meta["val"]`` on recomputed nodes. Refreshing the fake metadata here makes
    every GraphTrainer caller see correct storage aliasing by default.
    """
    saved = {n: n.meta.get("recompute", _NO_TAG) for n in gm.graph.nodes}
    _normalize_soft_tags_for_remat(gm)
    try:
        try:
            remat = _torch_remat_using_tags_for_fwd_loss_bwd_graph(gm)
        except RuntimeError as exc:
            if "disjoint backward regions" not in str(exc):
                raise
            # Rigi chunked loss creates one backward region per vocab chunk.
            # Profile the same multi-region remat implementation that the
            # GraphTrainer pass pipeline uses at runtime.
            from torchtitan.experiments.graph_trainer.selective_activation_remat import (
                selective_activation_remat_pass,
            )

            remat = selective_activation_remat_pass(_clone_graph_module_for_profile(gm))
    finally:
        _restore_recompute_tags(gm, saved)
    _refresh_fake_tensor_meta(remat)
    return remat


def _peak_after_remat(gm: fx.GraphModule, example_inputs: tuple | None = None) -> int:
    """Modelled allocated peak (bytes) of the current tags via throwaway remat.

    ``remat_using_tags_for_fwd_loss_bwd_graph`` mutates ``recompute`` metadata on
    ``gm`` while cleaning tags, so snapshot and restore the tags after profiling.
    """
    saved = {n: n.meta.get("recompute", _NO_TAG) for n in gm.graph.nodes}
    remat = remat_using_tags_for_fwd_loss_bwd_graph(gm)
    peak = max(build_memory_profile(remat.graph, _is_releasable))
    _restore_recompute_tags(gm, saved)
    return peak


def _memory_profile_after_remat(
    gm: fx.GraphModule,
    example_inputs: tuple | None,
    runtime_estimator: Callable[[fx.Node], float],
) -> tuple[int, float]:
    """Return (modelled peak bytes, recompute closure cost ms) for current tags."""
    saved = {n: n.meta.get("recompute", _NO_TAG) for n in gm.graph.nodes}
    remat = remat_using_tags_for_fwd_loss_bwd_graph(gm)
    peak = max(build_memory_profile(remat.graph, _is_releasable))
    recompute_cost = sum(
        runtime_estimator(n)
        for n in remat.graph.nodes
        if n.op == "call_function" and _is_recomputed_node(n)
    )
    _restore_recompute_tags(gm, saved)
    return peak, recompute_cost


@dataclass(frozen=True)
class _EventPoints:
    after_alloc: dict[fx.Node, int]
    after_free: dict[fx.Node, int]


@dataclass
class _MemoryCurve:
    curve: list[float]
    peak_target: float
    events: _EventPoints
    alias: GraphAliasTracker
    order: dict[fx.Node, int]
    bwd_start_order: int
    free_point_by_storage: dict[object, int]
    remat_intervals: dict[fx.Node, list[tuple[int, int, float]]]
    remat_after_free: dict[fx.Node, int]


def _storage_nbytes(storage_key: object) -> int:
    return optimization_hint(storage_key.storage.nbytes(), fallback=0)


def _nodes_with_fresh_cuda_storage(gm: fx.GraphModule) -> set[fx.Node]:
    """Return nodes that allocate their own non-empty CUDA storage."""
    alias = GraphAliasTracker(list(gm.graph.nodes))
    fresh_nodes: set[fx.Node] = set()
    for node in gm.graph.nodes:
        if any(
            storage_key.device.type != "cpu" and _storage_nbytes(storage_key) > 0
            for storage_key in alias.get_fresh_allocations(node)
        ):
            fresh_nodes.add(node)
    return fresh_nodes


def _cuda_output_storage_nbytes_by_node(gm: fx.GraphModule) -> dict[fx.Node, int]:
    """Return non-empty CUDA output storage bytes per node, including aliases."""
    alias = GraphAliasTracker(list(gm.graph.nodes))
    storage_bytes: dict[fx.Node, int] = {}
    for node in gm.graph.nodes:
        nbytes = sum(
            _storage_nbytes(storage_key)
            for storage_key in alias.node_to_output_storages[node]
            if storage_key.device.type != "cpu" and _storage_nbytes(storage_key) > 0
        )
        if nbytes > 0:
            storage_bytes[node] = nbytes
    return storage_bytes


def _is_executable_node(node: fx.Node) -> bool:
    # get_attr reads module state; its storage is initial live memory, not a
    # scheduled runtime allocation/free event.
    return node.op not in ("placeholder", "get_attr", "output")


def _last_use_before(
    uses: set[fx.Node],
    order: dict[fx.Node, int],
    before: int,
) -> fx.Node | None:
    return max(
        (u for u in uses if order[u] < before),
        key=order.__getitem__,
        default=None,
    )


def _last_executable_use(
    uses: set[fx.Node],
    order: dict[fx.Node, int],
    events: _EventPoints,
) -> fx.Node | None:
    return max(
        (u for u in uses if u in events.after_free),
        key=order.__getitem__,
        default=None,
    )


def _recompute_nodes_closure_and_uses(
    gm: fx.GraphModule,
    extra_saved: set[fx.Node] | None = None,
    extra_recomputed: set[fx.Node] | None = None,
) -> tuple[set[fx.Node], dict[fx.Node, set[fx.Node]]]:
    """Mirror remat's dependency walk without materializing the remat graph."""
    if extra_saved is None:
        extra_saved = set()
    if extra_recomputed is None:
        extra_recomputed = set()

    closure: set[fx.Node] = set()
    uses: dict[fx.Node, set[fx.Node]] = {}

    def gather_for_use(use: fx.Node) -> None:
        local: set[fx.Node] = set()

        def walk(node: fx.Node) -> None:
            if (
                node in local
                or node in extra_saved
                or (
                    node not in extra_recomputed
                    and node.meta.get("recompute") not in _RECOMPUTE_POLICIES
                )
            ):
                return
            local.add(node)
            for inp in node.all_input_nodes:
                walk(inp)

        for inp in use.all_input_nodes:
            walk(inp)
        closure.update(local)
        for dep in local:
            uses.setdefault(dep, set()).add(use)

    for node in gm.graph.nodes:
        if _is_backward_node(node):
            gather_for_use(node)

    return closure, uses


def _recompute_schedule_before_bwd(
    nodes: list[fx.Node],
    order: dict[fx.Node, int],
) -> dict[fx.Node, list[fx.Node]]:
    """Return recompute deps inserted before each backward node."""
    recomputed: set[fx.Node] = set()
    schedule: dict[fx.Node, list[fx.Node]] = {}

    def gather_for_use(use: fx.Node) -> list[fx.Node]:
        deps: set[fx.Node] = set()

        def walk(node: fx.Node) -> None:
            if (
                node in deps
                or node in recomputed
                or node.meta.get("recompute") not in _RECOMPUTE_POLICIES
            ):
                return
            deps.add(node)
            for inp in node.all_input_nodes:
                walk(inp)

        for inp in use.all_input_nodes:
            walk(inp)
        return sorted(deps, key=order.__getitem__)

    for node in nodes:
        if not _is_backward_node(node):
            continue
        deps = gather_for_use(node)
        if deps:
            schedule[node] = deps
            recomputed.update(deps)
    return schedule


def _build_event_points_with_remat(
    gm: fx.GraphModule,
    schedule_before_bwd: dict[fx.Node, list[fx.Node]],
) -> tuple[_EventPoints, list[tuple[str, fx.Node]], dict[tuple[str, fx.Node], int]]:
    after_alloc: dict[fx.Node, int] = {}
    after_free: dict[fx.Node, int] = {}
    event_sequence: list[tuple[str, fx.Node]] = []
    after_free_by_event: dict[tuple[str, fx.Node], int] = {}

    point = 1
    for node in gm.graph.nodes:
        if not _is_executable_node(node):
            continue
        if _is_backward_node(node):
            for dep in schedule_before_bwd.get(node, ()):
                event = ("remat", dep)
                event_sequence.append(event)
                point += 1
                after_free_by_event[event] = point
                point += 1

        event = ("node", node)
        event_sequence.append(event)
        after_alloc[node] = point
        point += 1
        after_free[node] = point
        after_free_by_event[event] = point
        point += 1

    return (
        _EventPoints(after_alloc=after_alloc, after_free=after_free),
        event_sequence,
        after_free_by_event,
    )


def _add_interval(
    delta: dict[int, float],
    start: int | None,
    end: int | None,
    amount: float,
    curve_len: int,
) -> None:
    if start is None or end is None:
        return
    start = max(0, min(start, curve_len))
    end = max(0, min(end, curve_len))
    if start >= end or amount == 0:
        return
    for point in range(start, end):
        delta[point] = delta.get(point, 0.0) + amount


def _build_memory_curve(
    gm: fx.GraphModule,
    *,
    exact_current_peak: int,
    exact_target_peak: float,
) -> _MemoryCurve:
    """Approximate current remat memory with original-graph storage intervals.

    The curve uses GraphAliasTracker for real storage identity. Recomputed
    forward values are freed at their last forward use, saved values live to
    their backward use, and recompute-closure nodes add transient backward
    intervals. The curve is a planner heuristic; candidate acceptance still uses
    exact remat + build_memory_profile.
    """
    nodes = list(gm.graph.nodes)
    order = {node: i for i, node in enumerate(nodes)}
    alias = GraphAliasTracker(nodes)
    schedule_before_bwd = _recompute_schedule_before_bwd(nodes, order)
    events, event_sequence, after_free_by_event = _build_event_points_with_remat(
        gm, schedule_before_bwd
    )
    bwd_start_order = min(
        (order[n] for n in nodes if _is_backward_node(n)),
        default=len(nodes),
    )

    free_point_by_storage: dict[object, int] = {}
    for storage_key, allocator in alias.storage_to_allocator.items():
        if storage_key.device.type == "cpu" or not _is_releasable(allocator):
            continue
        uses = set(alias.storage_to_uses.get(storage_key, ()))
        if any(use.op == "output" for use in uses):
            continue
        if allocator.meta.get("recompute") in _RECOMPUTE_POLICIES:
            last_user = _last_use_before(uses, order, bwd_start_order) or allocator
        else:
            last_user = _last_executable_use(uses, order, events) or allocator
        if last_user in events.after_free:
            free_point_by_storage[storage_key] = events.after_free[last_user]

    storages_by_free_point: dict[int, list[object]] = {}
    for storage_key, point in free_point_by_storage.items():
        storages_by_free_point.setdefault(point, []).append(storage_key)

    remat_event_by_node: dict[fx.Node, tuple[str, fx.Node]] = {}
    for event in event_sequence:
        if event[0] == "remat":
            remat_event_by_node[event[1]] = event
    event_position = {event: i for i, event in enumerate(event_sequence)}
    remat_after_free = {
        node: after_free_by_event[event] for node, event in remat_event_by_node.items()
    }

    remat_storage_last_use: dict[tuple[fx.Node, object], tuple[str, fx.Node]] = {}
    for dep, dep_event in remat_event_by_node.items():
        for storage_key in alias.get_fresh_allocations(dep):
            if storage_key.device.type == "cpu":
                continue
            for event in event_sequence:
                if event_position[event] <= event_position[dep_event]:
                    continue
                use_node = event[1]
                if event[0] == "node" and not _is_backward_node(use_node):
                    continue
                if storage_key not in alias.get_storage_uses(use_node):
                    continue
                remat_storage_last_use[(dep, storage_key)] = event

    remat_storages_by_free_event: dict[
        tuple[str, fx.Node], list[tuple[fx.Node, object]]
    ] = {}
    remat_intervals: dict[fx.Node, list[tuple[int, int, float]]] = {}
    for (dep, storage_key), last_use_event in remat_storage_last_use.items():
        remat_storages_by_free_event.setdefault(last_use_event, []).append(
            (dep, storage_key)
        )
        dep_event = remat_event_by_node[dep]
        # Remat event alloc points are one point before their after-free.
        start = after_free_by_event[dep_event] - 1
        end = after_free_by_event[last_use_event]
        amount = float(_storage_nbytes(storage_key))
        remat_intervals.setdefault(dep, []).append((start, end, amount))

    current = 0.0
    for node in nodes:
        if node.op not in ("placeholder", "get_attr"):
            continue
        for storage_key in alias.get_fresh_allocations(node):
            if storage_key.device.type != "cpu":
                current += _storage_nbytes(storage_key)

    curve = [current]
    for event in event_sequence:
        kind, node = event
        if kind == "remat":
            for storage_key in alias.get_fresh_allocations(node):
                if storage_key.device.type != "cpu":
                    current += _storage_nbytes(storage_key)
        else:
            for storage_key in alias.get_fresh_allocations(node):
                if storage_key.device.type != "cpu":
                    current += _storage_nbytes(storage_key)
        curve.append(current)

        if kind == "node":
            for storage_key in storages_by_free_point.get(events.after_free[node], ()):
                current -= _storage_nbytes(storage_key)
        for dep, storage_key in remat_storages_by_free_event.get(event, ()):
            current -= _storage_nbytes(storage_key)
        curve.append(current)

    curve_peak = max(curve) if curve else 0.0
    headroom = float(exact_target_peak - exact_current_peak)
    return _MemoryCurve(
        curve=curve,
        peak_target=max(0.0, curve_peak + headroom),
        events=events,
        alias=alias,
        order=order,
        bwd_start_order=bwd_start_order,
        free_point_by_storage=free_point_by_storage,
        remat_intervals=remat_intervals,
        remat_after_free=remat_after_free,
    )


def _candidate_memory_curve_delta(
    memory_curve: _MemoryCurve,
    node: fx.Node,
    removed_recompute: set[fx.Node],
) -> tuple[dict[int, float], int]:
    delta: dict[int, float] = {}
    saved_bytes = 0
    curve_len = len(memory_curve.curve)

    for storage_key in memory_curve.alias.node_to_output_storages[node]:
        if storage_key.device.type == "cpu":
            continue
        amount = _storage_nbytes(storage_key)
        saved_bytes += amount
        start = memory_curve.free_point_by_storage.get(storage_key)
        use_points: list[int] = []
        for use in memory_curve.alias.storage_to_uses.get(storage_key, ()):
            if (
                memory_curve.order[use] >= memory_curve.bwd_start_order
                and use in memory_curve.events.after_free
            ):
                use_points.append(memory_curve.events.after_free[use])
        for remat_node, after_free in memory_curve.remat_after_free.items():
            if remat_node in removed_recompute:
                continue
            if storage_key in memory_curve.alias.get_storage_uses(remat_node):
                use_points.append(after_free)
        end = max(use_points, default=None)
        if end is None:
            uses = {
                u
                for u in memory_curve.alias.storage_to_uses.get(storage_key, ())
                if memory_curve.order[u] >= memory_curve.bwd_start_order
            }
            end_node = _last_executable_use(
                uses, memory_curve.order, memory_curve.events
            )
            end = (
                memory_curve.events.after_free[end_node]
                if end_node is not None
                else None
            )
        _add_interval(delta, start, end, float(amount), curve_len)

    for removed in removed_recompute:
        for start, end, amount in memory_curve.remat_intervals.get(removed, ()):
            _add_interval(delta, start, end, -amount, curve_len)

    return delta, saved_bytes


def _delta_fits(
    memory_curve: _MemoryCurve,
    selected_delta: list[float],
    local_delta: dict[int, float],
) -> bool:
    """Return whether one candidate fits the current memory curve target.

    At each event point P, the modeled memory after accepting this candidate is:

        baseline curve[P]       40 GiB
        accepted saves delta    +3 GiB   # selected_delta[P]
        this candidate delta    +2 GiB   # local_delta[P]
                                -------
        modeled memory          45 GiB > 44 GiB target, so reject

    local_delta is sparse. Positive intervals keep this candidate's forward
    storage live until backward uses it. Negative intervals remove backward
    remat transients when this save makes an upstream recompute closure
    unnecessary:

        forward:  A -> B -> C
        save B:   +live(B) before backward, -remat(A), -remat(B) in backward

    selected_delta is the sum of previous candidate deltas against the same
    curve. If the caller rebuilds the curve after each accepted save, those
    saves are already in the new curve and selected_delta stays zero.
    """
    for point, amount in local_delta.items():
        if (
            memory_curve.curve[point] + selected_delta[point] + amount
            > memory_curve.peak_target
        ):
            return False
    return True


def _apply_delta(selected_delta: list[float], local_delta: dict[int, float]) -> None:
    for point, amount in local_delta.items():
        selected_delta[point] += amount


def _memory_curve_peak(
    memory_curve: _MemoryCurve, selected_delta: list[float]
) -> float:
    return max(
        base + delta
        for base, delta in zip(memory_curve.curve, selected_delta, strict=True)
    )
