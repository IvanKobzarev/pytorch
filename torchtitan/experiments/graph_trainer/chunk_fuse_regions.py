# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import sys
from collections import defaultdict
from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any

import torch
from torch._inductor.fx_passes.fuse_regions import FUSE_REGION
from torch.fx import Node
from torch.utils._pytree import tree_flatten

from torchtitan.experiments.graph_trainer.subgraph_regions import SUBGRAPH_REGION
from torchtitan.tools.logging import logger


_DEFAULT_MIN_TENSOR_BYTES = 64 * 1024 * 1024
_DEFAULT_MIN_REGIONS = 2
_DEFAULT_MIN_REGION_NODES = 3
_DEFAULT_SLICE_SEARCH_DEPTH = 8
_IGNORED_SLICE_END = sys.maxsize // 2


@dataclass(frozen=True)
class _SliceInfo:
    base: Node
    dim: int
    start: int
    end: int

    @property
    def span(self) -> int:
        return self.end - self.start


class _BackwardNodeIter:
    def __init__(self, node: Node) -> None:
        self.queue = list(_node_inputs(node))

    def next(self) -> Node | None:
        if not self.queue:
            return None
        return self.queue.pop(0)

    def add_inputs(self, node: Node) -> None:
        self.queue.extend(_node_inputs(node))


def _node_inputs(node: Node) -> list[Node]:
    args, _ = tree_flatten((node.args, node.kwargs))
    result: list[Node] = []
    seen: set[Node] = set()
    for arg in args:
        if isinstance(arg, Node) and arg not in seen:
            seen.add(arg)
            result.append(arg)
    return result


def _node_val(node: Node) -> Any:
    if "val" in node.meta:
        return node.meta["val"]
    return node.meta.get("example_value")


def _shape_key(shape: torch.Size) -> tuple[Hashable, ...]:
    return tuple(dim if isinstance(dim, int) else str(dim) for dim in shape)


def _value_key(value: Any) -> Hashable:
    if isinstance(value, torch.Tensor):
        return ("tensor", _shape_key(value.shape), str(value.dtype))
    if isinstance(value, (tuple, list)):
        return tuple(_value_key(v) for v in value)
    if value is None:
        return None
    return type(value).__name__


def _const_key(value: Any) -> Hashable:
    if isinstance(value, Node):
        return _value_key(_node_val(value))
    if isinstance(value, (str, int, float, bool, type(None), torch.dtype)):
        return value
    if isinstance(value, (tuple, list)):
        return tuple(_const_key(v) for v in value)
    if isinstance(value, dict):
        return tuple(
            (k, _const_key(v))
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        )
    return type(value).__name__


def _value_nbytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        try:
            return int(value.numel()) * value.dtype.itemsize
        except (TypeError, ValueError):
            return 0
    if isinstance(value, (tuple, list)):
        return max((_value_nbytes(v) for v in value), default=0)
    return 0


def _region_max_nbytes(region: list[Node]) -> int:
    return max((_value_nbytes(_node_val(node)) for node in region), default=0)


def _has_region_boundary(node: Node) -> bool:
    custom = node.meta.get("custom")
    if not isinstance(custom, dict):
        return False
    if FUSE_REGION in custom or SUBGRAPH_REGION in custom:
        return True
    compile_with_inductor = custom.get("compile_with_inductor")
    return (
        isinstance(compile_with_inductor, dict)
        and (
            FUSE_REGION in compile_with_inductor
            or SUBGRAPH_REGION in compile_with_inductor
        )
    )


def _is_higher_order_node(node: Node) -> bool:
    return node.op == "call_function" and isinstance(
        node.target, torch._ops.HigherOrderOperator
    )


def _node_fingerprint(node: Node) -> Hashable | None:
    if node.op in ("placeholder", "output", "get_attr") or _is_higher_order_node(node):
        return None
    if _has_region_boundary(node):
        return None
    stack_trace = node.meta.get("stack_trace")
    if not stack_trace:
        return None
    return (
        node.op,
        str(node.target),
        stack_trace,
        _value_key(_node_val(node)),
        _const_key(node.args),
        _const_key(node.kwargs),
    )


def _slice_info(node: Node) -> _SliceInfo | None:
    if node.op != "call_function" or node.target != torch.ops.aten.slice.Tensor:
        return None
    if len(node.args) < 4:
        return None
    base, dim, start, end = node.args[:4]
    if not (
        isinstance(base, Node)
        and isinstance(dim, int)
        and isinstance(start, int)
        and isinstance(end, int)
    ):
        return None
    if end >= _IGNORED_SLICE_END or end <= start:
        return None
    return _SliceInfo(base, dim, start, end)


def _collect_slice_infos(node: Node, *, max_depth: int) -> list[_SliceInfo]:
    infos: list[_SliceInfo] = []
    queue = [(node, 0)]
    seen: set[Node] = set()
    while queue:
        cur, depth = queue.pop(0)
        if cur in seen:
            continue
        seen.add(cur)
        info = _slice_info(cur)
        if info is not None:
            infos.append(info)
        if depth >= max_depth:
            continue
        queue.extend((arg, depth + 1) for arg in _node_inputs(cur))
    return infos


def _region_external_inputs(region: list[Node]) -> list[Node]:
    region_set = set(region)
    inputs: list[Node] = []
    seen: set[Node] = set()
    for node in region:
        for arg in _node_inputs(node):
            if arg in region_set or arg in seen:
                continue
            seen.add(arg)
            inputs.append(arg)
    return inputs


def _find_contiguous_chunk_slices(
    regions: list[list[Node]],
    *,
    slice_search_depth: int,
) -> tuple[_SliceInfo, ...] | None:
    intervals_by_key: dict[tuple[Node, int, int], dict[int, _SliceInfo]] = (
        defaultdict(dict)
    )
    for region_idx, region in enumerate(regions):
        for external_input in _region_external_inputs(region):
            for info in _collect_slice_infos(
                external_input, max_depth=slice_search_depth
            ):
                intervals_by_key[(info.base, info.dim, info.span)][region_idx] = info

    for intervals_by_region in intervals_by_key.values():
        if len(intervals_by_region) != len(regions):
            continue
        intervals_in_graph_order = tuple(
            intervals_by_region[idx] for idx in range(len(regions))
        )
        intervals_by_start = sorted(
            intervals_in_graph_order, key=lambda info: info.start
        )
        if all(
            lhs.end == rhs.start
            for lhs, rhs in zip(intervals_by_start, intervals_by_start[1:])
        ):
            return intervals_in_graph_order
    return None


def _is_contiguous_region(region: list[Node], node_to_rank: dict[Node, int]) -> bool:
    ranks = sorted(node_to_rank[node] for node in region)
    return ranks == list(range(ranks[0], ranks[-1] + 1))


def _expand_duplicate_regions(
    group: list[Node],
    node_to_duplicate_group: dict[Node, list[Node]],
    claimed: set[Node],
) -> list[list[Node]]:
    regions = [[node] for node in group]
    node_iters = [_BackwardNodeIter(node) for node in group]

    while True:
        first_candidate = node_iters[0].next()
        if first_candidate is None:
            break
        candidates = [first_candidate]
        duplicate_group = node_to_duplicate_group.get(first_candidate)
        add_to_all = (
            first_candidate not in claimed
            and not _is_higher_order_node(first_candidate)
            and duplicate_group is not None
        )
        for node_iter in node_iters[1:]:
            candidate = node_iter.next()
            if candidate is None:
                add_to_all = False
                continue
            candidates.append(candidate)
            add_to_all &= (
                candidate not in claimed
                and candidate not in candidates[:-1]
                and not _is_higher_order_node(candidate)
                and node_to_duplicate_group.get(candidate) is duplicate_group
            )

        if add_to_all:
            for region, node_iter, candidate in zip(regions, node_iters, candidates):
                region.append(candidate)
                node_iter.add_inputs(candidate)

    for region in regions:
        region.reverse()
    return regions


def annotate_auto_chunk_fuse_regions_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
    *,
    min_tensor_bytes: int = _DEFAULT_MIN_TENSOR_BYTES,
    min_regions: int = _DEFAULT_MIN_REGIONS,
    min_region_nodes: int = _DEFAULT_MIN_REGION_NODES,
    slice_search_depth: int = _DEFAULT_SLICE_SEARCH_DEPTH,
) -> torch.fx.GraphModule:
    """Annotate unrolled chunk bodies with Inductor fuse-region boundaries.

    The pass looks for repeated source/op/shape-identical graph regions with a
    large tensor anchor. It only accepts them when every repetition is fed by a
    same-span slice and those slices are contiguous on the same source tensor.
    """
    annotated_groups = 0
    annotated_regions = 0

    for module_idx, module in enumerate(gm.modules()):
        if not isinstance(module, torch.fx.GraphModule):
            continue
        node_to_rank = {node: idx for idx, node in enumerate(module.graph.nodes)}
        duplicates_by_fingerprint: dict[Hashable, list[Node]] = defaultdict(list)
        for node in module.graph.nodes:
            fingerprint = _node_fingerprint(node)
            if fingerprint is not None:
                duplicates_by_fingerprint[fingerprint].append(node)

        duplicate_groups = [
            nodes
            for nodes in duplicates_by_fingerprint.values()
            if len(nodes) >= min_regions
        ]
        duplicate_groups.sort(
            key=lambda nodes: -min(node_to_rank[node] for node in nodes)
        )
        anchor_groups = [
            nodes
            for nodes in duplicate_groups
            if _value_nbytes(_node_val(nodes[0])) >= min_tensor_bytes
        ]
        node_to_duplicate_group = {
            node: group for group in duplicate_groups for node in group
        }

        claimed: set[Node] = set()
        for duplicate_group in anchor_groups:
            group = [node for node in duplicate_group if node not in claimed]
            if len(group) < min_regions:
                continue
            regions = _expand_duplicate_regions(group, node_to_duplicate_group, claimed)
            if len(regions[0]) < min_region_nodes:
                continue
            if _region_max_nbytes(regions[0]) < min_tensor_bytes:
                continue
            if any(
                not _is_contiguous_region(region, node_to_rank) for region in regions
            ):
                continue
            if (
                _find_contiguous_chunk_slices(
                    regions, slice_search_depth=slice_search_depth
                )
                is None
            ):
                continue

            region_name_prefix = f"auto_chunk_m{module_idx}_g{annotated_groups}"
            for region_idx, region in enumerate(regions):
                region_name = f"{region_name_prefix}_r{region_idx}"
                for node in region:
                    node.meta.setdefault("custom", {}).setdefault(
                        FUSE_REGION, region_name
                    )
                    claimed.add(node)
                annotated_regions += 1
            annotated_groups += 1

    if annotated_regions:
        logger.info(
            "Annotated %d auto chunk fuse regions across %d repeated chunk groups",
            annotated_regions,
            annotated_groups,
        )
    return gm
