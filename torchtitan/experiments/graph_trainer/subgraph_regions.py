# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from contextlib import nullcontext
from operator import attrgetter
from typing import Any

import torch
from torch.fx import Node
from torch.fx.traceback import annotate

from torchtitan.tools.logging import logger


SUBGRAPH_REGION = "graph_trainer_subgraph_region"


def subgraph(name: str | None):
    if name is None:
        return nullcontext()
    if not isinstance(name, str):
        raise AssertionError(
            f"expected subgraph region name to be str, got {type(name)}"
        )
    return annotate({SUBGRAPH_REGION: name})


def _getattr_or_none(module: torch.fx.GraphModule, target: str) -> Any:
    try:
        return attrgetter(target)(module)
    except AttributeError:
        return None


def _has_graph_module_arg(node: Node) -> bool:
    gm = node.graph.owning_module
    if gm is None:
        return False
    return any(
        inp.op == "get_attr"
        and isinstance(inp.target, str)
        and isinstance(_getattr_or_none(gm, inp.target), torch.fx.GraphModule)
        for inp in node.all_input_nodes
    )


def _subgraph_region_from_custom(custom: dict[str, Any]) -> str | None:
    region = custom.get(SUBGRAPH_REGION)
    if region is not None:
        if not isinstance(region, str):
            raise AssertionError(
                f"expected custom {SUBGRAPH_REGION} to be a str, got {type(region)}"
            )
        return region

    compile_with_inductor = custom.get("compile_with_inductor")
    if isinstance(compile_with_inductor, dict):
        region = compile_with_inductor.get(SUBGRAPH_REGION)
        if region is not None:
            if not isinstance(region, str):
                raise AssertionError(
                    f"expected custom compile_with_inductor {SUBGRAPH_REGION} "
                    f"to be a str, got {type(region)}"
                )
            return region
    return None


def subgraph_region_key(node: Node) -> str | None:
    if node.op in ("placeholder", "output", "get_attr"):
        return None
    if node.op == "call_function" and isinstance(
        node.target, torch._ops.HigherOrderOperator
    ):
        return None
    if _has_graph_module_arg(node):
        return None

    custom = node.meta.get("custom")
    if not isinstance(custom, dict):
        return None
    region = _subgraph_region_from_custom(custom)
    if region is None:
        return None
    phase = "bwd" if node.meta.get("autograd_backward") is True else "fwd"
    return f"{region}_{phase}"


def collect_subgraph_region_groups(
    graph: torch.fx.Graph,
) -> list[tuple[str, list[Node]]]:
    groups: list[tuple[str, list[Node]]] = []
    current_key: str | None = None
    current_nodes: list[Node] = []

    def flush() -> None:
        nonlocal current_key, current_nodes
        if current_key is not None and len(current_nodes) > 1:
            groups.append((current_key, current_nodes))
        current_key = None
        current_nodes = []

    for node in list(graph.nodes):
        key = subgraph_region_key(node)
        if key is None:
            flush()
            continue
        if key != current_key:
            flush()
            current_key = key
        current_nodes.append(node)
    flush()
    return groups


def _record_subgraph_region(
    module: torch.fx.GraphModule, region_node: Node, region: str
) -> None:
    region_node.meta[SUBGRAPH_REGION] = region
    get_subgraph = region_node.args[0]
    if not (
        isinstance(get_subgraph, Node)
        and get_subgraph.op == "get_attr"
        and isinstance(get_subgraph.target, str)
    ):
        return
    submod = getattr(module, get_subgraph.target, None)
    if isinstance(submod, torch.fx.GraphModule):
        submod.meta[SUBGRAPH_REGION] = region


def apply_subgraph_region_annotations_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
) -> torch.fx.GraphModule:
    from torch._inductor.fx_passes.fuse_regions import mark_invoke_subgraph

    outlined_regions = 0
    for module in list(gm.modules()):
        if not isinstance(module, torch.fx.GraphModule):
            continue
        groups = collect_subgraph_region_groups(module.graph)
        if not groups:
            continue
        for region, nodes in groups:
            region_node = mark_invoke_subgraph(
                module.graph,
                nodes,
                region_name_prefix=f"subgraph_region_{outlined_regions}",
                strip_custom_keys=(SUBGRAPH_REGION,),
            )
            _record_subgraph_region(module, region_node, region)
            outlined_regions += 1
        module.graph.lint()
        module.recompile()

    if outlined_regions:
        logger.info("Outlined %d annotated subgraph regions", outlined_regions)
    return gm
