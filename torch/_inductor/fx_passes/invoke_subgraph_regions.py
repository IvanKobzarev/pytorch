"""FX helper for outlining explicit invoke_subgraph regions."""

# mypy: allow-untyped-defs
from collections.abc import Callable
from dataclasses import dataclass
from operator import attrgetter, getitem
from typing import Any

import torch
import torch.fx as fx
from torch.fx._lazy_graph_module import _LazyGraphModule
from torch.utils._ordered_set import OrderedSet


@dataclass(frozen=True)
class _OutlinedInvokeSubgraph:
    region_node: fx.Node
    replacements: tuple[fx.Node, ...]


def _get_subgraph_name(gm: fx.GraphModule, name: str) -> str:
    i = 0
    while hasattr(gm, f"{name}_{i}"):
        i += 1
    return f"{name}_{i}"


def _copy_placeholder_meta(
    placeholder: fx.Node, input_node: fx.Node, owning_module: fx.GraphModule
) -> None:
    if "val" in input_node.meta:
        placeholder.meta.update(input_node.meta)
    elif input_node.op == "get_attr" and isinstance(input_node.target, str):
        placeholder.meta["val"] = attrgetter(input_node.target)(owning_module)


def _strip_custom_keys_from_meta(meta: dict[str, Any], keys: tuple[str, ...]) -> None:
    if not keys:
        return
    custom = meta.get("custom")
    if not isinstance(custom, dict):
        return

    custom = custom.copy()
    for key in keys:
        custom.pop(key, None)
    compile_with_inductor = custom.get("compile_with_inductor")
    if isinstance(compile_with_inductor, dict):
        compile_with_inductor = compile_with_inductor.copy()
        for key in keys:
            compile_with_inductor.pop(key, None)
        custom["compile_with_inductor"] = compile_with_inductor

    if custom:
        meta["custom"] = custom
    else:
        meta.pop("custom", None)


def _outline_invoke_subgraph(
    graph: fx.Graph,
    nodes: list[fx.Node],
    *,
    region_name_prefix: str,
    output_name_suffix: str,
    always_return_tuple: bool = False,
    strip_custom_keys: tuple[str, ...] = (),
    sort_region_outputs: Callable[[fx.Node], Any] | None = None,
) -> _OutlinedInvokeSubgraph:
    owning_module = graph.owning_module
    if owning_module is None:
        raise AssertionError("expected graph to have an owning_module")
    if not nodes:
        raise AssertionError("expected non-empty nodes")

    node_set = OrderedSet(nodes)
    ordered_nodes = [node for node in graph.nodes if node in node_set]
    if len(ordered_nodes) != len(node_set):
        raise AssertionError("expected all nodes to belong to graph")
    if any(node.op in ("placeholder", "output") for node in ordered_nodes):
        raise AssertionError(
            "expected invoke_subgraph nodes to exclude graph boundaries"
        )

    region_outputs = [
        node
        for node in ordered_nodes
        if any(user not in node_set for user in node.users)
    ]
    if sort_region_outputs is not None:
        region_outputs.sort(key=sort_region_outputs)

    subgraph = fx.Graph(owning_module)
    env: dict[fx.Node, fx.Node] = {}
    input_replacements: dict[fx.Node, Any] = {}
    boundary_args: list[tuple[fx.Node, tuple[int, ...], Any]] = []

    external_inputs: OrderedSet[fx.Node] = OrderedSet()
    preserved_getattrs: OrderedSet[fx.Node] = OrderedSet()

    def collect_external_input(node: fx.Node) -> fx.Node:
        if node not in node_set:
            if (
                node.op == "get_attr"
                and isinstance(node.target, str)
                and isinstance(attrgetter(node.target)(owning_module), fx.GraphModule)
            ):
                preserved_getattrs.add(node)
            else:
                external_inputs.add(node)
        return node

    for node in ordered_nodes:
        fx.map_arg((node.args, node.kwargs), collect_external_input)

    node_order = {node: idx for idx, node in enumerate(graph.nodes)}
    latest_input = max(
        external_inputs,
        key=lambda node: node_order[node],
        default=None,
    )
    first_external_user = min(
        (
            user
            for output_node in region_outputs
            for user in output_node.users
            if user not in node_set
        ),
        key=lambda node: node_order[node],
        default=None,
    )
    if (
        first_external_user is not None
        and latest_input is not None
        and node_order[latest_input] >= node_order[first_external_user]
    ):
        raise AssertionError("expected invoke_subgraph boundary to be acyclic")

    def add_boundary_arg(
        input_node: fx.Node, path: tuple[int, ...], meta_val: Any
    ) -> fx.Node:
        placeholder = subgraph.placeholder(f"arg_{len(boundary_args)}")
        _copy_placeholder_meta(placeholder, input_node, owning_module)
        if path:
            placeholder.meta["val"] = meta_val
        boundary_args.append((input_node, path, meta_val))
        return placeholder

    def make_input_replacement(
        input_node: fx.Node, value: Any, path: tuple[int, ...] = ()
    ) -> Any:
        if isinstance(value, (tuple, list)):
            return type(value)(
                make_input_replacement(input_node, item, (*path, idx))
                for idx, item in enumerate(value)
            )
        return add_boundary_arg(input_node, path, value)

    for input_node in external_inputs:
        value = input_node.meta.get("val")
        if isinstance(value, (tuple, list)):
            input_replacements[input_node] = make_input_replacement(input_node, value)
        else:
            input_replacements[input_node] = add_boundary_arg(input_node, (), value)

    def load_arg(node: fx.Node) -> Any:
        if node in env:
            return env[node]
        if node in node_set:
            raise AssertionError("expected invoke_subgraph nodes to be topological")
        if node in preserved_getattrs:
            if not isinstance(node.target, str):
                raise AssertionError("expected get_attr target to be a string")
            get_attr_node = subgraph.get_attr(node.target)
            get_attr_node.meta.update(node.meta)
            env[node] = get_attr_node
            return get_attr_node
        return input_replacements[node]

    for node in ordered_nodes:
        env[node] = subgraph.node_copy(node, load_arg)

    subgraph_outputs = tuple(env[node] for node in region_outputs)
    if len(subgraph_outputs) == 0:
        out = subgraph.output(())
        out.meta["val"] = ()
    elif len(subgraph_outputs) == 1 and not always_return_tuple:
        out = subgraph.output(subgraph_outputs[0])
        if "val" in region_outputs[0].meta:
            out.meta["val"] = region_outputs[0].meta["val"]
    else:
        out = subgraph.output(subgraph_outputs)
        out.meta["val"] = tuple(node.meta.get("val") for node in region_outputs)
    subgraph.lint()

    subgraph_module = _LazyGraphModule(owning_module, subgraph)
    first_name = ordered_nodes[0].name
    last_name = ordered_nodes[-1].name
    region_name = f"{region_name_prefix}_{first_name}_{last_name}"
    subgraph_attr_name = _get_subgraph_name(owning_module, region_name)
    setattr(owning_module, subgraph_attr_name, subgraph_module)

    if latest_input is None or node_order[latest_input] < node_order[ordered_nodes[0]]:
        with graph.inserting_before(ordered_nodes[0]):
            get_subgraph = graph.get_attr(subgraph_attr_name)
    else:
        with graph.inserting_after(latest_input):
            get_subgraph = graph.get_attr(subgraph_attr_name)

    outer_args: list[fx.Node] = []
    insert_after = get_subgraph

    def make_outer_arg(
        input_node: fx.Node, path: tuple[int, ...], meta_val: Any
    ) -> fx.Node:
        nonlocal insert_after
        source = input_node
        for idx in path:
            with graph.inserting_after(insert_after):
                source = graph.call_function(
                    getitem,
                    args=(source, idx),
                    name=(
                        f"{input_node.name}_{output_name_suffix}_arg_{len(outer_args)}"
                    ),
                )
            insert_after = source
        if path:
            source.meta["val"] = meta_val
        return source

    for input_node, path, meta_val in boundary_args:
        outer_args.append(make_outer_arg(input_node, path, meta_val))

    with graph.inserting_after(insert_after):
        region_node = graph.call_function(
            torch.ops.higher_order.invoke_subgraph,
            args=(get_subgraph, subgraph_attr_name, *outer_args),
            name=region_name,
        )

    replacements: list[fx.Node] = []
    if len(region_outputs) == 0:
        region_node.meta["val"] = ()
    elif len(region_outputs) == 1 and not always_return_tuple:
        replacement = region_node
        replacement.meta = region_outputs[0].meta.copy()
        replacement.meta.pop("eager_input_vals", None)
        _strip_custom_keys_from_meta(replacement.meta, strip_custom_keys)
        replacements.append(replacement)
    else:
        region_node.meta["val"] = tuple(node.meta.get("val") for node in region_outputs)
        insert_after = region_node
        for idx, output_node in enumerate(region_outputs):
            with graph.inserting_after(insert_after):
                replacement = graph.call_function(
                    getitem,
                    args=(region_node, idx),
                    name=f"{output_node.name}_{output_name_suffix}",
                )
            replacement.meta = output_node.meta.copy()
            replacement.meta.pop("eager_input_vals", None)
            _strip_custom_keys_from_meta(replacement.meta, strip_custom_keys)
            replacements.append(replacement)
            insert_after = replacement

    for output_node, replacement in zip(region_outputs, replacements, strict=True):
        for user in list(output_node.users):
            if user not in node_set:
                user.replace_input_with(output_node, replacement)

    for node in reversed(ordered_nodes):
        graph.erase_node(node)
    graph.lint()

    return _OutlinedInvokeSubgraph(region_node, tuple(replacements))


def mark_invoke_subgraph(
    graph: fx.Graph,
    nodes: list[fx.Node],
    *,
    region_name_prefix: str = "invoke_subgraph_region",
    strip_custom_keys: tuple[str, ...] = (),
    sort_region_outputs: Callable[[fx.Node], Any] | None = None,
) -> fx.Node:
    """Outline FX nodes into an invoke_subgraph HOP and return the HOP node."""
    return _outline_invoke_subgraph(
        graph,
        nodes,
        region_name_prefix=region_name_prefix,
        output_name_suffix=region_name_prefix,
        always_return_tuple=True,
        strip_custom_keys=strip_custom_keys,
        sort_region_outputs=sort_region_outputs,
    ).region_node
