# mypy: allow-untyped-defs
"""
Effect ordering pass for inductor.

This pass adds ordering dependencies to FX graphs using the control_deps HOP
for precise control over scheduling constraints. When you need exact ordering between
operations (e.g., collective_start -> mm -> wait), this pass wraps operations
with control_deps to make dependencies explicit.
"""

from operator import attrgetter, getitem
from typing import Any

import torch.fx as fx
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._higher_order_ops.utils import register_fake
from torch._ops import HigherOrderOperator
from torch.fx._lazy_graph_module import _LazyGraphModule
from torch.utils._ordered_set import OrderedSet


class ControlDeps(HigherOrderOperator):
    """
    Higher-order operator that enforces ordering by making dependencies explicit.

    Schema: control_deps(additional_deps, target, *args, **kwargs) -> result
    where:
    - additional_deps: tuple of tensors that must be computed before this op
    - subgraph: GraphModule containing the exact operation to execute
    - args/kwargs: arguments for the target function

    This ensures all tensors in additional_deps are computed before the target
    executes, creating explicit scheduling dependencies.
    """

    def __init__(self) -> None:
        super().__init__("control_deps")

    def __call__(self, additional_deps, subgraph, *args, **kwargs):
        """Call the operator with dependencies and subgraph.

        Args:
            additional_deps: Tuple of tensors that must be computed first
            subgraph: GraphModule containing the exact operation to execute
            *args: Arguments to pass to the subgraph
        """
        if not isinstance(additional_deps, (tuple, list)):
            raise TypeError(
                f"additional_deps must be tuple/list, got {type(additional_deps).__name__}"
            )
        if not (isinstance(subgraph, fx.GraphModule) or callable(subgraph)):
            raise TypeError(
                f"subgraph must be GraphModule or callable, got {type(subgraph).__name__}"
            )
        # pyrefly: ignore [missing-attribute]
        return super().__call__(additional_deps, subgraph, *args, **kwargs)


control_deps = ControlDeps()

META_OVERLAP_DEPS = "inductor_overlap_deps"

# control_deps wraps side-effecting ops (e.g. record_event, wait_event)
# and must not be eliminated by DCE even when its outputs are unused.
from torch.fx.node import has_side_effect


has_side_effect(control_deps)


# Register fake implementation for tracing
@register_fake(control_deps)
def _(additional_deps, subgraph, *args, **kwargs):
    """Fake tensor implementation - execute the subgraph."""
    return subgraph(*args, **kwargs)


# Register eager execution implementation
@control_deps.py_impl(DispatchKey.CompositeExplicitAutograd)
def control_deps_eager(additional_deps, subgraph, *args, **kwargs):
    """Eager implementation - just execute the subgraph."""
    return subgraph(*args, **kwargs)


# Autograd impl needed because additional_deps tensors may have autograd state,
# causing dispatch through AutogradCUDA even in post-autograd graphs.
@control_deps.py_impl(DispatchKey.Autograd)
def control_deps_autograd(additional_deps, subgraph, *args, **kwargs):
    return subgraph(*args, **kwargs)


def get_subgraph_name(gm: fx.GraphModule, name):
    name = f"subgraph_{name}"

    if not hasattr(gm, name):
        return name

    i = 0
    while hasattr(gm, f"{name}_{i}"):
        i += 1

    return f"{name}_{i}"


def _extract_unique_nodes(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[list[fx.Node], list[Any], Any]:
    """Extract unique fx.Node instances from args/kwargs using pytree.

    Args:
        args: The positional arguments (may contain nested structures with fx.Node)
        kwargs: The keyword arguments (may contain nested structures with fx.Node)

    Returns:
        - Ordered list of unique fx.Node instances (preserves first occurrence order)
        - Flattened list of all items from args/kwargs
        - The pytree spec for reconstructing the original structure
    """
    flat_args_kwargs, spec = pytree.tree_flatten((args, kwargs))
    unique_nodes: list[fx.Node] = []
    seen: OrderedSet[fx.Node] = OrderedSet()
    for item in flat_args_kwargs:
        if isinstance(item, fx.Node) and item not in seen:
            unique_nodes.append(item)
            seen.add(item)
    return unique_nodes, flat_args_kwargs, spec


def preserve_node_ordering(
    graph: fx.Graph,
    additional_deps_map: dict[fx.Node, OrderedSet[fx.Node]],
    verbose: bool = False,
) -> None:
    """
    Preserve node ordering using control_deps HOP with subgraph.

    This function wraps operations with control_deps that:
    1. Makes additional dependencies explicit (first argument)
    2. Creates a subgraph internally to preserve the exact original operation
    3. Preserves the original node names

    Args:
        graph: The FX graph to modify
        additional_deps_map: Mapping from dependent nodes to their dependencies
        verbose: If True, print debug information
    """
    if not additional_deps_map:
        return

    # Track replacements so we can update dependencies
    replacements: dict[fx.Node, fx.Node] = {}

    # Process each node that needs additional dependencies
    for dependent_node, dep_nodes in additional_deps_map.items():
        if dependent_node.op != "call_function":
            raise AssertionError(dependent_node.op)

        original_name = dependent_node.name
        original_args = dependent_node.args
        original_kwargs = dependent_node.kwargs
        original_meta = dependent_node.meta.copy()

        updated_dep_nodes = [replacements.get(dep, dep) for dep in dep_nodes]

        # Create a subgraph that preserves the exact original operation
        subgraph_module = _create_subgraph_for_node(graph, dependent_node)

        owning_mod = graph.owning_module
        if owning_mod is None:
            raise AssertionError("expected graph to have an owning_module")
        subgraph_attr_name = get_subgraph_name(owning_mod, original_name)
        setattr(graph.owning_module, subgraph_attr_name, subgraph_module)

        # Create control_deps call with:
        # 1. Additional dependencies as first arg (explicit)
        # 2. Subgraph via get_attr (like b2b gemm pass)
        # 3. Original arguments (only fx.Node args and kwargs are passed)
        with graph.inserting_before(dependent_node):
            # Create get_attr node for the subgraph
            get_subgraph = graph.get_attr(subgraph_attr_name)

            # Extract unique nodes from nested args/kwargs
            node_args, _, _ = _extract_unique_nodes(original_args, original_kwargs)

            # Create with temporary name first
            ordered_node = graph.call_function(
                control_deps,
                args=(
                    tuple(updated_dep_nodes),  # additional_deps
                    get_subgraph,  # subgraph via get_attr (like b2b gemm)
                    *node_args,  # original node arguments (from both args and kwargs)
                ),
                kwargs={},
                name=f"__temp_{original_name}",  # Temporary name to avoid conflict
            )

        # Copy metadata from original node
        ordered_node.meta = original_meta
        # this will be constrained on the target node in subgraph if it exists
        ordered_node.meta.pop("eager_input_vals", None)

        # Replace all uses of the original node with the ordered version
        dependent_node.replace_all_uses_with(ordered_node)

        # Remove the original node from the graph
        graph.erase_node(dependent_node)

        # Now rename the ordered node to the original name
        ordered_node.name = original_name  # PRESERVE ORIGINAL NAME

        # Track the replacement for future dependencies
        replacements[dependent_node] = ordered_node


def preserve_node_ordering_with_region_subgraphs(
    graph: fx.Graph,
    additional_deps_map: dict[fx.Node, OrderedSet[fx.Node]],
    regions: list[list[fx.Node]],
    verbose: bool = False,
) -> None:
    """
    Preserve ordering with one control_deps HOP per fusible overlap region.

    The fallback control_deps implementation wraps every ordered node in its own
    subgraph.  That preserves the schedule but prevents fusion across adjacent
    hidden compute nodes.  Region wrapping keeps the same external ordering
    constraints while leaving nodes inside the selected region visible to the
    normal Inductor lowering/fusion pipeline.
    """
    if not regions:
        preserve_node_ordering(graph, additional_deps_map, verbose=verbose)
        return

    replacements: dict[fx.Node, fx.Node] = {}
    region_nodes: OrderedSet[fx.Node] = OrderedSet()

    for region in regions:
        live_region = [n for n in region if not n._erased]
        if len(live_region) < 2:
            continue
        if any(n in region_nodes for n in live_region):
            raise AssertionError("overlap control_deps regions must be disjoint")
        region_nodes.update(live_region)
        replacements.update(
            _wrap_region_with_control_deps(
                graph,
                live_region,
                additional_deps_map,
                replacements,
            )
        )

    rewritten_deps: dict[fx.Node, OrderedSet[fx.Node]] = {}
    for node, deps in additional_deps_map.items():
        if node in region_nodes or node._erased:
            continue
        updated_node = replacements.get(node, node)
        if updated_node._erased:
            continue
        updated_deps: OrderedSet[fx.Node] = OrderedSet()
        for dep in deps:
            if dep._erased and dep not in replacements:
                continue
            updated_dep = replacements.get(dep, dep)
            if updated_dep is updated_node or updated_dep._erased:
                continue
            updated_deps.add(updated_dep)
        if updated_deps:
            rewritten_deps.setdefault(updated_node, OrderedSet()).update(updated_deps)

    if rewritten_deps:
        preserve_node_ordering(graph, rewritten_deps, verbose=verbose)


def preserve_node_ordering_with_meta(
    graph: fx.Graph,
    additional_deps_map: dict[fx.Node, OrderedSet[fx.Node]],
    verbose: bool = False,
) -> None:
    """
    Preserve node ordering by storing dependency metadata on FX nodes.

    The scheduler consumes this metadata after lowering and fusion. Unlike
    control_deps, this does not wrap nodes in subgraphs, so normal Inductor
    fusion is still available.
    """
    if not additional_deps_map:
        return

    for dependent_node, dep_nodes in additional_deps_map.items():
        if dependent_node.op != "call_function":
            raise AssertionError(dependent_node.op)

        deps = _get_meta_overlap_deps(dependent_node)

        for dep_node in dep_nodes:
            deps.add(dep_node)


def _get_meta_overlap_deps(node: fx.Node) -> OrderedSet[fx.Node]:
    deps = node.meta.get(META_OVERLAP_DEPS)
    if deps is None:
        deps = OrderedSet()
        node.meta[META_OVERLAP_DEPS] = deps
    elif not isinstance(deps, OrderedSet):
        deps = OrderedSet(deps)
        node.meta[META_OVERLAP_DEPS] = deps
    return deps


def _rewrite_meta_overlap_deps(
    deps: OrderedSet[fx.Node],
    replacements: dict[fx.Node, fx.Node | None],
    target: fx.Node,
) -> OrderedSet[fx.Node]:
    rewritten: OrderedSet[fx.Node] = OrderedSet()
    for dep in deps:
        dep = replacements.get(dep, dep)
        if dep is None or dep is target or dep._erased:
            continue
        rewritten.add(dep)
    return rewritten


def transfer_meta_overlap_deps(
    graph: fx.Graph,
    replacements: dict[fx.Node, fx.Node | None],
) -> None:
    """
    Transfer metadata ordering constraints across graph rewrites.

    Collective bucketing erases the original start/wait nodes and replaces them
    with bucketed nodes. Metadata deps must follow those replacements; otherwise
    lowering cannot map the erased FX nodes to scheduler operation names.
    """
    if not replacements:
        return

    for old_node, new_node in replacements.items():
        if new_node is None:
            continue
        deps = old_node.meta.get(META_OVERLAP_DEPS)
        if not deps:
            continue
        rewritten = _rewrite_meta_overlap_deps(OrderedSet(deps), replacements, new_node)
        if not rewritten:
            continue
        new_deps = _get_meta_overlap_deps(new_node)
        new_deps.update(rewritten)

    for node in graph.nodes:
        deps = node.meta.get(META_OVERLAP_DEPS)
        if not deps:
            continue
        rewritten = _rewrite_meta_overlap_deps(OrderedSet(deps), replacements, node)
        if rewritten:
            node.meta[META_OVERLAP_DEPS] = rewritten
        else:
            node.meta.pop(META_OVERLAP_DEPS, None)


def preserve_node_ordering_from_config(
    graph: fx.Graph,
    additional_deps_map: dict[fx.Node, OrderedSet[fx.Node]],
    verbose: bool = False,
) -> None:
    from torch._inductor import config

    impl = config.aten_distributed_optimizations.insert_overlap_deps_impl
    if impl == "control_deps":
        preserve_node_ordering(graph, additional_deps_map, verbose=verbose)
    elif impl == "meta":
        preserve_node_ordering_with_meta(graph, additional_deps_map, verbose=verbose)
    else:
        raise RuntimeError(f"Unknown insert_overlap_deps_impl: {impl}")


def _copy_placeholder_meta(
    placeholder: fx.Node, orig_node: fx.Node, owning_module: fx.GraphModule
) -> None:
    if "val" in orig_node.meta:
        placeholder.meta.update(orig_node.meta)
    elif orig_node.op == "get_attr" and isinstance(orig_node.target, str):
        placeholder.meta["val"] = attrgetter(orig_node.target)(owning_module)


def _wrap_region_with_control_deps(
    graph: fx.Graph,
    region: list[fx.Node],
    additional_deps_map: dict[fx.Node, OrderedSet[fx.Node]],
    replacements: dict[fx.Node, fx.Node],
) -> dict[fx.Node, fx.Node]:
    owning_mod = graph.owning_module
    if owning_mod is None:
        raise AssertionError("expected graph to have an owning_module")

    region_set = OrderedSet(region)
    for node in region:
        if node.op != "call_function":
            raise AssertionError(node.op)

    external_inputs: list[fx.Node] = []
    seen_inputs: OrderedSet[fx.Node] = OrderedSet()
    for node in region:
        flat_args_kwargs, _ = pytree.tree_flatten((node.args, node.kwargs))
        for item in flat_args_kwargs:
            if (
                isinstance(item, fx.Node)
                and item not in region_set
                and item not in seen_inputs
            ):
                external_inputs.append(item)
                seen_inputs.add(item)

    externally_depended_on: OrderedSet[fx.Node] = OrderedSet()
    for node, deps in additional_deps_map.items():
        if node in region_set:
            continue
        for dep in deps:
            if dep in region_set:
                externally_depended_on.add(dep)

    user_output_nodes: OrderedSet[fx.Node] = OrderedSet()
    for node in region:
        if any(user not in region_set for user in node.users):
            user_output_nodes.add(node)

    candidate_dep_outputs = OrderedSet(user_output_nodes)
    candidate_dep_outputs.update(externally_depended_on)
    dep_output_nodes: OrderedSet[fx.Node] = OrderedSet()
    for node in externally_depended_on:
        if any(
            node is not candidate
            and _has_internal_data_path(node, candidate, region_set)
            for candidate in candidate_dep_outputs
        ):
            continue
        dep_output_nodes.add(node)

    output_nodes = OrderedSet(user_output_nodes)
    output_nodes.update(dep_output_nodes)
    if not output_nodes:
        output_nodes.add(region[-1])

    subgraph_module = _create_subgraph_for_region(
        graph, region, external_inputs, list(output_nodes)
    )
    subgraph_attr_name = get_subgraph_name(owning_mod, f"{region[0].name}_region")
    setattr(owning_mod, subgraph_attr_name, subgraph_module)

    region_deps: OrderedSet[fx.Node] = OrderedSet()
    for node in region:
        for dep in additional_deps_map.get(node, ()):
            if dep in region_set:
                continue
            dep = replacements.get(dep, dep)
            if dep._erased:
                continue
            region_deps.add(dep)

    with graph.inserting_before(region[0]):
        get_subgraph = graph.get_attr(subgraph_attr_name)
        region_result = graph.call_function(
            control_deps,
            args=(tuple(region_deps), get_subgraph, *external_inputs),
            kwargs={},
            name=f"__temp_{region[0].name}_region",
        )
        if len(output_nodes) == 1:
            only_output = next(iter(output_nodes))
            region_result.meta = only_output.meta.copy()
            region_result.meta.pop("eager_input_vals", None)
            output_replacements = {only_output: region_result}
        else:
            region_result.meta["val"] = tuple(
                node.meta.get("val") for node in output_nodes
            )
            output_replacements = {}
            for idx, output_node in enumerate(output_nodes):
                output_replacement = graph.call_function(
                    getitem,
                    args=(region_result, idx),
                    kwargs={},
                    name=f"__temp_{output_node.name}_region",
                )
                output_replacement.meta = output_node.meta.copy()
                output_replacements[output_node] = output_replacement

    for old_node, replacement in output_replacements.items():
        for user in list(old_node.users):
            if user not in region_set:
                user.replace_input_with(old_node, replacement)

    for node in reversed(region):
        graph.erase_node(node)

    if len(output_nodes) == 1:
        only_output = next(iter(output_nodes))
        region_result.name = only_output.name
    else:
        for old_node, replacement in output_replacements.items():
            replacement.name = old_node.name

    node_replacements: dict[fx.Node, fx.Node] = {}
    for node in region:
        replacement = output_replacements.get(node)
        if replacement is None:
            for output_node, output_replacement in output_replacements.items():
                if _has_internal_data_path(node, output_node, region_set):
                    replacement = output_replacement
                    break
        node_replacements[node] = (
            replacement if replacement is not None else region_result
        )
    return node_replacements


def _create_subgraph_for_region(
    graph: fx.Graph,
    region: list[fx.Node],
    external_inputs: list[fx.Node],
    output_nodes: list[fx.Node],
) -> fx.GraphModule:
    owning_module = graph.owning_module
    if owning_module is None:
        raise AssertionError("graph.owning_module must not be None")

    region_set = OrderedSet(region)
    subgraph = fx.Graph(owning_module)

    external_placeholders: dict[fx.Node, fx.Node] = {}
    for idx, orig_node in enumerate(external_inputs):
        placeholder = subgraph.placeholder(f"arg_{idx}")
        _copy_placeholder_meta(placeholder, orig_node, owning_module)
        external_placeholders[orig_node] = placeholder

    node_to_subgraph: dict[fx.Node, fx.Node] = {}

    def replace_nodes(item: Any) -> Any:
        if isinstance(item, fx.Node):
            if item in region_set:
                return node_to_subgraph[item]
            return external_placeholders[item]
        return item

    for node in region:
        flat_args_kwargs, spec = pytree.tree_flatten((node.args, node.kwargs))
        new_flat = [replace_nodes(item) for item in flat_args_kwargs]
        new_args, new_kwargs = pytree.tree_unflatten(new_flat, spec)
        if not callable(node.target):
            raise AssertionError(
                f"expected node.target to be callable, got {node.target}"
            )
        result = subgraph.call_function(
            node.target,
            tuple(new_args),
            new_kwargs,  # type: ignore[arg-type]
        )
        result.meta.update(node.meta)
        node_to_subgraph[node] = result

    outputs = [node_to_subgraph[node] for node in output_nodes]
    if len(outputs) == 1:
        out = subgraph.output(outputs[0])
        if "val" in output_nodes[0].meta:
            out.meta["val"] = output_nodes[0].meta["val"]
    else:
        out = subgraph.output(tuple(outputs))
        out.meta["val"] = tuple(node.meta.get("val") for node in output_nodes)

    return _LazyGraphModule(owning_module, subgraph)


def _has_internal_data_path(
    source: fx.Node, target: fx.Node, region_set: OrderedSet[fx.Node]
) -> bool:
    stack = [user for user in source.users if user in region_set]
    seen: OrderedSet[fx.Node] = OrderedSet()
    while stack:
        node = stack.pop()
        if node is target:
            return True
        if node in seen:
            continue
        seen.add(node)
        stack.extend(user for user in node.users if user in region_set)
    return False


def _create_subgraph_for_node(
    graph: fx.Graph, node: fx.Node, additional_deps=None
) -> fx.GraphModule:
    """
    Create a subgraph that exactly recreates a node's operation optionally passing through additional dependencies.

    The subgraph takes only the fx.Node arguments and recreates the operation
    with the exact target, args structure, and kwargs.

    Args:
        graph: The parent graph
        node: The node to wrap in a subgraph
        additional_deps: Additional dependencies to pass through the subgraph

    Returns:
        A GraphModule containing the subgraph
    """
    # Get the owning module
    owning_module = graph.owning_module
    if owning_module is None:
        raise AssertionError("graph.owning_module must not be None")

    # Create a new graph for the subgraph
    subgraph = fx.Graph(owning_module)

    # Extract unique nodes and get flattened structure + spec
    unique_nodes, flat_args_kwargs, spec = _extract_unique_nodes(node.args, node.kwargs)

    # Create placeholders for each unique node
    node_to_placeholder: dict[fx.Node, fx.Node] = {}
    for idx, orig_node in enumerate(unique_nodes):
        placeholder = subgraph.placeholder(f"arg_{idx}")
        _copy_placeholder_meta(placeholder, orig_node, owning_module)
        node_to_placeholder[orig_node] = placeholder

    # Replace fx.Node instances with their placeholders
    def replace_nodes(item: Any) -> Any:
        if isinstance(item, fx.Node):
            return node_to_placeholder[item]
        return item

    additional_deps_placeholders = []
    for idx, dep in enumerate(additional_deps or ()):
        placeholder = subgraph.placeholder(f"dep_{idx}")
        if "val" in dep.meta:
            placeholder.meta.update(dep.meta)
        additional_deps_placeholders.append(placeholder)

    new_flat = [replace_nodes(item) for item in flat_args_kwargs]
    new_args, new_kwargs = pytree.tree_unflatten(new_flat, spec)

    # Recreate the exact original operation in the subgraph
    if not callable(node.target):
        raise AssertionError(f"expected node.target to be callable, got {node.target}")
    result = subgraph.call_function(
        node.target,
        tuple(new_args),
        new_kwargs,  # type: ignore[arg-type]
    )

    # Copy metadata from the original node
    result.meta.update(node.meta)

    if additional_deps_placeholders:
        outputs = tuple([result] + additional_deps_placeholders)
        out = subgraph.output(outputs)
        out.meta["val"] = tuple(output.meta.get("val") for output in outputs)
    else:
        out = subgraph.output(result)
        if "val" in result.meta:
            out.meta["val"] = result.meta["val"]

    return _LazyGraphModule(owning_module, subgraph)
