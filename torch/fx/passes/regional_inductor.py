import contextlib
import functools
import logging
import os
from collections.abc import Callable, Iterator, Mapping
from typing import Any, ParamSpec, TypeVar


_P = ParamSpec("_P")
_R = TypeVar("_R")

import torch
from torch.fx._compatibility import compatibility


logger = logging.getLogger(__name__)

__all__ = ["regional_inductor"]


# standalone_inductor returns a callable class object - this does not sit well
# with Fx graph node op call_function which expects a function. So this is just
# a wrapper function to make Fx graph codegen happy.
def _dummy_wrapper(fn: Callable[_P, _R]) -> Callable[_P, _R]:
    @functools.wraps(fn)
    def inner(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        return fn(*args, **kwargs)

    return inner


@contextlib.contextmanager
def _disable_remat_for_regional_subcompile() -> Iterator[None]:
    # In torch.compile, regional_inductor subcompiles run after the enclosing
    # non-strict full graph has already been partitioned, so any graph-SAC
    # remat pass has already run before we reach this nested compile.
    # Rerunning remat here can see stage-2-reordered backward nodes that
    # violate remat's contiguous-backward-region assumption.
    with torch._functorch.config.patch(remat_using_tags_for_fwd_loss_bwd_graph=False):
        yield


def _compile_submod(gm: torch.fx.GraphModule, prefix: str) -> torch.fx.GraphModule:
    from torch._inductor.standalone_compile import AOTCompiledArtifact

    for node in gm.graph.nodes:
        if node.op == "call_module" and node.target.startswith(prefix):
            submod = getattr(gm, node.target)
            node_order = {graph_node: i for i, graph_node in enumerate(gm.graph.nodes)}
            fake_inputs = []
            donated_inputs = []
            for inp_node in node.all_input_nodes:
                if hasattr(inp_node, "meta") and "val" in inp_node.meta:
                    fake_inputs.append(
                        _standalone_region_fake_input(inp_node.meta["val"])
                    )
                    donated_inputs.append(
                        _can_donate_region_input(gm, submod, node_order, node, inp_node)
                    )
                else:
                    raise RuntimeError(
                        f"Partition is bad because non fake tensor value is seen {inp_node}"
                    )

            for placeholder, fake_input, donated_input in zip(
                (n for n in submod.graph.nodes if n.op == "placeholder"),
                fake_inputs,
                donated_inputs,
            ):
                placeholder.meta["val"] = fake_input
                if donated_input:
                    placeholder.meta["inductor_donated_input"] = True

            # Get inductor configs from annotation
            # TODO we should change partition when there are multiple differently
            # annotated regions.
            inductor_options: dict[str, Any] = {}
            for sub_node in submod.graph.nodes:
                if hasattr(sub_node, "meta") and sub_node.meta.get("custom", None):
                    custom = sub_node.meta["custom"]
                    if isinstance(custom, dict) and "compile_with_inductor" in custom:
                        compile_value = custom["compile_with_inductor"]
                        if (
                            isinstance(compile_value, dict)
                            and "inductor_configs" in compile_value
                        ):
                            inductor_options = compile_value["inductor_configs"]
                            break

            # Log the options being used
            logger.info(
                "Compiling submodule %s with inductor options: %s",
                node.target,
                inductor_options,
            )

            # Apply config patches before compilation
            import torch._inductor.config as inductor_config

            # Validate that all config keys exist
            for key in inductor_options:
                if not hasattr(inductor_config, key):
                    raise ValueError(
                        f"Invalid inductor config key '{key}' in regional_inductor annotation. "
                        f"Available config keys can be found in torch._inductor.config"
                    )

            donated_input_idxs = tuple(
                i for i, donated_input in enumerate(donated_inputs) if donated_input
            )
            if donated_input_idxs:
                inductor_options = {
                    **inductor_options,
                    "regional_inductor_donated_input_idxs": donated_input_idxs,
                }

            with (
                inductor_config.patch(inductor_options),
                _disable_remat_for_regional_subcompile(),
                torch.fx.traceback._set_regional_inductor_subgraph_name(node.target),
            ):
                compiled_fn = torch._inductor.standalone_compile(
                    submod,
                    fake_inputs,
                    dynamic_shapes="from_tracing_context",
                    aot=True,
                    donate_graph_module=True,
                )
            if not isinstance(compiled_fn, AOTCompiledArtifact):
                raise AssertionError(
                    f"Expected AOTCompiledArtifact, got {type(compiled_fn)}"
                )
            # _dummy_wrapper is to make call_function happy
            compiled_submod = _dummy_wrapper(compiled_fn)
            with gm.graph.inserting_after(node):
                new_node = gm.graph.call_function(
                    compiled_submod, args=node.args, kwargs=node.kwargs
                )
                new_node.meta = node.meta
                node.replace_all_uses_with(new_node)
                gm.graph.erase_node(node)
                del gm._modules[node.target]

    gm.recompile()
    return gm


def _standalone_region_fake_input(value: Any) -> Any:
    if isinstance(value, torch.Tensor) and value._is_view():
        # Donation legality is checked on the parent graph before cloning.
        return value.clone()
    return value


def _same_donation_layout(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and a.stride() == b.stride()


def _tensor_nbytes(value: torch.Tensor) -> int | None:
    try:
        return int(value.numel()) * value.dtype.itemsize
    except Exception:
        return None


def _storage_base(value: torch.Tensor) -> torch.Tensor | None:
    base = value
    while isinstance(next_base := getattr(base, "_base", None), torch.Tensor):
        base = next_base
    return base


def _contiguous_storage_interval(value: torch.Tensor) -> tuple[int, int] | None:
    if not value.is_contiguous():
        return None
    try:
        start = int(value.storage_offset())
        end = start + int(value.numel())
    except Exception:
        return None
    return (start, end)


def _storage_intervals_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(a[0], b[0]) < min(a[1], b[1])


def _has_future_overlapping_alias_use(
    gm: torch.fx.GraphModule,
    node_order: dict[torch.fx.Node, int],
    region_node: torch.fx.Node,
    node: torch.fx.Node,
    value: torch.Tensor,
) -> bool:
    base = _storage_base(value)
    interval = _contiguous_storage_interval(value)
    region_pos = node_order[region_node]
    if base is None or interval is None:
        return True

    for alias_node in gm.graph.nodes:
        alias_value = alias_node.meta.get("val")
        if not isinstance(alias_value, torch.Tensor):
            continue
        if _storage_base(alias_value) is not base:
            continue

        alias_interval = _contiguous_storage_interval(alias_value)
        if alias_interval is None or _storage_intervals_overlap(
            interval, alias_interval
        ):
            for user in alias_node.users:
                if alias_node is node and user is region_node:
                    continue
                if node_order.get(user, -1) > region_pos:
                    return True
    return False


def _can_overwrite_region_input_storage(
    gm: torch.fx.GraphModule,
    node_order: dict[torch.fx.Node, int],
    region_node: torch.fx.Node,
    node: torch.fx.Node,
    value: torch.Tensor,
) -> bool:
    return _contiguous_storage_interval(
        value
    ) is not None and not _has_future_overlapping_alias_use(
        gm, node_order, region_node, node, value
    )


def _has_matching_region_output(
    submod: torch.fx.GraphModule, value: torch.Tensor
) -> bool:
    output = next(iter(reversed(submod.graph.find_nodes(op="output"))), None)
    if output is None:
        return False
    for output_node in output.all_input_nodes:
        output_value = output_node.meta.get("val")
        if isinstance(output_value, torch.Tensor) and _same_donation_layout(
            value, output_value
        ):
            return True
    return False


def _can_donate_region_input(
    gm: torch.fx.GraphModule,
    submod: torch.fx.GraphModule,
    node_order: dict[torch.fx.Node, int],
    region_node: torch.fx.Node,
    node: torch.fx.Node,
) -> bool:
    from torch._inductor import config as inductor_config

    if not inductor_config.regional_inductor_donate_intermediate_view_inputs:
        return False

    value = node.meta.get("val")
    nbytes = _tensor_nbytes(value) if isinstance(value, torch.Tensor) else None
    min_bytes = inductor_config.regional_inductor_donate_intermediate_inputs_min_bytes
    if not (
        isinstance(value, torch.Tensor)
        and value.is_floating_point()
        and nbytes is not None
        and nbytes >= min_bytes
    ):
        return False
    if not _has_matching_region_output(submod, value):
        return False
    if not _can_overwrite_region_input_storage(
        gm, node_order, region_node, node, value
    ):
        return False

    return (
        node.op not in ("placeholder", "get_attr")
        and set(node.users) == {region_node}
        and all(
            input_node.op not in ("placeholder", "get_attr")
            for input_node in node.all_input_nodes
        )
    )


def _maybe_dump_regional_graph(gm: torch.fx.GraphModule, phase: str) -> None:
    path = os.environ.get("TORCHINDUCTOR_DEBUG_REGIONAL_GRAPH")
    if not path:
        return
    with open(path, "a") as f:
        f.write(f"\n## {phase}: {gm._get_name()}\n")
        for node in gm.graph.nodes:
            val = node.meta.get("val")
            shape = getattr(val, "shape", None)
            dtype = getattr(val, "dtype", None)
            is_view = val._is_view() if isinstance(val, torch.Tensor) else None
            users = ",".join(user.name for user in node.users)
            custom = node.meta.get("custom")
            args = ",".join(arg.name for arg in node.all_input_nodes)
            donated = node.meta.get("inductor_donated_input", False)
            f.write(
                f"{node.name}: op={node.op} target={node.target} shape={shape} dtype={dtype} view={is_view} donate={donated} args=[{args}] users=[{users}] custom={custom}\n"
            )
        f.write("\n")


def _needs_inductor_compile(node: torch.fx.Node) -> bool:
    return bool(
        node.op not in ("placeholder", "output")
        and hasattr(node, "meta")
        and node.meta.get("custom", None)
        and "compile_with_inductor" in node.meta["custom"]
    )


def _has_explicit_inductor_region(node: torch.fx.Node) -> bool:
    if not _needs_inductor_compile(node):
        return False
    compile_value = node.meta["custom"]["compile_with_inductor"]
    return isinstance(compile_value, dict) and "inductor_region" in compile_value


def _same_meta_shape(node: torch.fx.Node, other: torch.fx.Node) -> bool:
    node_val = node.meta.get("val")
    other_val = other.meta.get("val")
    return (
        getattr(node_val, "shape", None) is not None
        and getattr(node_val, "shape", None) == getattr(other_val, "shape", None)
        and getattr(node_val, "dtype", None) == getattr(other_val, "dtype", None)
    )


def _find_view_backward_tail_nodes(gm: torch.fx.GraphModule) -> set[torch.fx.Node]:
    tail_nodes: set[torch.fx.Node] = set()
    worklist: list[torch.fx.Node] = []
    for node in gm.graph.nodes:
        if (
            _has_explicit_inductor_region(node)
            and node.op == "call_function"
            and node.target is torch.ops.aten.slice_backward.default
        ):
            tail_nodes.add(node)
            worklist.append(node)

    while worklist:
        node = worklist.pop()
        for user in node.users:
            if not _has_explicit_inductor_region(user) or user in tail_nodes:
                continue
            if (
                user.op == "call_function"
                and user.target is torch.ops.aten.detach.default
                and _same_meta_shape(user, node)
            ):
                tail_nodes.add(user)
                worklist.append(user)
                continue
            if (
                user.op == "call_function"
                and user.target
                in (torch.ops.aten.add.Tensor, torch.ops.aten.add_.Tensor)
                and _same_meta_shape(user, node)
            ):
                tail_nodes.add(user)
                worklist.append(user)

    return tail_nodes


class _RegionScooper:
    """
    Scoops out the inductor marked regions. It does NOT compile them.
    """

    @staticmethod
    def scoop_regions(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
        from torch._inductor import config as inductor_config
        from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
        from torch.fx.passes.operator_support import create_op_support
        from torch.fx.passes.utils.fuser_utils import fuse_by_partitions

        # Group tagged nodes by region ID.  The region ID comes from the
        # optional "inductor_region" key inside the compile_with_inductor
        # annotation. When absent, all tagged nodes share a single default region
        _DEFAULT_REGION = object()
        _VIEW_BACKWARD_TAIL_REGION = object()
        view_backward_tail_nodes = (
            _find_view_backward_tail_nodes(gm)
            if inductor_config.regional_inductor_sink_view_backward
            else set()
        )
        regions: dict[object, set[torch.fx.Node]] = {}
        for node in gm.graph.nodes:
            if _needs_inductor_compile(node):
                if node in view_backward_tail_nodes:
                    rid = _VIEW_BACKWARD_TAIL_REGION
                else:
                    compile_value = node.meta["custom"]["compile_with_inductor"]
                    if (
                        isinstance(compile_value, dict)
                        and "inductor_region" in compile_value
                    ):
                        rid = compile_value["inductor_region"]
                    else:
                        rid = _DEFAULT_REGION
                regions.setdefault(rid, set()).add(node)

        if not regions:
            logger.info("No inductor marked nodes found")
            return gm

        # Run CapabilityBasedPartitioner per region to get cycle-safe partitions
        # without merging across region boundaries.
        def _is_in_region(
            region_nodes: set[torch.fx.Node],
        ) -> Callable[[Mapping[str, torch.nn.Module], torch.fx.Node], bool]:
            def is_node_supported(
                _submodules: Mapping[str, torch.nn.Module], node: torch.fx.Node
            ) -> bool:
                return node in region_nodes

            return is_node_supported

        all_partitions: list[dict[torch.fx.Node, int | None]] = []
        for region_nodes in regions.values():
            support = create_op_support(_is_in_region(region_nodes))
            partitioner = CapabilityBasedPartitioner(
                gm, support, allows_single_node_partition=True
            )
            for partition in partitioner.propose_partitions():
                all_partitions.append(partition.nodes)

        return fuse_by_partitions(
            gm,
            all_partitions,
            prefix="__marked_inductor_submod",
            always_return_tuple=True,
        )

    @staticmethod
    def recursively_scoop_regions(
        gm: torch.fx.GraphModule, _processed: set[int] | None = None
    ) -> torch.fx.GraphModule:
        if _processed is None:
            _processed = set()
        for node in gm.graph.find_nodes(op="get_attr"):
            if _needs_inductor_compile(node):
                # If the get_attr itself is marked for compile, the outer graph will
                # take care of it. If we don't do that, we end up with nested
                # regional inductor compiles that do not work well.
                continue
            submod = getattr(gm, node.target)
            # Track by id: multiple get_attr nodes may reference the same GraphModule
            if (
                isinstance(submod, torch.fx.GraphModule)
                and id(submod) not in _processed
            ):
                _processed.add(id(submod))
                _RegionScooper.recursively_scoop_regions(submod, _processed)

        return _RegionScooper.scoop_regions(gm)

    def __call__(self, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
        with torch.fx.traceback.preserve_node_meta(enable=False):
            return _RegionScooper.recursively_scoop_regions(gm)


class _RegionCompiler:
    """
    Compiles the scooped out regions.
    """

    @staticmethod
    def compile_region(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
        from torch.fx.graph import _BoxedCodeGen

        gm = _compile_submod(gm, "__marked_inductor_submod")
        gm.graph.set_codegen(_BoxedCodeGen())
        gm.recompile()
        return gm

    @staticmethod
    def recursively_compile_regions(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
        # Find if the graph module has a scooped out region
        found_region = False
        for node in gm.graph.find_nodes(op="call_module"):
            submod = getattr(gm, node.target)
            if isinstance(submod, torch.fx.GraphModule):
                if node.target.startswith("__marked_inductor_submod"):
                    found_region = True

        # Recurse through the subgraphs
        for node in gm.graph.find_nodes(op="get_attr"):
            submod = getattr(gm, node.target)
            if isinstance(submod, torch.fx.GraphModule):
                _RegionCompiler.recursively_compile_regions(submod)

        if found_region:
            return _RegionCompiler.compile_region(gm)
        return gm

    def __call__(self, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
        with torch.fx.traceback.preserve_node_meta(enable=False):
            return _RegionCompiler.recursively_compile_regions(gm)


def _create_inductor_marked_regions(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    with torch.fx.traceback.preserve_node_meta(enable=False):
        _maybe_dump_regional_graph(gm, "before_scoop")
        gm = _RegionScooper()(gm)
        _maybe_dump_regional_graph(gm, "after_scoop")
        return gm


def _compile_inductor_marked_regions(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    with torch.fx.traceback.preserve_node_meta(enable=False):
        gm = _RegionCompiler()(gm)
        _maybe_dump_regional_graph(gm, "after_compile")
        return gm


@compatibility(is_backward_compatible=False)
def regional_inductor(
    gm: torch.fx.GraphModule, *example_args: object
) -> torch.fx.GraphModule:
    """
    Scoops out inductor marked regions and compiles them with inductor.

    Inductor options should be provided via the annotation API::

        with fx_traceback.annotate(
            {
                "compile_with_inductor": {
                    "inductor_configs": {
                        "max_autotune": True,
                        "triton.cudagraphs": False,
                    }
                }
            }
        ):
            ...
    """

    # fuser utils create new nodes using create_proxy which retains the seq_nr
    # metadata and cause issues

    with torch.fx.traceback.preserve_node_meta(enable=False):
        gm = _create_inductor_marked_regions(gm)
        gm = _compile_inductor_marked_regions(gm)
        if torch._functorch.config.force_autograd_cache:
            from torch._inductor.output_code import RegionalOutputCode

            return RegionalOutputCode(gm)  # type: ignore[return-value]
        return gm
