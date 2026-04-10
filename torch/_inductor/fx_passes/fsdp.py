import json
import logging
import math
from collections.abc import Callable

import torch
import torch.fx as fx
from torch._inductor.fx_passes.bucketing import (
    bucket_all_gather_by_mb,
    bucket_reduce_scatter_by_mb,
    BucketMode,
    is_all_gather_into_tensor as is_all_gather,
    is_reduce_scatter_tensor,
    merge_all_gather,
    merge_reduce_scatter,
)
from torch._logging import trace_structured
from torch.utils._ordered_set import OrderedSet


logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def is_graph_input(node: torch.fx.Node) -> bool:
    return node.op == "placeholder"


def is_fsdp_all_gather(n):
    assert is_all_gather(n)
    while len(n.all_input_nodes) == 1:
        n = n.all_input_nodes[0]
        if n.op == "placeholder":
            return True
    return False


def is_fsdp_all_gather_wait(wait: torch.fx.Node) -> bool:
    # Assume all_gather_into_tensor input is either graph input
    # or dtype conversion of graph input
    ag_node = wait.args[0]  # type: ignore[arg-type, union-attr]
    return is_fsdp_all_gather(ag_node)


def is_graph_output(node: torch.fx.Node) -> bool:
    return all(user.op == "output" for user in node.users)


def is_fsdp_reduce_scatter_wait(wait: torch.fx.Node) -> bool:
    if is_graph_output(wait):
        return True

    if len(wait.users) == 1:
        user = next(iter(wait.users))
        assert user is not None
        return (
            is_graph_output(user)
            and user.op == "call_function"
            and user.target is torch.ops.prims.convert_element_type.default
        )

    return False


def bucket_fsdp_all_gather(
    gm: torch.fx.GraphModule,
    bucket_cap_mb_by_bucket_idx: Callable[[int], float] | None = None,
    mode: BucketMode | None = None,
) -> None:
    """
    Bucketing pass for SimpleFSDP all_gather ops.

    Attributes:
        gm (torch.fx.GraphModule): Graph module of the graph.
        bucket_cap_mb_by_bucket_idx (Callable[[int], float] | None): callback function that
            takes in bucket id and returns size of a bucket in megabytes.
    """
    if bucket_cap_mb_by_bucket_idx is None:
        from torch._inductor.fx_passes.bucketing import (
            bucket_cap_mb_by_bucket_idx_default,
        )

        bucket_cap_mb_by_bucket_idx = bucket_cap_mb_by_bucket_idx_default
    assert bucket_cap_mb_by_bucket_idx is not None
    ag_buckets = bucket_all_gather_by_mb(
        gm,
        bucket_cap_mb_by_bucket_idx,
        filter_wait_node=is_fsdp_all_gather_wait,
    )
    if len(ag_buckets) == 0:
        return
    merge_all_gather(gm, ag_buckets, mode)


def bucket_fsdp_reduce_scatter(
    gm: torch.fx.GraphModule,
    bucket_cap_mb_by_bucket_idx: Callable[[int], float] | None = None,
    mode: BucketMode | None = None,
) -> None:
    """
    Bucketing pass for SimpleFSDP reduce_scatter ops.

    Attributes:
        gm (torch.fx.GraphModule): Graph module of the graph.
        bucket_cap_mb_by_bucket_idx (Callable[[int], float] | None): callback function that
            takes in bucket idx and returns size of a bucket in megabytes. By default
            torch._inductor.fx_passes.bucketing.bucket_cap_mb_by_bucket_idx_default is used.

    """
    if bucket_cap_mb_by_bucket_idx is None:
        from torch._inductor.fx_passes.bucketing import (
            bucket_cap_mb_by_bucket_idx_default,
        )

        bucket_cap_mb_by_bucket_idx = bucket_cap_mb_by_bucket_idx_default
    # reduce_scatter bucketing does not support multidtype mode;
    # resolve None to the default and strip multidtype if present.
    from torch._inductor.fx_passes.bucketing import _default_bucket_mode

    rs_bucket_mode: BucketMode = mode or _default_bucket_mode()
    if "multidtype" in rs_bucket_mode:
        rs_bucket_mode = rs_bucket_mode.replace("_multidtype", "")  # type: ignore[assignment]
    rs_buckets = bucket_reduce_scatter_by_mb(
        gm,
        bucket_cap_mb_by_bucket_idx,
        filter_wait_node=is_fsdp_reduce_scatter_wait,
        mode=rs_bucket_mode,
    )
    if len(rs_buckets) == 0:
        return
    merge_reduce_scatter(gm, rs_buckets, mode)


def _get_group_name(n: fx.Node) -> str:
    """Extract group_name from a collective node's args."""
    from torch.fx.operator_schemas import normalize_function

    opt = normalize_function(
        n.target,  # type: ignore[arg-type]
        args=n.args,
        kwargs=n.kwargs,
        normalize_to_only_use_kwargs=True,
    )
    assert opt is not None
    _, kwargs = opt
    return kwargs["group_name"]


def _get_group_size_from_node(n: fx.Node) -> int:
    """Extract group_size from a collective node's args."""
    from torch.fx.operator_schemas import normalize_function

    opt = normalize_function(
        n.target,  # type: ignore[arg-type]
        args=n.args,
        kwargs=n.kwargs,
        normalize_to_only_use_kwargs=True,
    )
    assert opt is not None
    _, kwargs = opt
    return kwargs["group_size"]


def identify_fsdp_group_names(gm: torch.fx.GraphModule) -> OrderedSet[str]:
    """Identify process group names used by FSDP collectives.

    Uses is_fsdp_all_gather heuristic on all_gather nodes to find FSDP groups,
    then returns those group names. All collectives on these groups (AG, RS, AR)
    are considered FSDP via group-name transitivity.
    """
    fsdp_groups: OrderedSet[str] = OrderedSet()
    for n in gm.graph.nodes:
        if is_all_gather(n) and is_fsdp_all_gather(n):
            fsdp_groups.add(_get_group_name(n))
    return fsdp_groups


def compute_pre_bucket_cap_mb(
    group_size: int,
    bucket_cap_mb_override: float | None = None,
) -> float:
    """Compute the bucket cap for pre-bucketing based on bandwidth saturation.

    Returns a conservative bucket size in MB that guarantees saturation of
    the process group's network bandwidth. Uses the NCCL analytical model
    with a safety multiplier to account for model inaccuracy.

    If bucket_cap_mb_override is set, returns that directly.
    """
    if bucket_cap_mb_override is not None:
        return bucket_cap_mb_override

    import torch._inductor.config as inductor_config
    from torch._inductor.comm_analysis import compute_min_saturation_bytes, NCCL_COLL

    dist_opts = inductor_config.aten_distributed_optimizations
    target_eff = dist_opts.pre_bucketing_fsdp_collectives_target_efficiency
    safety_mult = dist_opts.pre_bucketing_fsdp_collectives_safety_multiplier
    floor_mb = dist_opts.pre_bucketing_fsdp_collectives_min_bucket_cap_mb
    ceil_mb = dist_opts.pre_bucketing_fsdp_collectives_max_bucket_cap_mb

    min_bytes = compute_min_saturation_bytes(
        group_size, NCCL_COLL.ALL_GATHER, target_efficiency=target_eff
    )
    cap_mb = safety_mult * min_bytes / (1024 * 1024)
    cap_mb = max(floor_mb, min(ceil_mb, cap_mb))

    return cap_mb


def _collect_collective_sizes(
    gm: torch.fx.GraphModule, fsdp_groups: OrderedSet[str]
) -> list[dict[str, object]]:
    """Collect per-collective sizes for FSDP collectives in graph order."""
    sizes: list[dict[str, object]] = []
    for n in gm.graph.nodes:
        if is_all_gather(n) and _get_group_name(n) in fsdp_groups:
            val = n.meta["val"]
            size_mb = val.numel() * val.element_size() / (1024 * 1024)
            sizes.append({"type": "AG", "size_mb": round(size_mb, 3), "name": n.name})
        elif is_reduce_scatter_tensor(n) and _get_group_name(n) in fsdp_groups:
            inp = n.all_input_nodes[0].meta["val"]
            size_mb = inp.numel() * inp.element_size() / (1024 * 1024)
            sizes.append({"type": "RS", "size_mb": round(size_mb, 3), "name": n.name})
    return sizes


def pre_bucket_fsdp_collectives(
    gm: torch.fx.GraphModule,
    mode: BucketMode | None = None,
    bucket_cap_mb: float | None = None,
) -> None:
    """Pre-bucket FSDP collectives before overlap scheduling.

    Identifies FSDP process groups via all_gather structural heuristics,
    then merges all_gather and reduce_scatter ops on those groups into
    bandwidth-saturating buckets.
    """
    import torch._inductor.config as inductor_config

    dist_opts = inductor_config.aten_distributed_optimizations
    verbose = dist_opts.pre_bucketing_fsdp_collectives_verbose

    fsdp_groups = identify_fsdp_group_names(gm)
    if not fsdp_groups:
        return

    # Count collectives before bucketing
    ag_count = sum(1 for n in gm.graph.nodes if is_all_gather(n))
    rs_count = sum(1 for n in gm.graph.nodes if is_reduce_scatter_tensor(n))

    # Verbose: log per-collective sizes before bucketing
    if verbose:
        coll_sizes = _collect_collective_sizes(gm, fsdp_groups)
        logger.info(
            "pre_bucket_fsdp: %d collectives before bucketing, sizes (MB): %s",
            len(coll_sizes),
            ", ".join(f"{s['type']}({s['size_mb']})" for s in coll_sizes[:50]),
        )
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "pre_bucketing_collective_sizes",
                "encoding": "json",
            },
            payload_fn=lambda: json.dumps(coll_sizes),
        )

    # Determine bucket cap from first FSDP collective's group size
    group_size = None
    for n in gm.graph.nodes:
        if is_all_gather(n) and _get_group_name(n) in fsdp_groups:
            group_size = _get_group_size_from_node(n)
            break

    if group_size is not None:
        cap_mb = compute_pre_bucket_cap_mb(group_size, bucket_cap_mb)
    else:
        cap_mb = bucket_cap_mb if bucket_cap_mb is not None else 500.0

    bucket_cap_fn: Callable[[int], float] = lambda _idx: cap_mb

    bucket_fsdp_all_gather(gm, bucket_cap_mb_by_bucket_idx=bucket_cap_fn, mode=mode)
    bucket_fsdp_reduce_scatter(gm, bucket_cap_mb_by_bucket_idx=bucket_cap_fn, mode=mode)

    ag_count_after = sum(1 for n in gm.graph.nodes if is_all_gather(n))
    rs_count_after = sum(1 for n in gm.graph.nodes if is_reduce_scatter_tensor(n))

    nNodes = math.ceil((group_size or 1) / 8)

    # Verbose: log per-collective sizes after bucketing
    if verbose:
        coll_sizes_after = _collect_collective_sizes(gm, fsdp_groups)
        logger.info(
            "pre_bucket_fsdp: %d collectives after bucketing, sizes (MB): %s",
            len(coll_sizes_after),
            ", ".join(f"{s['type']}({s['size_mb']})" for s in coll_sizes_after[:50]),
        )
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "pre_bucketing_collective_sizes_after",
                "encoding": "json",
            },
            payload_fn=lambda: json.dumps(coll_sizes_after),
        )

    logger.info(
        "pre_bucket_fsdp_collectives: fsdp_groups=%s, group_size=%s, nNodes=%d, "
        "bucket_cap_mb=%.1f, all_gather %d->%d, reduce_scatter %d->%d",
        list(fsdp_groups),
        group_size,
        nNodes,
        cap_mb,
        ag_count,
        ag_count_after,
        rs_count,
        rs_count_after,
    )

    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "pre_bucketing_fsdp_collectives",
            "encoding": "string",
        },
        payload_fn=lambda: (
            f"fsdp_groups={list(fsdp_groups)}, group_size={group_size}, "
            f"nNodes={nNodes}, bucket_cap_mb={cap_mb:.1f}, "
            f"all_gather {ag_count}->{ag_count_after}, "
            f"reduce_scatter {rs_count}->{rs_count_after}"
        ),
    )
