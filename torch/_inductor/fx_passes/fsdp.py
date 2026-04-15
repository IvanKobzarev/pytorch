import json
import logging
import math
from collections.abc import Callable

import torch
import torch.fx as fx
from torch._inductor.fx_passes.bucketing import (
    bucket_all_gather_by_mb,
    bucket_all_reduce_by_mb,
    bucket_reduce_scatter_by_mb,
    BucketMode,
    is_all_gather_into_tensor as is_all_gather,
    is_all_reduce_tensor,
    is_reduce_scatter_tensor,
    merge_all_gather,
    merge_all_reduce_bucket,
    merge_reduce_scatter,
)
from torch._logging import trace_structured
from torch.utils._ordered_set import OrderedSet


logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def is_fsdp_all_gather(n: torch.fx.Node) -> bool:
    """Check if an all_gather derives from exactly one placeholder (parameter).

    Uses backward BFS to count placeholder ancestors across all input branches.
    Handles multi-input chains (e.g. cat(param, zeros) for padding) that the old
    single-input-chain walk would miss.
    """
    assert is_all_gather(n)
    visited: OrderedSet[torch.fx.Node] = OrderedSet()
    queue = list(n.all_input_nodes)
    placeholders = 0
    while queue:
        node = queue.pop()
        if node in visited:
            continue
        visited.add(node)
        if node.op == "placeholder":
            placeholders += 1
            if placeholders > 1:
                return False
        else:
            queue.extend(node.all_input_nodes)
    return placeholders == 1


def is_fsdp_all_gather_wait(wait: torch.fx.Node) -> bool:
    ag_node = wait.args[0]  # type: ignore[arg-type, union-attr]
    return is_fsdp_all_gather(ag_node)  # type: ignore[arg-type]


def is_fsdp_reduce_scatter_wait(wait: torch.fx.Node) -> bool:
    """Check if a reduce_scatter wait flows only to graph outputs through unary ops.

    Uses forward BFS to verify every path from *wait* reaches an output node and
    every intermediate node is unary (single input). Handles arbitrary chains of
    view/reshape/cast ops between RS wait and output.

    Note: this would return False for compiled multi-microbatch gradient
    accumulation where add(existing_grad, rs_result) appears in the chain.
    Current FSDP2 compile patterns don't produce this (each microbatch is
    compiled separately).
    """
    if not wait.users:
        return False
    visited: OrderedSet[torch.fx.Node] = OrderedSet()
    queue = [wait]
    while queue:
        node = queue.pop()
        if node in visited:
            continue
        visited.add(node)
        for user in node.users:
            if user.op == "output":
                continue
            if len(user.all_input_nodes) != 1:
                return False
            queue.append(user)
    return True


def bucket_fsdp_all_gather(
    gm: torch.fx.GraphModule,
    bucket_cap_mb_by_bucket_idx: Callable[[int], float] | None = None,
    mode: BucketMode = "default",
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
    mode: BucketMode = "default",
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
    rs_bucket_mode: BucketMode = mode or "default"
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


def bucket_fsdp_all_reduce(
    gm: torch.fx.GraphModule,
    bucket_cap_mb_by_bucket_idx: Callable[[int], float] | None = None,
    fsdp_groups: OrderedSet[str] | None = None,
) -> None:
    """Bucketing pass for FSDP all_reduce ops.

    Filters by group name (no structural heuristic for AR unlike AG/RS).
    """
    if bucket_cap_mb_by_bucket_idx is None:
        from torch._inductor.fx_passes.bucketing import (
            bucket_cap_mb_by_bucket_idx_default,
        )

        bucket_cap_mb_by_bucket_idx = bucket_cap_mb_by_bucket_idx_default

    def is_fsdp_all_reduce_wait(wait: torch.fx.Node) -> bool:
        ar_node = wait.args[0]
        if not is_all_reduce_tensor(ar_node):  # type: ignore[arg-type]
            return False
        if fsdp_groups is None:
            return True
        return _get_group_name(ar_node) in fsdp_groups  # type: ignore[arg-type]

    ar_buckets = bucket_all_reduce_by_mb(
        gm, bucket_cap_mb_by_bucket_idx, filter_wait_node=is_fsdp_all_reduce_wait
    )
    for bucket in ar_buckets:
        merge_all_reduce_bucket(gm.graph, bucket)


def _get_collective_kwargs(n: fx.Node) -> dict[str, object]:
    """Normalize a collective node's args into keyword args."""
    from torch.fx.operator_schemas import normalize_function

    opt = normalize_function(
        n.target,  # type: ignore[arg-type]
        args=n.args,
        kwargs=n.kwargs,
        normalize_to_only_use_kwargs=True,
    )
    assert opt is not None
    _, kwargs = opt
    return kwargs


def _get_group_name(n: fx.Node) -> str:
    return _get_collective_kwargs(n)["group_name"]  # type: ignore[return-value]


def _get_group_size_from_node(n: fx.Node) -> int:
    return _get_collective_kwargs(n)["group_size"]  # type: ignore[return-value]


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
    with a calibration multiplier to account for model inaccuracy.

    If bucket_cap_mb_override is set, returns that directly.
    """
    if bucket_cap_mb_override is not None:
        return bucket_cap_mb_override

    import torch._inductor.config as inductor_config
    from torch._inductor.comm_analysis import compute_min_saturation_bytes, NCCL_COLL

    dist_opts = inductor_config.aten_distributed_optimizations
    cal_mult = (
        dist_opts.pre_bucketing_fsdp_collectives_saturation_calibration_multiplier
    )
    floor_mb = dist_opts.pre_bucketing_fsdp_collectives_min_bucket_cap_mb
    ceil_mb = dist_opts.pre_bucketing_fsdp_collectives_max_bucket_cap_mb

    min_bytes = compute_min_saturation_bytes(
        group_size, NCCL_COLL.ALL_GATHER, target_efficiency=0.95
    )
    cap_mb = cal_mult * min_bytes / (1024 * 1024)
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
        elif is_all_reduce_tensor(n) and _get_group_name(n) in fsdp_groups:
            val = n.all_input_nodes[0].meta["val"]
            size_mb = val.numel() * val.element_size() / (1024 * 1024)
            sizes.append({"type": "AR", "size_mb": round(size_mb, 3), "name": n.name})
    return sizes


def pre_bucket_fsdp_collectives(
    gm: torch.fx.GraphModule,
    mode: BucketMode | None = None,
    bucket_cap_mb: float | None = None,
) -> None:
    """Pre-bucket FSDP collectives before overlap scheduling.

    Identifies FSDP process groups via all_gather structural heuristics,
    then merges all_gather, reduce_scatter, and all_reduce ops on those
    groups into bandwidth-saturating buckets.
    """
    import torch._inductor.config as inductor_config

    dist_opts = inductor_config.aten_distributed_optimizations
    verbose = dist_opts.pre_bucketing_fsdp_collectives_verbose

    fsdp_groups = identify_fsdp_group_names(gm)
    if not fsdp_groups:
        return

    def _is_fsdp_coll(n, is_coll_fn):
        return is_coll_fn(n) and _get_group_name(n) in fsdp_groups

    # Count FSDP collectives before bucketing
    ag_count = sum(1 for n in gm.graph.nodes if _is_fsdp_coll(n, is_all_gather))
    rs_count = sum(
        1 for n in gm.graph.nodes if _is_fsdp_coll(n, is_reduce_scatter_tensor)
    )
    ar_count = sum(1 for n in gm.graph.nodes if _is_fsdp_coll(n, is_all_reduce_tensor))

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
        # Shouldn't be reachable: fsdp_groups is populated from AG nodes, so
        # the loop above should always find at least one matching AG.
        logger.warning("pre_bucket_fsdp: no FSDP all_gather found for group_size")
        cap_mb = bucket_cap_mb if bucket_cap_mb is not None else 500.0

    def bucket_cap_fn(_idx: int) -> float:
        return cap_mb

    resolved_mode: BucketMode = mode or "default"
    bucket_fsdp_all_gather(
        gm, bucket_cap_mb_by_bucket_idx=bucket_cap_fn, mode=resolved_mode
    )
    bucket_fsdp_reduce_scatter(
        gm, bucket_cap_mb_by_bucket_idx=bucket_cap_fn, mode=resolved_mode
    )
    bucket_fsdp_all_reduce(
        gm, bucket_cap_mb_by_bucket_idx=bucket_cap_fn, fsdp_groups=fsdp_groups
    )

    ag_count_after = sum(1 for n in gm.graph.nodes if _is_fsdp_coll(n, is_all_gather))
    rs_count_after = sum(
        1 for n in gm.graph.nodes if _is_fsdp_coll(n, is_reduce_scatter_tensor)
    )
    ar_count_after = sum(
        1 for n in gm.graph.nodes if _is_fsdp_coll(n, is_all_reduce_tensor)
    )

    nNodes = math.ceil(group_size / 8) if group_size is not None else 1

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
        "bucket_cap_mb=%.1f, all_gather %d->%d, reduce_scatter %d->%d, "
        "all_reduce %d->%d",
        list(fsdp_groups),
        group_size,
        nNodes,
        cap_mb,
        ag_count,
        ag_count_after,
        rs_count,
        rs_count_after,
        ar_count,
        ar_count_after,
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
            f"reduce_scatter {rs_count}->{rs_count_after}, "
            f"all_reduce {ar_count}->{ar_count_after}"
        ),
    )
