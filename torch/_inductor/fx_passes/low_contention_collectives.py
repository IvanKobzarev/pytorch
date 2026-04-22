from __future__ import annotations

import logging
import operator
import warnings

import torch
from torch.utils._ordered_set import OrderedSet


log = logging.getLogger(__name__)


def _get_collective_info(node):
    """Return (is_ag, group_name) if node is an AG/RS collective, else None."""
    from torch._inductor.fx_passes.bucketing import (
        is_all_gather_into_tensor,
        is_reduce_scatter_tensor,
    )
    from torch._inductor.fx_passes.overlap_scheduling import get_group_name

    if is_all_gather_into_tensor(node):
        return True, get_group_name(node)
    if is_reduce_scatter_tensor(node):
        return False, get_group_name(node)
    return None


def replace_collectives_with_low_contention(
    graph: torch.fx.Graph,
) -> None:
    """Replace FSDP collectives with copy-engine symm_mem variants."""
    symm_mem = torch.ops.symm_mem

    collectives = []
    groups: OrderedSet[str] = OrderedSet()
    for node in list(graph.nodes):
        info = _get_collective_info(node)
        if info is None:
            continue
        is_ag, group_name = info
        collectives.append((node, is_ag, group_name))
        groups.add(group_name)

    if not collectives:
        return

    # Some group names can't be resolved at compile time — skip them.
    valid_groups: OrderedSet[str] = OrderedSet()
    for group_name in groups:
        if _enable_symm_mem(group_name):
            valid_groups.add(group_name)

    # Filter to collectives whose groups we can actually resolve
    collectives = [
        (node, is_ag, gn) for node, is_ag, gn in collectives if gn in valid_groups
    ]
    if not collectives:
        return

    from torch._inductor import config

    min_bytes = config.aten_distributed_optimizations.low_contention_min_bytes_per_rank
    use_ag_v2 = config.aten_distributed_optimizations.low_contention_all_gather_v2
    use_nccl_ce = config.aten_distributed_optimizations.low_contention_use_nccl_ce
    allow_nvlink_contention = config.aten_distributed_optimizations.low_contention_allow_nvlink_contention
    max_coalesce = config.aten_distributed_optimizations.low_contention_ce_max_coalesce_size

    node_positions = {n: i for i, n in enumerate(graph.nodes)}

    # First pass: filter collectives, collect candidates for replacement
    candidates = []
    skipped_small = 0
    skipped_no_overlap = 0
    skipped_nvlink_contention = 0
    for node, is_ag, group_name in collectives:
        coll_type = "AG" if is_ag else "RS"

        if min_bytes > 0:
            per_rank_bytes = _get_per_rank_bytes(node, is_ag)
            if per_rank_bytes is not None and per_rank_bytes < min_bytes:
                skipped_small += 1
                log.debug(
                    "LC skip %s %s: size %d < min_bytes %d",
                    coll_type,
                    node.name,
                    per_rank_bytes,
                    min_bytes,
                )
                continue

        if not _has_compute_bound_overlap(node, graph, node_positions):
            skipped_no_overlap += 1
            log.debug("LC skip %s %s: no compute-bound overlap", coll_type, node.name)
            continue

        if not use_nccl_ce and not allow_nvlink_contention and _has_other_group_collectives(
            node, group_name, graph, node_positions
        ):
            skipped_nvlink_contention += 1
            log.debug(
                "LC skip %s %s: overlaps other-group collectives (NVLink contention)",
                coll_type,
                node.name,
            )
            continue

        candidates.append((node, is_ag, group_name))

    # Compute per-shape pool size from the graph: count the max number of
    # same-shape AGs whose output buffers are simultaneously live.
    # Buffer liveness extends from AG start to the last consumer of the
    # output (following view chains that share storage), not just to
    # wait_tensor — this is the minimum buffer count to avoid races.
    pool_size_cfg = config.aten_distributed_optimizations.low_contention_ce_buffer_pool_size
    shape_pool_sizes: dict[tuple, int] = {}

    if use_nccl_ce:
        # Build per-shape (start_pos, last_consumer_pos) lists
        shape_intervals: dict[tuple, list[tuple[int, int]]] = {}
        for node, is_ag, group_name in candidates:
            if not is_ag:
                continue
            input_val = node.args[0].meta.get("val") if node.args else None
            if isinstance(input_val, torch.Tensor):
                shape_key = (group_name, tuple(input_val.shape), input_val.dtype)
            else:
                shape_key = (group_name, node.name)
            start_pos = node_positions[node]
            wait_node = _find_wait_for_collective(node)
            if wait_node is not None:
                end_pos = _find_last_consumer_pos(wait_node, node_positions)
            else:
                end_pos = start_pos
            shape_intervals.setdefault(shape_key, []).append((start_pos, end_pos))

        for shape_key, intervals in shape_intervals.items():
            # Sweep line: count max simultaneously-live same-shape AG buffers
            events: list[tuple[int, int]] = []
            for s, w in intervals:
                events.append((s, +1))
                events.append((w, -1))
            events.sort()
            max_c = cur = 0
            for _, delta in events:
                cur += delta
                max_c = max(max_c, cur)
            # Clamp to config if set (pool_size_cfg > 0 overrides auto)
            if pool_size_cfg > 0:
                max_c = min(max_c, pool_size_cfg)
            shape_pool_sizes[shape_key] = max(max_c, 1)

        if shape_pool_sizes:
            log.info(
                "CE auto pool sizes: %s",
                {str(k): v for k, v in shape_pool_sizes.items()},
            )

    ag_counters: dict[tuple, int] = {}

    def _get_buffer_id(node, group_name):
        input_val = node.args[0].meta.get("val") if node.args else None
        if isinstance(input_val, torch.Tensor):
            shape_key = (group_name, tuple(input_val.shape), input_val.dtype)
        else:
            shape_key = (group_name, node.name)
        counter = ag_counters.get(shape_key, 0)
        ps = shape_pool_sizes.get(shape_key, 0)
        bid = counter % ps if ps > 0 else counter
        ag_counters[shape_key] = counter + 1
        return bid

    # Group consecutive CE AG candidates for coalescing.
    # Two AGs are "consecutive" if no non-AG candidate node or any user
    # of an earlier AG appears between them in the node ordering.
    replacements = 0
    coalesced_groups = 0
    fallback_groups = 0
    if use_nccl_ce:
        candidate_set = {n for n, _, _ in candidates}
        ag_candidate_set = {n for n, is_ag, _ in candidates if is_ag}

        # Build groups: walk graph nodes in order, grouping consecutive AGs
        # that have no interleaved dependencies (users of previous AG outputs).
        groups: list[list[tuple]] = []
        current_group: list[tuple] = []
        ag_outputs_in_group: set = set()

        candidate_map = {n: (n, is_ag, gn) for n, is_ag, gn in candidates}
        for node in graph.nodes:
            if node in ag_candidate_set:
                current_group.append(candidate_map[node])
                ag_outputs_in_group.add(node)
            elif node in candidate_set:
                # RS candidate — breaks the AG group
                if current_group:
                    groups.append(current_group)
                    current_group = []
                    ag_outputs_in_group = set()
                _replace_collective(
                    node, graph, symm_mem, False, candidate_map[node][2],
                    use_ag_v2, use_nccl_ce, 0,
                )
                replacements += 1
            else:
                # Non-candidate node: break the group if this node
                # uses any AG output from the current group AND is a
                # compute consumer (matmul, attention, etc.).
                # Wait_tensor and lightweight ops (views, casts) don't
                # break groups — they'll be repositioned after the
                # coalesced call in _replace_ag_group_coalesced.
                if current_group and ag_outputs_in_group:
                    uses_group_ag = any(
                        inp in ag_outputs_in_group
                        for inp in node.all_input_nodes
                    )
                    if uses_group_ag and _is_compute_consumer(node):
                        groups.append(current_group)
                        current_group = []
                        ag_outputs_in_group = set()

        if current_group:
            groups.append(current_group)

        # Split groups that exceed max coalesce size.
        if max_coalesce > 0:
            split_groups = []
            for g in groups:
                for i in range(0, len(g), max_coalesce):
                    split_groups.append(g[i:i + max_coalesce])
            groups = split_groups

        group_sizes = [len(g) for g in groups]
        log.info(
            "CE AG coalescing: %d groups, sizes=%s, max=%d",
            len(groups),
            group_sizes[:20],
            max(group_sizes) if group_sizes else 0,
        )

        for group in groups:
            if len(group) == 1:
                node, is_ag, group_name = group[0]
                bid = _get_buffer_id(node, group_name)
                _replace_collective(
                    node, graph, symm_mem, is_ag, group_name,
                    use_ag_v2, use_nccl_ce, bid,
                )
                replacements += 1
            elif _can_coalesce_group(group, node_positions):
                _replace_ag_group_coalesced(
                    group, graph, symm_mem, _get_buffer_id,
                )
                replacements += len(group)
                coalesced_groups += 1
            else:
                # Fall back to individual CE replacements when coalescing
                # is unsafe (e.g. TP outputs interleaved with FSDP AGs).
                fallback_groups += 1
                log.info(
                    "CE skipping coalescing for group of %d AGs "
                    "(inputs not all available before first AG)",
                    len(group),
                )
                for node, is_ag, gn in group:
                    bid = _get_buffer_id(node, gn)
                    _replace_collective(
                        node, graph, symm_mem, is_ag, gn,
                        use_ag_v2, use_nccl_ce, bid,
                    )
                    replacements += 1
    else:
        for node, is_ag, group_name in candidates:
            _replace_collective(
                node, graph, symm_mem, is_ag, group_name,
                use_ag_v2, use_nccl_ce, 0,
            )
            replacements += 1

    total_ce_buffers = sum(
        min(count, shape_pool_sizes.get(sk, count))
        for sk, count in ag_counters.items()
    )

    log.info(
        "Replaced %d/%d FSDP collectives "
        "(skipped_small=%d, skipped_no_overlap=%d, "
        "skipped_nvlink_contention=%d, min_bytes=%d, ce_buffers=%d, "
        "coalesced_groups=%d, fallback_groups=%d)",
        replacements,
        len(collectives),
        skipped_small,
        skipped_no_overlap,
        skipped_nvlink_contention,
        min_bytes,
        total_ce_buffers,
        coalesced_groups,
        fallback_groups,
    )


def _can_coalesce_group(group, node_positions):
    """Check if all AG inputs are defined before the first AG in the group.

    With TP, AG inputs can be TP collective outputs positioned between
    AGs. Coalescing requires all inputs available at a single insertion
    point, which is only safe when all inputs precede the first AG.
    """
    ag_nodes = [n for n, _, _ in group]
    first_ag_pos = node_positions[ag_nodes[0]]
    for ag_node in ag_nodes:
        inp = ag_node.args[0]
        inp_pos = node_positions.get(inp, -1)
        if inp_pos >= first_ag_pos:
            return False
    return True


def _replace_ag_group_coalesced(group, graph, symm_mem, get_buffer_id_fn):
    """Replace a group of consecutive AG nodes with a single coalesced CE call.

    Precondition: _can_coalesce_group() returned True, meaning all AG inputs
    are defined before the first AG. This lets us insert the coalesced call
    at the first AG position safely.

    Intermediate nodes (wait_tensor, views, casts) between AGs are moved
    after the getitems to maintain topological order.
    """
    group_name = group[0][2]
    ag_nodes = [n for n, _, _ in group]
    input_nodes = [n.args[0] for n in ag_nodes]
    buffer_ids = [get_buffer_id_fn(n, gn) for n, _, gn in group]

    ag_set = set(ag_nodes)
    first_ag, last_ag = ag_nodes[0], ag_nodes[-1]

    # Collect intermediate nodes between first and last AG (exclusive).
    intermediate_nodes = []
    in_range = False
    for node in list(graph.nodes):
        if node is first_ag:
            in_range = True
            continue
        if node is last_ag:
            break
        if in_range and node not in ag_set:
            intermediate_nodes.append(node)

    with graph.inserting_before(first_ag):
        new_node = graph.call_function(
            symm_mem._nccl_ce_all_gather_coalesced.default,
            args=(input_nodes, group_name, buffer_ids),
        )

    prev_node = new_node
    getitem_nodes = []
    for i in range(len(group)):
        with graph.inserting_after(prev_node):
            getitem_node = graph.call_function(
                operator.getitem, args=(new_node, i)
            )
        getitem_nodes.append(getitem_node)
        prev_node = getitem_node

    for i, (node, _, _) in enumerate(group):
        getitem_nodes[i].meta.update(node.meta)
        node.replace_all_uses_with(getitem_nodes[i])

    # Move intermediate nodes after the last getitem, preserving order.
    insert_after = prev_node
    for node in intermediate_nodes:
        node.prepend(insert_after.next)
        insert_after = node

    for node, _, _ in group:
        graph.erase_node(node)


def _enable_symm_mem(group_name):
    """Try to enable symmetric memory for a group. Returns True on success."""
    from torch.distributed._symmetric_memory import (
        enable_symm_mem_for_group,
        is_symm_mem_enabled_for_group,
    )

    if is_symm_mem_enabled_for_group(group_name):
        return True
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            enable_symm_mem_for_group(group_name)
        return True
    except (TypeError, RuntimeError, KeyError) as e:
        log.debug("LC: cannot enable symm_mem for group %s: %s", group_name, e)
        return False


def _replace_collective(
    node, graph, symm_mem, is_ag, group_name,
    use_ag_v2=False, use_nccl_ce=False,
    buffer_id=0,
):
    input_node = node.args[0]
    if is_ag:
        if use_nccl_ce:
            target = symm_mem._nccl_ce_all_gather.default
            args = (input_node, group_name, buffer_id)
        elif use_ag_v2:
            target = symm_mem._low_contention_all_gather_v2.default
            args = (input_node, group_name)
        else:
            target = symm_mem._low_contention_all_gather.default
            args = (input_node, group_name)
    else:
        reduce_op = node.args[1]
        if use_nccl_ce:
            target = symm_mem._nccl_efficiency_reduce_scatter.default
        else:
            target = symm_mem._low_contention_reduce_scatter.default
        args = (input_node, reduce_op, group_name)

    with graph.inserting_before(node):
        new_node = graph.call_function(target, args=args)
    new_node.meta.update(node.meta)
    node.replace_all_uses_with(new_node)
    graph.erase_node(node)


def _get_per_rank_bytes(node, is_ag):
    """Return per-rank message bytes for a collective, or None if unknown."""
    input_val = node.args[0].meta.get("val") if node.args else None
    if not isinstance(input_val, torch.Tensor):
        return None
    total_bytes = input_val.nelement() * input_val.element_size()
    if is_ag:
        return total_bytes
    # For RS, input is the full tensor; per-rank = total / group_size
    group_size = node.args[2] if len(node.args) > 2 else None
    if not isinstance(group_size, int) or group_size <= 0:
        return None
    return total_bytes // group_size


def _is_compute_consumer(node):
    """Check if a node is a heavy compute consumer (matmul, attention, conv).

    Used to decide whether to break coalescing groups. Lightweight ops
    like wait_tensor, views, casts, and element-wise ops don't justify
    breaking a coalescing group.
    """
    from torch._inductor.fx_passes.overlap_scheduling import is_compute_node

    return is_compute_node(node)


def _has_compute_bound_overlap(start_node, graph, node_positions):
    """Check if compute-bound ops overlap with this collective.

    Uses overlap scheduler metadata when available (stream-level analysis),
    falls back to position-based heuristic (compute nodes between start/wait).
    """
    # Overlap scheduler stamps this after stream-level scheduling
    overlap_hidden = start_node.meta.get("overlap_hidden")
    if overlap_hidden is not None:
        return overlap_hidden

    # Fallback: position-based check for when overlap scheduler didn't run
    from torch._inductor.fx_passes.overlap_scheduling import is_compute_node

    wait_node = _find_wait_for_collective(start_node)
    if wait_node is None:
        return False

    start_pos = node_positions[start_node]
    wait_pos = node_positions[wait_node]

    for node in graph.nodes:
        pos = node_positions.get(node)
        if pos is None or pos <= start_pos or pos >= wait_pos:
            continue
        if is_compute_node(node):
            return True
    return False


def _has_other_group_collectives(start_node, group_name, graph, node_positions):
    """Check if other groups' collectives overlap, competing for NVLink."""
    wait_node = _find_wait_for_collective(start_node)
    if wait_node is None:
        return False

    start_pos = node_positions[start_node]
    wait_pos = node_positions[wait_node]

    for node in graph.nodes:
        pos = node_positions.get(node)
        if pos is None or pos <= start_pos or pos >= wait_pos:
            continue
        info = _get_collective_info(node)
        if info is not None:
            _, other_group = info
            if other_group != group_name:
                log.debug(
                    "LC contention %s: found %s (group %s) between start/wait",
                    start_node.name,
                    node.name,
                    other_group,
                )
                return True
    return False


def _find_last_consumer_pos(node, node_positions):
    """Find the graph position of the last transitive consumer of a node.

    Follows view-like ops (reshape, slice, etc.) that share storage,
    since the underlying CE buffer remains live through views.
    Skips output/placeholder nodes which aren't real buffer consumers.
    """
    _VIEW_OPS = {
        "reshape", "view", "expand", "permute", "transpose",
        "slice", "select", "unflatten", "flatten", "contiguous",
        "as_strided", "unsqueeze", "squeeze", "t", "narrow",
    }
    last_pos = node_positions.get(node, 0)
    stack = list(node.users)
    visited = {node}
    while stack:
        user = stack.pop()
        if user in visited:
            continue
        visited.add(user)
        if user.op in ("output", "placeholder"):
            continue
        pos = node_positions.get(user)
        if pos is not None:
            last_pos = max(last_pos, pos)
        if user.op == "call_function":
            target_name = getattr(user.target, "__name__", str(user.target))
            is_view = any(v in target_name for v in _VIEW_OPS)
            if is_view:
                stack.extend(user.users)
    return last_pos


def _is_wait_tensor(node):
    """Check if node is a wait_tensor op (direct or wrapped in ControlDeps)."""
    if node.op != "call_function":
        return False
    if node.target is torch.ops._c10d_functional.wait_tensor.default:
        return True
    # Handles public namespace (c10d_functional.wait_tensor) and
    # ControlDeps-wrapped wait_tensor (from TBB manual scheduling)
    return "wait_tensor" in node.name


def _find_wait_for_collective(start_node):
    """Find the wait_tensor node for a collective.

    Handles multiple graph patterns:
    1. Direct: start -> wait_tensor(start)
    2. _out variant: start(out=buf) -> wait_tensor(buf)
    3. ControlDeps-wrapped: start -> control_deps(wait_tensor_subgraph, start)
    """
    for user in start_node.users:
        if _is_wait_tensor(user):
            return user

    # For _out variants, check users of the out-buffer keyword argument.
    c10d = torch.ops._c10d_functional
    if start_node.target in (
        c10d.all_gather_into_tensor_out.default,
        c10d.reduce_scatter_tensor_out.default,
    ):
        out_buf = start_node.kwargs.get("out")
        if isinstance(out_buf, torch.fx.Node):
            for user in out_buf.users:
                if _is_wait_tensor(user):
                    return user

    return None
