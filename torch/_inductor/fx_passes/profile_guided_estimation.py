"""
Profile-Guided Estimation (PGE) for overlap scheduling.

Parses a Chrome Trace JSON (from torch.profiler) and builds lookup tables
for collective, matmul, and attention kernel runtimes. These are used as
a custom_runtime_estimation hook in the overlap scheduler.

When the same profile is loaded on all ranks, estimates are deterministic
and no cross-rank synchronization is needed.
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.fx as fx
from torch._inductor.fx_passes.bucketing import (
    is_all_gather_into_tensor,
    is_all_reduce_tensor,
    is_all_to_all_tensor,
    is_reduce_scatter_tensor,
)
from torch._logging import trace_structured
from torch.utils._ordered_set import OrderedSet


log = logging.getLogger(__name__)


def _rank_stride(ranks: tuple[int, ...]) -> int | None:
    """Compute the stride of a sorted rank tuple, or None if non-uniform.

    Examples:
        (0, 2, 4, 6) → stride 2
        (0, 1)       → stride 1
        (1, 3, 5, 7) → stride 2
        (0, 1, 4, 5) → None (non-uniform)
    """
    if len(ranks) <= 1:
        return None
    stride = ranks[1] - ranks[0]
    if stride <= 0:
        return None
    for i in range(2, len(ranks)):
        if ranks[i] - ranks[i - 1] != stride:
            return None
    return stride


@dataclass
class CollectiveRecord:
    """A single collective kernel observation from the profile."""

    collective_name: str  # "all_gather_into_tensor", "reduce_scatter_tensor", etc.
    pg_ranks: tuple[int, ...]  # sorted rank tuple
    group_size: int
    in_nelems: int  # "In msg nelems" from profile
    out_nelems: int  # "Out msg nelems" from profile
    dtype: str  # "Float", "BFloat16", etc.
    duration_us: float


@dataclass
class MatmulRecord:
    """A single aten::mm observation from the profile."""

    # input shapes: ((M, K), (K, N))
    input_shapes: tuple[tuple[int, ...], tuple[int, ...]]
    dtype: str
    duration_us: float  # sum of all GPU kernels for this CPU op


@dataclass
class SdpaRecord:
    """A single SDPA (scaled dot product attention) observation."""

    batch: int
    num_heads: int
    seq_len: int
    head_dim: int
    dtype: str
    is_backward: bool
    duration_us: float  # sum of all GPU kernels for this CPU op


_DTYPE_BYTES: dict[str, int] = {
    "Float": 4,
    "Half": 2,
    "BFloat16": 2,
    "Double": 8,
    "Int": 4,
    "Long": 8,
    "Char": 1,
    "Byte": 1,
}


@dataclass
class ProfileData:
    """Parse Chrome Trace JSON and build lookup tables for kernel runtimes."""

    collectives: list[CollectiveRecord] = field(default_factory=list)
    matmuls: list[MatmulRecord] = field(default_factory=list)
    sdpa_records: list[SdpaRecord] = field(default_factory=list)
    pg_configs: dict[str, tuple[int, ...]] = field(default_factory=dict)

    # Lookup indices built after loading
    _collective_index: dict[
        tuple[str, tuple[int, ...], str], list[tuple[int, float]]
    ] = field(default_factory=dict)
    # Fallback index by (name, stride, group_size, dtype) — matches PGs
    # belonging to the same mesh dimension regardless of specific ranks.
    # E.g. (0,2,4,6) and (1,3,5,7) both have stride=2, size=4.
    _collective_index_by_stride: dict[
        tuple[str, int, int, str], list[tuple[int, float]]
    ] = field(default_factory=dict)
    # Count of distinct strides per (stride, group_size) — used for
    # ambiguity check (skip fallback if multiple PGs share stride+size).
    _pg_count_by_stride: dict[tuple[int, int], int] = field(default_factory=dict)
    _matmul_index: dict[tuple[tuple[int, ...], tuple[int, ...], str], float] = field(
        default_factory=dict
    )
    _sdpa_index: dict[tuple[int, int, int, int, str, bool], float] = field(
        default_factory=dict
    )
    # Peak observed bandwidth per PG (GB/s), computed from largest messages
    _pg_peak_bw: dict[tuple[int, ...], float] = field(default_factory=dict)
    # Stride-based fallback: (stride, group_size) -> peak BW (GB/s)
    _stride_peak_bw: dict[tuple[int, int], float] = field(default_factory=dict)
    # Source of last lookup_collective result: "profile" or "pg_bandwidth"
    last_estimation_source: str = ""

    def load(self, trace_path: str) -> None:
        """Load and parse a Chrome Trace JSON file."""
        with open(trace_path) as f:
            data = json.load(f)

        self._parse_pg_configs(data)
        self._parse_events(data.get("traceEvents", []))
        self._build_indices()

        log.info(
            "PGE loaded: %d collectives, %d matmuls, %d sdpa records, %d PGs",
            len(self.collectives),
            len(self.matmuls),
            len(self.sdpa_records),
            len(self.pg_configs),
        )

    def _parse_pg_configs(self, data: dict[str, Any]) -> None:
        dist_info = data.get("distributedInfo", {})
        pg_config = dist_info.get("pg_config", {})
        # pg_config can be a list of dicts or a dict of dicts
        if isinstance(pg_config, list):
            for pg_info in pg_config:
                pg_name = str(pg_info.get("pg_name", ""))
                ranks = pg_info.get("ranks", [])
                if ranks:
                    self.pg_configs[pg_name] = tuple(sorted(ranks))
        elif isinstance(pg_config, dict):
            for pg_name, pg_info in pg_config.items():
                ranks = pg_info.get("ranks", [])
                if ranks:
                    self.pg_configs[pg_name] = tuple(sorted(ranks))

    def _parse_events(self, events: list[dict[str, Any]]) -> None:
        # Build External id -> CPU op mapping
        cpu_ops: dict[int, dict[str, Any]] = {}
        for ev in events:
            cat = ev.get("cat", "")
            if cat == "cpu_op":
                eid = ev.get("args", {}).get("External id")
                if eid is not None:
                    cpu_ops[eid] = ev

        # Build External id -> list of GPU kernel durations
        gpu_kernels: dict[int, list[tuple[str, float]]] = defaultdict(list)
        for ev in events:
            if ev.get("cat") != "kernel":
                continue
            args = ev.get("args", {})
            eid = args.get("External id")
            dur = ev.get("dur", 0.0)
            name = ev.get("name", "")
            if eid is not None and dur > 0:
                gpu_kernels[eid].append((name, dur))

        # Parse collectives from GPU kernel events directly
        # (NCCL kernels carry collective metadata in args)
        for ev in events:
            if ev.get("cat") != "kernel":
                continue
            args = ev.get("args", {})
            coll_name = args.get("Collective name")
            if coll_name is None:
                continue
            pg_name = args.get("Process Group Name", "")
            pg_ranks_str = args.get("Process Group Ranks", "")
            group_size = args.get("Group size", 0)
            in_nelems = args.get("In msg nelems", 0)
            out_nelems = args.get("Out msg nelems", 0)
            dtype = args.get("dtype", "")
            dur = ev.get("dur", 0.0)
            if dur <= 0:
                continue

            pg_ranks = self._parse_ranks(pg_ranks_str, pg_name)

            self.collectives.append(
                CollectiveRecord(
                    collective_name=coll_name,
                    pg_ranks=pg_ranks,
                    group_size=group_size,
                    in_nelems=in_nelems,
                    out_nelems=out_nelems,
                    dtype=dtype,
                    duration_us=dur,
                )
            )

        # Parse matmuls and SDPA from CPU ops correlated to GPU kernels
        for eid, cpu_ev in cpu_ops.items():
            name = cpu_ev.get("name", "")
            cpu_args = cpu_ev.get("args", {})
            kernels = gpu_kernels.get(eid, [])
            if not kernels:
                continue
            total_dur = sum(dur for _, dur in kernels)

            if name == "aten::mm":
                self._parse_mm(cpu_args, total_dur)
            elif "attention" in name.lower() or "sdpa" in name.lower():
                self._parse_sdpa(name, cpu_args, total_dur)

    def _parse_ranks(self, ranks_str: str, pg_name: str) -> tuple[int, ...]:
        """Parse rank list from profile string or fall back to pg_configs."""
        if isinstance(ranks_str, str) and ranks_str.startswith("["):
            try:
                ranks = json.loads(ranks_str)
                return tuple(sorted(ranks))
            except (json.JSONDecodeError, TypeError):
                pass
        # Fall back to pg_configs
        if pg_name in self.pg_configs:
            return self.pg_configs[pg_name]
        return ()

    def _parse_mm(self, args: dict[str, Any], total_dur: float) -> None:
        input_dims = args.get("Input Dims", [])
        input_types = args.get("Input type", [])
        if len(input_dims) < 2:
            return
        dtype = input_types[0] if input_types else ""
        shapes = (tuple(input_dims[0]), tuple(input_dims[1]))
        self.matmuls.append(
            MatmulRecord(input_shapes=shapes, dtype=dtype, duration_us=total_dur)
        )

    def _parse_sdpa(self, op_name: str, args: dict[str, Any], total_dur: float) -> None:
        input_dims = args.get("Input Dims", [])
        input_types = args.get("Input type", [])
        if not input_dims or not input_dims[0]:
            return
        # Q tensor shape: [batch, num_heads, seq_len, head_dim]
        q_shape = input_dims[0]
        if len(q_shape) != 4:
            return
        dtype = input_types[0] if input_types else ""
        is_backward = "backward" in op_name.lower()
        self.sdpa_records.append(
            SdpaRecord(
                batch=q_shape[0],
                num_heads=q_shape[1],
                seq_len=q_shape[2],
                head_dim=q_shape[3],
                dtype=dtype,
                is_backward=is_backward,
                duration_us=total_dur,
            )
        )

    def _build_indices(self) -> None:
        """Build lookup indices from parsed records."""
        coll_idx: dict[tuple[str, tuple[int, ...], str], list[tuple[int, float]]] = (
            defaultdict(list)
        )
        coll_idx_by_stride: dict[tuple[str, int, int, str], list[tuple[int, float]]] = (
            defaultdict(list)
        )
        # Track distinct PG rank sets per (stride, group_size) for ambiguity check
        pg_sets_by_stride: dict[tuple[int, int], OrderedSet[tuple[int, ...]]] = (
            defaultdict(OrderedSet)
        )
        for rec in self.collectives:
            norm_name = self._normalize_collective_name(rec.collective_name)
            gs = len(rec.pg_ranks) if rec.pg_ranks else rec.group_size
            coll_idx[(norm_name, rec.pg_ranks, rec.dtype)].append(
                (rec.out_nelems, rec.duration_us)
            )
            stride = _rank_stride(rec.pg_ranks)
            if stride is not None:
                coll_idx_by_stride[(norm_name, stride, gs, rec.dtype)].append(
                    (rec.out_nelems, rec.duration_us)
                )
                pg_sets_by_stride[(stride, gs)].add(rec.pg_ranks)
        # Sort by nelems for interpolation
        self._collective_index = {
            k: sorted(v, key=lambda x: x[0]) for k, v in coll_idx.items()
        }
        self._collective_index_by_stride = {
            k: sorted(v, key=lambda x: x[0]) for k, v in coll_idx_by_stride.items()
        }
        self._pg_count_by_stride = {k: len(pgs) for k, pgs in pg_sets_by_stride.items()}

        # Matmul index: (shape_a, shape_b, dtype) -> avg_dur_us
        mm_groups: dict[tuple[tuple[int, ...], tuple[int, ...], str], list[float]] = (
            defaultdict(list)
        )
        for rec in self.matmuls:
            key = (rec.input_shapes[0], rec.input_shapes[1], rec.dtype)
            mm_groups[key].append(rec.duration_us)
        self._matmul_index = {k: sum(v) / len(v) for k, v in mm_groups.items()}

        # SDPA index: (batch, heads, seq_len, head_dim, dtype, is_bwd) -> avg_dur_us
        sdpa_groups: dict[tuple[int, int, int, int, str, bool], list[float]] = (
            defaultdict(list)
        )
        for rec in self.sdpa_records:
            key = (
                rec.batch,
                rec.num_heads,
                rec.seq_len,
                rec.head_dim,
                rec.dtype,
                rec.is_backward,
            )
            sdpa_groups[key].append(rec.duration_us)
        self._sdpa_index = {k: sum(v) / len(v) for k, v in sdpa_groups.items()}

        # Per-PG peak bandwidth: compute bytes/us for each collective observation,
        # then take the max from the top-N largest messages per PG (where bandwidth
        # is most representative of hardware speed, not dominated by startup latency).
        # Uses output-convention bytes (matching _estimate_with_pg_bandwidth).
        _TOP_N = 5  # consider top N largest messages for peak BW
        pg_bw_samples: dict[tuple[int, ...], list[tuple[int, float]]] = defaultdict(
            list
        )
        stride_bw_samples: dict[tuple[int, int], list[tuple[int, float]]] = defaultdict(
            list
        )
        for rec in self.collectives:
            if rec.out_nelems <= 0 or rec.duration_us <= 0:
                continue
            gs = len(rec.pg_ranks) if rec.pg_ranks else rec.group_size
            elem_bytes = self._dtype_elem_bytes(rec.dtype)
            total_bytes = rec.out_nelems * elem_bytes
            bw_gbps = total_bytes / (rec.duration_us * 1e-6) / 1e9  # GB/s
            pg_bw_samples[rec.pg_ranks].append((total_bytes, bw_gbps))
            stride = _rank_stride(rec.pg_ranks)
            if stride is not None:
                stride_bw_samples[(stride, gs)].append((total_bytes, bw_gbps))

        def _peak_bw_from_samples(
            samples: list[tuple[int, float]],
        ) -> float:
            """Get peak BW from the top-N largest messages."""
            # Sort by message size descending, take top N, return max BW
            sorted_samples = sorted(samples, key=lambda x: x[0], reverse=True)
            top = sorted_samples[:_TOP_N]
            return max(bw for _, bw in top) if top else 0.0

        self._pg_peak_bw = {
            pg: _peak_bw_from_samples(samples)
            for pg, samples in pg_bw_samples.items()
            if samples
        }
        self._stride_peak_bw = {
            key: _peak_bw_from_samples(samples)
            for key, samples in stride_bw_samples.items()
            if samples
        }

    @staticmethod
    def _dtype_elem_bytes(dtype: str) -> int:
        """Return bytes per element for a profile dtype string."""
        return _DTYPE_BYTES.get(dtype, 2)  # default bf16

    @staticmethod
    def _normalize_collective_name(name: str) -> str:
        """Normalize collective name between profile and FX conventions.

        Profile uses: _allgather_base, allreduce, reduce_scatter_tensor_coalesced
        FX uses: all_gather_into_tensor, all_reduce, reduce_scatter_tensor
        """
        n = name.lower()
        if "allgather" in n or "all_gather" in n:
            return "all_gather"
        if "reduce_scatter" in n:
            return "reduce_scatter"
        if "allreduce" in n or "all_reduce" in n:
            return "all_reduce"
        if "all_to_all" in n or "alltoall" in n:
            return "all_to_all"
        return name

    # Maximum ratio of target_nelems / max_observed before switching from
    # log-log extrapolation to bandwidth-based estimation.
    EXTRAPOLATION_CAP = 2.0

    def _estimate_with_pg_bandwidth(
        self,
        pg_ranks: tuple[int, ...],
        nelems: int,
        dtype: str,
    ) -> float | None:
        """Estimate collective duration using peak observed bandwidth for this PG.

        Used when the target size exceeds the extrapolation cap. Returns ms or None.
        """
        bw_gbps = self._pg_peak_bw.get(pg_ranks)
        if bw_gbps is None or bw_gbps <= 0:
            # Try stride-based fallback
            stride = _rank_stride(pg_ranks)
            gs = len(pg_ranks)
            if stride is not None:
                bw_gbps = self._stride_peak_bw.get((stride, gs))
        if bw_gbps is None or bw_gbps <= 0:
            return None  # fall through to analytical
        elem_bytes = self._dtype_elem_bytes(dtype)
        total_bytes = nelems * elem_bytes
        dur_ms = total_bytes / (bw_gbps * 1e6)  # GB/s → bytes/ms = 1e6
        return dur_ms

    def lookup_collective(
        self,
        collective_name: str,
        pg_ranks: tuple[int, ...],
        nelems: int,
        dtype: str,
    ) -> float | None:
        """Look up collective duration in ms. Interpolates by nelems in log-log space.

        Tries exact rank match first, then falls back to stride-based match
        (same mesh dimension across ranks: e.g. (0,2,4,6) and (1,3,5,7) both
        have stride=2, size=4 → same dimension).

        When the target size exceeds EXTRAPOLATION_CAP * max_observed, uses
        bandwidth-based estimation from peak observed bandwidth instead of
        linear extrapolation (which overestimates for large messages).
        """
        self.last_estimation_source = ""
        norm_name = self._normalize_collective_name(collective_name)
        # Try exact rank match first
        key = (norm_name, pg_ranks, dtype)
        entries = self._collective_index.get(key)
        if not entries:
            # Fall back to stride-based match: same mesh dimension
            gs = len(pg_ranks)
            stride = _rank_stride(pg_ranks)
            if (
                stride is not None
                and self._pg_count_by_stride.get((stride, gs), 0) == 1
            ):
                stride_key = (norm_name, stride, gs, dtype)
                entries = self._collective_index_by_stride.get(stride_key)
            if not entries:
                return None

        # Exact match
        for n, dur in entries:
            if n == nelems:
                self.last_estimation_source = "profile"
                return dur / 1e3  # us -> ms

        # Check extrapolation distance: if target is far beyond observed range,
        # use bandwidth-based model instead of log-log extrapolation
        max_observed = max((n for n, _ in entries if n > 0), default=0)
        if max_observed > 0 and nelems > max_observed * self.EXTRAPOLATION_CAP:
            est = self._estimate_with_pg_bandwidth(pg_ranks, nelems, dtype)
            if est is not None:
                self.last_estimation_source = "pg_bandwidth"
                return est
            # Fall through to log-log if no BW data available

        # Interpolation in log-log space
        result = self._interpolate_log_log(entries, nelems)
        if result is not None:
            self.last_estimation_source = "profile"
        return result

    def _interpolate_log_log(
        self, entries: list[tuple[int, float]], target_nelems: int
    ) -> float | None:
        """Interpolate duration in log-log space (log(nelems) vs log(dur))."""
        if not entries or target_nelems <= 0:
            return None

        log_target = math.log(target_nelems)

        # Find bracketing entries
        lower: tuple[int, float] | None = None
        upper: tuple[int, float] | None = None
        for n, dur in entries:
            if n <= 0 or dur <= 0:
                continue
            if n <= target_nelems:
                lower = (n, dur)
            if n >= target_nelems and upper is None:
                upper = (n, dur)

        if lower is not None and upper is not None:
            log_n0, log_d0 = math.log(lower[0]), math.log(lower[1])
            log_n1, log_d1 = math.log(upper[0]), math.log(upper[1])
            if log_n1 == log_n0:
                return lower[1] / 1e3
            t = (log_target - log_n0) / (log_n1 - log_n0)
            log_dur = log_d0 + t * (log_d1 - log_d0)
            return math.exp(log_dur) / 1e3  # us -> ms
        elif lower is not None:
            # Extrapolate from nearest lower (scale proportionally)
            return (lower[1] * target_nelems / lower[0]) / 1e3
        elif upper is not None:
            # Extrapolate from nearest upper
            return (upper[1] * target_nelems / upper[0]) / 1e3

        return None

    def lookup_mm(
        self,
        input_shapes: tuple[tuple[int, ...], tuple[int, ...]],
        dtype: str,
    ) -> float | None:
        """Look up matmul duration in ms.

        Tries exact shape match first, then interpolates by FLOP ratio from
        the nearest matmul with the same dtype.
        """
        key = (input_shapes[0], input_shapes[1], dtype)
        dur_us = self._matmul_index.get(key)
        if dur_us is not None:
            return dur_us / 1e3  # us -> ms
        # Interpolate: scale by FLOP ratio from nearest same-dtype matmul
        target_flops = self._mm_flops(input_shapes[0], input_shapes[1])
        if target_flops <= 0:
            return None
        best_ratio = float("inf")
        best_dur: float | None = None
        for (sa, sb, dt), d in self._matmul_index.items():
            if dt != dtype:
                continue
            ref_flops = self._mm_flops(sa, sb)
            if ref_flops <= 0:
                continue
            ratio = max(target_flops / ref_flops, ref_flops / target_flops)
            if ratio < best_ratio:
                best_ratio = ratio
                best_dur = d * (target_flops / ref_flops)
        if best_dur is not None:
            return best_dur / 1e3
        return None

    @staticmethod
    def _mm_flops(a: tuple[int, ...], b: tuple[int, ...]) -> int:
        """Compute 2*M*N*K for a matmul (A=[M,K] @ B=[K,N])."""
        if len(a) >= 2 and len(b) >= 2:
            return 2 * a[-2] * b[-1] * a[-1]
        return 0

    def lookup_sdpa(
        self,
        batch: int,
        num_heads: int,
        seq_len: int,
        head_dim: int,
        dtype: str,
        is_backward: bool,
    ) -> float | None:
        """Look up SDPA duration in ms.

        Tries exact shape match first, then interpolates by FLOP ratio from
        the nearest SDPA with the same dtype and direction (fwd/bwd).
        """
        key = (batch, num_heads, seq_len, head_dim, dtype, is_backward)
        dur_us = self._sdpa_index.get(key)
        if dur_us is not None:
            return dur_us / 1e3  # us -> ms
        # Interpolate: SDPA FLOPs ~ batch * heads * seq_len^2 * head_dim
        target_flops = batch * num_heads * seq_len * seq_len * head_dim
        if target_flops <= 0:
            return None
        best_ratio = float("inf")
        best_dur: float | None = None
        for (b2, h2, s2, d2, dt2, bwd2), dur in self._sdpa_index.items():
            if dt2 != dtype or bwd2 != is_backward:
                continue
            ref_flops = b2 * h2 * s2 * s2 * d2
            if ref_flops <= 0:
                continue
            ratio = max(target_flops / ref_flops, ref_flops / target_flops)
            if ratio < best_ratio:
                best_ratio = ratio
                best_dur = dur * (target_flops / ref_flops)
        if best_dur is not None:
            return best_dur / 1e3
        return None


# Mapping from torch dtype to profile dtype strings
_DTYPE_TO_PROFILE_STR: dict[torch.dtype, str] = {
    torch.float32: "Float",
    torch.float16: "Half",
    torch.bfloat16: "BFloat16",
    torch.float64: "Double",
    torch.int32: "Int",
    torch.int64: "Long",
    torch.int8: "Char",
    torch.uint8: "Byte",
}

# Also map C10 type strings to normalized form
_C10_DTYPE_TO_PROFILE_STR: dict[str, str] = {
    "c10::Float": "Float",
    "c10::Half": "Half",
    "c10::BFloat16": "BFloat16",
    "c10::Double": "Double",
    "c10::Int": "Int",
    "c10::Long": "Long",
    "c10::Char": "Char",
    "c10::Byte": "Byte",
}


def _dtype_to_profile_str(dtype: torch.dtype) -> str:
    return _DTYPE_TO_PROFILE_STR.get(dtype, str(dtype))


def _normalize_profile_dtype(dtype_str: str) -> str:
    """Normalize profile dtype string (may be 'c10::BFloat16' or 'BFloat16')."""
    return _C10_DTYPE_TO_PROFILE_STR.get(dtype_str, dtype_str)


def _get_node_dtype_str(node: fx.Node) -> str:
    """Extract dtype string from FX node metadata."""
    val = node.meta.get("val")
    if isinstance(val, torch.Tensor):
        return _dtype_to_profile_str(val.dtype)
    if isinstance(val, (list, tuple)) and val:
        first = val[0]
        if isinstance(first, torch.Tensor):
            return _dtype_to_profile_str(first.dtype)
    return ""


def _is_mm_node(node: fx.Node) -> bool:
    """Check if node is a matrix multiplication."""
    return node.target in (
        torch.ops.aten.mm.default,
        torch.ops.aten.mm.out,
    )


def _is_sdpa_node(node: fx.Node) -> bool:
    """Check if node is a scaled dot-product attention op."""
    target = node.target
    if not hasattr(target, "__name__"):
        return False
    name = target.__name__
    return any(
        kw in name.lower()
        for kw in ("scaled_dot_product", "cudnn_attention", "flash_attention", "sdpa")
    )


def _is_sdpa_backward(node: fx.Node) -> bool:
    """Check if SDPA node is a backward op."""
    target = node.target
    name = target.__name__ if hasattr(target, "__name__") else str(target)
    return "backward" in name.lower()


def _get_mm_shapes(
    node: fx.Node,
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """Extract (A_shape, B_shape) from mm node metadata."""
    if len(node.args) < 2:
        return None
    a_node, b_node = node.args[0], node.args[1]
    if not isinstance(a_node, fx.Node) or not isinstance(b_node, fx.Node):
        return None
    a_val = a_node.meta.get("val")
    b_val = b_node.meta.get("val")
    if not isinstance(a_val, torch.Tensor) or not isinstance(b_val, torch.Tensor):
        return None

    def _resolve_shape(t: torch.Tensor) -> tuple[int, ...] | None:
        from torch._inductor.fx_passes.node_runtime_estimation import get_hint

        shape = [get_hint(s) for s in t.shape]
        if any(s is None for s in shape):
            log.debug("PGE: unresolved symbolic dims in shape %s", t.shape)
            return None
        return tuple(shape)  # type: ignore[arg-type]

    a_shape = _resolve_shape(a_val)
    b_shape = _resolve_shape(b_val)
    if a_shape is None or b_shape is None:
        return None
    return (a_shape, b_shape)


def _get_sdpa_key(
    node: fx.Node,
) -> tuple[int, int, int, int, str, bool] | None:
    """Extract (batch, heads, seq_len, head_dim, dtype, is_bwd) from SDPA node."""
    # Q is first input
    if not node.args:
        return None
    q_node = node.args[0]
    if not isinstance(q_node, fx.Node):
        return None
    q_val = q_node.meta.get("val")
    if not isinstance(q_val, torch.Tensor):
        return None
    shape = q_val.shape
    if len(shape) != 4:
        return None
    try:
        batch, heads, seq_len, head_dim = (int(s) for s in shape)
    except (TypeError, ValueError):
        return None
    dtype = _dtype_to_profile_str(q_val.dtype)
    is_bwd = _is_sdpa_backward(node)
    return (batch, heads, seq_len, head_dim, dtype, is_bwd)


def _is_collective_node(node: fx.Node) -> bool:
    """Check if node is a collective communication op."""
    return (
        is_all_gather_into_tensor(node)
        or is_reduce_scatter_tensor(node)
        or is_all_reduce_tensor(node)
        or is_all_to_all_tensor(node)
    )


def _get_collective_info(
    node: fx.Node,
) -> tuple[str, tuple[int, ...], int, str] | None:
    """Extract (collective_name, pg_ranks, nelems, dtype) from collective node."""
    from torch.fx.operator_schemas import normalize_function

    try:
        target = node.target
        assert isinstance(target, torch._ops.OpOverload)
        collective_name = target.name().split("::")[-1].split(".")[0]

        opt = normalize_function(
            target,
            args=node.args,
            kwargs=node.kwargs,
            normalize_to_only_use_kwargs=True,
        )
        if opt is None:
            return None
        _, kwargs = opt
        group_name = kwargs.get("group_name", "")

        from torch.distributed.distributed_c10d import (
            _resolve_process_group,
            get_process_group_ranks,
        )

        pg = _resolve_process_group(group_name)
        pg_ranks = tuple(sorted(get_process_group_ranks(pg)))

        # Get nelems from input tensor
        val = node.meta.get("val")
        if isinstance(val, torch.Tensor):
            nelems = 1
            for s in val.shape:
                nelems *= int(s)
            dtype = _dtype_to_profile_str(val.dtype)
        else:
            # Try first arg
            if node.args and isinstance(node.args[0], fx.Node):
                inp_val = node.args[0].meta.get("val")
                if isinstance(inp_val, torch.Tensor):
                    nelems = 1
                    for s in inp_val.shape:
                        nelems *= int(s)
                    dtype = _dtype_to_profile_str(inp_val.dtype)
                else:
                    return None
            else:
                return None

        return (collective_name, pg_ranks, nelems, dtype)
    except Exception:
        log.debug(
            "PGE: failed to extract collective info for %s", node.name, exc_info=True
        )
        return None


def _normalize_profile_indices(profile: ProfileData) -> None:
    """Normalize dtype strings in profile indices to match profile format."""
    # Rebuild collective indices with normalized dtypes
    new_coll: dict[tuple[str, tuple[int, ...], str], list[tuple[int, float]]] = {}
    for (coll_name, pg_ranks, dtype), entries in profile._collective_index.items():
        norm_dtype = _normalize_profile_dtype(dtype)
        key = (coll_name, pg_ranks, norm_dtype)
        if key in new_coll:
            new_coll[key].extend(entries)
        else:
            new_coll[key] = list(entries)
    profile._collective_index = {
        k: sorted(v, key=lambda x: x[0]) for k, v in new_coll.items()
    }

    new_coll_by_stride: dict[tuple[str, int, int, str], list[tuple[int, float]]] = {}
    for (
        coll_name,
        stride,
        gs,
        dtype,
    ), entries in profile._collective_index_by_stride.items():
        norm_dtype = _normalize_profile_dtype(dtype)
        key = (coll_name, stride, gs, norm_dtype)
        if key in new_coll_by_stride:
            new_coll_by_stride[key].extend(entries)
        else:
            new_coll_by_stride[key] = list(entries)
    profile._collective_index_by_stride = {
        k: sorted(v, key=lambda x: x[0]) for k, v in new_coll_by_stride.items()
    }

    # Rebuild matmul index with normalized dtypes (average on collision)
    mm_groups: dict[tuple[tuple[int, ...], tuple[int, ...], str], list[float]] = (
        defaultdict(list)
    )
    for (sa, sb, dtype), dur in profile._matmul_index.items():
        norm_dtype = _normalize_profile_dtype(dtype)
        mm_groups[(sa, sb, norm_dtype)].append(dur)
    profile._matmul_index = {k: sum(v) / len(v) for k, v in mm_groups.items()}

    # Rebuild sdpa index with normalized dtypes (average on collision)
    sdpa_groups: dict[tuple[int, int, int, int, str, bool], list[float]] = defaultdict(
        list
    )
    for (b, h, s, d, dtype, bwd), dur in profile._sdpa_index.items():
        norm_dtype = _normalize_profile_dtype(dtype)
        sdpa_groups[(b, h, s, d, norm_dtype, bwd)].append(dur)
    profile._sdpa_index = {k: sum(v) / len(v) for k, v in sdpa_groups.items()}


class ProfileGuidedEstimator:
    """Profile-guided runtime estimator for FX nodes.

    Implements the ``custom_runtime_estimation`` interface:
    ``(fx.Node, int | None) -> float | None`` (returns ms or None for fallback).
    """

    profile: ProfileData
    estimation_log: list[dict[str, Any]]
    miss_log: list[dict[str, Any]]

    def __init__(self, trace_path: str) -> None:
        self.profile = ProfileData()
        self.estimation_log: list[dict[str, Any]] = []
        self.miss_log: list[dict[str, Any]] = []
        try:
            self.profile.load(trace_path)
            _normalize_profile_indices(self.profile)
        except Exception:
            log.exception("PGE: failed to load profile from %s", trace_path)

    def __call__(self, node: fx.Node, override_size: int | None = None) -> float | None:
        # Collectives
        if _is_collective_node(node):
            return self._estimate_collective(node, override_size)
        # Matmul
        if _is_mm_node(node):
            return self._estimate_mm(node)
        # SDPA
        if _is_sdpa_node(node):
            return self._estimate_sdpa(node)
        return None

    def _estimate_collective(
        self, node: fx.Node, override_size: int | None
    ) -> float | None:
        info = _get_collective_info(node)
        if info is None:
            self.miss_log.append(
                {
                    "node": node.name,
                    "target": str(node.target),
                    "reason": "get_collective_info returned None",
                }
            )
            return None
        coll_name, pg_ranks, nelems, dtype = info
        if override_size is not None:
            if override_size == 0:
                return 0.0
            val = node.meta.get("val")
            if isinstance(val, torch.Tensor):
                elem_size = val.element_size()
                if elem_size > 0:
                    nelems = override_size // elem_size
        dtype_bytes = 0
        val = node.meta.get("val")
        if isinstance(val, torch.Tensor):
            dtype_bytes = val.element_size()
        est = self.profile.lookup_collective(coll_name, pg_ranks, nelems, dtype)
        if est is not None:
            self.estimation_log.append(
                {
                    "node": node.name,
                    "op": coll_name,
                    "nelems": nelems,
                    "dtype": dtype,
                    "dtype_bytes": dtype_bytes,
                    "group_size": len(pg_ranks),
                    "stride": _rank_stride(pg_ranks),
                    "pge_ms": est,
                    "source": self.profile.last_estimation_source or "profile",
                }
            )
        else:
            self.miss_log.append(
                {
                    "node": node.name,
                    "op": coll_name,
                    "nelems": nelems,
                    "dtype": dtype,
                    "group_size": len(pg_ranks),
                    "reason": "no match in profile",
                }
            )
        return est

    def _estimate_mm(self, node: fx.Node) -> float | None:
        shapes = _get_mm_shapes(node)
        if shapes is None:
            self.miss_log.append(
                {
                    "node": node.name,
                    "target": str(node.target),
                    "reason": "get_mm_shapes returned None",
                }
            )
            return None
        dtype = _get_node_dtype_str(node)
        est = self.profile.lookup_mm(shapes, dtype)
        if est is not None:
            self.estimation_log.append(
                {
                    "node": node.name,
                    "op": "mm",
                    "shapes": [list(s) for s in shapes],
                    "dtype": dtype,
                    "pge_ms": est,
                    "source": "profile",
                }
            )
        else:
            self.miss_log.append(
                {
                    "node": node.name,
                    "op": "mm",
                    "shapes": [list(s) for s in shapes],
                    "dtype": dtype,
                    "reason": "no match in profile",
                }
            )
        return est

    def _estimate_sdpa(self, node: fx.Node) -> float | None:
        sdpa_key = _get_sdpa_key(node)
        if sdpa_key is None:
            self.miss_log.append(
                {
                    "node": node.name,
                    "target": str(node.target),
                    "reason": "get_sdpa_key returned None",
                }
            )
            return None
        batch, heads, seq_len, head_dim, dtype, is_bwd = sdpa_key
        est = self.profile.lookup_sdpa(batch, heads, seq_len, head_dim, dtype, is_bwd)
        if est is not None:
            self.estimation_log.append(
                {
                    "node": node.name,
                    "op": "sdpa_bwd" if is_bwd else "sdpa_fwd",
                    "shape": [batch, heads, seq_len, head_dim],
                    "dtype": dtype,
                    "pge_ms": est,
                    "source": "profile",
                }
            )
        else:
            self.miss_log.append(
                {
                    "node": node.name,
                    "op": "sdpa",
                    "shape": [batch, heads, seq_len, head_dim],
                    "dtype": dtype,
                    "reason": "no match in profile",
                }
            )
        return est


def log_pge_estimations(
    estimator: ProfileGuidedEstimator,
    analytical_estimates: dict[str, float] | None = None,
) -> None:
    """Dump PGE estimation results via trace_structured for tlparse."""
    rows = []
    for entry in estimator.estimation_log:
        row = dict(entry)
        node_name = entry.get("node", "")
        if analytical_estimates and node_name in analytical_estimates:
            analytical_ms = analytical_estimates[node_name]
            row["analytical_ms"] = analytical_ms
            pge_ms = entry.get("pge_ms", 0)
            if analytical_ms > 0:
                row["pge_vs_analytical_pct"] = round(
                    (pge_ms - analytical_ms) / analytical_ms * 100, 1
                )
        rows.append(row)

    table = _format_pge_table(rows, estimator.miss_log)
    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "pge_estimations_table",
            "encoding": "string",
        },
        payload_fn=lambda: table,
    )

    log.info(
        "PGE: %d estimations, %d misses logged to trace_structured",
        len(rows),
        len(estimator.miss_log),
    )


def _format_bytes(nbytes: int) -> str:
    """Format byte count as human-readable K/M/G string."""
    if nbytes >= 1 << 30:
        return f"{nbytes / (1 << 30):.1f}G"
    if nbytes >= 1 << 20:
        return f"{nbytes / (1 << 20):.1f}M"
    if nbytes >= 1 << 10:
        return f"{nbytes / (1 << 10):.0f}K"
    return f"{nbytes}B"


def _format_pge_table(
    rows: list[dict[str, Any]],
    miss_log: list[dict[str, Any]] | None = None,
) -> str:
    """Format PGE estimations + misses as a single aligned text table."""
    misses: list[dict[str, Any]] = list(miss_log) if miss_log is not None else []
    lines: list[str] = []
    try:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_count = torch.cuda.device_count()
        lines.append(f"GPU: {gpu_name} x{gpu_count}")
    except Exception:
        lines.append("GPU: unknown")
    lines.append("")

    has_analytical = any("analytical_ms" in r for r in rows)

    if has_analytical:
        header = (
            f"{'node':<45} {'op':<30} {'size':>8} {'gs':>4} {'st':>3}"
            f" {'pge_ms':>10} {'analytical_ms':>15} {'diff%':>8} {'':>3}"
            f" {'pge_GB/s':>10} {'analytical_GB/s':>15}"
        )
    else:
        header = (
            f"{'node':<45} {'op':<30} {'size':>8} {'gs':>4} {'st':>3} {'pge_ms':>10}"
        )
    lines.append(header)
    lines.append("-" * len(header))

    for row in rows:
        node = row.get("node", "")[:44]
        op = row.get("op", "")[:29]
        gs = row.get("group_size", "")
        stride = row.get("stride", "")
        stride_str = str(stride) if stride is not None else "-"
        pge_ms = row.get("pge_ms", 0)
        dtype_bytes = row.get("dtype_bytes", 0)
        n = row.get("nelems", 0) if isinstance(row.get("nelems"), int) else 0
        size_str = _format_bytes(n * dtype_bytes) if dtype_bytes > 0 and n > 0 else "-"

        if has_analytical:
            anal_ms = row.get("analytical_ms", None)
            diff_pct = row.get("pge_vs_analytical_pct", None)
            anal_str = f"{anal_ms:.4f}" if anal_ms is not None else "-"
            diff_str = f"{diff_pct:+.1f}%" if diff_pct is not None else "-"
            flag = ""
            if diff_pct is not None:
                adp = abs(diff_pct)
                if adp > 50:
                    flag = "***"
                elif adp > 15:
                    flag = "**"
            pge_bw_str = ""
            anal_bw_str = ""
            if dtype_bytes > 0 and n > 0:
                data_bytes = n * dtype_bytes
                if pge_ms > 0:
                    pge_bw = data_bytes / (pge_ms * 1e-3) / 1e9
                    pge_bw_str = f"{pge_bw:.1f}"
                if anal_ms is not None and anal_ms > 0:
                    anal_bw = data_bytes / (anal_ms * 1e-3) / 1e9
                    anal_bw_str = f"{anal_bw:.1f}"
            line = (
                f"{node:<45} {op:<30} {size_str:>8} {gs:>4} {stride_str:>3}"
                f" {pge_ms:>10.4f} {anal_str:>15} {diff_str:>8} {flag:>3}"
                f" {pge_bw_str:>10} {anal_bw_str:>15}"
            )
        else:
            line = (
                f"{node:<45} {op:<30} {size_str:>8} {gs:>4} {stride_str:>3}"
                f" {pge_ms:>10.4f}"
            )
        lines.append(line.rstrip())

    lines.append("")
    lines.append(f"Total: {len(rows)} estimations")
    if has_analytical:
        f1 = sum(
            1
            for r in rows
            if r.get("pge_vs_analytical_pct") is not None
            and abs(r["pge_vs_analytical_pct"]) > 50
        )
        f2 = sum(
            1
            for r in rows
            if r.get("pge_vs_analytical_pct") is not None
            and 15 < abs(r["pge_vs_analytical_pct"]) <= 50
        )
        lines.append(f"Flagged: {f2} ** (>15%), {f1} *** (>50%)")

    if misses:
        lines.append("")
        lines.append(f"=== MISSES ({len(misses)}) ===")
        miss_header = f"{'node':<45} {'op':<30} {'reason'}"
        lines.append(miss_header)
        lines.append("-" * len(miss_header))
        for m in misses:  # pyrefly: ignore[unsupported-operation]
            node = m.get("node", "")[:44]
            op = (m.get("op") or m.get("target") or "")[:29]
            reason = m.get("reason", "")
            shapes = m.get("shapes")
            shape_info = m.get("shape")
            extra = ""
            if shapes:
                extra = f" shapes={shapes}"
            elif shape_info:
                extra = f" shape={shape_info}"
            lines.append(f"{node:<45} {op:<30} {reason}{extra}".rstrip())

    return "\n".join(lines)
