# mypy: allow-untyped-defs
from __future__ import annotations

from dataclasses import fields, replace
from functools import cache
from typing import Any, TYPE_CHECKING, TypeAlias

import sympy

import torch
import torch._vendor.quack.gemm_config as quack_gemm_config
from torch.utils._ordered_set import OrderedSet


if TYPE_CHECKING:
    from collections.abc import Sequence

GemmConfigKey: TypeAlias = tuple[tuple[str, Any], ...]


def gemm_config_key(config: quack_gemm_config.GemmConfig) -> GemmConfigKey:
    """Project a QuACK GEMM config using the dataclass schema as the contract."""
    return tuple(
        (field.name, getattr(config, field.name))
        for field in fields(quack_gemm_config.GemmConfig)
    )


def gemm_config_from_key(config_key: GemmConfigKey) -> quack_gemm_config.GemmConfig:
    """Reconstruct a QuACK GEMM config from its generated-code cache key."""
    return quack_gemm_config.GemmConfig(**dict(config_key))


def explicit_gemm_configs_for_device(
    config: dict[str, Any], device: torch.device
) -> tuple[quack_gemm_config.GemmConfig, ...]:
    """Return device configs matching every explicitly pinned field.

    Exact type matching prevents bool/int aliases from selecting a different key.
    """
    field_names = tuple(field.name for field in fields(quack_gemm_config.GemmConfig))
    unexpected = [name for name in config if name not in field_names]
    if unexpected:
        raise NotImplementedError(
            "FlexGEMM explicit QUACK config contains unexpected GemmConfig fields: "
            f"{unexpected}"
        )

    candidates = candidate_gemm_configs_for_device(device)
    expected_device_capacity = candidates[0].device_capacity
    requested_device_capacity = config.get("device_capacity")
    if (
        type(requested_device_capacity) is type(expected_device_capacity)
        and requested_device_capacity != expected_device_capacity
    ):
        raise NotImplementedError(
            f"FlexGEMM explicit QUACK config targets SM{requested_device_capacity}0, "
            f"but {device} uses SM{expected_device_capacity}0 configs"
        )
    split_k = config.get("split_k", 1)
    if type(split_k) is not int or split_k < 1:
        raise NotImplementedError(
            f"FlexGEMM explicit QUACK split_k must be a positive int, got {split_k!r}"
        )
    if split_k > 1 and expected_device_capacity != 10:
        raise NotImplementedError(
            "FlexGEMM split-K currently requires an SM100 or SM110 config"
        )
    split_k_mode = config.get(
        "split_k_mode", int(quack_gemm_config.SplitKMode.SERIAL)
    )
    if type(split_k_mode) is not int or split_k_mode not in tuple(
        quack_gemm_config.SplitKMode
    ):
        raise NotImplementedError(
            f"FlexGEMM explicit QUACK split_k_mode is invalid: {split_k_mode!r}"
        )
    base_config = {
        name: value
        for name, value in config.items()
        if name not in ("split_k", "split_k_mode")
    }
    matches = tuple(
        replace(candidate, split_k=split_k, split_k_mode=split_k_mode)
        for candidate in candidates
        if all(
            type(value) is type(getattr(candidate, name))
            and value == getattr(candidate, name)
            for name, value in base_config.items()
        )
    )
    if matches:
        return matches
    raise NotImplementedError(
        f"FlexGEMM explicit QUACK config constraints are not supported on {device}: "
        f"{config}"
    )


@cache
def dense_gemm_config_priority_keys() -> tuple[GemmConfigKey, ...]:
    """Return the measured dense FlexGEMM QuACK preference order."""
    configs = (
        quack_gemm_config.GemmConfig(
            tile_m=128,
            tile_n=256,
            pingpong=False,
            is_dynamic_persistent=True,
            cluster_m=2,
            device_capacity=10,
        ),
        quack_gemm_config.GemmConfig(
            tile_m=128,
            tile_n=192,
            pingpong=False,
            is_dynamic_persistent=True,
            cluster_m=2,
            device_capacity=10,
        ),
        quack_gemm_config.GemmConfig(
            tile_m=256,
            tile_n=256,
            pingpong=False,
            is_dynamic_persistent=True,
            cluster_m=2,
            device_capacity=10,
        ),
        quack_gemm_config.GemmConfig(
            tile_m=256,
            tile_n=256,
            pingpong=False,
            is_dynamic_persistent=True,
            cluster_m=2,
            cluster_n=2,
            device_capacity=10,
        ),
        quack_gemm_config.GemmConfig(
            tile_m=256,
            tile_n=192,
            pingpong=False,
            is_dynamic_persistent=True,
            cluster_m=2,
            device_capacity=10,
        ),
        quack_gemm_config.GemmConfig(
            tile_m=128,
            tile_n=128,
            pingpong=False,
            is_dynamic_persistent=False,
            cluster_m=1,
            device_capacity=10,
        ),
        quack_gemm_config.GemmConfig(
            tile_m=128,
            tile_n=256,
            pingpong=False,
            is_dynamic_persistent=True,
            cluster_m=1,
            device_capacity=10,
        ),
        quack_gemm_config.GemmConfig(
            tile_m=128,
            tile_n=256,
            pingpong=False,
            is_dynamic_persistent=False,
            cluster_m=1,
            device_capacity=10,
        ),
        quack_gemm_config.GemmConfig(
            tile_m=128,
            tile_n=128,
            pingpong=False,
            is_dynamic_persistent=True,
            cluster_m=2,
            device_capacity=10,
        ),
        quack_gemm_config.GemmConfig(
            tile_m=256,
            tile_n=128,
            pingpong=False,
            is_dynamic_persistent=True,
            cluster_m=2,
            device_capacity=10,
        ),
        quack_gemm_config.GemmConfig(
            tile_m=128,
            tile_n=224,
            pingpong=False,
            is_dynamic_persistent=True,
            cluster_m=1,
            device_capacity=10,
        ),
        quack_gemm_config.GemmConfig(
            tile_m=128,
            tile_n=160,
            pingpong=False,
            is_dynamic_persistent=True,
            cluster_m=1,
            device_capacity=10,
        ),
    )
    return tuple(gemm_config_key(config) for config in configs)


def candidate_gemm_configs_for_device(
    device: torch.device,
    device_capacity_override: tuple[int, int] | None = None,
):
    """Return device-compatible QuACK configs without requiring CUDA in workers."""
    device_capacity = (
        torch.cuda.get_device_capability(device)[0]
        if device_capacity_override is None
        else device_capacity_override[0]
    )
    if device_capacity == 11:
        device_capacity = 10
    priority_map = {
        key: priority for priority, key in enumerate(dense_gemm_config_priority_keys())
    }
    configs = sorted(
        (
            config
            for config in quack_gemm_config.get_all_configs()
            if config.device_capacity == device_capacity and not config.use_tma_gather
        ),
        key=lambda config: (
            priority_map.get(gemm_config_key(config), len(priority_map)),
            config.tile_m,
            config.tile_n,
            config.cluster_m,
            config.cluster_n,
            int(config.is_dynamic_persistent),
        ),
    )
    if not configs:
        raise RuntimeError(
            f"FlexGEMM found no QuACK configs for CUDA device capability "
            f"SM{device_capacity}0"
        )
    return configs


def expand_split_k_configs_for_problem(
    configs: Sequence[quack_gemm_config.GemmConfig],
    device: torch.device,
    m,
    n,
    k,
    batch_size=1,
) -> tuple[quack_gemm_config.GemmConfig, ...]:
    """Add split-K variants for statically known, occupancy-starved SM100 GEMMs."""
    values = tuple(sympy.sympify(value) for value in (m, n, k, batch_size))
    if any(value.free_symbols for value in values):
        return tuple(configs)
    m, n, k, batch_size = (int(value) for value in values)
    if torch.cuda.get_device_capability(device)[0] not in (10, 11):
        return tuple(configs)
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    expanded = list(configs)
    for config in configs:
        # Expanding every base config made graph compilation impractical. Benchmark
        # split-K only on the static 256x256 seed that won the target workloads.
        if config.split_k != 1 or (
            config.tile_m,
            config.tile_n,
            config.is_dynamic_persistent,
            config.cluster_m,
            config.cluster_n,
            config.cluster_k,
            config.swap_ab,
        ) != (256, 256, False, 2, 1, 1, False):
            continue
        cta_tile_m = config.tile_m // config.cluster_m
        tile_m, tile_n = (
            (cta_tile_m, config.tile_n)
            if not config.swap_ab
            else (config.tile_n, cta_tile_m)
        )
        output_tiles = (
            (m + tile_m - 1) // tile_m
            * ((n + tile_n - 1) // tile_n)
            * batch_size
        )
        k_tiles = (k + (config.tile_k or 64) - 1) // (config.tile_k or 64)
        for split_k in (2, 4):
            fills_device = (
                output_tiles < 2 * sm_count
                and output_tiles * split_k <= 8 * sm_count
            )
            enough_k_work = 2 * split_k <= k_tiles
            valid_grid_z = batch_size * split_k <= 65535
            if fills_device and enough_k_work and valid_grid_z:
                expanded.append(
                    replace(
                        config,
                        split_k=split_k,
                        split_k_mode=int(quack_gemm_config.SplitKMode.PARALLEL),
                    )
                )
    return tuple(expanded)


def default_gemm_config_key(
    device: torch.device,
    m,
    n,
    configs: Sequence[quack_gemm_config.GemmConfig] | None = None,
) -> GemmConfigKey:
    """Return the untuned default QuACK config key for generated code."""
    configs = candidate_gemm_configs_for_device(device) if configs is None else configs
    config_keys = OrderedSet([gemm_config_key(config) for config in configs])
    default_key, skinny_key, large_rect_key, large_key = (
        dense_gemm_config_priority_keys()[:4]
    )

    from torch._inductor.virtualized import V

    guard_or_false = V.graph.sizevars.guard_or_false
    if guard_or_false(sympy.Le(m, n)):
        min_dim, max_dim = m, n
    elif guard_or_false(sympy.Lt(n, m)):
        min_dim, max_dim = n, m
    else:
        return (
            default_key if default_key in config_keys else gemm_config_key(configs[0])
        )

    if guard_or_false(sympy.Lt(min_dim, 512)):
        preferred_keys = (skinny_key, default_key)
    elif guard_or_false(sympy.And(sympy.Eq(min_dim, 1024), sympy.Eq(max_dim, 1024))):
        preferred_keys = (skinny_key, default_key)
    elif guard_or_false(
        sympy.And(
            sympy.Ge(max_dim, 4096), sympy.Ge(min_dim, 768), sympy.Lt(min_dim, 1024)
        )
    ):
        preferred_keys = (large_key, default_key)
    elif guard_or_false(sympy.And(sympy.Ge(max_dim, 4096), sympy.Eq(min_dim, 1024))):
        preferred_keys = (large_rect_key, default_key)
    elif guard_or_false(sympy.Ge(min_dim, 2048)):
        preferred_keys = (large_key, default_key)
    else:
        preferred_keys = (default_key,)

    for key in preferred_keys:
        if key in config_keys:
            return key
    return gemm_config_key(configs[0])
