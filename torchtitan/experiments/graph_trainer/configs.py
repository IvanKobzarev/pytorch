# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from dataclasses import dataclass, field, fields, replace
from typing import Literal

from torchtitan.components.loss import ChunkedCELoss, CrossEntropyLoss
from torchtitan.config.configs import CompileConfig
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.trainer import Trainer


@dataclass(kw_only=True, slots=True)
class GraphTrainerCompileConfig(CompileConfig):
    mode: Literal["jit", "aot_fx_trace"] | None = "aot_fx_trace"
    """
    Compilation mode. Options:
        aot_fx_trace: non-strict tracing of fwd+loss+bwd via make_fx
        jit: standard torch.compile() with custom backend (deprecated)
    """

    backend: str = "aot_eager"

    passes: list[str] = field(default_factory=list)
    """
    Compiler pass names to apply.
    In JIT mode: applied as graph passes (e.g., auto_bucketing, transformer_block_bucketing)
    """

    enable_passes: bool = True
    """When False, skip all graph passes (both default and user-configured)."""

    disable_passes: list[str] = field(default_factory=list)
    """Pass names to selectively disable for debugging and ablation
    studies. A pass is skipped if its name exactly matches any entry.
    Example: --compile.disable_passes custom_codegen_pass,cudagraph_pass"""

    debug_graph_passes: bool = False
    """Log timing, op-count diffs, and before/after graphs for each pass to tlparse."""

    memory_policy: Literal[
        "default", "full", "eager", "sac_and_offload", "save_layer_inputs"
    ] = "default"
    """
    Memory optimization policy for activation management (SAC, offload).
        default: SAC — save all compute-intensive ops and FSDP all_gathers.
        full: full recompute — only layer outputs are saved. Mirrors
            eager's full AC (checkpoint_wrapper with no context_fn).
        eager: SAC alternating mm ops between save/recompute, matching the
            eager AC policy in torchtitan.distributed.activation_checkpoint.
        sac_and_offload: SAC + CPU offload — apply default SAC first,
            then offload surviving MUST_SAVE activations to CPU within
            the cpu_offload_budget_gb budget.
        save_layer_inputs: save each layer's input, recompute the interior.
    """

    ac_min_cut_enabled: bool = False
    """Enable min-cut activation-checkpointing refinement on top of the
    memory_policy floor (details in min_cut_ac.py). The pass first profiles the
    input AC policy as the reference, then greedily chooses save cutpoints under
    reference_peak + ac_min_cut_max_peak_increase_gb. The default budget is
    0GB, so enabling min-cut does not allow modelled peak regression.
    Tag-decision only;
    selective_activation_remat materializes downstream, so it stays correct under
    CPU offload."""

    ac_min_cut_max_peak_increase_gb: float | None = 0.0
    """Peak budget in GB relative to the pre-min-cut reference policy. The
    default 0GB forbids modelled peak regression. None means no hard peak
    requirement and should be treated as an explicit experiment. Positive values
    spend extra peak memory for less recompute; negative values require a lower
    peak than the reference policy. The peak is build_memory_profile on the joint
    forward-loss-backward graph, a static-graph upper bound that holds the FSDP
    all-gathered params, so this tracks relative changes, not absolute GPU peak."""

    ac_min_cut_save_scope: Literal["min_cut", "all"] = "min_cut"
    """Candidate pool for min-cut AC:
        min_cut: only search the min-cut frontier, the byte-minimal structural
            cutpoints.
        all: process the min-cut frontier first, then remaining eligible
            PREFER_SAVE/PREFER_RECOMPUTE activations. Over-target phases use a
            peak-progress ranker; otherwise candidates are ranked by remat-closure
            cost avoided per saved byte. Exact remat/profile validation checks
            the peak budget. If the requested target is infeasible, min-cut keeps
            the best lower-peak plan found and logs the miss."""

    ac_min_cut_min_broad_candidate_gb: float | None = None
    ac_min_cut_max_broad_candidate_gb: float | None = None
    """Optional size window for non-frontier candidates when
    ac_min_cut_save_scope="all". The min-cut frontier is never capped. This is a
    debugging and tuning knob for broad candidate pools with very large
    activations."""

    ac_min_cut_selection_mode: Literal["budgeted", "compile_like", "chain_cut"] = (
        "budgeted"
    )
    """Selection strategy for min-cut AC:
        budgeted: start from the current memory policy and greedily save useful
            candidates while respecting ac_min_cut_max_peak_increase_gb.
        compile_like: widen the recomputable pool, save only the byte-minimal
            min-cut frontier, and recompute the rest of the closure. This more
            closely matches AOTAutograd's partitioner and is experimental.
        chain_cut: only rematerialize deterministic unsaveable softmax chains by
            saving their anchor and recomputing the pre-softmax/log-softmax chain."""

    ac_save_final_layer_output: bool = True
    """When memory_policy=save_layer_inputs, also save the final transformer
    layer output boundary. This preserves the eager per-layer checkpointing
    policy. Disabling it lets the final layer output be recomputed so the graph
    can avoid holding both the final boundary and post-layer head activation."""

    ac_relax_relaxable_must_saves: bool = False
    """Opt-in experiment: before min-cut AC, downgrade eligible MUST_SAVE
    activations to PREFER_SAVE so min-cut can replace rigid saves with a new save
    set. When min-cut AC is enabled, the peak budget remains relative to the
    original memory-policy floor before this relaxation. Unsafe/uncuttable saves
    remain hard. Disabled by default to preserve the normal MUST_* contract."""

    ac_allow_allowed_saves: bool = False
    """Opt-in experiment: before min-cut AC, mark every eligible non-MUST forward
    CUDA activation as PREFER_SAVE so ac_min_cut_save_scope=all can search the
    broad activation pool. When min-cut AC is enabled, the peak budget remains
    relative to the original memory-policy floor before widening the pool. Hard
    MUST_* tags remain constraints. This pass only applies cheap semantic filters
    (collectives, RNG/nondeterministic/mutable ops, CPU/non-tensor values, and
    nodes without tensor metadata); min-cut candidate filtering later checks
    fresh storage ownership."""

    ac_allow_unsaveable_recomputes: bool = False
    """Opt-in experiment: before min-cut AC, mark deterministic unsaveable
    boundaries as PREFER_RECOMPUTE without widening the ordinary save-candidate
    pool. This is narrower than ac_allow_allowed_saves and is disabled by
    default."""

    ac_min_cut_decompose_log_softmax: bool = False
    """Opt-in experiment: decompose fp32 aten._log_softmax into row-stat pieces
    before min-cut so the pass can save compact stats instead of the full
    softmax output. Disabled by default because it changes the graph seen by
    Inductor and can affect fusion decisions."""

    ac_min_cut_memory_estimator: Literal["approximate", "exact"] = "approximate"
    """Debug/validation knob for min-cut AC candidate checking:
        approximate: use the memory curve for ordering/pruning, but
            exact-check every kept candidate with build_memory_profile on a
            throwaway remat, then fully recheck the final result. Accepted plans
            are guaranteed not to exceed the configured peak budget.
        exact: slow debugging mode for approximate; tentatively save candidates
            one at a time, remat, and measure build_memory_profile for each
            candidate without memory-curve pruning.
    """

    pass_pipeline: str = "default"
    """Pass pipeline selection. Controls which graph pass pipeline, post-init
    hooks, and pre-train-step hooks are activated."""

    inductor_compilation: Literal["regional", "full"] = "regional"
    """
    Inductor compilation strategy. Mutually exclusive options:
        regional: compile tagged regions (e.g. FlexAttention HOPs) with
            regional_inductor while leaving the rest interpreted.
        full: compile the entire graph with inductor into optimized
            Triton kernels. Provides better performance but may change
            bitwise numerics compared to regional/interpreted execution.
    """

    numerics_changing_optim: bool = False
    """Enable passes that improve performance but may change numerics
    compared to the uncompiled path (e.g. RMSNorm Inductor fusion)."""

    cpu_offload_prefetch_n_layers: int = 1
    """Prefetch reloads this many layers ahead in the backward graph
    to overlap H2D transfers with compute."""

    cpu_offload_defer_n_layers: int = 1
    """Defer forward wait_tensor ops this many layers past the last consumer
    to overlap D2H transfers with compute."""

    cpu_offload_budget_gb: float = 100.0
    """Maximum CPU memory budget (in GB per rank) for offloaded activations.
    Tensors are selected largest-first until the budget is exhausted."""

    enable_fsdp_ag_rs_overlap: bool = False
    """When True, run ``overlap_fsdp_ag_rs_pass``. The pass moves backward
    FSDP all-gathers onto a separate CUDA stream from reduce-scatters so the
    two collectives can overlap. It is a no-op when the graph contains no
    FSDP all-gathers."""

    precompile_artifact_dir: str = ""
    """
    Directory for precompiled artifacts. Setting this enables precompile:
    precompile_main.py saves the artifact here, and training loads it from
    here to skip compilation. For multi-node setups use a shared filesystem
    path.
    """

    enable_autoparallel: bool = False
    """Use AutoParallelGraph (ILP solver-based SPMD sharding) instead of
    manual TP/FSDP/EP. Forces the AOT compilation path internally."""


def validate_autoparallel_config(
    compile_config: GraphTrainerCompileConfig,
) -> None:
    if compile_config.enable_autoparallel and compile_config.mode != "aot_fx_trace":
        raise ValueError(
            "AutoParallel graph_trainer integration only supports "
            "--compile.mode aot_fx_trace"
        )


def to_graph_trainer_config(
    base_config: Trainer.Config,
    model_registry: Callable[[str], ModelSpec],
) -> "GraphTrainer.Config":
    """Convert a base Trainer.Config to a GraphTrainer.Config.

    Copies all fields from the base config and replaces the model_spec with one
    from the graph_trainer model_registry. The compile field is removed and
    left as the GraphTrainer.Config default; callers should explicitly set it.
    """
    from .cudagraph import cudagraph_annotate_trace_post_processor
    from .trainer import GraphTrainer

    d = {f.name: getattr(base_config, f.name) for f in fields(base_config)}
    graph_spec = model_registry(base_config.model_spec.flavor)
    # Wrap the base model config in the graph_trainer's model config class
    # (e.g. GraphTrainerQwen3Model.Config) while preserving all field values
    # (including moe_comm_backend etc.).
    graph_model_cls = type(graph_spec.model)
    graph_model = graph_model_cls(
        **{
            f.name: getattr(base_config.model_spec.model, f.name)
            for f in fields(base_config.model_spec.model)
        }
    )
    d["model_spec"] = replace(
        base_config.model_spec,
        parallelize_fn=graph_spec.parallelize_fn,
        model=graph_model,
    )
    d.pop("compile")

    # graph_trainer uses graph-based SAC instead of eager AC. Override any
    # enabled AC policy with the default selective one so callers don't need
    # per-config fixups.
    ac = d.get("activation_checkpoint")
    if ac is not None:
        d["activation_checkpoint"] = SelectiveAC.Config()

    # TODO: graph_trainer doesn't yet support ChunkedCELoss
    if isinstance(d.get("loss"), ChunkedCELoss.Config):
        d["loss"] = CrossEntropyLoss.Config()

    # Merge CUDA graph kernel annotations into profiler traces when profiling
    # is active.  No-op otherwise (and no-op when requirements aren't met).
    # It's also a no-op if there is CUDA graph is not enabled.
    profiler = d.get("profiler")
    if profiler is not None:
        profiler.trace_post_processor = cudagraph_annotate_trace_post_processor()

    return GraphTrainer.Config(**d)
