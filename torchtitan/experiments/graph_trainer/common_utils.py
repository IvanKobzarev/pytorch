# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time
from collections.abc import Callable
from contextlib import contextmanager

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor, Replicate
from torch.fx.traceback import annotate_fn
from torch.utils._pytree import register_constant, register_pytree_node, tree_map

from torchtitan.config import TORCH_DTYPE_MAP, TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.context_parallel import apply_cp_to_forward
from torchtitan.experiments.graph_trainer.simple_fsdp import (
    data_parallel,
    MixedPrecisionPolicy,
)
from torchtitan.tools.logging import logger


@contextmanager
def log_timer(label: str):
    start = time.perf_counter()
    yield
    elapsed_s = time.perf_counter() - start
    logger.info("%s took %.3fs", label, elapsed_s)


def build_decoder_config_for_backend(
    config_builder: Callable, attn_backend: str, **builder_kwargs
):
    """Build a Decoder model config for ``attn_backend``, allowing test-only SDPA.

    ``SDPA`` is not a valid production language-model backend — ``get_attention_config``
    rejects it because the dataloaders always emit per-document positions and SDPA
    cannot consume them (it only has a boolean ``is_causal``). The graph_trainer
    tests, however, use SDPA to exercise *backend-agnostic* graph machinery
    (precompile-artifact serialization, custom codegen, context parallel, bitwise
    determinism) without FlexAttention's ``BlockMask``, which is unpicklable (its
    ``mask_mod`` closures are Python code objects), is not a tensor (so it breaks
    pipeline-parallel split-backward, which calls ``.requires_grad`` on every stage
    input), and overflows the fp32 Triton shared-memory limit on large head dims.

    For SDPA we build the flex config (a valid backend) and swap each layer's
    ``inner_attention`` to ``ScaledDotProductAttention.Config()``. Production code
    never reaches this path: ``get_attention_config`` still rejects ``sdpa``, so no
    model registry can construct an SDPA language model outside these tests.
    """
    if attn_backend != "sdpa":
        return config_builder(attn_backend=attn_backend, **builder_kwargs)

    from torchtitan.models.common.attention import ScaledDotProductAttention

    config = config_builder(attn_backend="flex", **builder_kwargs)
    for layer in config.layers:
        layer.attention.inner_attention = ScaledDotProductAttention.Config()
    return config


_MODULE_FQN = "module_fqn"
_NOT_IN_LAYERS = -1


def _is_backward_node(node: torch.fx.Node) -> bool:
    return node.meta.get("autograd_backward", False)


def _get_layer_id(node: torch.fx.Node, layer_prefix: str = "layers") -> int:
    """Extract the layer index from the node's module_fqn metadata.

    Nodes under ``<layer_prefix>.<N>`` return ``N``.
    All other nodes (tok_embeddings, norm, output) return ``_NOT_IN_LAYERS``.
    ``layer_prefix`` is the layer container's attribute name.
    """
    fqn = node.meta.get("custom", {}).get(_MODULE_FQN, "")
    parts = fqn.split(".")
    if len(parts) >= 2 and parts[0] == layer_prefix:
        try:
            return int(parts[1])
        except ValueError:
            pass
    return _NOT_IN_LAYERS


def apply_save_layer_inputs_ac(
    gm: torch.fx.GraphModule,
    *,
    layer_prefix: str = "layers",
    save_final_layer_output: bool = True,
) -> torch.fx.GraphModule:
    """Tag the joint graph with the "save block input" AC policy.

    Pass 1: tag every forward node inside a transformer layer PREFER_RECOMPUTE.
    Pass 2: upgrade to MUST_SAVE any layer node whose output is consumed by a
    later layer -- that crossing tensor is the next block's input, so saving it
    avoids recomputing the whole preceding layer. The final layer output can
    optionally stay recomputable to avoid pinning both it and the post-layer
    head activation at the peak.
    Nodes outside layers (embeddings, final norm, loss head) are left untagged
    and are therefore saved.

    This is the tag-based equivalent of eager full per-layer activation
    checkpointing (``checkpoint`` around each block): save the layer boundary,
    recompute the interior. Expressed as FX ``node.meta["recompute"]`` tags so
    that ``remat_using_tags_for_fwd_loss_bwd_graph`` materializes the recompute
    and ``min_cut_ac_pass`` can refine it. Requires ``annotate_module_fqns`` to
    have run on the model before tracing.

    Reusable across repos: operates on any forward-loss-backward joint fx graph
    whose backward nodes carry ``node.meta["autograd_backward"]``.
    """
    from torch.utils.checkpoint import CheckpointPolicy

    def _is_recomputable(n: torch.fx.Node) -> bool:
        return n.meta.get("recompute") in (
            CheckpointPolicy.PREFER_RECOMPUTE,
            CheckpointPolicy.MUST_RECOMPUTE,
        )

    def _is_rng(n: torch.fx.Node) -> bool:
        # RNG / nondeterministic ops (e.g. dropout) cannot live in a recompute
        # region: remat refuses to duplicate them (has_recomputable_rng_ops).
        return (
            hasattr(n.target, "tags")
            and torch.Tag.nondeterministic_seeded in n.target.tags
        )

    # Pass 1: tag in-layer forward nodes PREFER_RECOMPUTE (but keep RNG ops
    # MUST_SAVE so their recompute-needing consumers read the saved value).
    n_recompute = 0
    for node in gm.graph.nodes:
        if node.op != "call_function" or _is_backward_node(node):
            continue
        if _get_layer_id(node, layer_prefix) == _NOT_IN_LAYERS:
            continue
        if _is_rng(node):
            node.meta["recompute"] = CheckpointPolicy.MUST_SAVE
            continue
        node.meta["recompute"] = CheckpointPolicy.PREFER_RECOMPUTE
        n_recompute += 1

    if n_recompute == 0:
        logger.warning(
            "apply_save_layer_inputs_ac: no nodes under '%s.<N>' -- did "
            "annotate_module_fqns run before tracing? (no-op)",
            layer_prefix,
        )
        return gm

    # Pass 2: force MUST_SAVE on layer-boundary (residual stream) tensors.
    n_save = 0
    for node in gm.graph.nodes:
        if _is_backward_node(node) or not _is_recomputable(node):
            continue
        layer_id = _get_layer_id(node, layer_prefix)
        if layer_id == _NOT_IN_LAYERS:
            continue
        for user in node.users:
            if _is_backward_node(user):
                continue
            user_layer_id = _get_layer_id(user, layer_prefix)
            if user_layer_id != layer_id and (
                save_final_layer_output or user_layer_id != _NOT_IN_LAYERS
            ):
                node.meta["recompute"] = CheckpointPolicy.MUST_SAVE
                n_save += 1
                break

    logger.info(
        "apply_save_layer_inputs_ac: %d MUST_SAVE (layer boundaries), "
        "%d PREFER_RECOMPUTE (layer interiors)",
        n_save,
        n_recompute - n_save,
    )
    return gm


def annotate_module_fqns(model: nn.Module) -> None:
    """Annotate all modules' forward with their fully-qualified names.

    Every named submodule (excluding the root) gets its forward method wrapped
    with ``annotate_fn`` so that FX nodes carry ``module_fqn`` in
    ``node.meta["custom"]``.

    Call once after model construction, before tracing/compilation.
    """
    for fqn, submodule in model.named_modules():
        if fqn:  # skip root module
            submodule.forward = annotate_fn({_MODULE_FQN: fqn})(submodule.forward)


def parallelize_inputs(parallel_dims, args, kwargs):
    if not parallel_dims.tp_enabled:
        return args, kwargs

    def to_dtensor(tensor):
        if isinstance(tensor, torch.Tensor):
            return DTensor.from_local(
                tensor, parallel_dims.get_mesh("tp"), [Replicate()]
            )
        return tensor

    dt_args = tree_map(to_dtensor, args)

    # TODO: When using flex_attention, BlockMask would show up in kwargs,
    # and it's unclear how to convert it to DTensor. If I use to_dtensor,
    # it would fail with Dynamo Error: P2011360347
    # dt_kwargs = tree_map(to_dtensor, kwargs)

    dt_kwargs = kwargs

    return dt_args, dt_kwargs


def register_blockmask_pytree_node():
    from torch.nn.attention.flex_attention import BlockMask

    if BlockMask not in torch.utils._pytree.SUPPORTED_NODES:
        register_pytree_node(
            BlockMask,
            BlockMask._flatten,
            BlockMask._unflatten,
            flatten_with_keys_fn=BlockMask._flatten_with_keys,
            serialized_type_name="torch.nn.attention.flex_attention.BlockMask",
        )


def maybe_register_blockmask_pytree_node() -> None:
    """Register flex-attention pytree helpers if they are missing."""
    from torch.nn.attention.flex_attention import _MaskModWrapper, BlockMask

    if BlockMask not in torch.utils._pytree.SUPPORTED_NODES:
        register_blockmask_pytree_node()
    if _MaskModWrapper not in torch.utils._pytree.SUPPORTED_NODES:
        register_constant(_MaskModWrapper)


def end_with_pass(passes: list[Callable], names: list[str]) -> bool:
    return (
        len(passes) > 0
        and (last_pass_name := getattr(passes[-1], "__name__", None))
        and (last_pass_name in names)
    )


def get_default_transformer_block_buckets(
    n_layers: int,
) -> list[list[str] | str]:
    """Get default transformer block buckets for manual bucketing passes.

    Assumes the standard Decoder layout: tok_embeddings, layers.0..N-1,
    norm, and output (e.g., Llama3, DeepSeekV3, Qwen3).
    """
    return [
        "tok_embeddings",
        *[f"layers.{i}" for i in range(n_layers)],
        ["norm", "lm_head"],
    ]


def get_transformer_block_buckets(model) -> list[list[str] | str]:
    """Get transformer block buckets for manual bucketing passes.

    Works for any model with tok_embeddings, layers (OrderedDict), norm, and output
    attributes (e.g., Llama3, DeepSeekV3).
    """
    # [TODO](ruisizhang123) add EP support for transformer block bucketing
    module_list = [
        model.tok_embeddings,
        [model.norm, model.lm_head],
    ]
    for layer_id, transformer_block in model.layers.items():
        module_list.append(transformer_block)

    def convert_modules_to_fqns(modules, module_to_fqn_mapping):
        """Convert a (possibly nested) list of modules to FQN strings."""
        result = []
        for m in modules:
            if isinstance(m, list):
                if fqn_list := convert_modules_to_fqns(m, module_to_fqn_mapping):
                    result.append(fqn_list)
            else:
                if fqn := module_to_fqn_mapping.get(m):
                    result.append(fqn)
        return result

    module_to_name = {m: n for n, m in model.named_modules()}
    module_fqns = convert_modules_to_fqns(module_list, module_to_name)
    return module_fqns


def apply_cp_to_attention(
    model: nn.Module,
    parallel_dims: ParallelDims,
) -> None:
    """Wrap each layer's inner attention with CP logic."""
    attention_modules = [
        # pyrefly: ignore [missing-attribute]
        block.attention.inner_attention
        for block in model.layers.values()
    ]
    apply_cp_to_forward(attention_modules, parallel_dims.get_mesh("cp"))


def apply_simple_fsdp(
    model: nn.Module,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
) -> nn.Module:
    """Wrap the model (and any MoE experts) with graph_trainer's simple_fsdp.

    For MoE-enabled models, ``moe.experts`` submodules are separately wrapped
    on the EDP mesh when expert parallelism is enabled.
    """
    if parallel_dims.dp_replicate_enabled:
        if parallel_dims.dp_shard_enabled or parallel_dims.cp_enabled:
            dp_mesh_dim_names = ["dp_replicate", "fsdp"]
            dp_mode = "hybrid_shard"
        else:
            dp_mesh_dim_names = ["dp_replicate"]
            dp_mode = "replicate"
    else:
        dp_mesh_dim_names = ["fsdp"]
        dp_mode = "fully_shard"

    dp_mesh = parallel_dims.get_mesh(dp_mesh_dim_names)
    mp_policy = MixedPrecisionPolicy(
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
    )

    if parallel_dims.ep_enabled and hasattr(model, "layers"):
        edp_mesh_names = (
            ["dp_replicate", "efsdp"]
            if parallel_dims.dp_replicate_enabled
            else ["efsdp"]
        )
        edp_mesh = parallel_dims.get_optional_mesh(edp_mesh_names)
        assert edp_mesh is not None

        for _, transformer_block in model.layers.items():
            if not getattr(transformer_block, "moe_enabled", False):
                continue
            assert hasattr(transformer_block, "moe")
            experts_shard_dim = 0
            if (
                edp_mesh["efsdp"].size() * parallel_dims.ep
                > transformer_block.moe.experts.num_experts
            ):
                experts_shard_dim = 1

            transformer_block.moe.experts = data_parallel(
                transformer_block.moe.experts,
                edp_mesh,
                dp_mode,
                mp_policy=mp_policy,
                shard_dim=experts_shard_dim,
            )

    model = data_parallel(
        model,
        dp_mesh,
        dp_mode,
        mp_policy=mp_policy,
    )
    logger.info(
        "Applied Data Parallel (simple_fsdp) (dp mode=%s) to the model", dp_mode
    )
    return model
