# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
from unittest.mock import patch

import torch
from torch._inductor.fx_passes.fuse_regions import (
    FUSE_REGION,
    apply_fuse_region_annotations,
)
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import TestCase

from torchtitan.experiments.graph_trainer.chunk_fuse_regions import (
    annotate_auto_chunk_fuse_regions_pass,
)
from torchtitan.experiments.graph_trainer.inductor_passes import (
    full_inductor_compilation_pass,
)
from torchtitan.experiments.graph_trainer.performance_passes import (
    annotate_rmsnorm_for_regional_inductor_pass,
)
from torchtitan.experiments.graph_trainer.subgraph_regions import (
    SUBGRAPH_REGION,
    apply_subgraph_region_annotations_pass,
    subgraph,
)


class TestAnnotateRMSNormForRegionalInductorPass(TestCase):
    """Unit tests for annotate_rmsnorm_for_regional_inductor_pass."""

    def _build_rmsnorm_gm(self, node_specs):
        """Build a GraphModule with fused RMSNorm ops and other ops.

        Args:
            node_specs: List of op targets. For ``_fused_rms_norm`` and
                ``_fused_rms_norm_backward`` nodes, ``getitem`` users are
                automatically appended (mirroring the real traced graph
                structure).
        """
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        w = graph.placeholder("w")
        last = x

        _FUSED_TARGETS = {
            torch.ops.aten._fused_rms_norm.default,
            torch.ops.aten._fused_rms_norm_backward.default,
        }

        for target in node_specs:
            if target in _FUSED_TARGETS:
                # Mimic the real graph: fused op returns a tuple,
                # followed by getitem nodes extracting elements.
                if target == torch.ops.aten._fused_rms_norm.default:
                    fused = graph.call_function(target, args=(last, [256], w, 1e-5))
                else:
                    fused = graph.call_function(
                        target, args=(last, w, last, [256], 1e-5)
                    )
                gi0 = graph.call_function(operator.getitem, args=(fused, 0))
                gi1 = graph.call_function(operator.getitem, args=(fused, 1))
                last = gi0
            else:
                last = graph.call_function(target, args=(last,))

        graph.output(last)
        return torch.fx.GraphModule(torch.nn.Module(), graph)

    def _count_tagged_nodes(self, gm):
        """Count nodes that have compile_with_inductor in their custom metadata."""
        count = 0
        for node in gm.graph.nodes:
            custom = node.meta.get("custom", {})
            if "compile_with_inductor" in custom:
                count += 1
        return count

    def test_tags_fused_rmsnorm_and_getitems(self):
        """_fused_rms_norm nodes and their getitem users are tagged."""
        gm = self._build_rmsnorm_gm(
            [
                torch.ops.aten._fused_rms_norm.default,
                torch.ops.aten.mul.Tensor,
                torch.ops.aten.add.Tensor,
            ]
        )

        annotate_rmsnorm_for_regional_inductor_pass(gm)

        # 1 fused node + 2 getitem users = 3 tagged nodes
        self.assertEqual(self._count_tagged_nodes(gm), 3)

    def test_does_not_tag_non_rmsnorm_nodes(self):
        """Nodes that are not _fused_rms_norm targets are not tagged."""
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        n1 = graph.call_function(torch.ops.aten.mul.Tensor, args=(x, x))
        n2 = graph.call_function(torch.ops.aten.add.Tensor, args=(n1, x))
        graph.output(n2)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        annotate_rmsnorm_for_regional_inductor_pass(gm)

        self.assertEqual(self._count_tagged_nodes(gm), 0)

    def test_fwd_and_bwd_both_tagged(self):
        """Forward and backward fused norms and their getitems are all tagged."""
        gm = self._build_rmsnorm_gm(
            [
                torch.ops.aten._fused_rms_norm.default,
                torch.ops.aten.mul.Tensor,
                torch.ops.aten._fused_rms_norm_backward.default,
            ]
        )

        annotate_rmsnorm_for_regional_inductor_pass(gm)

        # 2 fused nodes + 2*2 getitem users = 6 tagged nodes
        self.assertEqual(self._count_tagged_nodes(gm), 6)

    def test_custom_compile_config_propagated(self):
        """A custom compile config is wrapped under inductor_configs."""
        gm = self._build_rmsnorm_gm([torch.ops.aten._fused_rms_norm.default])

        config = {"max_autotune": True, "coordinate_descent_tuning": True}
        annotate_rmsnorm_for_regional_inductor_pass(gm, rmsnorm_compile_config=config)

        for node in gm.graph.nodes:
            annotation = node.meta.get("custom", {}).get("compile_with_inductor")
            if annotation is not None:
                self.assertEqual(annotation["inductor_configs"], config)


class TestFullInductorCompilationPass(TestCase):
    def test_preserves_fuse_region_metadata(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        add = graph.call_function(torch.ops.aten.add.Tensor, args=(x, x))
        graph.output(add)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        add.meta["custom"] = {FUSE_REGION: "source_region"}

        cudagraph_path = (
            "torchtitan.experiments.graph_trainer.cudagraph."
            "is_cudagraph_compatible"
        )
        regional_inductor_path = (
            "torchtitan.experiments.graph_trainer.inductor_passes."
            "regional_inductor_pass"
        )
        with (
            patch(cudagraph_path, return_value=True),
            patch(regional_inductor_path, side_effect=lambda gm, example_inputs: gm),
        ):
            result = full_inductor_compilation_pass(gm, ())

        annotation = add.meta["custom"]["compile_with_inductor"]
        self.assertEqual(add.meta["custom"][FUSE_REGION], "source_region")
        self.assertEqual(annotation[FUSE_REGION], "source_region")
        self.assertEqual(
            annotation["inductor_configs"],
            {"size_threshold_for_succ_based_strategy": 1},
        )
        self.assertTrue(result.meta["cudagraph_compatible"])


class TestSubgraphRegionAnnotationsPass(TestCase):
    def _set_tensor_meta(self, node, shape):
        node.meta["val"] = torch.empty(shape, device="meta")
        return node

    def _build_region_gm(self, key):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        add = graph.call_function(torch.ops.aten.add.Tensor, args=(x, 1))
        relu = graph.call_function(torch.ops.aten.relu.default, args=(add,))
        sin = graph.call_function(torch.ops.aten.sin.default, args=(relu,))
        graph.output(sin)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        for node in (x, add, relu, sin):
            self._set_tensor_meta(node, (4,))
        for node in (add, relu):
            node.meta["custom"] = {key: "chunk"}
        return gm

    def _invoke_subgraph_nodes(self, gm):
        return [
            node
            for node in gm.graph.nodes
            if node.op == "call_function"
            and node.target is torch.ops.higher_order.invoke_subgraph
        ]

    def _subgraph_placeholder_count(self, gm, invoke_node):
        subgraph_name = invoke_node.args[0].target
        submod = getattr(gm, subgraph_name)
        return len(submod.graph.find_nodes(op="placeholder"))

    def test_subgraph_region_outlines_without_fuse_region(self):
        gm = self._build_region_gm(SUBGRAPH_REGION)

        apply_subgraph_region_annotations_pass(gm, ())

        invoke_nodes = self._invoke_subgraph_nodes(gm)
        self.assertEqual(len(invoke_nodes), 1)
        self.assertEqual(invoke_nodes[0].meta[SUBGRAPH_REGION], "chunk_fwd")
        self.assertNotIn(FUSE_REGION, invoke_nodes[0].meta)
        self.assertNotIn(FUSE_REGION, invoke_nodes[0].meta.get("custom", {}))

    def test_subgraph_region_matches_fuse_region_boundary_shape(self):
        fuse_gm = self._build_region_gm(FUSE_REGION)
        subgraph_gm = self._build_region_gm(SUBGRAPH_REGION)

        apply_fuse_region_annotations(fuse_gm.graph)
        fuse_gm.recompile()
        apply_subgraph_region_annotations_pass(subgraph_gm, ())

        fuse_node = self._invoke_subgraph_nodes(fuse_gm)[0]
        subgraph_node = self._invoke_subgraph_nodes(subgraph_gm)[0]
        self.assertEqual(fuse_node.meta[FUSE_REGION], "chunk_fwd")
        self.assertEqual(subgraph_node.meta[SUBGRAPH_REGION], "chunk_fwd")
        self.assertEqual(len(fuse_node.args), len(subgraph_node.args))
        self.assertEqual(
            self._subgraph_placeholder_count(fuse_gm, fuse_node),
            self._subgraph_placeholder_count(subgraph_gm, subgraph_node),
        )

    def test_subgraph_context_manager_annotation_is_consumed(self):
        def fn(x):
            y = torch.sin(x)
            with subgraph("ctx"):
                z = torch.relu(y + 1)
            return z * 2

        with torch.fx.traceback.preserve_node_meta():
            gm = make_fx(fn)(torch.randn(4))

        apply_subgraph_region_annotations_pass(gm, ())

        invoke_nodes = self._invoke_subgraph_nodes(gm)
        self.assertEqual(len(invoke_nodes), 1)
        self.assertEqual(invoke_nodes[0].meta[SUBGRAPH_REGION], "ctx_fwd")


class _ChunkedGmMixin:
    def _set_tensor_meta(self, node, shape, *, dtype=torch.float32):
        node.meta["val"] = torch.empty(shape, dtype=dtype, device="meta")
        return node

    def _set_body_meta(self, node, shape, source, *, dtype=torch.float32):
        self._set_tensor_meta(node, shape, dtype=dtype)
        node.meta["stack_trace"] = (
            f'  File "/model.py", line {source}, in chunk_body\n'
        )
        return node

    def _build_chunked_gm(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        w = graph.placeholder("w")
        self._set_tensor_meta(x, (8, 4))
        self._set_tensor_meta(w, (4, 16))

        chunk_outputs = []
        chunk_nodes = []
        for start in (0, 4):
            chunk = graph.call_function(
                torch.ops.aten.slice.Tensor, args=(x, 0, start, start + 4)
            )
            self._set_tensor_meta(chunk, (4, 4))
            mm = graph.call_function(torch.ops.aten.mm.default, args=(chunk, w))
            relu = graph.call_function(torch.ops.aten.relu.default, args=(mm,))
            add = graph.call_function(torch.ops.aten.add.Tensor, args=(relu, relu))
            total = graph.call_function(torch.ops.aten.sum.default, args=(add,))
            self._set_body_meta(mm, (4, 16), 10)
            self._set_body_meta(relu, (4, 16), 11)
            self._set_body_meta(add, (4, 16), 12)
            self._set_body_meta(total, (), 13)
            chunk_outputs.append(total)
            chunk_nodes.append((chunk, mm, relu, add, total))

        out = graph.call_function(torch.ops.aten.add.Tensor, args=tuple(chunk_outputs))
        self._set_tensor_meta(out, ())
        graph.output(out)
        return torch.fx.GraphModule(torch.nn.Module(), graph), chunk_nodes


class TestAnnotateAutoChunkFuseRegionsPass(_ChunkedGmMixin, TestCase):
    def test_annotates_repeated_chunk_bodies(self):
        gm, chunk_nodes = self._build_chunked_gm()

        annotate_auto_chunk_fuse_regions_pass(gm, (), min_tensor_bytes=64)

        regions = []
        for chunk, mm, relu, add, total in chunk_nodes:
            self.assertNotIn(FUSE_REGION, chunk.meta.get("custom", {}))
            region = mm.meta["custom"][FUSE_REGION]
            self.assertEqual(relu.meta["custom"][FUSE_REGION], region)
            self.assertEqual(add.meta["custom"][FUSE_REGION], region)
            self.assertNotIn(FUSE_REGION, total.meta.get("custom", {}))
            regions.append(region)
        self.assertNotEqual(regions[0], regions[1])

    def test_does_not_annotate_repeated_non_chunked_bodies(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        w = graph.placeholder("w")
        self._set_tensor_meta(x, (4, 4))
        self._set_tensor_meta(w, (4, 16))

        outputs = []
        for _ in range(2):
            mm = graph.call_function(torch.ops.aten.mm.default, args=(x, w))
            relu = graph.call_function(torch.ops.aten.relu.default, args=(mm,))
            self._set_body_meta(mm, (4, 16), 10)
            self._set_body_meta(relu, (4, 16), 11)
            outputs.append(relu)
        out = graph.call_function(torch.ops.aten.add.Tensor, args=tuple(outputs))
        self._set_tensor_meta(out, (4, 16))
        graph.output(out)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        annotate_auto_chunk_fuse_regions_pass(gm, (), min_tensor_bytes=1)

        for node in gm.graph.nodes:
            self.assertNotIn(FUSE_REGION, node.meta.get("custom", {}))


class TestSubgraphRegionsForChunking(_ChunkedGmMixin, TestCase):
    def _invoke_subgraph_nodes(self, gm):
        return [
            node
            for node in gm.graph.nodes
            if node.op == "call_function"
            and node.target is torch.ops.higher_order.invoke_subgraph
        ]

    def test_explicit_subgraph_regions_outline_chunk_bodies(self):
        gm, chunk_nodes = self._build_chunked_gm()
        for idx, (_, mm, relu, add, total) in enumerate(chunk_nodes):
            for node in (mm, relu, add, total):
                node.meta["custom"] = {SUBGRAPH_REGION: f"chunk_{idx}"}

        apply_subgraph_region_annotations_pass(gm, ())

        invoke_nodes = self._invoke_subgraph_nodes(gm)
        self.assertEqual(len(invoke_nodes), 2)
        self.assertEqual(
            [node.meta[SUBGRAPH_REGION] for node in invoke_nodes],
            ["chunk_0_fwd", "chunk_1_fwd"],
        )
        for node in invoke_nodes:
            self.assertNotIn(FUSE_REGION, node.meta)

    def test_auto_chunk_fuse_regions_skip_explicit_subgraphs(self):
        gm, chunk_nodes = self._build_chunked_gm()
        for idx, (_, mm, relu, add, total) in enumerate(chunk_nodes):
            for node in (mm, relu, add, total):
                node.meta["custom"] = {SUBGRAPH_REGION: f"chunk_{idx}"}

        apply_subgraph_region_annotations_pass(gm, ())
        annotate_auto_chunk_fuse_regions_pass(gm, (), min_tensor_bytes=64)

        for module in gm.modules():
            if not isinstance(module, torch.fx.GraphModule):
                continue
            for node in module.graph.nodes:
                self.assertNotIn(FUSE_REGION, node.meta.get("custom", {}))


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
