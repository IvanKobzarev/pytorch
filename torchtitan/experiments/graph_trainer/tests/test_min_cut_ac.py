# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
from unittest.mock import patch

import torch
import torch.fx as fx
from torch._functorch._activation_checkpointing.remat_using_tags_for_fwd_loss_bwd_graph_pass import (  # noqa: E501
    remat_using_tags_for_fwd_loss_bwd_graph as _torch_remat_using_tags_for_fwd_loss_bwd_graph,  # noqa: E501
)
from torch._inductor.fx_passes.memory_estimator import build_memory_profile
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils.checkpoint import CheckpointPolicy

from torchtitan.experiments.graph_trainer.common_utils import (
    _MODULE_FQN,
    apply_save_layer_inputs_ac,
)
from torchtitan.experiments.graph_trainer.memory_utils import (
    _build_memory_curve,
    _candidate_memory_curve_delta,
    _delta_fits,
    _is_releasable,
    _memory_profile_after_remat,
    _nodes_with_fresh_cuda_storage,
    _peak_after_remat,
    _recompute_nodes_closure_and_uses,
    _refresh_fake_tensor_meta,
    remat_using_tags_for_fwd_loss_bwd_graph,
)
from torchtitan.experiments.graph_trainer.min_cut_ac import (
    _budgeted_save_candidate_phases,
    _BudgetedSaveSelectionResult,
    _candidate_peak_fits_or_makes_progress,
    _decompose_log_softmax_for_min_cut,
    _GB,
    _min_cut,
    _optimize_under_peak_budget,
    _save_candidates,
    _select_saves_under_peak_budget,
    _set_mandatory_must_saves,
    _should_rank_by_peak_progress,
    ac_allow_allowed_saves,
    ac_allow_unsaveable_recomputes,
    ac_relax_relaxable_must_saves,
    min_cut_ac_pass,
)
from torchtitan.experiments.graph_trainer.selective_activation_remat import (
    selective_activation_remat_pass,
)


_NO_TAG = object()


def fake_c10d_functional_wait(x):
    return x


class TestMinCutACPeakModel(TestCase):
    """``_refresh_fake_tensor_meta`` retraces the rematerialized recompute chain so
    ``build_memory_profile`` keys liveness on the right storage: a recomputed
    compute op is a real new allocation (fresh storage), a recomputed view aliases
    its recomputed parent (no new storage). These tests pin both -- a naive per-dup
    clone (which the retrace replaced) gets the view case wrong: it gives the view
    fresh storage and double-counts its bytes."""

    def _make_remat_graph(self) -> tuple[fx.GraphModule, tuple[torch.Tensor, ...]]:
        # x -> sin -> cos -> {sum=loss, neg=bwd}; sin and cos recomputed. Two
        # compute dups, each a fresh allocation in the backward.
        graph = fx.Graph()
        x = graph.placeholder("x")
        a = graph.call_function(torch.ops.aten.sin.default, args=(x,))
        b = graph.call_function(torch.ops.aten.cos.default, args=(a,))
        loss = graph.call_function(torch.ops.aten.sum.default, args=(b,))
        bwd = graph.call_function(torch.ops.aten.neg.default, args=(b,))
        graph.output((loss, bwd, b))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        with FakeTensorMode() as fake_mode:
            fake_x = torch.empty(1024, device="cuda")
            FakeTensorProp(gm, mode=fake_mode).propagate_dont_convert_inputs(fake_x)

        a.meta["recompute"] = CheckpointPolicy.PREFER_RECOMPUTE
        b.meta["recompute"] = CheckpointPolicy.PREFER_RECOMPUTE
        bwd.meta["autograd_backward"] = True
        return gm, (fake_x,)

    def _make_remat_graph_with_view(
        self,
    ) -> tuple[fx.GraphModule, tuple[torch.Tensor, ...]]:
        # x -> sin (compute) -> t (transpose, a view of sin) -> {sum=loss,
        # neg=bwd}; sin and t recomputed. The view dup must ALIAS its recomputed
        # parent; only the compute dup is a fresh allocation.
        graph = fx.Graph()
        x = graph.placeholder("x")
        a = graph.call_function(torch.ops.aten.sin.default, args=(x,))
        v = graph.call_function(torch.ops.aten.t.default, args=(a,))
        loss = graph.call_function(torch.ops.aten.sum.default, args=(v,))
        bwd = graph.call_function(torch.ops.aten.neg.default, args=(v,))
        graph.output((loss, bwd))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        with FakeTensorMode() as fake_mode:
            fake_x = torch.empty(512, 512, device="cuda")
            FakeTensorProp(gm, mode=fake_mode).propagate_dont_convert_inputs(fake_x)

        a.meta["recompute"] = CheckpointPolicy.PREFER_RECOMPUTE
        v.meta["recompute"] = CheckpointPolicy.PREFER_RECOMPUTE
        bwd.meta["autograd_backward"] = True
        return gm, (fake_x,)

    def _make_save_candidate_graph(
        self,
    ) -> tuple[fx.GraphModule, tuple[torch.Tensor, ...]]:
        # x -> sin (large soft save candidate) -> sum(dim=1) (small activation)
        # -> bwd. min-cut can choose the smaller sum output instead of sin.
        graph = fx.Graph()
        x = graph.placeholder("x")
        a = graph.call_function(torch.ops.aten.sin.default, args=(x,))
        b = graph.call_function(torch.ops.aten.sum.dim_IntList, args=(a, [1], False))
        loss = graph.call_function(torch.ops.aten.sum.default, args=(b,))
        bwd = graph.call_function(torch.ops.aten.neg.default, args=(b,))
        graph.output((loss, bwd))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        with FakeTensorMode() as fake_mode:
            fake_x = torch.empty(512, 512, device="cuda")
            FakeTensorProp(gm, mode=fake_mode).propagate_dont_convert_inputs(fake_x)

        a.meta["recompute"] = CheckpointPolicy.PREFER_SAVE
        b.meta["recompute"] = CheckpointPolicy.PREFER_RECOMPUTE
        bwd.meta["autograd_backward"] = True
        return gm, (fake_x,)

    def _make_save_candidate_contract_graph(
        self,
    ) -> tuple[fx.GraphModule, tuple[torch.Tensor, ...]]:
        # x -> A(large save candidate) -> B(small) -> C(large candidate)
        # -> D -> bwd.
        # A and C are PREFER_SAVE, so they are soft candidates, not concrete
        # saves. The selector should still consider both PREFER_SAVE and
        # PREFER_RECOMPUTE nodes and prefer the most valuable cutpoints.
        graph = fx.Graph()
        x = graph.placeholder("x")
        a = graph.call_function(torch.ops.aten.sin.default, args=(x,))
        b = graph.call_function(torch.ops.aten.sum.dim_IntList, args=(a, [1], False))
        v = graph.call_function(torch.ops.aten.expand.default, args=(b, [512, 512]))
        c = graph.call_function(torch.ops.aten.clone.default, args=(v,))
        d = graph.call_function(torch.ops.aten.cos.default, args=(c,))
        loss = graph.call_function(torch.ops.aten.sum.default, args=(d,))
        bwd = graph.call_function(torch.ops.aten.neg.default, args=(d,))
        graph.output((loss, bwd))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        with FakeTensorMode() as fake_mode:
            fake_x = torch.empty(512, 512, device="cuda")
            FakeTensorProp(gm, mode=fake_mode).propagate_dont_convert_inputs(fake_x)

        a.meta["recompute"] = CheckpointPolicy.PREFER_SAVE
        b.meta["recompute"] = CheckpointPolicy.PREFER_RECOMPUTE
        v.meta["recompute"] = CheckpointPolicy.PREFER_RECOMPUTE
        c.meta["recompute"] = CheckpointPolicy.PREFER_SAVE
        d.meta["recompute"] = CheckpointPolicy.PREFER_RECOMPUTE
        bwd.meta["autograd_backward"] = True
        return gm, (fake_x,)

    def _make_interior_cutpoint_graph(
        self,
    ) -> tuple[fx.GraphModule, tuple[torch.Tensor, ...]]:
        # x -> A(saved boundary) -> B(small) -> C(large) -> bwd. B is the min-cut
        # frontier, but its fresh storage is not directly consumed by the original
        # backward node. min_cut mode must preserve this interior frontier
        # candidate.
        graph = fx.Graph()
        x = graph.placeholder("x")
        a = graph.call_function(torch.ops.aten.sin.default, args=(x,))
        b = graph.call_function(torch.ops.aten.sum.dim_IntList, args=(a, [1], False))
        v = graph.call_function(torch.ops.aten.expand.default, args=(b, [512, 512]))
        c = graph.call_function(torch.ops.aten.clone.default, args=(v,))
        loss = graph.call_function(torch.ops.aten.sum.default, args=(c,))
        bwd = graph.call_function(torch.ops.aten.neg.default, args=(c,))
        graph.output((loss, bwd))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        with FakeTensorMode() as fake_mode:
            fake_x = torch.empty(512, 512, device="cuda")
            FakeTensorProp(gm, mode=fake_mode).propagate_dont_convert_inputs(fake_x)

        a.meta["recompute"] = CheckpointPolicy.MUST_SAVE
        for node in (b, v, c):
            node.meta["recompute"] = CheckpointPolicy.PREFER_RECOMPUTE
        bwd.meta["autograd_backward"] = True
        return gm, (fake_x,)

    def _make_all_scope_graph(
        self,
    ) -> tuple[fx.GraphModule, tuple[torch.Tensor, ...], fx.Node, fx.Node]:
        # x -> A(saved boundary) -> B(small) -> C(medium) -> D(large) -> bwd.
        # min_cut only sees B as the smallest structural cut. all may choose C if
        # its saved bytes buy a better recompute-closure reduction.
        graph = fx.Graph()
        x = graph.placeholder("x")
        a = graph.call_function(torch.ops.aten.sin.default, args=(x,))
        b = graph.call_function(torch.ops.aten.sum.dim_IntList, args=(a, [1], False))
        c = graph.call_function(torch.ops.aten.cat.default, args=([b, b], 0))
        v = graph.call_function(torch.ops.aten.unsqueeze.default, args=(c, 1))
        e = graph.call_function(torch.ops.aten.expand.default, args=(v, [1024, 512]))
        d = graph.call_function(torch.ops.aten.clone.default, args=(e,))
        loss = graph.call_function(torch.ops.aten.sum.default, args=(d,))
        bwd = graph.call_function(torch.ops.aten.neg.default, args=(d,))
        graph.output((loss, bwd))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        with FakeTensorMode() as fake_mode:
            fake_x = torch.empty(512, 512, device="cuda")
            FakeTensorProp(gm, mode=fake_mode).propagate_dont_convert_inputs(fake_x)

        a.meta["recompute"] = CheckpointPolicy.MUST_SAVE
        for node in (b, c, v, e, d):
            node.meta["recompute"] = CheckpointPolicy.PREFER_RECOMPUTE
        bwd.meta["autograd_backward"] = True
        return gm, (fake_x,), b, c

    def _make_rng_graph(
        self,
        policy: CheckpointPolicy,
    ) -> tuple[fx.GraphModule, tuple[torch.Tensor, ...], fx.Node]:
        graph = fx.Graph()
        x = graph.placeholder("x")
        r = graph.call_function(torch.ops.aten.bernoulli.default, args=(x,))
        loss = graph.call_function(torch.ops.aten.sum.default, args=(r,))
        bwd = graph.call_function(torch.ops.aten.neg.default, args=(r,))
        graph.output((loss, bwd))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        with FakeTensorMode() as fake_mode:
            fake_x = torch.empty(32, device="cuda")
            FakeTensorProp(gm, mode=fake_mode).propagate_dont_convert_inputs(fake_x)

        r.meta["recompute"] = policy
        bwd.meta["autograd_backward"] = True
        return gm, (fake_x,), r

    def _make_collective_closure_graph(
        self,
    ) -> tuple[fx.GraphModule, tuple[torch.Tensor, ...], fx.Node]:
        graph = fx.Graph()
        x = graph.placeholder("x")
        c = graph.call_function(fake_c10d_functional_wait, args=(x,))
        y = graph.call_function(torch.ops.aten.sin.default, args=(c,))
        loss = graph.call_function(torch.ops.aten.sum.default, args=(y,))
        bwd = graph.call_function(torch.ops.aten.neg.default, args=(y,))
        graph.output((loss, bwd))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        with FakeTensorMode() as fake_mode:
            fake_x = torch.empty(32, device="cuda")
            FakeTensorProp(gm, mode=fake_mode).propagate_dont_convert_inputs(fake_x)

        c.meta["recompute"] = CheckpointPolicy.PREFER_RECOMPUTE
        y.meta["recompute"] = CheckpointPolicy.PREFER_RECOMPUTE
        bwd.meta["autograd_backward"] = True
        return gm, (fake_x,), y

    def _make_saved_uncuttable_plus_finite_frontier_graph(
        self,
    ) -> tuple[fx.GraphModule, tuple[torch.Tensor, ...], fx.Node]:
        graph = fx.Graph()
        x = graph.placeholder("x")
        saved_rng = graph.call_function(torch.ops.aten.bernoulli.default, args=(x,))
        rng_bwd = graph.call_function(torch.ops.aten.neg.default, args=(saved_rng,))
        a = graph.call_function(torch.ops.aten.sin.default, args=(x,))
        b = graph.call_function(torch.ops.aten.sum.dim_IntList, args=(a, [1], False))
        v = graph.call_function(torch.ops.aten.expand.default, args=(b, [512, 512]))
        c = graph.call_function(torch.ops.aten.clone.default, args=(v,))
        loss = graph.call_function(torch.ops.aten.sum.default, args=(c,))
        bwd = graph.call_function(torch.ops.aten.neg.default, args=(c,))
        graph.output((loss, rng_bwd, bwd))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        with FakeTensorMode() as fake_mode:
            fake_x = torch.empty(512, 512, device="cuda")
            FakeTensorProp(gm, mode=fake_mode).propagate_dont_convert_inputs(fake_x)

        saved_rng.meta["recompute"] = CheckpointPolicy.MUST_SAVE
        a.meta["recompute"] = CheckpointPolicy.MUST_SAVE
        for node in (b, v, c):
            node.meta["recompute"] = CheckpointPolicy.PREFER_RECOMPUTE
        rng_bwd.meta["autograd_backward"] = True
        bwd.meta["autograd_backward"] = True
        return gm, (fake_x,), b

    def _make_fanout_curve_graph(self) -> tuple[fx.GraphModule, dict[str, fx.Node]]:
        graph = fx.Graph()
        x = graph.placeholder("x")
        a = graph.call_function(torch.ops.aten.sin.default, args=(x,))
        b = graph.call_function(torch.ops.aten.cos.default, args=(a,))
        c = graph.call_function(torch.ops.aten.relu.default, args=(a,))
        d = graph.call_function(torch.ops.aten.add.Tensor, args=(b, c))
        loss = graph.call_function(torch.ops.aten.sum.default, args=(d,))
        bwd = graph.call_function(torch.ops.aten.neg.default, args=(d,))
        graph.output((loss, bwd))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        with FakeTensorMode() as fake_mode:
            fake_x = torch.empty(256, 256, device="cuda")
            FakeTensorProp(gm, mode=fake_mode).propagate_dont_convert_inputs(fake_x)

        for node in (a, b, c, d):
            node.meta["recompute"] = CheckpointPolicy.PREFER_RECOMPUTE
        bwd.meta["autograd_backward"] = True
        return gm, {"sin": a, "cos": b, "add": d}

    def _make_chain_curve_graph(self) -> tuple[fx.GraphModule, dict[str, fx.Node]]:
        graph = fx.Graph()
        x = graph.placeholder("x")
        a = graph.call_function(torch.ops.aten.sin.default, args=(x,))
        b = graph.call_function(torch.ops.aten.cos.default, args=(a,))
        c = graph.call_function(torch.ops.aten.relu.default, args=(b,))
        loss = graph.call_function(torch.ops.aten.sum.default, args=(c,))
        bwd = graph.call_function(torch.ops.aten.neg.default, args=(c,))
        graph.output((loss, bwd))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        with FakeTensorMode() as fake_mode:
            fake_x = torch.empty(256, 256, device="cuda")
            FakeTensorProp(gm, mode=fake_mode).propagate_dont_convert_inputs(fake_x)

        for node in (a, b, c):
            node.meta["recompute"] = CheckpointPolicy.PREFER_RECOMPUTE
        bwd.meta["autograd_backward"] = True
        return gm, {"sin": a, "cos": b, "relu": c}

    def _tag_snapshot(self, gm: fx.GraphModule) -> dict[fx.Node, object]:
        return {n: n.meta.get("recompute", _NO_TAG) for n in gm.graph.nodes}

    def _storage_id(self, node: fx.Node) -> int:
        return node.meta["val"].untyped_storage()._cdata

    def _find(self, gm: fx.GraphModule, target, recomputed: bool) -> fx.Node:
        node = next(
            (
                n
                for n in gm.graph.nodes
                if n.target is target and n.name.endswith("_recomputed") == recomputed
            ),
            None,
        )
        self.assertIsNotNone(node, f"{target} recomputed={recomputed} not in graph")
        return node

    def _curve_and_exact_candidate_peak(
        self,
        gm: fx.GraphModule,
        candidate: fx.Node,
    ) -> tuple[float, float, int, int, bool]:
        exact_baseline_peak = _peak_after_remat(gm)
        memory_curve = _build_memory_curve(
            gm,
            exact_current_peak=exact_baseline_peak,
            exact_target_peak=exact_baseline_peak,
        )
        current_closure, _ = _recompute_nodes_closure_and_uses(gm)
        next_closure, _ = _recompute_nodes_closure_and_uses(gm, {candidate})
        removed = current_closure - next_closure
        local_delta, _ = _candidate_memory_curve_delta(memory_curve, candidate, removed)
        modeled_candidate_curve = list(memory_curve.curve)
        for point, amount in local_delta.items():
            modeled_candidate_curve[point] += amount

        old_tag = candidate.meta.get("recompute", _NO_TAG)
        candidate.meta["recompute"] = CheckpointPolicy.MUST_SAVE
        try:
            exact_candidate_peak = _peak_after_remat(gm)
        finally:
            if old_tag is _NO_TAG:
                candidate.meta.pop("recompute", None)
            else:
                candidate.meta["recompute"] = old_tag

        return (
            max(memory_curve.curve),
            max(modeled_candidate_curve),
            exact_baseline_peak,
            exact_candidate_peak,
            _delta_fits(
                memory_curve,
                [0.0] * len(memory_curve.curve),
                local_delta,
            ),
        )

    def test_compute_recompute_gets_fresh_storage(self):
        gm, _ = self._make_remat_graph()
        remat = remat_using_tags_for_fwd_loss_bwd_graph(gm)
        for op in (torch.ops.aten.sin.default, torch.ops.aten.cos.default):
            self.assertNotEqual(
                self._storage_id(self._find(remat, op, recomputed=False)),
                self._storage_id(self._find(remat, op, recomputed=True)),
            )

    def test_view_recompute_aliases_recomputed_parent(self):
        gm, _ = self._make_remat_graph_with_view()
        remat = remat_using_tags_for_fwd_loss_bwd_graph(gm)
        sin_orig = self._find(remat, torch.ops.aten.sin.default, recomputed=False)
        sin_dup = self._find(remat, torch.ops.aten.sin.default, recomputed=True)
        t_dup = self._find(remat, torch.ops.aten.t.default, recomputed=True)
        # compute dup is a real new allocation
        self.assertNotEqual(self._storage_id(sin_orig), self._storage_id(sin_dup))
        # view dup aliases its recomputed parent -- not a fresh allocation. A naive
        # per-dup clone would give it fresh storage and double-count its bytes.
        self.assertEqual(self._storage_id(sin_dup), self._storage_id(t_dup))

    def test_view_is_not_a_save_candidate_or_min_cut(self):
        graph = fx.Graph()
        x = graph.placeholder("x")
        saved = graph.call_function(torch.ops.aten.sin.default, args=(x,))
        candidate = graph.call_function(torch.ops.aten.cos.default, args=(saved,))
        view_node = graph.call_function(torch.ops.aten.t.default, args=(candidate,))
        loss = graph.call_function(torch.ops.aten.sum.default, args=(view_node,))
        bwd = graph.call_function(torch.ops.aten.neg.default, args=(view_node,))
        graph.output((loss, bwd))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        with FakeTensorMode() as fake_mode:
            fake_x = torch.empty(512, 512, device="cuda")
            FakeTensorProp(gm, mode=fake_mode).propagate_dont_convert_inputs(fake_x)

        saved.meta["recompute"] = CheckpointPolicy.MUST_SAVE
        candidate.meta["recompute"] = CheckpointPolicy.PREFER_RECOMPUTE
        view_node.meta["recompute"] = CheckpointPolicy.PREFER_RECOMPUTE
        bwd.meta["autograd_backward"] = True

        self.assertIn(candidate, _nodes_with_fresh_cuda_storage(gm))
        self.assertNotIn(view_node, _nodes_with_fresh_cuda_storage(gm))
        self.assertIn(candidate, _save_candidates(gm))
        self.assertNotIn(view_node, _save_candidates(gm))

        cut = _min_cut(gm)
        self.assertIn(candidate, cut)
        self.assertNotIn(view_node, cut)

    def test_multi_output_getitem_can_be_min_cut(self):
        graph = fx.Graph()
        x = graph.placeholder("x")
        saved = graph.call_function(torch.ops.aten.sin.default, args=(x,))
        norm = graph.call_function(
            torch.ops.aten.native_layer_norm.default,
            args=(saved, [4], None, None, 1e-5),
        )
        out = graph.call_function(operator.getitem, args=(norm, 0))
        loss = graph.call_function(torch.ops.aten.sum.default, args=(out,))
        bwd = graph.call_function(torch.ops.aten.neg.default, args=(out,))
        graph.output((loss, bwd))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        with FakeTensorMode() as fake_mode:
            fake_x = torch.empty(8, 4, device="cuda")
            FakeTensorProp(gm, mode=fake_mode).propagate_dont_convert_inputs(fake_x)

        saved.meta["recompute"] = CheckpointPolicy.MUST_SAVE
        norm.meta["recompute"] = CheckpointPolicy.PREFER_RECOMPUTE
        out.meta["recompute"] = CheckpointPolicy.PREFER_RECOMPUTE
        bwd.meta["autograd_backward"] = True

        self.assertNotIn(out, _nodes_with_fresh_cuda_storage(gm))
        self.assertNotIn(norm, _save_candidates(gm))
        self.assertIn(out, _save_candidates(gm))
        self.assertEqual(_min_cut(gm), {out})

    def test_detached_multi_output_getitem_can_be_min_cut(self):
        graph = fx.Graph()
        x = graph.placeholder("x")
        saved = graph.call_function(torch.ops.aten.sin.default, args=(x,))
        norm = graph.call_function(
            torch.ops.aten.native_layer_norm.default,
            args=(saved, [4], None, None, 1e-5),
        )
        stat = graph.call_function(operator.getitem, args=(norm, 1))
        detached = graph.call_function(torch.ops.aten.detach.default, args=(stat,))
        loss = graph.call_function(torch.ops.aten.sum.default, args=(detached,))
        bwd = graph.call_function(torch.ops.aten.neg.default, args=(detached,))
        graph.output((loss, bwd))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        with FakeTensorMode() as fake_mode:
            fake_x = torch.empty(8, 4, device="cuda")
            FakeTensorProp(gm, mode=fake_mode).propagate_dont_convert_inputs(fake_x)

        saved.meta["recompute"] = CheckpointPolicy.MUST_SAVE
        norm.meta["recompute"] = CheckpointPolicy.PREFER_RECOMPUTE
        stat.meta["recompute"] = CheckpointPolicy.PREFER_RECOMPUTE
        detached.meta["recompute"] = CheckpointPolicy.PREFER_RECOMPUTE
        bwd.meta["autograd_backward"] = True

        self.assertNotIn(detached, _nodes_with_fresh_cuda_storage(gm))
        self.assertIn(detached, _save_candidates(gm))
        self.assertEqual(_min_cut(gm), {detached})

    def test_pre_decomp_log_softmax_is_not_a_save_candidate(self):
        graph = fx.Graph()
        x = graph.placeholder("x")
        logits = graph.call_function(torch.ops.aten.sin.default, args=(x,))
        log_softmax = graph.call_function(
            torch.ops.aten._log_softmax.default, args=(logits, -1, False)
        )
        loss = graph.call_function(torch.ops.aten.sum.default, args=(log_softmax,))
        bwd = graph.call_function(torch.ops.aten.neg.default, args=(log_softmax,))
        graph.output((loss, bwd))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        with FakeTensorMode() as fake_mode:
            fake_x = torch.empty(16, 128, device="cuda")
            FakeTensorProp(gm, mode=fake_mode).propagate_dont_convert_inputs(fake_x)

        logits.meta["recompute"] = CheckpointPolicy.PREFER_RECOMPUTE
        log_softmax.meta["recompute"] = CheckpointPolicy.PREFER_RECOMPUTE
        bwd.meta["autograd_backward"] = True

        self.assertIn(logits, _save_candidates(gm))
        self.assertNotIn(log_softmax, _save_candidates(gm))

        min_cut_ac_pass(
            gm,
            (fake_x,),
            max_peak_increase_gb=1.0,
            save_scope="all",
            memory_estimator="exact",
            runtime_estimator=lambda n: 100.0 if n is log_softmax else 1.0,
        )

        self.assertEqual(
            log_softmax.meta["recompute"], CheckpointPolicy.PREFER_RECOMPUTE
        )

    def test_memory_curve_does_not_allocate_for_view(self):
        gm, _ = self._make_remat_graph_with_view()
        view_node = self._find(gm, torch.ops.aten.t.default, recomputed=False)
        exact_peak = _peak_after_remat(gm)
        memory_curve = _build_memory_curve(
            gm,
            exact_current_peak=exact_peak,
            exact_target_peak=exact_peak,
        )

        point = memory_curve.events.after_alloc[view_node]
        self.assertEqual(memory_curve.curve[point], memory_curve.curve[point - 1])

    def test_reshape_copy_can_be_a_save_candidate(self):
        graph = fx.Graph()
        x = graph.placeholder("x")
        t = graph.call_function(torch.ops.aten.t.default, args=(x,))
        r = graph.call_function(torch.ops.aten.reshape.default, args=(t, [16]))
        loss = graph.call_function(torch.ops.aten.sum.default, args=(r,))
        bwd = graph.call_function(torch.ops.aten.neg.default, args=(r,))
        graph.output((loss, bwd))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        with FakeTensorMode() as fake_mode:
            fake_x = torch.empty(4, 4, device="cuda")
            FakeTensorProp(gm, mode=fake_mode).propagate_dont_convert_inputs(fake_x)

        t.meta["recompute"] = CheckpointPolicy.PREFER_RECOMPUTE
        r.meta["recompute"] = CheckpointPolicy.PREFER_RECOMPUTE
        bwd.meta["autograd_backward"] = True

        fresh_cuda_nodes = _nodes_with_fresh_cuda_storage(gm)
        self.assertNotIn(t, fresh_cuda_nodes)
        self.assertIn(r, fresh_cuda_nodes)
        self.assertNotIn(t, _save_candidates(gm))
        self.assertIn(r, _save_candidates(gm))

    def test_graph_trainer_remat_pass_refreshes_fake_tensor_meta(self):
        gm, _ = self._make_remat_graph_with_view()
        remat = selective_activation_remat_pass(gm)
        sin_dup = self._find(remat, torch.ops.aten.sin.default, recomputed=True)
        t_dup = self._find(remat, torch.ops.aten.t.default, recomputed=True)
        self.assertEqual(self._storage_id(sin_dup), self._storage_id(t_dup))

    def test_peak_after_remat_counts_recomputed_storage_and_restores_tags(self):
        expected_gm, _ = self._make_remat_graph()
        remat = _torch_remat_using_tags_for_fwd_loss_bwd_graph(expected_gm)
        raw_peak = max(build_memory_profile(remat.graph, _is_releasable))
        _refresh_fake_tensor_meta(remat)
        refreshed_peak = max(build_memory_profile(remat.graph, _is_releasable))

        gm, example_inputs = self._make_remat_graph()
        code_before = gm.code
        tags_before = self._tag_snapshot(gm)

        peak = _peak_after_remat(gm, example_inputs)

        # counting the recompute dups as fresh allocations raises the peak
        self.assertGreater(refreshed_peak, raw_peak)
        self.assertEqual(peak, refreshed_peak)
        # gm is left untouched (throwaway remat + tag restore)
        self.assertEqual(gm.code, code_before)
        self.assertEqual(self._tag_snapshot(gm), tags_before)

    def test_remat_materializes_only_must_save_as_saved(self):
        gm, _ = self._make_save_candidate_graph()
        tags_before = self._tag_snapshot(gm)

        remat = remat_using_tags_for_fwd_loss_bwd_graph(gm)

        self.assertEqual(self._tag_snapshot(gm), tags_before)
        self.assertTrue(
            any(n.name == "sin_default_recomputed" for n in remat.graph.nodes)
        )

    def test_multi_region_fallback_does_not_deepcopy_module_state(self):
        class NoDeepcopy:
            def __deepcopy__(self, memo):
                raise AssertionError("module state should not be deep-copied")

        gm, _ = self._make_remat_graph()
        gm.non_deepcopyable = NoDeepcopy()

        with patch(
            "torchtitan.experiments.graph_trainer.memory_utils._torch_remat_using_tags_for_fwd_loss_bwd_graph",
            side_effect=RuntimeError("Detected 8 disjoint backward regions"),
        ):
            remat = remat_using_tags_for_fwd_loss_bwd_graph(gm)

        self.assertTrue(
            any(n.name == "sin_default_recomputed" for n in remat.graph.nodes)
        )

    def test_memory_curve_keeps_output_storages_live(self):
        graph = fx.Graph()
        x = graph.placeholder("x")
        y = graph.call_function(torch.ops.aten.sin.default, args=(x,))
        graph.output(y)
        gm = fx.GraphModule(torch.nn.Module(), graph)

        with FakeTensorMode() as fake_mode:
            fake_x = torch.empty(1024, device="cuda")
            FakeTensorProp(gm, mode=fake_mode).propagate_dont_convert_inputs(fake_x)

        exact_profile = build_memory_profile(gm.graph, _is_releasable)
        memory_curve = _build_memory_curve(
            gm,
            exact_current_peak=max(exact_profile),
            exact_target_peak=max(exact_profile),
        )

        self.assertEqual(memory_curve.curve[-1], exact_profile[-1])

    def test_memory_curve_matches_exact_remat_fanout_candidate(self):
        gm, nodes = self._make_fanout_curve_graph()

        (
            modeled_baseline,
            modeled_candidate,
            exact_baseline,
            exact_candidate,
            fits,
        ) = self._curve_and_exact_candidate_peak(gm, nodes["sin"])

        self.assertEqual(modeled_baseline, exact_baseline)
        self.assertEqual(modeled_candidate, exact_candidate)
        self.assertFalse(fits)

    def test_memory_curve_ends_saved_lifetime_at_recompute_use(self):
        gm, nodes = self._make_chain_curve_graph()

        (
            modeled_baseline,
            modeled_candidate,
            exact_baseline,
            exact_candidate,
            fits,
        ) = self._curve_and_exact_candidate_peak(gm, nodes["cos"])

        self.assertEqual(modeled_baseline, exact_baseline)
        self.assertEqual(modeled_candidate, exact_candidate)
        self.assertTrue(fits)

    def test_alias_and_mutable_nodes_are_not_save_candidates(self):
        graph = fx.Graph()
        x = graph.placeholder("x")
        a = graph.call_function(torch.ops.aten.sin.default, args=(x,))
        d = graph.call_function(torch.ops.aten.detach.default, args=(a,))
        z = graph.call_function(torch.ops.aten.clone.default, args=(a,))
        c = graph.call_function(torch.ops.aten.copy_.default, args=(z, a))
        loss = graph.call_function(torch.ops.aten.sum.default, args=(c,))
        bwd = graph.call_function(torch.ops.aten.neg.default, args=(c,))
        graph.output((loss, bwd))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        with FakeTensorMode() as fake_mode:
            fake_x = torch.empty(1024, device="cuda")
            FakeTensorProp(gm, mode=fake_mode).propagate_dont_convert_inputs(fake_x)

        for node in (a, d, z, c):
            node.meta["recompute"] = CheckpointPolicy.PREFER_RECOMPUTE
        bwd.meta["autograd_backward"] = True

        candidates = _save_candidates(gm)

        fresh_cuda_nodes = _nodes_with_fresh_cuda_storage(gm)
        self.assertNotIn(d, fresh_cuda_nodes)
        self.assertNotIn(c, fresh_cuda_nodes)
        self.assertIn(a, candidates)
        self.assertIn(z, candidates)
        self.assertNotIn(d, candidates)
        self.assertNotIn(c, candidates)

    def test_allow_allowed_saves_marks_finite_candidate_pool(self):
        graph = fx.Graph()
        x = graph.placeholder("x")
        a = graph.call_function(torch.ops.aten.sin.default, args=(x,))
        view = graph.call_function(torch.ops.aten.t.default, args=(a,))
        collective = graph.call_function(fake_c10d_functional_wait, args=(view,))
        rng = graph.call_function(torch.ops.aten.bernoulli.default, args=(view,))
        hard = graph.call_function(torch.ops.aten.cos.default, args=(view,))
        loss = graph.call_function(torch.ops.aten.sum.default, args=(hard,))
        bwd = graph.call_function(torch.ops.aten.neg.default, args=(hard,))
        graph.output((loss, collective, rng, bwd))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        with FakeTensorMode() as fake_mode:
            fake_x = torch.empty(32, 32, device="cuda")
            FakeTensorProp(gm, mode=fake_mode).propagate_dont_convert_inputs(fake_x)

        hard.meta["recompute"] = CheckpointPolicy.MUST_RECOMPUTE
        bwd.meta["autograd_backward"] = True

        ac_allow_allowed_saves(gm)

        self.assertEqual(a.meta["recompute"], CheckpointPolicy.PREFER_SAVE)
        self.assertNotIn("recompute", view.meta)
        self.assertNotIn("recompute", collective.meta)
        self.assertNotIn("recompute", rng.meta)
        self.assertEqual(hard.meta["recompute"], CheckpointPolicy.MUST_RECOMPUTE)

        candidates = _save_candidates(gm)
        self.assertIn(a, candidates)
        self.assertNotIn(view, candidates)

    def test_mutable_recompute_is_forced_saved(self):
        graph = fx.Graph()
        x = graph.placeholder("x")
        a = graph.call_function(torch.ops.aten.sin.default, args=(x,))
        z = graph.call_function(torch.ops.aten.clone.default, args=(a,))
        c = graph.call_function(torch.ops.aten.copy_.default, args=(z, a))
        loss = graph.call_function(torch.ops.aten.sum.default, args=(c,))
        bwd = graph.call_function(torch.ops.aten.neg.default, args=(c,))
        graph.output((loss, bwd))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        with FakeTensorMode() as fake_mode:
            fake_x = torch.empty(1024, device="cuda")
            FakeTensorProp(gm, mode=fake_mode).propagate_dont_convert_inputs(fake_x)

        c.meta["recompute"] = CheckpointPolicy.PREFER_RECOMPUTE
        bwd.meta["autograd_backward"] = True

        _set_mandatory_must_saves(gm)

        self.assertEqual(c.meta["recompute"], CheckpointPolicy.MUST_SAVE)

    def test_budgeted_prefers_smaller_cutpoint(self):
        gm, example_inputs = self._make_save_candidate_graph()
        sin_node = self._find(gm, torch.ops.aten.sin.default, recomputed=False)
        sum_dim_node = self._find(gm, torch.ops.aten.sum.dim_IntList, recomputed=False)
        runtime_estimator = lambda n: 1.0

        baseline_peak, baseline_cost = _memory_profile_after_remat(
            gm, example_inputs, runtime_estimator
        )

        min_cut_ac_pass(
            gm,
            example_inputs,
            save_scope="all",
            runtime_estimator=runtime_estimator,
        )

        final_peak, final_cost = _memory_profile_after_remat(
            gm, example_inputs, runtime_estimator
        )
        self.assertEqual(sin_node.meta["recompute"], CheckpointPolicy.PREFER_SAVE)
        self.assertEqual(sum_dim_node.meta["recompute"], CheckpointPolicy.MUST_SAVE)
        self.assertLess(final_peak, baseline_peak)
        self.assertLessEqual(final_cost, baseline_cost)

    def test_budgeted_adds_replacement_cutpoint_without_peak_regression(self):
        gm, example_inputs = self._make_save_candidate_contract_graph()
        sin_node = self._find(gm, torch.ops.aten.sin.default, recomputed=False)
        sum_dim_node = self._find(gm, torch.ops.aten.sum.dim_IntList, recomputed=False)
        clone_node = self._find(gm, torch.ops.aten.clone.default, recomputed=False)
        cos_node = self._find(gm, torch.ops.aten.cos.default, recomputed=False)
        runtime_estimator = lambda n: 1.0

        baseline_peak, baseline_cost = _memory_profile_after_remat(
            gm, example_inputs, runtime_estimator
        )

        min_cut_ac_pass(
            gm,
            example_inputs,
            save_scope="all",
            memory_estimator="approximate",
            runtime_estimator=runtime_estimator,
        )

        final_peak, final_cost = _memory_profile_after_remat(
            gm, example_inputs, runtime_estimator
        )
        self.assertEqual(sin_node.meta["recompute"], CheckpointPolicy.PREFER_SAVE)
        self.assertIn(
            CheckpointPolicy.MUST_SAVE,
            (
                sum_dim_node.meta["recompute"],
                clone_node.meta["recompute"],
                cos_node.meta["recompute"],
            ),
        )
        self.assertLessEqual(final_peak, baseline_peak)
        self.assertLessEqual(final_cost, baseline_cost)

    def test_budgeted_all_scope_searches_full_candidate_pool(self):
        gm, example_inputs = self._make_save_candidate_contract_graph()
        sin_node = self._find(gm, torch.ops.aten.sin.default, recomputed=False)
        sum_dim_node = self._find(gm, torch.ops.aten.sum.dim_IntList, recomputed=False)
        clone_node = self._find(gm, torch.ops.aten.clone.default, recomputed=False)
        cos_node = self._find(gm, torch.ops.aten.cos.default, recomputed=False)
        runtime_estimator = lambda n: 1.0

        baseline_peak, baseline_cost = _memory_profile_after_remat(
            gm, example_inputs, runtime_estimator
        )

        min_cut_ac_pass(
            gm,
            example_inputs,
            save_scope="all",
            memory_estimator="approximate",
            runtime_estimator=runtime_estimator,
        )

        final_peak, final_cost = _memory_profile_after_remat(
            gm, example_inputs, runtime_estimator
        )
        self.assertEqual(sin_node.meta["recompute"], CheckpointPolicy.PREFER_SAVE)
        self.assertIn(
            CheckpointPolicy.MUST_SAVE,
            (
                sum_dim_node.meta["recompute"],
                clone_node.meta["recompute"],
                cos_node.meta["recompute"],
            ),
        )
        self.assertLessEqual(final_peak, baseline_peak)
        self.assertLessEqual(final_cost, baseline_cost)

    def test_budgeted_all_scope_can_filter_broad_candidates_by_size(self):
        gm, _ = self._make_save_candidate_contract_graph()
        sin_node = self._find(gm, torch.ops.aten.sin.default, recomputed=False)
        clone_node = self._find(gm, torch.ops.aten.clone.default, recomputed=False)
        cos_node = self._find(gm, torch.ops.aten.cos.default, recomputed=False)
        sum_dim_node = self._find(gm, torch.ops.aten.sum.dim_IntList, recomputed=False)

        plan = _budgeted_save_candidate_phases(
            gm,
            "all",
            set(),
            min_broad_candidate_bytes=1_000_000,
            max_broad_candidate_bytes=1_100_000,
        )

        self.assertIn(sin_node, plan.broad)
        self.assertIn(clone_node, plan.broad)
        self.assertIn(cos_node, plan.broad)
        self.assertNotIn(sum_dim_node, plan.broad)

    def test_budgeted_min_cut_profiles_candidate_peak(self):
        gm, example_inputs = self._make_interior_cutpoint_graph()
        sum_dim_node = self._find(gm, torch.ops.aten.sum.dim_IntList, recomputed=False)
        clone_node = self._find(gm, torch.ops.aten.clone.default, recomputed=False)
        clone_node.meta["recompute"] = CheckpointPolicy.PREFER_SAVE
        profile_calls = 0

        def fake_profile(*args, **kwargs):
            nonlocal profile_calls
            profile_calls += 1
            if profile_calls == 1:
                return 100, 1.0
            if profile_calls == 2:
                return 90, 2.0
            return 95, 1.5

        with (
            patch(
                "torchtitan.experiments.graph_trainer.min_cut_ac._memory_profile_after_remat",
                side_effect=fake_profile,
            ),
            patch(
                "torchtitan.experiments.graph_trainer.min_cut_ac._peak_after_remat",
                return_value=95,
            ),
        ):
            min_cut_ac_pass(
                gm,
                example_inputs,
                save_scope="min_cut",
                memory_estimator="approximate",
                runtime_estimator=lambda n: 1.0,
            )

        self.assertEqual(profile_calls, 2)
        self.assertEqual(sum_dim_node.meta["recompute"], CheckpointPolicy.MUST_SAVE)

    def test_budgeted_exact_rejects_peak_regressing_candidate(self):
        gm, example_inputs = self._make_save_candidate_graph()
        sin_node = self._find(gm, torch.ops.aten.sin.default, recomputed=False)
        sum_dim_node = self._find(gm, torch.ops.aten.sum.dim_IntList, recomputed=False)

        with (
            patch(
                "torchtitan.experiments.graph_trainer.min_cut_ac._memory_profile_after_remat",
                side_effect=[(100, 1.0), (90, 2.0)],
            ),
            patch(
                "torchtitan.experiments.graph_trainer.min_cut_ac._peak_after_remat",
                return_value=200,
            ),
        ):
            min_cut_ac_pass(
                gm,
                example_inputs,
                save_scope="min_cut",
                memory_estimator="exact",
                runtime_estimator=lambda n: 1.0,
            )

        self.assertEqual(sin_node.meta["recompute"], CheckpointPolicy.PREFER_SAVE)
        self.assertEqual(
            sum_dim_node.meta["recompute"], CheckpointPolicy.PREFER_RECOMPUTE
        )

    def test_budgeted_negative_budget_accepts_lower_peak_target(self):
        gm, example_inputs = self._make_save_candidate_graph()
        sin_node = self._find(gm, torch.ops.aten.sin.default, recomputed=False)
        sum_dim_node = self._find(gm, torch.ops.aten.sum.dim_IntList, recomputed=False)

        with (
            patch(
                "torchtitan.experiments.graph_trainer.min_cut_ac._memory_profile_after_remat",
                side_effect=[(100, 1.0), (90, 1.5)],
            ),
            patch(
                "torchtitan.experiments.graph_trainer.min_cut_ac._peak_after_remat",
                return_value=90,
            ),
            patch(
                "torchtitan.experiments.graph_trainer.min_cut_ac._min_cut",
                return_value={sum_dim_node},
            ),
        ):
            min_cut_ac_pass(
                gm,
                example_inputs,
                max_peak_increase_gb=-10 / _GB,
                save_scope="min_cut",
                memory_estimator="exact",
                runtime_estimator=lambda n: 1.0,
            )

        self.assertEqual(sin_node.meta["recompute"], CheckpointPolicy.PREFER_SAVE)
        self.assertEqual(sum_dim_node.meta["recompute"], CheckpointPolicy.MUST_SAVE)

    def test_budgeted_negative_budget_accepts_cumulative_progress(self):
        gm, example_inputs = self._make_save_candidate_contract_graph()
        candidates = _save_candidates(gm)

        peak_iter = iter([int(95 * _GB), int(90 * _GB)])

        def fake_peak(*args, **kwargs):
            return next(peak_iter, 90)

        with (
            patch(
                "torchtitan.experiments.graph_trainer.min_cut_ac._memory_profile_after_remat",
                side_effect=[(int(100 * _GB), 1.0), (int(90 * _GB), 1.5)],
            ),
            patch(
                "torchtitan.experiments.graph_trainer.min_cut_ac._peak_after_remat",
                side_effect=fake_peak,
            ),
            patch(
                "torchtitan.experiments.graph_trainer.min_cut_ac._min_cut",
                return_value=candidates,
            ),
        ):
            min_cut_ac_pass(
                gm,
                example_inputs,
                max_peak_increase_gb=-10.0,
                save_scope="min_cut",
                memory_estimator="exact",
                runtime_estimator=lambda n: 1.0,
            )

        saved = [
            n
            for n in gm.graph.nodes
            if n.meta.get("recompute") == CheckpointPolicy.MUST_SAVE
        ]
        self.assertGreaterEqual(len(saved), 2)

    def test_budgeted_negative_budget_keeps_peak_progress(self):
        gm, example_inputs = self._make_save_candidate_graph()
        sum_dim_node = self._find(gm, torch.ops.aten.sum.dim_IntList, recomputed=False)

        with patch(
            "torchtitan.experiments.graph_trainer.min_cut_ac._memory_profile_after_remat",
            side_effect=[(100, 1.0), (95, 1.0), (95, 1.0)],
        ):
            with patch(
                "torchtitan.experiments.graph_trainer.min_cut_ac._peak_after_remat",
                return_value=95,
            ):
                with patch(
                    "torchtitan.experiments.graph_trainer.min_cut_ac._min_cut",
                    return_value={sum_dim_node},
                ):
                    min_cut_ac_pass(
                        gm,
                        example_inputs,
                        max_peak_increase_gb=-10 / _GB,
                        save_scope="min_cut",
                        memory_estimator="exact",
                        runtime_estimator=lambda n: 1.0,
                    )

        self.assertEqual(sum_dim_node.meta["recompute"], CheckpointPolicy.MUST_SAVE)

    def test_budgeted_missed_target_keeps_lower_peak_candidate(self):
        gm, example_inputs = self._make_save_candidate_graph()
        sum_dim_node = self._find(gm, torch.ops.aten.sum.dim_IntList, recomputed=False)

        with patch(
            "torchtitan.experiments.graph_trainer.min_cut_ac._memory_profile_after_remat",
            side_effect=[(100, 1.0), (95, 1.0), (95, 1.0)],
        ):
            with patch(
                "torchtitan.experiments.graph_trainer.min_cut_ac._peak_after_remat",
                return_value=95,
            ):
                with patch(
                    "torchtitan.experiments.graph_trainer.min_cut_ac._min_cut",
                    return_value={sum_dim_node},
                ):
                    min_cut_ac_pass(
                        gm,
                        example_inputs,
                        max_peak_increase_gb=-10 / _GB,
                        save_scope="min_cut",
                        memory_estimator="exact",
                        runtime_estimator=lambda n: 1.0,
                    )

        self.assertEqual(sum_dim_node.meta["recompute"], CheckpointPolicy.MUST_SAVE)

    def test_over_budget_peak_progress_requires_material_reduction(self):
        self.assertFalse(
            _candidate_peak_fits_or_makes_progress(
                candidate_peak=int(100 * _GB + 512 * 1024),
                current_peak=int(100 * _GB),
                target_peak=90 * _GB,
            )
        )
        self.assertFalse(
            _candidate_peak_fits_or_makes_progress(
                candidate_peak=int(100 * _GB + 2 * 1024 * 1024),
                current_peak=int(100 * _GB),
                target_peak=90 * _GB,
            )
        )
        self.assertTrue(
            _candidate_peak_fits_or_makes_progress(
                candidate_peak=int(99 * _GB),
                current_peak=int(100 * _GB),
                target_peak=90 * _GB,
            )
        )
        self.assertFalse(
            _candidate_peak_fits_or_makes_progress(
                candidate_peak=int(101 * _GB),
                current_peak=int(90 * _GB),
                target_peak=90 * _GB,
            )
        )

    def test_peak_progress_ranking_requires_material_excess(self):
        self.assertFalse(
            _should_rank_by_peak_progress(
                current_peak=int(100 * _GB + 256 * 1024 * 1024),
                target_peak=100 * _GB,
            )
        )
        self.assertTrue(
            _should_rank_by_peak_progress(
                current_peak=int(102 * _GB),
                target_peak=100 * _GB,
            )
        )

    def test_negative_budget_best_effort_fills_under_reference(self):
        gm, example_inputs = self._make_save_candidate_graph()
        nodes = [n for n in gm.graph.nodes if n.op == "call_function"]
        first_result = _BudgetedSaveSelectionResult(
            peak=int(95 * _GB),
            kept=[nodes[0]],
            rejected_peak=1,
            pruned_memory_curve=2,
        )
        fill_result = _BudgetedSaveSelectionResult(
            peak=int(99 * _GB),
            kept=[nodes[1]],
            rejected_peak=3,
            pruned_memory_curve=4,
        )

        with (
            patch(
                "torchtitan.experiments.graph_trainer.min_cut_ac._memory_profile_after_remat",
                side_effect=[
                    (int(100 * _GB), 10.0),
                    (int(95 * _GB), 9.0),
                    (int(99 * _GB), 8.0),
                ],
            ),
            patch(
                "torchtitan.experiments.graph_trainer.min_cut_ac._select_saves_under_peak_budget",
                side_effect=[first_result, fill_result],
            ) as select_saves,
        ):
            result = _optimize_under_peak_budget(
                gm,
                example_inputs,
                max_peak_increase_gb=-10.0,
                memory_estimator="exact",
                save_scope="all",
                runtime_estimator=lambda n: 1.0,
            )

        self.assertEqual(select_saves.call_count, 2)
        self.assertEqual(
            select_saves.call_args_list[1].kwargs["current_peak"], int(95 * _GB)
        )
        self.assertEqual(
            select_saves.call_args_list[1].kwargs["target_peak"], int(100 * _GB)
        )
        self.assertEqual(result.final_peak, int(99 * _GB))
        self.assertFalse(result.target_met)
        self.assertTrue(result.filled_best_effort)

    def test_relax_budget_restores_reference_when_best_effort_worse(self):
        gm, example_inputs = self._make_interior_cutpoint_graph()
        sin_node = self._find(gm, torch.ops.aten.sin.default, recomputed=False)
        sum_dim_node = self._find(gm, torch.ops.aten.sum.dim_IntList, recomputed=False)

        with (
            patch(
                "torchtitan.experiments.graph_trainer.min_cut_ac._memory_profile_after_remat",
                side_effect=[
                    (int(100 * _GB), 1.0),
                    (int(200 * _GB), 2.0),
                    (int(150 * _GB), 1.5),
                    (int(100 * _GB), 1.0),
                ],
            ),
            patch(
                "torchtitan.experiments.graph_trainer.min_cut_ac._peak_after_remat",
                return_value=int(150 * _GB),
            ),
            patch(
                "torchtitan.experiments.graph_trainer.min_cut_ac.logger.info"
            ) as log_info,
        ):
            min_cut_ac_pass(
                gm,
                example_inputs,
                max_peak_increase_gb=0.0,
                save_scope="all",
                memory_estimator="exact",
                relax_relaxable_must_saves=True,
                runtime_estimator=lambda n: 1.0,
            )

        self.assertEqual(sin_node.meta["recompute"], CheckpointPolicy.MUST_SAVE)
        self.assertEqual(
            sum_dim_node.meta["recompute"], CheckpointPolicy.PREFER_RECOMPUTE
        )
        self.assertTrue(
            any(
                "restoring reference tags" in str(arg)
                for call in log_info.call_args_list
                for arg in call.args
            )
        )

    def test_relax_budget_keeps_best_effort_when_it_improves_reference(self):
        gm, example_inputs = self._make_interior_cutpoint_graph()
        sin_node = self._find(gm, torch.ops.aten.sin.default, recomputed=False)
        sum_dim_node = self._find(gm, torch.ops.aten.sum.dim_IntList, recomputed=False)

        with (
            patch(
                "torchtitan.experiments.graph_trainer.min_cut_ac._memory_profile_after_remat",
                side_effect=[
                    (int(100 * _GB), 1.0),
                    (int(200 * _GB), 2.0),
                    (int(95 * _GB), 1.5),
                    (int(95 * _GB), 1.5),
                ],
            ),
            patch(
                "torchtitan.experiments.graph_trainer.min_cut_ac._peak_after_remat",
                return_value=int(95 * _GB),
            ),
            patch(
                "torchtitan.experiments.graph_trainer.min_cut_ac._min_cut",
                return_value={sum_dim_node},
            ),
        ):
            min_cut_ac_pass(
                gm,
                example_inputs,
                max_peak_increase_gb=-10.0,
                save_scope="min_cut",
                memory_estimator="exact",
                relax_relaxable_must_saves=True,
                runtime_estimator=lambda n: 1.0,
            )

        self.assertEqual(sin_node.meta["recompute"], CheckpointPolicy.PREFER_SAVE)
        self.assertEqual(sum_dim_node.meta["recompute"], CheckpointPolicy.MUST_SAVE)

    def test_budgeted_missed_target_keeps_mandatory_safety_rewrites(self):
        gm, example_inputs, rng_node = self._make_rng_graph(
            CheckpointPolicy.PREFER_RECOMPUTE
        )

        with patch(
            "torchtitan.experiments.graph_trainer.min_cut_ac._memory_profile_after_remat",
            side_effect=[(100, 1.0), (100, 1.0), (100, 1.0)],
        ):
            min_cut_ac_pass(
                gm,
                example_inputs,
                max_peak_increase_gb=-10 / _GB,
                save_scope="min_cut",
                memory_estimator="exact",
                runtime_estimator=lambda n: 1.0,
            )

        self.assertEqual(rng_node.meta["recompute"], CheckpointPolicy.MUST_SAVE)

    def test_budgeted_missed_target_can_save_after_collective(self):
        gm, example_inputs, _ = self._make_collective_closure_graph()
        wait_node = self._find(gm, fake_c10d_functional_wait, recomputed=False)
        sin_node = self._find(gm, torch.ops.aten.sin.default, recomputed=False)

        with patch(
            "torchtitan.experiments.graph_trainer.min_cut_ac._memory_profile_after_remat",
            side_effect=[(100, 1.0), (95, 1.0), (95, 1.0)],
        ):
            with patch(
                "torchtitan.experiments.graph_trainer.min_cut_ac._peak_after_remat",
                return_value=95,
            ):
                min_cut_ac_pass(
                    gm,
                    example_inputs,
                    max_peak_increase_gb=-10 / _GB,
                    save_scope="all",
                    memory_estimator="exact",
                    runtime_estimator=lambda n: 1.0,
                )

        self.assertEqual(wait_node.meta["recompute"], CheckpointPolicy.PREFER_RECOMPUTE)
        self.assertEqual(sin_node.meta["recompute"], CheckpointPolicy.MUST_SAVE)

    def test_approx_budgeted_selection_exact_checks_candidate_peak(self):
        gm, example_inputs = self._make_save_candidate_graph()
        tags_before = self._tag_snapshot(gm)
        profile_calls = 0

        def fake_profile(*args, **kwargs):
            nonlocal profile_calls
            profile_calls += 1
            if profile_calls == 1:
                return 100, 1.0
            if profile_calls == 2:
                return 90, 2.0
            return 200, 1.5

        with (
            patch(
                "torchtitan.experiments.graph_trainer.min_cut_ac._memory_profile_after_remat",
                side_effect=fake_profile,
            ),
        ):
            min_cut_ac_pass(
                gm,
                example_inputs,
                max_peak_increase_gb=0.0,
                save_scope="all",
                memory_estimator="approximate",
                runtime_estimator=lambda n: 1.0,
            )

        self.assertEqual(profile_calls, 2)
        self.assertEqual(self._tag_snapshot(gm), tags_before)

    def test_approx_budgeted_selection_exact_checks_candidates_after_target_met(self):
        gm, example_inputs = self._make_save_candidate_contract_graph()
        for node in gm.graph.nodes:
            if node.meta.get("recompute") == CheckpointPolicy.PREFER_SAVE:
                node.meta["recompute"] = CheckpointPolicy.PREFER_RECOMPUTE

        candidates = {
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and not n.meta.get("autograd_backward")
        }
        peak_calls = 0

        def fake_peak(*args, **kwargs):
            nonlocal peak_calls
            peak_calls += 1
            return 200

        def fake_fitting_delta(memory_curve, node, removed):
            return ({i: -10 * _GB for i in range(len(memory_curve.curve))}, 1)

        with (
            patch(
                "torchtitan.experiments.graph_trainer.min_cut_ac._peak_after_remat",
                side_effect=fake_peak,
            ),
            patch(
                "torchtitan.experiments.graph_trainer.min_cut_ac._candidate_memory_curve_delta",
                side_effect=fake_fitting_delta,
            ),
        ):
            result = _select_saves_under_peak_budget(
                gm,
                example_inputs,
                candidate_phases=(candidates,),
                current_peak=200,
                target_peak=200,
                memory_estimator="approximate",
                runtime_estimator=lambda n: 1.0,
            )

        self.assertGreater(peak_calls, 0)
        self.assertEqual(result.peak, 200)
        self.assertGreater(len(result.kept), 0)

    def test_over_budget_approx_selection_prioritizes_peak_progress(self):
        gm, example_inputs = self._make_save_candidate_graph()
        sin_node = self._find(gm, torch.ops.aten.sin.default, recomputed=False)
        sum_dim_node = self._find(gm, torch.ops.aten.sum.dim_IntList, recomputed=False)
        trials = []

        def runtime_estimator(node):
            return 100.0 if node is sin_node else 1.0

        def fake_delta(memory_curve, node, removed):
            if node is sum_dim_node:
                return ({i: -2 * _GB for i in range(len(memory_curve.curve))}, 1)
            return ({}, 1)

        def fake_peak(*args, **kwargs):
            if sum_dim_node.meta["recompute"] == CheckpointPolicy.MUST_SAVE:
                trials.append("peak_progress")
                return int(190 * _GB)
            if sin_node.meta["recompute"] == CheckpointPolicy.MUST_SAVE:
                trials.append("runtime")
                return int(200 * _GB)
            trials.append("other")
            return int(200 * _GB)

        with (
            patch(
                "torchtitan.experiments.graph_trainer.min_cut_ac._candidate_memory_curve_delta",
                side_effect=fake_delta,
            ),
            patch(
                "torchtitan.experiments.graph_trainer.min_cut_ac._peak_after_remat",
                side_effect=fake_peak,
            ),
        ):
            _select_saves_under_peak_budget(
                gm,
                example_inputs,
                candidate_phases=({sin_node, sum_dim_node},),
                current_peak=int(200 * _GB),
                target_peak=100 * _GB,
                memory_estimator="approximate",
                runtime_estimator=runtime_estimator,
            )

        self.assertEqual(trials[0], "peak_progress")

    def test_approx_budgeted_exact_checks_candidate_peak(self):
        gm, example_inputs = self._make_interior_cutpoint_graph()
        sum_dim_node = self._find(gm, torch.ops.aten.sum.dim_IntList, recomputed=False)

        with (
            patch(
                "torchtitan.experiments.graph_trainer.min_cut_ac._memory_profile_after_remat",
                side_effect=[(100, 1.0), (100, 1.0), (100, 1.0)],
            ),
            patch(
                "torchtitan.experiments.graph_trainer.min_cut_ac._peak_after_remat",
                return_value=200,
            ),
        ):
            min_cut_ac_pass(
                gm,
                example_inputs,
                max_peak_increase_gb=0.0,
                save_scope="min_cut",
                memory_estimator="approximate",
                runtime_estimator=lambda n: 1.0,
            )

        self.assertEqual(
            sum_dim_node.meta["recompute"], CheckpointPolicy.PREFER_RECOMPUTE
        )

    def test_budgeted_missed_target_keeps_non_regressing_candidate_tags(
        self,
    ):
        gm, example_inputs = self._make_interior_cutpoint_graph()
        sum_dim_node = self._find(gm, torch.ops.aten.sum.dim_IntList, recomputed=False)

        with (
            patch(
                "torchtitan.experiments.graph_trainer.min_cut_ac._memory_profile_after_remat",
                side_effect=[(100, 1.0), (100, 1.0), (100, 1.0)],
            ),
            patch(
                "torchtitan.experiments.graph_trainer.min_cut_ac._peak_after_remat",
                return_value=100,
            ),
        ):
            min_cut_ac_pass(
                gm,
                example_inputs,
                max_peak_increase_gb=-10 / _GB,
                save_scope="min_cut",
                memory_estimator="exact",
                runtime_estimator=lambda n: 1.0,
            )

        self.assertEqual(sum_dim_node.meta["recompute"], CheckpointPolicy.MUST_SAVE)

    def test_recompute_closure_does_not_overcount_shared_fanout_prefix(self):
        graph = fx.Graph()
        x = graph.placeholder("x")
        a = graph.call_function(torch.ops.aten.sin.default, args=(x,))
        b = graph.call_function(torch.ops.aten.cos.default, args=(a,))
        c = graph.call_function(torch.ops.aten.neg.default, args=(a,))
        bwd_b = graph.call_function(torch.ops.aten.relu.default, args=(b,))
        bwd_c = graph.call_function(torch.ops.aten.sigmoid.default, args=(c,))
        graph.output((bwd_b, bwd_c))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        for node in (a, b, c):
            node.meta["recompute"] = CheckpointPolicy.PREFER_RECOMPUTE
        bwd_b.meta["autograd_backward"] = True
        bwd_c.meta["autograd_backward"] = True

        closure, _ = _recompute_nodes_closure_and_uses(gm)
        self.assertEqual(closure, {a, b, c})

        closure_after_saving_b, _ = _recompute_nodes_closure_and_uses(gm, {b})
        self.assertEqual(closure_after_saving_b, {a, c})

    def test_min_cut_save_scope_preserves_interior_frontier(self):
        gm, example_inputs = self._make_interior_cutpoint_graph()
        sum_dim_node = self._find(gm, torch.ops.aten.sum.dim_IntList, recomputed=False)

        min_cut_ac_pass(
            gm,
            example_inputs,
            save_scope="min_cut",
            memory_estimator="exact",
            runtime_estimator=lambda n: 1.0,
        )

        self.assertEqual(sum_dim_node.meta["recompute"], CheckpointPolicy.MUST_SAVE)

    def test_min_cut_ignores_already_saved_uncuttable_uses(self):
        (
            gm,
            example_inputs,
            finite_cutpoint,
        ) = self._make_saved_uncuttable_plus_finite_frontier_graph()

        min_cut_ac_pass(
            gm,
            example_inputs,
            save_scope="min_cut",
            memory_estimator="exact",
            runtime_estimator=lambda n: 1.0,
        )

        self.assertEqual(finite_cutpoint.meta["recompute"], CheckpointPolicy.MUST_SAVE)

    def test_min_cut_does_not_cut_already_saved_boundary(self):
        graph = fx.Graph()
        x = graph.placeholder("x")
        saved = graph.call_function(
            torch.ops.aten.sum.dim_IntList, args=(x, [1], False)
        )
        view = graph.call_function(
            torch.ops.aten.expand.default, args=(saved, [512, 512])
        )
        candidate = graph.call_function(torch.ops.aten.clone.default, args=(view,))
        loss = graph.call_function(torch.ops.aten.sum.default, args=(candidate,))
        bwd = graph.call_function(torch.ops.aten.neg.default, args=(candidate,))
        graph.output((loss, bwd))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        with FakeTensorMode() as fake_mode:
            fake_x = torch.empty(512, 512, device="cuda")
            FakeTensorProp(gm, mode=fake_mode).propagate_dont_convert_inputs(fake_x)

        saved.meta["recompute"] = CheckpointPolicy.MUST_SAVE
        for node in (view, candidate):
            node.meta["recompute"] = CheckpointPolicy.PREFER_RECOMPUTE
        bwd.meta["autograd_backward"] = True

        self.assertEqual(_min_cut(gm), {candidate})

        min_cut_ac_pass(
            gm,
            (fake_x,),
            save_scope="min_cut",
            memory_estimator="exact",
            runtime_estimator=lambda n: 1.0,
        )

        self.assertEqual(candidate.meta["recompute"], CheckpointPolicy.MUST_SAVE)

    def test_save_layer_inputs_saves_final_layer_output_boundary(self):
        graph = fx.Graph()
        x = graph.placeholder("x")
        layer_out = graph.call_function(torch.ops.aten.sin.default, args=(x,))
        final_norm = graph.call_function(torch.ops.aten.relu.default, args=(layer_out,))
        bwd = graph.call_function(torch.ops.aten.neg.default, args=(layer_out,))
        graph.output((final_norm, bwd))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        layer_out.meta["custom"] = {_MODULE_FQN: "layers.0"}
        final_norm.meta["custom"] = {_MODULE_FQN: "norm"}
        bwd.meta["autograd_backward"] = True

        apply_save_layer_inputs_ac(gm)

        self.assertEqual(layer_out.meta["recompute"], CheckpointPolicy.MUST_SAVE)
        self.assertNotIn("recompute", final_norm.meta)

    def test_save_layer_inputs_can_recompute_final_layer_output_boundary(self):
        graph = fx.Graph()
        x = graph.placeholder("x")
        layer_out = graph.call_function(torch.ops.aten.sin.default, args=(x,))
        final_norm = graph.call_function(torch.ops.aten.relu.default, args=(layer_out,))
        bwd = graph.call_function(torch.ops.aten.neg.default, args=(layer_out,))
        graph.output((final_norm, bwd))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        layer_out.meta["custom"] = {_MODULE_FQN: "layers.0"}
        final_norm.meta["custom"] = {_MODULE_FQN: "norm"}
        bwd.meta["autograd_backward"] = True

        apply_save_layer_inputs_ac(gm, save_final_layer_output=False)

        self.assertEqual(layer_out.meta["recompute"], CheckpointPolicy.PREFER_RECOMPUTE)
        self.assertNotIn("recompute", final_norm.meta)

    def test_all_scope_processes_min_cut_before_broad_candidates(self):
        gm, example_inputs, b_node, c_node = self._make_all_scope_graph()
        trials = []

        def runtime_estimator(n):
            return 100.0 if n is c_node else 1.0

        peak_calls = 0

        def fake_peak(*args, **kwargs):
            nonlocal peak_calls
            peak_calls += 1
            if c_node.meta["recompute"] == CheckpointPolicy.MUST_SAVE:
                trials.append("broad")
                return 200
            if b_node.meta["recompute"] == CheckpointPolicy.MUST_SAVE:
                trials.append("min_cut")
                return 100
            trials.append("other")
            return 200

        with patch(
            "torchtitan.experiments.graph_trainer.min_cut_ac._peak_after_remat",
            side_effect=fake_peak,
        ):
            min_cut_ac_pass(
                gm,
                example_inputs,
                save_scope="all",
                memory_estimator="exact",
                runtime_estimator=runtime_estimator,
            )

        self.assertGreaterEqual(len(trials), 2)
        self.assertEqual(trials[:2], ["min_cut", "broad"])

    def test_all_scope_considers_broad_only_saves_for_zero_budget(self):
        gm, example_inputs, _, c_node = self._make_all_scope_graph()

        with (
            patch(
                "torchtitan.experiments.graph_trainer.min_cut_ac._min_cut",
                return_value=set(),
            ),
            patch(
                "torchtitan.experiments.graph_trainer.min_cut_ac._peak_after_remat",
                return_value=0,
            ) as peak_after_remat,
        ):
            min_cut_ac_pass(
                gm,
                example_inputs,
                max_peak_increase_gb=0.0,
                save_scope="all",
                memory_estimator="exact",
                runtime_estimator=lambda n: 1.0,
            )

        peak_after_remat.assert_called()
        self.assertEqual(c_node.meta["recompute"], CheckpointPolicy.MUST_SAVE)

    def test_all_scope_considers_broad_only_saves_for_negative_budget(self):
        gm, example_inputs, _, c_node = self._make_all_scope_graph()

        with (
            patch(
                "torchtitan.experiments.graph_trainer.min_cut_ac._min_cut",
                return_value=set(),
            ),
            patch(
                "torchtitan.experiments.graph_trainer.min_cut_ac._peak_after_remat",
                return_value=0,
            ) as peak_after_remat,
        ):
            min_cut_ac_pass(
                gm,
                example_inputs,
                max_peak_increase_gb=-1 / _GB,
                save_scope="all",
                memory_estimator="exact",
                runtime_estimator=lambda n: 1.0,
            )

        peak_after_remat.assert_called()
        self.assertEqual(c_node.meta["recompute"], CheckpointPolicy.MUST_SAVE)

    def test_all_scope_allows_broad_only_saves_for_positive_budget(self):
        gm, example_inputs, _, c_node = self._make_all_scope_graph()

        with (
            patch(
                "torchtitan.experiments.graph_trainer.min_cut_ac._min_cut",
                return_value=set(),
            ),
            patch(
                "torchtitan.experiments.graph_trainer.min_cut_ac._peak_after_remat",
                return_value=0,
            ) as peak_after_remat,
        ):
            min_cut_ac_pass(
                gm,
                example_inputs,
                max_peak_increase_gb=1.0,
                save_scope="all",
                memory_estimator="exact",
                runtime_estimator=lambda n: 100.0 if n is c_node else 1.0,
            )

        peak_after_remat.assert_called()
        self.assertEqual(c_node.meta["recompute"], CheckpointPolicy.MUST_SAVE)

    def test_all_scope_allows_saves_that_remove_collective_recompute(self):
        graph = fx.Graph()
        x = graph.placeholder("x")
        collective = graph.call_function(fake_c10d_functional_wait, args=(x,))
        floor_save = graph.call_function(
            torch.ops.aten.sum.dim_IntList, args=(collective, [1], False)
        )
        broad = graph.call_function(torch.ops.aten.sin.default, args=(collective,))
        loss = graph.call_function(torch.ops.aten.sum.default, args=(broad,))
        bwd = graph.call_function(torch.ops.aten.neg.default, args=(broad,))
        graph.output((loss, bwd, floor_save))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        with FakeTensorMode() as fake_mode:
            fake_x = torch.empty(512, 512, device="cuda")
            FakeTensorProp(gm, mode=fake_mode).propagate_dont_convert_inputs(fake_x)

        collective.meta["recompute"] = CheckpointPolicy.PREFER_RECOMPUTE
        floor_save.meta["recompute"] = CheckpointPolicy.MUST_SAVE
        broad.meta["recompute"] = CheckpointPolicy.PREFER_RECOMPUTE
        bwd.meta["autograd_backward"] = True

        min_cut_ac_pass(
            gm,
            (fake_x,),
            save_scope="all",
            memory_estimator="exact",
            runtime_estimator=lambda n: 1.0,
        )

        self.assertEqual(floor_save.meta["recompute"], CheckpointPolicy.MUST_SAVE)
        self.assertEqual(
            collective.meta["recompute"], CheckpointPolicy.PREFER_RECOMPUTE
        )
        self.assertEqual(broad.meta["recompute"], CheckpointPolicy.MUST_SAVE)

    def test_allow_allowed_saves_marks_unsaveable_softmax_for_recompute(self):
        graph = fx.Graph()
        x = graph.placeholder("x")
        logits = graph.call_function(torch.ops.aten.clone.default, args=(x,))
        log_softmax = graph.call_function(
            torch.ops.aten._log_softmax.default, args=(logits, 1, False)
        )
        detach = graph.call_function(torch.ops.aten.detach.default, args=(log_softmax,))
        loss = graph.call_function(torch.ops.aten.sum.default, args=(log_softmax,))
        bwd = graph.call_function(torch.ops.aten.neg.default, args=(detach,))
        graph.output((loss, bwd))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        with FakeTensorMode() as fake_mode:
            fake_x = torch.empty(128, 256, device="cuda")
            FakeTensorProp(gm, mode=fake_mode).propagate_dont_convert_inputs(fake_x)

        bwd.meta["autograd_backward"] = True

        ac_allow_allowed_saves(gm)

        self.assertEqual(logits.meta["recompute"], CheckpointPolicy.MUST_SAVE)
        self.assertEqual(
            log_softmax.meta["recompute"], CheckpointPolicy.PREFER_RECOMPUTE
        )
        self.assertEqual(detach.meta["recompute"], CheckpointPolicy.PREFER_RECOMPUTE)
        self.assertNotIn(log_softmax, _save_candidates(gm))
        self.assertNotIn(detach, _save_candidates(gm))

        gm = selective_activation_remat_pass(gm, (fake_x,))
        self.assertTrue(
            any(
                node.target == torch.ops.aten._log_softmax.default
                and node.meta.get("ac_recomputed") is True
                for node in gm.graph.nodes
            )
        )
        self.assertTrue(
            any(
                node.target == torch.ops.aten.detach.default
                and node.meta.get("ac_recomputed") is True
                for node in gm.graph.nodes
            )
        )
        self.assertFalse(
            any(
                node.target == torch.ops.aten.clone.default
                and node.meta.get("ac_recomputed") is True
                for node in gm.graph.nodes
            )
        )

    def test_allow_unsaveable_recomputes_keeps_save_pool_unchanged(self):
        graph = fx.Graph()
        x = graph.placeholder("x")
        logits = graph.call_function(torch.ops.aten.clone.default, args=(x,))
        log_softmax = graph.call_function(
            torch.ops.aten._log_softmax.default, args=(logits, 1, False)
        )
        detach = graph.call_function(torch.ops.aten.detach.default, args=(log_softmax,))
        loss = graph.call_function(torch.ops.aten.sum.default, args=(log_softmax,))
        bwd = graph.call_function(torch.ops.aten.neg.default, args=(detach,))
        graph.output((loss, bwd))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        with FakeTensorMode() as fake_mode:
            fake_x = torch.empty(128, 512, device="cuda")
            FakeTensorProp(gm, mode=fake_mode).propagate_dont_convert_inputs(fake_x)

        bwd.meta["autograd_backward"] = True
        ac_allow_unsaveable_recomputes(gm)

        self.assertEqual(logits.meta["recompute"], CheckpointPolicy.MUST_SAVE)
        self.assertEqual(
            log_softmax.meta["recompute"], CheckpointPolicy.PREFER_RECOMPUTE
        )
        self.assertEqual(detach.meta["recompute"], CheckpointPolicy.PREFER_RECOMPUTE)
        self.assertNotIn(loss, _save_candidates(gm))
        self.assertIsNone(loss.meta.get("recompute"))

    def test_min_cut_decomposes_log_softmax_to_compact_stats(self):
        graph = fx.Graph()
        x = graph.placeholder("x")
        logits = graph.call_function(torch.ops.aten.clone.default, args=(x,))
        logits_fp32 = graph.call_function(
            torch.ops.aten._to_copy.default,
            args=(logits,),
            kwargs={"dtype": torch.float32},
        )
        log_softmax = graph.call_function(
            torch.ops.aten._log_softmax.default, args=(logits_fp32, 1, False)
        )
        log_softmax.meta["custom"] = {"full_inductor_fuse_region": "txt_unemb_chunk_0"}
        detach = graph.call_function(torch.ops.aten.detach.default, args=(log_softmax,))
        loss = graph.call_function(torch.ops.aten.sum.default, args=(log_softmax,))
        bwd = graph.call_function(torch.ops.aten.neg.default, args=(detach,))
        graph.output((loss, bwd))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        with FakeTensorMode() as fake_mode:
            fake_x = torch.empty(64, 512, device="cuda", dtype=torch.bfloat16)
            FakeTensorProp(gm, mode=fake_mode).propagate_dont_convert_inputs(fake_x)

        bwd.meta["autograd_backward"] = True
        self.assertEqual(_decompose_log_softmax_for_min_cut(gm), 1)

        self.assertFalse(
            any(
                node.target == torch.ops.aten._log_softmax.default
                for node in gm.graph.nodes
            )
        )
        amax = next(
            node
            for node in gm.graph.nodes
            if node.target == torch.ops.aten.amax.default
        )
        log = next(
            node for node in gm.graph.nodes if node.target == torch.ops.aten.log.default
        )
        result = detach.args[0]
        self.assertIsInstance(result, fx.Node)
        self.assertEqual(logits.meta["recompute"], CheckpointPolicy.MUST_SAVE)
        self.assertEqual(logits_fp32.meta["recompute"], CheckpointPolicy.MUST_RECOMPUTE)
        self.assertEqual(amax.meta["recompute"], CheckpointPolicy.MUST_SAVE)
        self.assertEqual(log.meta["recompute"], CheckpointPolicy.MUST_SAVE)
        self.assertEqual(result.meta["recompute"], CheckpointPolicy.MUST_RECOMPUTE)
        self.assertEqual(detach.meta["recompute"], CheckpointPolicy.MUST_RECOMPUTE)
        self.assertEqual(amax.meta["val"].shape, torch.Size([64, 1]))
        self.assertEqual(log.meta["val"].shape, torch.Size([64, 1]))
        self.assertEqual(result.meta["val"].shape, torch.Size([64, 512]))
        self.assertEqual(
            amax.meta["custom"]["full_inductor_fuse_region"], "txt_unemb_chunk_0"
        )

    def test_log_softmax_anchor_walks_pointwise_chain_to_matmul(self):
        graph = fx.Graph()
        x = graph.placeholder("x")
        w = graph.placeholder("w")
        scale = graph.placeholder("scale")
        bias = graph.placeholder("bias")
        mm = graph.call_function(torch.ops.aten.mm.default, args=(x, w))
        scaled = graph.call_function(torch.ops.aten.mul.Tensor, args=(mm, scale))
        logits = graph.call_function(torch.ops.aten.add.Tensor, args=(scaled, bias))
        logits_fp32 = graph.call_function(
            torch.ops.aten._to_copy.default,
            args=(logits,),
            kwargs={"dtype": torch.float32},
        )
        log_softmax = graph.call_function(
            torch.ops.aten._log_softmax.default, args=(logits_fp32, 1, False)
        )
        detach = graph.call_function(torch.ops.aten.detach.default, args=(log_softmax,))
        loss = graph.call_function(torch.ops.aten.sum.default, args=(log_softmax,))
        bwd = graph.call_function(torch.ops.aten.neg.default, args=(detach,))
        graph.output((loss, bwd))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        with FakeTensorMode() as fake_mode:
            fake_x = torch.empty(16, 64, device="cuda", dtype=torch.bfloat16)
            fake_w = torch.empty(64, 128, device="cuda", dtype=torch.bfloat16)
            fake_scale = torch.empty(128, device="cuda", dtype=torch.bfloat16)
            fake_bias = torch.empty(128, device="cuda", dtype=torch.bfloat16)
            FakeTensorProp(gm, mode=fake_mode).propagate_dont_convert_inputs(
                fake_x, fake_w, fake_scale, fake_bias
            )

        bwd.meta["autograd_backward"] = True
        self.assertEqual(_decompose_log_softmax_for_min_cut(gm), 1)

        self.assertEqual(mm.meta["recompute"], CheckpointPolicy.MUST_SAVE)
        self.assertEqual(scaled.meta["recompute"], CheckpointPolicy.MUST_RECOMPUTE)
        self.assertEqual(logits.meta["recompute"], CheckpointPolicy.MUST_RECOMPUTE)
        self.assertEqual(logits_fp32.meta["recompute"], CheckpointPolicy.MUST_RECOMPUTE)
        amax = next(
            node
            for node in gm.graph.nodes
            if node.target == torch.ops.aten.amax.default
        )
        log = next(
            node for node in gm.graph.nodes if node.target == torch.ops.aten.log.default
        )
        self.assertEqual(amax.meta["recompute"], CheckpointPolicy.MUST_SAVE)
        self.assertEqual(log.meta["recompute"], CheckpointPolicy.MUST_SAVE)

    def test_allow_unsaveable_recomputes_marks_softmax_input_chain(self):
        graph = fx.Graph()
        x = graph.placeholder("x")
        logits = graph.call_function(torch.ops.aten.clone.default, args=(x,))
        logits_fp32 = graph.call_function(
            torch.ops.aten._to_copy.default,
            args=(logits,),
            kwargs={"dtype": torch.float32},
        )
        logits_view = graph.call_function(
            torch.ops.aten.view.default, args=(logits_fp32, [64, 512])
        )
        log_softmax = graph.call_function(
            torch.ops.aten._log_softmax.default, args=(logits_view, 1, False)
        )
        detach = graph.call_function(torch.ops.aten.detach.default, args=(log_softmax,))
        loss = graph.call_function(torch.ops.aten.sum.default, args=(log_softmax,))
        bwd = graph.call_function(torch.ops.aten.neg.default, args=(detach,))
        graph.output((loss, bwd))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        with FakeTensorMode() as fake_mode:
            fake_x = torch.empty(128, 256, device="cuda", dtype=torch.bfloat16)
            FakeTensorProp(gm, mode=fake_mode).propagate_dont_convert_inputs(fake_x)

        bwd.meta["autograd_backward"] = True
        ac_allow_unsaveable_recomputes(gm)

        self.assertEqual(logits.meta["recompute"], CheckpointPolicy.MUST_SAVE)
        self.assertEqual(logits_fp32.meta["recompute"], CheckpointPolicy.MUST_RECOMPUTE)
        self.assertEqual(logits_view.meta["recompute"], CheckpointPolicy.MUST_RECOMPUTE)
        self.assertEqual(
            log_softmax.meta["recompute"], CheckpointPolicy.PREFER_RECOMPUTE
        )
        self.assertEqual(detach.meta["recompute"], CheckpointPolicy.PREFER_RECOMPUTE)

        with patch(
            "torchtitan.experiments.graph_trainer.min_cut_ac._log_payload"
        ) as log_payload:
            self.assertEqual(_min_cut(gm), set())

        self.assertFalse(
            any(
                call.args[0] == "unbounded_cut"
                for call in log_payload.mock_calls
                if call.args
            )
        )

        gm = selective_activation_remat_pass(gm, (fake_x,))
        self.assertTrue(
            any(
                node.target == torch.ops.aten._to_copy.default
                and node.meta.get("ac_recomputed") is True
                for node in gm.graph.nodes
            )
        )
        self.assertTrue(
            any(
                node.target == torch.ops.aten.view.default
                and node.meta.get("ac_recomputed") is True
                for node in gm.graph.nodes
            )
        )
        self.assertTrue(
            any(
                node.target == torch.ops.aten._log_softmax.default
                and node.meta.get("ac_recomputed") is True
                for node in gm.graph.nodes
            )
        )
        self.assertFalse(
            any(
                node.target == torch.ops.aten.clone.default
                and node.meta.get("ac_recomputed") is True
                for node in gm.graph.nodes
            )
        )

    def test_recomputable_rng_is_forced_saved_before_remat(self):
        gm, example_inputs, rng_node = self._make_rng_graph(
            CheckpointPolicy.PREFER_RECOMPUTE
        )

        min_cut_ac_pass(
            gm,
            example_inputs,
            memory_estimator="exact",
            runtime_estimator=lambda n: 1.0,
        )

        self.assertEqual(rng_node.meta["recompute"], CheckpointPolicy.MUST_SAVE)
        _peak_after_remat(gm, example_inputs)

    def test_relax_relaxable_must_saves_keeps_uncuttable_hard(self):
        gm, _ = self._make_save_candidate_graph()
        sin_node = self._find(gm, torch.ops.aten.sin.default, recomputed=False)
        sin_node.meta["recompute"] = CheckpointPolicy.MUST_SAVE

        ac_relax_relaxable_must_saves(gm)

        self.assertEqual(sin_node.meta["recompute"], CheckpointPolicy.PREFER_SAVE)

        rng_gm, _, rng_node = self._make_rng_graph(CheckpointPolicy.MUST_SAVE)

        ac_relax_relaxable_must_saves(rng_gm)

        self.assertEqual(rng_node.meta["recompute"], CheckpointPolicy.MUST_SAVE)

    def test_must_recompute_rng_errors_before_remat(self):
        gm, example_inputs, _ = self._make_rng_graph(CheckpointPolicy.MUST_RECOMPUTE)

        with self.assertRaisesRegex(RuntimeError, "MUST_RECOMPUTE"):
            min_cut_ac_pass(
                gm,
                example_inputs,
                memory_estimator="exact",
                runtime_estimator=lambda n: 1.0,
            )

    def test_saved_collective_tag_is_preserved(self):
        gm, example_inputs, sin_node = self._make_collective_closure_graph()
        collective = next(
            n for n in gm.graph.nodes if n.target is fake_c10d_functional_wait
        )
        collective.meta["recompute"] = CheckpointPolicy.MUST_SAVE

        min_cut_ac_pass(
            gm,
            example_inputs,
            max_peak_increase_gb=1.0,
            save_scope="all",
            memory_estimator="exact",
            runtime_estimator=lambda n: 1.0,
        )

        self.assertEqual(collective.meta["recompute"], CheckpointPolicy.MUST_SAVE)
        self.assertEqual(sin_node.meta["recompute"], CheckpointPolicy.MUST_SAVE)

    def test_recomputed_collective_is_not_a_save_candidate(self):
        gm, example_inputs, _ = self._make_collective_closure_graph()
        collective = next(
            n for n in gm.graph.nodes if n.target is fake_c10d_functional_wait
        )

        min_cut_ac_pass(
            gm,
            example_inputs,
            max_peak_increase_gb=1.0,
            memory_estimator="exact",
            runtime_estimator=lambda n: 1.0,
        )

        self.assertEqual(
            collective.meta["recompute"], CheckpointPolicy.PREFER_RECOMPUTE
        )
        self.assertNotIn(collective, _save_candidates(gm))

    def test_peak_budget_rejects_negative_absolute_target(self):
        gm, example_inputs = self._make_interior_cutpoint_graph()

        with self.assertRaisesRegex(ValueError, "target peak negative"):
            min_cut_ac_pass(
                gm,
                example_inputs,
                max_peak_increase_gb=-1.0,
                runtime_estimator=lambda n: 1.0,
            )


if __name__ == "__main__":
    run_tests()
