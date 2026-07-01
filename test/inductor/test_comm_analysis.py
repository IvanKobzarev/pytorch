# Owner(s): ["module: inductor"]

import os
import sys
import warnings
from unittest import mock

import torch
import torch.distributed as dist


if not dist.is_available() or not dist.is_nccl_available():
    print("c10d NCCL not available, skipping tests", file=sys.stderr)
    sys.exit(0)

try:
    from torch.testing._internal.common_distributed import (
        requires_nccl,
        skip_if_lt_x_gpu,
    )
except ImportError:
    print("common_distributed not importable, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import run_tests, TestCase


class TestCommTopologyDetection(TestCase):
    def test_local_world_size_handles_masked_local_ranks(self):
        import torch._inductor.config as inductor_config
        from torch._inductor import comm_analysis

        with (
            inductor_config.patch(gpus_per_node=None),
            mock.patch.dict(
                os.environ,
                {"LOCAL_WORLD_SIZE": "8", "WORLD_SIZE": "8", "LOCAL_RANK": "0"},
                clear=True,
            ),
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "device_count", return_value=1),
        ):
            self.assertEqual(comm_analysis.get_gpus_per_node(), 8)

    def test_loopback_single_visible_gpu_launch_is_single_node(self):
        import torch._inductor.config as inductor_config
        from torch._inductor import comm_analysis

        with (
            inductor_config.patch(gpus_per_node=None),
            mock.patch.dict(
                os.environ,
                {
                    "WORLD_SIZE": "8",
                    "LOCAL_RANK": "0",
                    "MASTER_ADDR": "127.0.0.1",
                },
                clear=True,
            ),
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "device_count", return_value=1),
            mock.patch.object(
                comm_analysis,
                "get_gpu_type",
                return_value=comm_analysis.NVIDIA_GPU_TYPE.HOPPER,
            ),
            mock.patch.object(comm_analysis, "_has_nvlink", return_value=True),
        ):
            self.assertEqual(comm_analysis.get_gpus_per_node(), 8)
            self.assertEqual(
                comm_analysis.detect_interconnect(8),
                comm_analysis.InterconnectType.NVLINK_H100,
            )

    def test_single_visible_gpu_non_loopback_launch_stays_multi_node(self):
        import torch._inductor.config as inductor_config
        from torch._inductor import comm_analysis

        with (
            inductor_config.patch(gpus_per_node=None),
            mock.patch.dict(
                os.environ,
                {"WORLD_SIZE": "8", "LOCAL_RANK": "0", "MASTER_ADDR": "host0"},
                clear=True,
            ),
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "device_count", return_value=1),
            mock.patch.object(
                comm_analysis,
                "get_gpu_type",
                return_value=comm_analysis.NVIDIA_GPU_TYPE.HOPPER,
            ),
        ):
            self.assertEqual(comm_analysis.get_gpus_per_node(), 1)
            self.assertEqual(
                comm_analysis.detect_interconnect(8),
                comm_analysis.InterconnectType.IB_NDR,
            )


def _get_all_gather_node(group_size, group_name):
    """Trace a simple all_gather function and return the collective FX node."""

    def func(inp, group_size, group_name):
        out = torch.ops._c10d_functional.all_gather_into_tensor(
            inp, group_size, group_name
        )
        wait = torch.ops._c10d_functional.wait_tensor(out)
        return wait

    gm = make_fx(func)(torch.ones(4, 4), group_size, group_name)
    for n in gm.graph.nodes:
        if n.op == "call_function" and "all_gather_into_tensor" in str(n.target):
            return n
    raise RuntimeError("No all_gather_into_tensor node found in traced graph")


class TestNcclEstimateDeviceResolution(TestCase):
    """
    Tests for the device resolution fix in _nccl_estimate() inside
    estimate_nccl_collective_runtime_from_fx_node.
    """

    def _init_pg(self, backend, world_size=2):
        from torch.testing._internal.distributed.fake_pg import FakeStore

        store = FakeStore()
        dist.init_process_group(
            backend=backend, rank=0, world_size=world_size, store=store
        )
        pg = dist.group.WORLD
        group_name = "test_comm_analysis"
        torch._C._distributed_c10d._register_process_group(group_name, pg)
        return pg, group_name, pg.size()

    def _init_pg_real_store(self, backend, world_size=1):
        store = dist.HashStore()
        dist.init_process_group(
            backend=backend, rank=0, world_size=world_size, store=store
        )
        pg = dist.group.WORLD
        group_name = "test_comm_analysis"
        torch._C._distributed_c10d._register_process_group(group_name, pg)
        return pg, group_name, pg.size()

    def _destroy_pg(self):
        dist.destroy_process_group()

    def test_fake_backend_falls_back_to_analytical(self):
        """FAKE backend: _nccl_estimate returns None, falls back to analytical formula."""
        pg, group_name, group_size = self._init_pg("fake")
        try:
            node = _get_all_gather_node(group_size, group_name)
            from torch._inductor.comm_analysis import (
                estimate_nccl_collective_runtime_from_fx_node,
            )

            est_ms = estimate_nccl_collective_runtime_from_fx_node(
                node, use_nccl_estimator=True
            )
            self.assertGreater(est_ms, 0)

            est_ms_analytical = estimate_nccl_collective_runtime_from_fx_node(
                node, use_nccl_estimator=False
            )
            self.assertEqual(est_ms, est_ms_analytical)
        finally:
            self._destroy_pg()

    @requires_nccl()
    @skip_if_lt_x_gpu(1)
    def test_multi_backend_pg_resolves_to_nccl(self):
        """
        Multi-backend PG ("cpu:gloo,cuda:nccl"): We should resolve to the cuda device's backend.
        """
        torch.cuda.set_device(0)
        pg, group_name, group_size = self._init_pg_real_store("cpu:gloo,cuda:nccl")
        try:
            from torch.distributed.distributed_c10d import _get_pg_default_device

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                default_device = _get_pg_default_device(pg)
            self.assertEqual(default_device, torch.device("cpu"))

            nccl_backend = pg._get_backend(torch.device("cuda"))
            self.assertTrue(nccl_backend.supports_time_estimate)

            gloo_backend = pg._get_backend(torch.device("cpu"))
            self.assertFalse(gloo_backend.supports_time_estimate)
        finally:
            self._destroy_pg()

    @requires_nccl()
    @skip_if_lt_x_gpu(1)
    def test_single_nccl_backend_resolves_correctly(self):
        """Single NCCL backend PG: cuda device resolves to NCCL with time estimation."""
        torch.cuda.set_device(0)
        pg, group_name, group_size = self._init_pg_real_store("nccl")
        try:
            backend = pg._get_backend(torch.device("cuda"))
            self.assertTrue(backend.supports_time_estimate)
        finally:
            self._destroy_pg()


if __name__ == "__main__":
    run_tests()
