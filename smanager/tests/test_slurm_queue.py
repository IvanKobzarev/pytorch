import importlib.util
import sys
import types
import unittest
from pathlib import Path


def load_server():
    old = sys.modules.get("smanager")
    pkg = types.ModuleType("smanager")
    pkg.__version__ = "test"
    sys.modules["smanager"] = pkg
    try:
        path = Path(__file__).parents[1] / "smanager" / "server.py"
        spec = importlib.util.spec_from_file_location("smanager_server_under_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if old is None:
            sys.modules.pop("smanager", None)
        else:
            sys.modules["smanager"] = old


s = load_server()


class SlurmQueueTest(unittest.TestCase):
    def test_parse_gpu_count(self):
        self.assertEqual(s._parse_gpu_count("gpu:a100:8(S:0-7)"), 8)
        self.assertEqual(s._parse_gpu_count("gres/gpu:h100:0(IDX:N/A)"), 0)
        self.assertEqual(s._parse_gpu_count("cpu=128,mem=1024G,gres/gpu=4"), 4)
        self.assertEqual(s._parse_gpu_count({"gres": "gpu:a100:2", "used": "gpu:a100:1"}), 3)

    def test_queue_summary_and_pending_rank(self):
        now = 1760000000
        jobs = [
            s._format_slurm_queue_job({
                "job_id": "100",
                "name": "260101_120000/run-0",
                "user": "alice",
                "account": "rigi",
                "partition": "gpu-a",
                "state": "RUNNING",
                "num_nodes": "2",
                "tres_per_node": "gpu:a100:4",
                "qos": "normal",
                "priority": "10",
            }, now),
            s._format_slurm_queue_job({
                "job_id": "101",
                "name": "wait-a",
                "user": "bob",
                "account": "rigi",
                "partition": "gpu-a",
                "state": "PENDING",
                "num_nodes": "1",
                "tres_per_node": "gpu:a100:8",
                "qos": "high",
                "reason": "Priority",
                "priority": "50",
            }, now),
            s._format_slurm_queue_job({
                "job_id": "102",
                "name": "wait-b",
                "user": "alice",
                "account": "rigi",
                "partition": "gpu-b",
                "state": "PENDING",
                "num_nodes": "1",
                "tres_per_node": "gpu:h100:8",
                "qos": "normal",
                "reason": "Resources",
                "priority": "100",
            }, now),
        ]
        pending = s._rank_pending_jobs(jobs)
        summary = s._summarize_slurm_queue(jobs, pending, None, now)

        self.assertEqual([job["job_id"] for job in pending], [102, 101])
        self.assertEqual(pending[0]["pending_rank"], 1)
        self.assertEqual(summary["jobs"], 3)
        self.assertEqual(summary["running"], 1)
        self.assertEqual(summary["pending"], 2)
        self.assertEqual(summary["gpus_running"], 8)
        self.assertEqual(summary["gpus_pending"], 16)
        self.assertEqual(summary["by_user"]["alice"]["jobs"], 2)
        self.assertEqual(summary["by_partition"]["gpu-a"]["pending"], 1)
        self.assertEqual(summary["pending_reasons"], {"Resources": 1, "Priority": 1})

    def test_node_summary(self):
        nodes = [
            s._format_slurm_node({
                "name": "n1",
                "partitions": ["gpu-a"],
                "state": ["IDLE"],
                "gres": "gpu:a100:8",
                "gres_used": "gpu:a100:0(IDX:N/A)",
                "cpus": 128,
            }),
            s._format_slurm_node({
                "name": "n2",
                "partitions": ["gpu-a", "gpu-b"],
                "state": ["MIXED"],
                "gres": "gpu:h100:8",
                "gres_used": "gpu:h100:3(IDX:0-2)",
                "cpus": 128,
            }),
            s._format_slurm_node({
                "name": "n3",
                "partitions": ["gpu-b"],
                "state": ["DRAIN"],
                "gres": "gpu:h100:8",
                "gres_used": "gpu:h100:0(IDX:N/A)",
                "cpus": 128,
            }),
        ]
        summary = s._summarize_slurm_nodes(nodes)

        self.assertEqual(summary["nodes"], 3)
        self.assertEqual(summary["gpus_total"], 24)
        self.assertEqual(summary["gpus_free"], 21)
        self.assertEqual(summary["gpus_available"], 13)
        self.assertEqual(summary["states"], {"idle": 1, "mixed": 1, "unavailable": 1})
        self.assertEqual(summary["by_partition"]["gpu-a"]["gpus_available"], 13)
        self.assertEqual(summary["by_partition"]["gpu-b"]["gpus_available"], 5)


if __name__ == "__main__":
    unittest.main()
