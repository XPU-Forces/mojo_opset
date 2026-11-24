"""Check cross-job sharding using real pytest collection without an NPU."""

import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pytest
import yaml

from npu_pytest import GroupedShard


class Collected:
    def pytest_collection_finish(self, session):
        self.nodes = {item.nodeid for item in session.items}


class ShardingTests(unittest.TestCase):
    def test_workflow_covers_each_backend_without_gaps_or_overlap(self):
        workflow = Path(__file__).parents[1] / "workflows" / "npu-910b.yml"
        jobs = yaml.safe_load(workflow.read_text())["jobs"]["accuracy"]["strategy"]["matrix"]["include"]
        self.assertEqual(len({job["runner"] for job in jobs}), len(jobs))
        self.assertEqual({job["implementation"] for job in jobs}, {"triton", "torch_npu", "native"})
        for backend in ("triton", "torch_npu", "native"):
            backend_jobs = [job for job in jobs if job["implementation"] == backend]
            totals = {job["shards"] for job in backend_jobs}
            self.assertEqual(len(totals), 1)
            ranks = [rank for job in backend_jobs
                     for rank in range(job["offset"], job["offset"] + job["cards"])]
            self.assertEqual(sorted(ranks), list(range(totals.pop())))

    def test_unequal_jobs_cover_each_case_once_and_keep_groups_together(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for module in ("a", "b"):
                path = Path(directory) / f"test_shards_{module}.py"
                path.write_text("import pytest\n" + "\n".join(
                    f'@pytest.mark.parametrize("value", range({count}))\n'
                    f'def test_group_{index}(value): raise AssertionError("must only collect")\n'
                    for index, count in enumerate((13, 11, 7, 5, 3, 2, 1))
                ))
                paths.append(str(path))

            def collect(*plugins):
                collected = Collected()
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    status = pytest.main(["--collect-only", "-q", "-c", "/dev/null", "--rootdir", directory, *paths],
                                         plugins=[*plugins, collected])
                self.assertIn(status, (pytest.ExitCode.OK, pytest.ExitCode.NO_TESTS_COLLECTED))
                return collected.nodes

            all_nodes = collect()
            seen = set()
            group_owner = {}
            # Separate jobs receive 1, 2 and 4 cards, sharing seven global shards.
            for offset, cards in ((0, 1), (1, 2), (3, 4)):
                for worker in range(cards):
                    rank = offset + worker
                    nodes = collect(GroupedShard(rank, 7))
                    self.assertFalse(seen & nodes)
                    seen |= nodes
                    for node in nodes:
                        group = node.split("[", 1)[0]
                        self.assertEqual(group_owner.setdefault(group, rank), rank)
            self.assertEqual(seen, all_nodes)
            self.assertEqual(len(seen), 84)
            # More shards than parameter groups can leave a valid empty worker.
            self.assertEqual(collect(GroupedShard(20, 21)), set())

    def test_invalid_job_ranges_fail_before_device_initialization(self):
        script = str(Path(__file__).with_name("npu_pytest.py"))
        for extra in (("--shard-offset", "-1"), ("--shard-count", "1"),
                      ("--shard-offset", "2", "--shard-count", "3")):
            result = subprocess.run(
                [sys.executable, script, "--implementation", "triton", "--cards", "2", *extra],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("card range", result.stderr)


if __name__ == "__main__":
    unittest.main()
