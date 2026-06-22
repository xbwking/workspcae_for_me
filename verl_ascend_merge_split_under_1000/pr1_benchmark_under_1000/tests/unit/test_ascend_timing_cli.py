# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "bench_ascend_verl_timing.py"
WRAPPER = ROOT / "tests" / "special_npu" / "run_ascend_timing_breakdown_bench.sh"


def run_command(argv, **kwargs):
    return subprocess.run(argv, shell=False, **kwargs)

def load_bench_module():
    spec = importlib.util.spec_from_file_location("_bench_ascend_verl_timing_cli_ut", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_bench_ascend_verl_timing_cli_ut"] = module
    spec.loader.exec_module(module)
    return module


class AscendTimingCliDryRunTest(unittest.TestCase):
    def run_dry_run(self, tmpdir, *extra_args):
        completed = run_command(
            [
                sys.executable,
                str(SCRIPT),
                "run",
                "--output-dir",
                str(tmpdir),
                "--model-path",
                "/models/qwen",
                "--train-files",
                "/data/train.parquet",
                "--val-files",
                "/data/test.parquet",
                "--total-steps",
                "4",
                "--profile-steps",
                "2,3",
                "--dry-run",
                *extra_args,
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(completed.stdout)

    def test_run_dry_run_builds_npu_profiler_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            payload = self.run_dry_run(tmpdir)

        self.assertTrue(payload["env"]["VERL_FILE_LOGGER_PATH"].endswith("metrics.jsonl"))
        self.assertTrue(payload["metrics_jsonl"].endswith("metrics.jsonl"))
        self.assertTrue(payload["stdout_log"].endswith("stdout.log"))
        self.assertEqual(payload["command"][:3], ["python3", "-m", "verl.trainer.main_ppo"])
        self.assertIn("trainer.device=npu", payload["command"])
        self.assertIn("trainer.logger=['file','console']", payload["command"])
        self.assertIn("global_profiler.tool=npu", payload["command"])
        self.assertIn("global_profiler.steps=[2,3]", payload["command"])

    def test_dry_run_appends_user_overrides_after_default_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            payload = self.run_dry_run(
                tmpdir,
                "actor_rollout_ref.rollout.checkpoint_engine.backend=hccl",
                "trainer.logger=['console']",
            )

        self.assertEqual(payload["command"][-2], "actor_rollout_ref.rollout.checkpoint_engine.backend=hccl")
        self.assertEqual(payload["command"][-1], "trainer.logger=['console']")

    def test_dry_run_places_profiler_output_under_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            payload = self.run_dry_run(tmpdir)

        self.assertIn(f"global_profiler.save_path={tmpdir / 'npu_profile'}", payload["command"])
        self.assertEqual(payload["env"]["VERL_FILE_LOGGER_PATH"], str(tmpdir / "metrics.jsonl"))


class AscendTimingSubprocessCliTest(unittest.TestCase):
    def test_summarize_cli_writes_summary_and_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            metrics_path = tmpdir / "metrics.jsonl"
            stdout_path = tmpdir / "stdout.log"
            summary_path = tmpdir / "summary.json"
            csv_path = tmpdir / "timing.csv"
            metrics_path.write_text(
                json.dumps(
                    {
                        "step": 3,
                        "data": {
                            "timing_s/step": 2.0,
                            "timing_s/gen": 1.0,
                            "perf/throughput": 10.0,
                        },
                    }
                )
                + "\n"
            )
            stdout_path.write_text(
                "CheckpointEngineManager.update_weights timing: "
                "{'param_sync/total_ms': 100.0, 'param_sync/send_recv_update_ms': 80.0}\n"
            )

            completed = run_command(
                [
                    sys.executable,
                    str(SCRIPT),
                    "summarize",
                    "--metrics-jsonl",
                    str(metrics_path),
                    "--stdout-log",
                    str(stdout_path),
                    "--output-summary",
                    str(summary_path),
                    "--output-csv",
                    str(csv_path),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            payload = json.loads(summary_path.read_text())
            csv_exists = csv_path.exists()

        self.assertIn("timing_s/step", completed.stdout)
        self.assertEqual(payload["metrics"]["timing_s/gen"]["mean"], 1.0)
        self.assertEqual(payload["metrics"]["param_sync/send_recv_update_ms"]["mean"], 80.0)
        self.assertTrue(csv_exists)

    def test_compare_cli_writes_verdict_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            baseline = tmpdir / "baseline.json"
            patched = tmpdir / "patched.json"
            output = tmpdir / "compare.json"
            baseline.write_text(json.dumps({"metrics": {"timing_s/step": {"mean": 10.0}}}))
            patched.write_text(json.dumps({"metrics": {"timing_s/step": {"mean": 5.0}}}))

            completed = run_command(
                [
                    sys.executable,
                    str(SCRIPT),
                    "compare",
                    "--baseline-summary",
                    str(baseline),
                    "--patched-summary",
                    str(patched),
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(output.read_text())

        self.assertIn("end_to_end step", completed.stdout)
        self.assertEqual(payload["metrics"]["timing_s/step"]["speedup"], 2.0)
        self.assertTrue(payload["verdicts"]["end_to_end step"]["effective"])

    def test_run_benchmark_dry_run_returns_zero_without_subprocess_run(self):
        bench = load_bench_module()
        with tempfile.TemporaryDirectory() as tmp:
            args = bench.parse_args(
                [
                    "run",
                    "--output-dir",
                    tmp,
                    "--model-path",
                    "/models/qwen",
                    "--train-files",
                    "/data/train.parquet",
                    "--val-files",
                    "/data/test.parquet",
                    "--dry-run",
                ]
            )
            with mock.patch.object(bench.subprocess, "run") as run_mock:
                code = bench.run_benchmark(args)

        self.assertEqual(code, 0)
        run_mock.assert_not_called()

    def test_run_benchmark_invokes_subprocess_and_summarizes_when_not_dry_run(self):
        bench = load_bench_module()
        with tempfile.TemporaryDirectory() as tmp:
            args = bench.parse_args(
                [
                    "run",
                    "--output-dir",
                    tmp,
                    "--model-path",
                    "/models/qwen",
                    "--train-files",
                    "/data/train.parquet",
                    "--val-files",
                    "/data/test.parquet",
                    "--total-steps",
                    "1",
                ]
            )
            with mock.patch.object(bench.subprocess, "run", return_value=mock.Mock(returncode=7)) as run_mock:
                code = bench.run_benchmark(args)

        self.assertEqual(code, 7)
        run_mock.assert_called_once()


class AscendTimingShellWrapperTest(unittest.TestCase):
    def test_shell_wrapper_dry_run_passes_env_and_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            completed = run_command(
                [
                    "bash",
                    str(WRAPPER),
                    "--dry-run",
                    "actor_rollout_ref.rollout.checkpoint_engine.backend=hccl",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
                env={
                    "PATH": os.environ.get("PATH", "/usr/bin:/bin:/usr/sbin:/sbin"),
                    "MODEL_PATH": "/models/qwen",
                    "TRAIN_FILES": "/data/train.parquet",
                    "VAL_FILES": "/data/test.parquet",
                    "OUTPUT_DIR": str(tmpdir),
                    "TOTAL_STEPS": "5",
                    "PROFILE_STEPS": "2",
                },
            )
            payload_start = completed.stdout.index("{")
            payload_end = completed.stdout.rindex("}") + 1
            payload = json.loads(completed.stdout[payload_start:payload_end])

        self.assertEqual(payload["env"]["VERL_FILE_LOGGER_PATH"], str(tmpdir / "metrics.jsonl"))
        self.assertIn("trainer.total_training_steps=5", payload["command"])
        self.assertIn("global_profiler.steps=[2]", payload["command"])
        self.assertIn("actor_rollout_ref.rollout.checkpoint_engine.backend=hccl", payload["command"])

if __name__ == "__main__":
    unittest.main()
