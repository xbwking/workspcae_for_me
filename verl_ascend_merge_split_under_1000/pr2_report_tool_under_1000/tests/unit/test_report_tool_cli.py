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
REPORT_SCRIPT = ROOT / "scripts" / "ascend_verl_timing_report.py"
WRAPPER_SCRIPT = ROOT / "scripts" / "report_ascend_verl_timing.py"
INSTALL_SCRIPT = ROOT / "install_into_verl.sh"


def load_report_module():
    spec = importlib.util.spec_from_file_location("_ascend_verl_timing_report_cli_ut", REPORT_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_ascend_verl_timing_report_cli_ut"] = module
    spec.loader.exec_module(module)
    return module


def write_run_dir(run_dir: Path) -> None:
    run_dir.mkdir()
    (run_dir / "npu_profile").mkdir()
    (run_dir / "npu_profile" / "trace.json").write_text("{}")
    (run_dir / "metrics.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "step": 1,
                        "data": {
                            "timing_s/step": 10.0,
                            "timing_s/gen": 6.0,
                            "timing_s/update_actor": 2.0,
                            "perf/throughput": 100.0,
                            "fully_async/message_queue_get_rpc_count": 3,
                            "fully_async/cloudpickle_load_time": 0.3,
                        },
                    }
                ),
                json.dumps(
                    {
                        "step": 2,
                        "data": {
                            "timing_s/step": 12.0,
                            "timing_s/gen": 7.0,
                            "timing_s/update_actor": 2.5,
                            "perf/throughput": 110.0,
                            "fully_async/message_queue_get_rpc_count": 1,
                            "fully_async/cloudpickle_load_time": 0.1,
                        },
                    }
                ),
            ]
        )
        + "\n"
    )
    (run_dir / "stdout.log").write_text(
        "\n".join(
            [
                "CheckpointEngineManager.update_weights timing: "
                "{'param_sync/total_ms': 100.0, 'param_sync/send_recv_update_ms': 80.0}",
                "BucketedWeightSender stats: "
                "{'bucket_count': 2, 'bucket_bytes': 1024, 'sender_copy_ms': 3.0}",
            ]
        )
    )


class ReportArgparseTest(unittest.TestCase):
    def setUp(self):
        self.report_mod = load_report_module()

    def test_parse_args_requires_run_dir(self):
        with self.assertRaises(SystemExit):
            self.report_mod.parse_args([])

    def test_parse_args_uses_expected_defaults(self):
        args = self.report_mod.parse_args(["--run-dir", "/tmp/run"])

        self.assertEqual(args.run_dir, Path("/tmp/run"))
        self.assertIsNone(args.summary)
        self.assertIsNone(args.metrics_jsonl)
        self.assertIsNone(args.stdout_log)
        self.assertIsNone(args.output_md)
        self.assertIsNone(args.output_json)
        self.assertIsNone(args.output_csv)
        self.assertEqual(args.warmup_steps, 0)
        self.assertIsNone(args.measured_steps)
        self.assertEqual(args.top_n, 8)

    def test_parse_args_accepts_all_optional_paths(self):
        args = self.report_mod.parse_args(
            [
                "--run-dir",
                "/tmp/run",
                "--summary",
                "/tmp/summary.json",
                "--metrics-jsonl",
                "/tmp/metrics.jsonl",
                "--stdout-log",
                "/tmp/stdout.log",
                "--output-md",
                "/tmp/report.md",
                "--output-json",
                "/tmp/report.json",
                "--output-csv",
                "/tmp/top.csv",
                "--warmup-steps",
                "2",
                "--measured-steps",
                "3,4",
                "--top-n",
                "5",
            ]
        )

        self.assertEqual(args.summary, Path("/tmp/summary.json"))
        self.assertEqual(args.metrics_jsonl, Path("/tmp/metrics.jsonl"))
        self.assertEqual(args.stdout_log, Path("/tmp/stdout.log"))
        self.assertEqual(args.output_md, Path("/tmp/report.md"))
        self.assertEqual(args.output_json, Path("/tmp/report.json"))
        self.assertEqual(args.output_csv, Path("/tmp/top.csv"))
        self.assertEqual(args.warmup_steps, 2)
        self.assertEqual(args.measured_steps, "3,4")
        self.assertEqual(args.top_n, 5)


class ReportCliExecutionTest(unittest.TestCase):
    def test_wrapper_cli_writes_default_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            write_run_dir(run_dir)
            completed = subprocess.run(
                [sys.executable, str(WRAPPER_SCRIPT), "--run-dir", str(run_dir), "--top-n", "4"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            outputs = json.loads(completed.stdout)
            markdown = Path(outputs["markdown"]).read_text()
            top_csv = Path(outputs["csv"]).read_text()
            payload = json.loads(Path(outputs["json"]).read_text())
            top_param_metrics = [item["metric"] for item in payload["top_param_sync_costs"]]

        self.assertIn("Ascend verl Benchmark 一页式报告", markdown)
        self.assertIn("## Step 耗时主项", markdown)
        self.assertIn("`timing_s/gen`", markdown)
        self.assertIn("`param_sync/send_recv_update_ms`", markdown)
        self.assertIn("`weight_transfer/sender_copy_ms`", markdown)
        self.assertIn("## 缺失指标", markdown)
        self.assertIn("section,metric,mean,p50,p95", top_csv)
        self.assertIn("param_sync/total_ms", top_param_metrics)

    def test_direct_cli_writes_default_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            write_run_dir(run_dir)
            completed = subprocess.run(
                [sys.executable, str(REPORT_SCRIPT), "--run-dir", str(run_dir)],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            outputs = json.loads(completed.stdout)
            md_exists = (run_dir / "report.md").exists()
            json_exists = (run_dir / "report.json").exists()
            csv_exists = (run_dir / "top_metrics.csv").exists()

        self.assertEqual(outputs["markdown"], str(run_dir / "report.md"))
        self.assertEqual(outputs["json"], str(run_dir / "report.json"))
        self.assertEqual(outputs["csv"], str(run_dir / "top_metrics.csv"))
        self.assertTrue(md_exists)
        self.assertTrue(json_exists)
        self.assertTrue(csv_exists)

    def test_cli_supports_custom_output_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            run_dir = tmpdir / "run"
            write_run_dir(run_dir)
            output_md = tmpdir / "custom.md"
            output_json = tmpdir / "custom.json"
            output_csv = tmpdir / "custom.csv"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(WRAPPER_SCRIPT),
                    "--run-dir",
                    str(run_dir),
                    "--output-md",
                    str(output_md),
                    "--output-json",
                    str(output_json),
                    "--output-csv",
                    str(output_csv),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            outputs = json.loads(completed.stdout)
            md_exists = output_md.exists()
            json_exists = output_json.exists()
            csv_exists = output_csv.exists()

        self.assertEqual(outputs, {"markdown": str(output_md), "json": str(output_json), "csv": str(output_csv)})
        self.assertTrue(md_exists)
        self.assertTrue(json_exists)
        self.assertTrue(csv_exists)

    def test_cli_applies_warmup_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            write_run_dir(run_dir)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(WRAPPER_SCRIPT),
                    "--run-dir",
                    str(run_dir),
                    "--warmup-steps",
                    "1",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            outputs = json.loads(completed.stdout)
            payload = json.loads(Path(outputs["json"]).read_text())

        self.assertEqual(payload["step_count"], 1)
        self.assertEqual(payload["key_metrics"]["timing_s/step"]["mean"], 12.0)

    def test_cli_applies_measured_steps_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            write_run_dir(run_dir)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(WRAPPER_SCRIPT),
                    "--run-dir",
                    str(run_dir),
                    "--measured-steps",
                    "1",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            outputs = json.loads(completed.stdout)
            payload = json.loads(Path(outputs["json"]).read_text())

        self.assertEqual(payload["step_count"], 1)
        self.assertEqual(payload["key_metrics"]["timing_s/step"]["mean"], 10.0)

    def test_cli_uses_existing_summary_when_summary_argument_is_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            run_dir = tmpdir / "run"
            run_dir.mkdir()
            summary = tmpdir / "custom_summary.json"
            summary.write_text(
                json.dumps(
                    {
                        "warmup_steps": 0,
                        "measured_steps": None,
                        "step_count": 1,
                        "metrics": {
                            "timing_s/step": {
                                "mean": 7.0,
                                "p50": 7.0,
                                "p95": 7.0,
                                "min": 7.0,
                                "max": 7.0,
                                "count": 1.0,
                            }
                        },
                    }
                )
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(WRAPPER_SCRIPT),
                    "--run-dir",
                    str(run_dir),
                    "--summary",
                    str(summary),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            outputs = json.loads(completed.stdout)
            payload = json.loads(Path(outputs["json"]).read_text())

        self.assertEqual(payload["step_count"], 1)
        self.assertEqual(payload["key_metrics"]["timing_s/step"]["mean"], 7.0)

    def test_main_prints_output_mapping(self):
        report_mod = load_report_module()
        fake_args = report_mod.parse_args(["--run-dir", "/tmp/run"])
        fake_report = {"outputs": {"markdown": "/tmp/report.md", "json": "/tmp/report.json", "csv": "/tmp/top.csv"}}
        with mock.patch.object(report_mod, "parse_args", return_value=fake_args) as parse_mock:
            with mock.patch.object(report_mod, "report_run", return_value=fake_report) as run_mock:
                with mock.patch("builtins.print") as print_mock:
                    code = report_mod.main(["--run-dir", "/tmp/run"])

        self.assertEqual(code, 0)
        parse_mock.assert_called_once_with(["--run-dir", "/tmp/run"])
        run_mock.assert_called_once_with(fake_args)
        printed = print_mock.call_args[0][0]
        self.assertIn("markdown", printed)
        self.assertIn("/tmp/report.md", printed)


class InstallScriptTest(unittest.TestCase):
    def test_install_script_copies_report_scripts(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "verl"
            (target / "scripts").mkdir(parents=True)
            completed = subprocess.run(
                ["bash", str(INSTALL_SCRIPT), str(target)],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            installed_core = (target / "scripts" / "ascend_verl_timing_report.py").exists()
            installed_wrapper = (target / "scripts" / "report_ascend_verl_timing.py").exists()

        self.assertIn("Installed report tool", completed.stdout)
        self.assertTrue(installed_core)
        self.assertTrue(installed_wrapper)

    def test_install_script_rejects_target_without_scripts_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "verl"
            target.mkdir()
            completed = subprocess.run(
                ["bash", str(INSTALL_SCRIPT), str(target)],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("target does not look like a verl repo root", completed.stdout + completed.stderr)

    def test_install_script_fails_when_target_argument_missing(self):
        completed = subprocess.run(
            ["bash", str(INSTALL_SCRIPT)],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("Usage:", completed.stderr + completed.stdout)

    def test_installed_wrapper_can_generate_report_in_target_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            target = tmpdir / "verl"
            run_dir = tmpdir / "run"
            (target / "scripts").mkdir(parents=True)
            write_run_dir(run_dir)
            subprocess.run(
                ["bash", str(INSTALL_SCRIPT), str(target)],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(target / "scripts" / "report_ascend_verl_timing.py"),
                    "--run-dir",
                    str(run_dir),
                ],
                cwd=target,
                check=True,
                capture_output=True,
                text=True,
            )
            outputs = json.loads(completed.stdout)
            md_exists = Path(outputs["markdown"]).exists()
            json_exists = Path(outputs["json"]).exists()
            csv_exists = Path(outputs["csv"]).exists()

        self.assertTrue(md_exists)
        self.assertTrue(json_exists)
        self.assertTrue(csv_exists)


class MarkdownContentTest(unittest.TestCase):
    def test_generated_markdown_lists_artifact_statuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            write_run_dir(run_dir)
            subprocess.run(
                [sys.executable, str(WRAPPER_SCRIPT), "--run-dir", str(run_dir)],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            markdown = (run_dir / "report.md").read_text()

        self.assertIn("`metrics_jsonl`", markdown)
        self.assertIn("`stdout_log`", markdown)
        self.assertIn("`summary_json`", markdown)
        self.assertIn("`timing_breakdown_csv`", markdown)
        self.assertIn("`npu_profile`", markdown)

    def test_generated_markdown_contains_usage_advice(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            write_run_dir(run_dir)
            subprocess.run(
                [sys.executable, str(WRAPPER_SCRIPT), "--run-dir", str(run_dir)],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            markdown = (run_dir / "report.md").read_text()

        self.assertIn("## 使用建议", markdown)
        self.assertIn("timing_s/step", markdown)
        self.assertIn("perf/throughput", markdown)
        self.assertIn("param_sync/*", markdown)
        self.assertIn("ray/*", markdown)

    def test_generated_json_keeps_grouped_metrics_for_follow_up_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            write_run_dir(run_dir)
            subprocess.run(
                [sys.executable, str(WRAPPER_SCRIPT), "--run-dir", str(run_dir)],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads((run_dir / "report.json").read_text())

        self.assertIn("groups", payload)
        self.assertIn("end_to_end", payload["groups"])
        self.assertIn("step_breakdown", payload["groups"])
        self.assertIn("parameter_sync", payload["groups"])
        self.assertIn("ray_and_serialization", payload["groups"])

    def test_generated_csv_is_small_top_metric_index_not_full_raw_dump(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            write_run_dir(run_dir)
            subprocess.run(
                [sys.executable, str(WRAPPER_SCRIPT), "--run-dir", str(run_dir), "--top-n", "2"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            lines = (run_dir / "top_metrics.csv").read_text().splitlines()

        self.assertGreaterEqual(len(lines), 2)
        self.assertLessEqual(len(lines), 1 + 2 * 3)


if __name__ == "__main__":
    unittest.main()
