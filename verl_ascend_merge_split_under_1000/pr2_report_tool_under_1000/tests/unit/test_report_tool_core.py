# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "ascend_verl_timing_report.py"


def load_report_module():
    spec = importlib.util.spec_from_file_location("_ascend_verl_timing_report_core_ut", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_ascend_verl_timing_report_core_ut"] = module
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
                            "timing_s/update_weights": 1.0,
                            "perf/throughput": 100.0,
                            "fully_async/message_queue_get_rpc_count": 3,
                            "fully_async/cloudpickle_load_time": 0.3,
                            "fully_async/total_wait_time": 0.6,
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
                            "timing_s/update_weights": 1.5,
                            "perf/throughput": 110.0,
                            "fully_async/message_queue_get_rpc_count": 1,
                            "fully_async/cloudpickle_load_time": 0.1,
                            "fully_async/total_wait_time": 0.2,
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
                "{'bucket_count': 2, 'bucket_bytes': 1024, 'sender_copy_ms': 3.0, 'metadata_send_ms': 1.0}",
                "BucketedWeightReceiver stats: "
                "{'bucket_count': 2, 'bucket_bytes': 1024, 'clone_or_to_device_ms': 4.0, 'metadata_recv_ms': 1.5}",
            ]
        )
    )


class ReportParsingAndSummaryTest(unittest.TestCase):
    def setUp(self):
        self.report_mod = load_report_module()
        self.tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_parse_step_list_accepts_none_empty_and_values(self):
        self.assertIsNone(self.report_mod._parse_step_list(None))
        self.assertIsNone(self.report_mod._parse_step_list(""))
        self.assertEqual(self.report_mod._parse_step_list("1, 3,5"), [1, 3, 5])

    def test_read_jsonl_metrics_filters_warmup_and_measured_steps(self):
        path = self.tmpdir / "metrics.jsonl"
        path.write_text(
            "\n".join(
                [
                    json.dumps({"step": 1, "data": {"timing_s/step": 1.0}}),
                    json.dumps({"step": 2, "data": {"timing_s/step": 2.0}}),
                    json.dumps({"step": 3, "data": {"timing_s/step": 3.0}}),
                ]
            )
            + "\n"
        )

        rows = self.report_mod._read_jsonl_metrics(path, warmup_steps=1, measured_steps={3})

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["step"], 3)

    def test_read_jsonl_metrics_ignores_non_dict_data(self):
        path = self.tmpdir / "metrics.jsonl"
        path.write_text(
            json.dumps({"step": 1, "data": ["bad"]})
            + "\n"
            + json.dumps({"step": 2, "data": {"timing_s/step": 2.0}})
            + "\n"
        )

        rows = self.report_mod._read_jsonl_metrics(path)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["step"], 2)

    def test_literal_dict_from_line_parses_embedded_dict(self):
        parsed = self.report_mod._literal_dict_from_line("x {'a': 1, 'b': 2.5} y")

        self.assertEqual(parsed, {"a": 1, "b": 2.5})

    def test_literal_dict_from_line_returns_none_for_bad_input(self):
        self.assertIsNone(self.report_mod._literal_dict_from_line("no dict"))
        self.assertIsNone(self.report_mod._literal_dict_from_line("{'a':"))

    def test_parse_stdout_metrics_collects_checkpoint_and_bucket_stats(self):
        path = self.tmpdir / "stdout.log"
        path.write_text(
            "\n".join(
                [
                    "CheckpointEngineManager.update_weights timing: {'param_sync/total_ms': 100.0}",
                    "BucketedWeightSender stats: {'bucket_count': 2, 'bucket_bytes': 1024, 'sender_copy_ms': 3.0}",
                    "BucketedWeightReceiver stats: {'bucket_count': 3, 'clone_or_to_device_ms': 4.0}",
                ]
            )
        )

        values = self.report_mod._parse_stdout_metrics(path)

        self.assertEqual(values["param_sync/total_ms"], [100.0])
        self.assertEqual(values["weight_transfer/sender_bucket_count"], [2.0])
        self.assertEqual(values["weight_transfer/sender_bucket_bytes"], [1024.0])
        self.assertEqual(values["weight_transfer/sender_copy_ms"], [3.0])
        self.assertEqual(values["weight_transfer/receiver_bucket_count"], [3.0])
        self.assertEqual(values["weight_transfer/receiver_copy_ms"], [4.0])

    def test_add_metric_value_creates_ray_and_serialization_aliases(self):
        values = defaultdict(list)
        self.report_mod._add_metric_value(values, "fully_async/message_queue_get_rpc_count", 9)
        self.report_mod._add_metric_value(values, "fully_async/cloudpickle_load_time", 0.5)
        self.report_mod._add_metric_value(values, "fully_async/total_wait_time", 1.5)

        self.assertEqual(values["ray/message_queue_get_rpc_count"], [9.0])
        self.assertEqual(values["serialization/cloudpickle_load_s"], [0.5])
        self.assertEqual(values["ray/message_queue_get_wait_s"], [1.5])

    def test_percentile_handles_empty_singleton_and_interpolation(self):
        self.assertEqual(self.report_mod._percentile([], 0.95), 0.0)
        self.assertEqual(self.report_mod._percentile([5.0], 0.95), 5.0)
        self.assertEqual(self.report_mod._percentile([1.0, 3.0], 0.5), 2.0)

    def test_summarize_values_computes_pct_of_step_mean(self):
        metrics = self.report_mod._summarize_values({"timing_s/step": [10.0], "timing_s/gen": [6.0]})

        self.assertEqual(metrics["timing_s/gen"]["pct_of_step_mean"], 60.0)

    def test_summarize_writes_summary_and_timing_csv(self):
        run_dir = self.tmpdir / "run"
        write_run_dir(run_dir)
        summary_path = run_dir / "summary.json"
        csv_path = run_dir / "timing_breakdown.csv"

        summary = self.report_mod.summarize(
            metrics_jsonl=run_dir / "metrics.jsonl",
            stdout_log=run_dir / "stdout.log",
            output_summary=summary_path,
            output_csv=csv_path,
            warmup_steps=0,
            measured_steps=None,
        )

        self.assertEqual(summary["step_count"], 2)
        self.assertEqual(summary["metrics"]["timing_s/step"]["mean"], 11.0)
        self.assertEqual(summary["metrics"]["param_sync/send_recv_update_ms"]["mean"], 80.0)
        self.assertTrue(summary_path.exists())
        self.assertTrue(csv_path.exists())


class ReportFormattingTest(unittest.TestCase):
    def setUp(self):
        self.report_mod = load_report_module()

    def test_metric_unit_classifies_seconds_milliseconds_bytes_and_throughput(self):
        self.assertEqual(self.report_mod._metric_unit("timing_s/step"), "s")
        self.assertEqual(self.report_mod._metric_unit("serialization/cloudpickle_load_s"), "s")
        self.assertEqual(self.report_mod._metric_unit("rollout/token_latency"), "s")
        self.assertEqual(self.report_mod._metric_unit("param_sync/total_ms"), "ms")
        self.assertEqual(self.report_mod._metric_unit("weight_transfer/sender_copy_ms"), "ms")
        self.assertEqual(self.report_mod._metric_unit("weight_transfer/sender_bucket_bytes"), "ms")
        self.assertEqual(self.report_mod._metric_unit("perf/throughput"), "samples/s")
        self.assertEqual(self.report_mod._metric_unit("custom/count"), "")

    def test_format_bytes_scales_units(self):
        self.assertEqual(self.report_mod._format_bytes(10), "10.00 B")
        self.assertEqual(self.report_mod._format_bytes(2048), "2.00 KiB")
        self.assertEqual(self.report_mod._format_bytes(1024 * 1024), "1.00 MiB")

    def test_format_number_applies_precision_and_unit(self):
        self.assertEqual(self.report_mod._format_number(120.1234, "ms"), "120.12 ms")
        self.assertEqual(self.report_mod._format_number(12.1234, "ms"), "12.123 ms")
        self.assertEqual(self.report_mod._format_number(1.12349, "ms"), "1.1235 ms")

    def test_metric_matches_prefixes_matches_exact_and_prefix(self):
        self.assertTrue(self.report_mod._metric_matches_prefixes("timing_s/step", ("timing_s/",)))
        self.assertTrue(self.report_mod._metric_matches_prefixes("perf/throughput", ("perf/throughput",)))
        self.assertFalse(self.report_mod._metric_matches_prefixes("param_sync/total_ms", ("timing_s/",)))

    def test_markdown_metric_table_includes_pct_column_when_requested(self):
        lines = self.report_mod._markdown_metric_table(
            [
                {
                    "metric": "timing_s/gen",
                    "mean": 6.0,
                    "p50": 6.0,
                    "p95": 7.0,
                    "count": 2.0,
                    "pct_of_step_mean": 60.0,
                    "unit": "s",
                }
            ],
            include_pct=True,
        )

        self.assertIn("占 step", lines[0])
        self.assertIn("60.00%", "\n".join(lines))

    def test_markdown_metric_table_omits_pct_column_when_disabled(self):
        lines = self.report_mod._markdown_metric_table(
            [
                {
                    "metric": "perf/throughput",
                    "mean": 100.0,
                    "p50": 100.0,
                    "p95": 100.0,
                    "count": 2.0,
                    "unit": "samples/s",
                }
            ],
            include_pct=False,
        )

        self.assertNotIn("占 step", lines[0])
        self.assertIn("100.00 samples/s", "\n".join(lines))


class ArtifactInfoTest(unittest.TestCase):
    def setUp(self):
        self.report_mod = load_report_module()
        self.tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_artifact_info_reports_missing_path(self):
        info = self.report_mod._artifact_info(self.tmpdir / "missing")

        self.assertFalse(info["exists"])
        self.assertEqual(info["path"], str(self.tmpdir / "missing"))

    def test_artifact_info_reports_file_size(self):
        path = self.tmpdir / "stdout.log"
        path.write_text("hello")

        info = self.report_mod._artifact_info(path)

        self.assertTrue(info["exists"])
        self.assertEqual(info["type"], "file")
        self.assertEqual(info["bytes"], 5)

    def test_artifact_info_reports_directory_file_count_and_total_bytes(self):
        path = self.tmpdir / "npu_profile"
        path.mkdir()
        (path / "a.json").write_text("123")
        (path / "b.json").write_text("4567")

        info = self.report_mod._artifact_info(path)

        self.assertTrue(info["exists"])
        self.assertEqual(info["type"], "directory")
        self.assertEqual(info["file_count"], 2)
        self.assertEqual(info["bytes"], 7)
        self.assertEqual(info["sample_files"], ["a.json", "b.json"])


class ReportModelTest(unittest.TestCase):
    def setUp(self):
        self.report_mod = load_report_module()
        self.tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def summary(self):
        return {
            "warmup_steps": 0,
            "measured_steps": None,
            "step_count": 2,
            "metrics": {
                "timing_s/step": {"mean": 11.0, "p50": 11.0, "p95": 11.9, "min": 10.0, "max": 12.0, "count": 2.0},
                "timing_s/gen": {
                    "mean": 6.5,
                    "p50": 6.5,
                    "p95": 6.95,
                    "min": 6.0,
                    "max": 7.0,
                    "count": 2.0,
                    "pct_of_step_mean": 59.09,
                },
                "timing_s/update_actor": {
                    "mean": 2.25,
                    "p50": 2.25,
                    "p95": 2.48,
                    "min": 2.0,
                    "max": 2.5,
                    "count": 2.0,
                    "pct_of_step_mean": 20.45,
                },
                "perf/throughput": {"mean": 105.0, "p50": 105.0, "p95": 109.5, "min": 100.0, "max": 110.0, "count": 2.0},
                "param_sync/total_ms": {"mean": 100.0, "p50": 100.0, "p95": 100.0, "min": 100.0, "max": 100.0, "count": 1.0},
                "weight_transfer/sender_copy_ms": {"mean": 3.0, "p50": 3.0, "p95": 3.0, "min": 3.0, "max": 3.0, "count": 1.0},
                "ray/message_queue_get_rpc_count": {"mean": 2.0, "p50": 2.0, "p95": 2.0, "min": 1.0, "max": 3.0, "count": 2.0},
                "serialization/cloudpickle_load_s": {"mean": 0.2, "p50": 0.2, "p95": 0.29, "min": 0.1, "max": 0.3, "count": 2.0},
            },
        }

    def test_build_report_model_groups_metrics(self):
        run_dir = self.tmpdir / "run"
        run_dir.mkdir()

        report = self.report_mod._build_report_model(self.summary(), run_dir=run_dir, top_n=4)

        self.assertIn("end_to_end", report["groups"])
        self.assertIn("step_breakdown", report["groups"])
        self.assertIn("parameter_sync", report["groups"])
        self.assertIn("ray_and_serialization", report["groups"])
        self.assertEqual(report["step_count"], 2)

    def test_build_report_model_collects_key_metrics_and_missing_keys(self):
        run_dir = self.tmpdir / "run"
        run_dir.mkdir()

        report = self.report_mod._build_report_model(self.summary(), run_dir=run_dir, top_n=4)

        self.assertIn("timing_s/step", report["key_metrics"])
        self.assertIn("perf/throughput", report["key_metrics"])
        self.assertIn("param_sync/send_recv_update_ms", report["missing_key_metrics"])

    def test_build_report_model_orders_step_costs_by_pct(self):
        run_dir = self.tmpdir / "run"
        run_dir.mkdir()

        report = self.report_mod._build_report_model(self.summary(), run_dir=run_dir, top_n=4)

        self.assertEqual(report["top_step_costs"][0]["metric"], "timing_s/gen")
        self.assertEqual(report["top_step_costs"][1]["metric"], "timing_s/update_actor")

    def test_build_report_model_honors_top_n(self):
        run_dir = self.tmpdir / "run"
        run_dir.mkdir()

        report = self.report_mod._build_report_model(self.summary(), run_dir=run_dir, top_n=1)

        self.assertEqual(len(report["top_step_costs"]), 1)
        self.assertEqual(len(report["top_param_sync_costs"]), 1)
        self.assertEqual(len(report["top_ray_serialization_costs"]), 1)

    def test_write_top_metrics_csv_writes_expected_sections(self):
        run_dir = self.tmpdir / "run"
        run_dir.mkdir()
        report = self.report_mod._build_report_model(self.summary(), run_dir=run_dir, top_n=4)
        output = self.tmpdir / "top_metrics.csv"

        self.report_mod._write_top_metrics_csv(output, report)

        with output.open() as fp:
            rows = list(csv.DictReader(fp))

        sections = {row["section"] for row in rows}
        self.assertIn("top_step_costs", sections)
        self.assertIn("top_param_sync_costs", sections)
        self.assertIn("top_ray_serialization_costs", sections)

    def test_write_markdown_report_contains_core_sections(self):
        run_dir = self.tmpdir / "run"
        run_dir.mkdir()
        report = self.report_mod._build_report_model(self.summary(), run_dir=run_dir, top_n=4)
        output = self.tmpdir / "report.md"

        self.report_mod._write_markdown_report(output, report)

        text = output.read_text()
        self.assertIn("Ascend verl Benchmark 一页式报告", text)
        self.assertIn("## 结论视图", text)
        self.assertIn("## Step 耗时主项", text)
        self.assertIn("## 参数同步 / 权重传输", text)
        self.assertIn("## Ray / 序列化 / 异步队列", text)
        self.assertIn("## 产物索引", text)


class ReportRunTest(unittest.TestCase):
    def setUp(self):
        self.report_mod = load_report_module()
        self.tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_report_run_rebuilds_summary_and_writes_outputs(self):
        run_dir = self.tmpdir / "run"
        write_run_dir(run_dir)
        args = self.report_mod.parse_args(["--run-dir", str(run_dir), "--top-n", "4"])

        report = self.report_mod.report_run(args)

        self.assertTrue(Path(report["outputs"]["markdown"]).exists())
        self.assertTrue(Path(report["outputs"]["json"]).exists())
        self.assertTrue(Path(report["outputs"]["csv"]).exists())
        self.assertTrue((run_dir / "summary.json").exists())
        self.assertTrue((run_dir / "timing_breakdown.csv").exists())
        self.assertEqual(report["step_count"], 2)
        self.assertEqual(report["key_metrics"]["timing_s/step"]["mean"], 11.0)
        self.assertEqual(report["key_metrics"]["ray/message_queue_get_rpc_count"]["mean"], 2.0)
        self.assertEqual(report["artifacts"]["npu_profile"]["file_count"], 1)

    def test_report_run_uses_existing_summary_when_present(self):
        run_dir = self.tmpdir / "run"
        run_dir.mkdir()
        (run_dir / "metrics.jsonl").write_text("")
        (run_dir / "stdout.log").write_text("")
        (run_dir / "summary.json").write_text(
            json.dumps(
                {
                    "warmup_steps": 0,
                    "measured_steps": None,
                    "step_count": 1,
                    "metrics": {
                        "timing_s/step": {
                            "mean": 1.0,
                            "p50": 1.0,
                            "p95": 1.0,
                            "min": 1.0,
                            "max": 1.0,
                            "count": 1.0,
                        }
                    },
                }
            )
        )
        args = self.report_mod.parse_args(["--run-dir", str(run_dir)])

        report = self.report_mod.report_run(args)

        self.assertEqual(report["step_count"], 1)
        self.assertEqual(report["key_metrics"]["timing_s/step"]["mean"], 1.0)
        self.assertFalse((run_dir / "timing_breakdown.csv").exists())

    def test_report_run_supports_custom_summary_metrics_and_stdout_paths(self):
        run_dir = self.tmpdir / "run"
        run_dir.mkdir()
        metrics_path = self.tmpdir / "custom_metrics.jsonl"
        stdout_path = self.tmpdir / "custom_stdout.log"
        metrics_path.write_text(json.dumps({"step": 2, "data": {"timing_s/step": 2.0}}) + "\n")
        stdout_path.write_text("")
        args = self.report_mod.parse_args(
            [
                "--run-dir",
                str(run_dir),
                "--metrics-jsonl",
                str(metrics_path),
                "--stdout-log",
                str(stdout_path),
            ]
        )

        report = self.report_mod.report_run(args)

        self.assertEqual(report["step_count"], 1)
        self.assertEqual(report["key_metrics"]["timing_s/step"]["mean"], 2.0)


if __name__ == "__main__":
    unittest.main()
