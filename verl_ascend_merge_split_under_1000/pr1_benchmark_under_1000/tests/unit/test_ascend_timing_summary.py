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
SCRIPT = ROOT / "scripts" / "bench_ascend_verl_timing.py"


def load_bench_module():
    spec = importlib.util.spec_from_file_location("_bench_ascend_verl_timing_summary_ut", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_bench_ascend_verl_timing_summary_ut"] = module
    spec.loader.exec_module(module)
    return module


class StepListParsingTest(unittest.TestCase):
    def setUp(self):
        self.bench = load_bench_module()

    def test_parse_step_list_returns_none_for_none(self):
        self.assertIsNone(self.bench._parse_step_list(None))

    def test_parse_step_list_parses_single_value(self):
        self.assertEqual(self.bench._parse_step_list("3"), [3])

    def test_parse_step_list_ignores_spaces(self):
        self.assertEqual(self.bench._parse_step_list(" 1, 2, 3 "), [1, 2, 3])

    def test_parse_step_list_ignores_empty_items(self):
        self.assertEqual(self.bench._parse_step_list("1,,3,"), [1, 3])


class JsonlMetricReadingTest(unittest.TestCase):
    def setUp(self):
        self.bench = load_bench_module()
        self.tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write_metrics(self, rows):
        path = self.tmpdir / "metrics.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        return path

    def test_read_jsonl_metrics_returns_empty_for_missing_file(self):
        rows = self.bench._read_jsonl_metrics(self.tmpdir / "missing.jsonl")

        self.assertEqual(rows, [])

    def test_read_jsonl_metrics_skips_blank_lines(self):
        path = self.tmpdir / "metrics.jsonl"
        path.write_text('\n{"step": 1, "data": {"timing_s/step": 1.0}}\n\n')

        rows = self.bench._read_jsonl_metrics(path)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["step"], 1)

    def test_read_jsonl_metrics_filters_warmup_steps(self):
        path = self.write_metrics(
            [
                {"step": 1, "data": {"timing_s/step": 1.0}},
                {"step": 2, "data": {"timing_s/step": 2.0}},
                {"step": 3, "data": {"timing_s/step": 3.0}},
            ]
        )

        rows = self.bench._read_jsonl_metrics(path, warmup_steps=1)

        self.assertEqual([row["step"] for row in rows], [2, 3])

    def test_read_jsonl_metrics_filters_measured_steps(self):
        path = self.write_metrics(
            [
                {"step": 1, "data": {"timing_s/step": 1.0}},
                {"step": 2, "data": {"timing_s/step": 2.0}},
                {"step": 3, "data": {"timing_s/step": 3.0}},
            ]
        )

        rows = self.bench._read_jsonl_metrics(path, measured_steps={1, 3})

        self.assertEqual([row["step"] for row in rows], [1, 3])

    def test_read_jsonl_metrics_ignores_non_dict_data(self):
        path = self.write_metrics(
            [
                {"step": 1, "data": ["not", "a", "dict"]},
                {"step": 2, "data": {"timing_s/step": 2.0}},
            ]
        )

        rows = self.bench._read_jsonl_metrics(path)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["step"], 2)


class StdoutMetricParsingTest(unittest.TestCase):
    def setUp(self):
        self.bench = load_bench_module()
        self.tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_literal_dict_from_line_parses_python_dict_fragment(self):
        parsed = self.bench._literal_dict_from_line("prefix {'a': 1.0, 'b': 2} suffix")

        self.assertEqual(parsed, {"a": 1.0, "b": 2})

    def test_literal_dict_from_line_returns_none_without_braces(self):
        self.assertIsNone(self.bench._literal_dict_from_line("no dict here"))

    def test_literal_dict_from_line_returns_none_for_malformed_dict(self):
        self.assertIsNone(self.bench._literal_dict_from_line("prefix {'a': "))

    def test_parse_stdout_metrics_returns_empty_for_missing_file(self):
        values = self.bench._parse_stdout_metrics(self.tmpdir / "missing.log")

        self.assertEqual(dict(values), {})

    def test_parse_stdout_metrics_collects_checkpoint_timings(self):
        path = self.tmpdir / "stdout.log"
        path.write_text(
            "CheckpointEngineManager.update_weights timing: "
            "{'param_sync/total_ms': 100.0, 'param_sync/send_recv_update_ms': 80.0, 'ignored': 'x'}\n"
        )

        values = self.bench._parse_stdout_metrics(path)

        self.assertEqual(values["param_sync/total_ms"], [100.0])
        self.assertEqual(values["param_sync/send_recv_update_ms"], [80.0])
        self.assertNotIn("ignored", values)

    def test_parse_stdout_metrics_collects_sender_stats(self):
        path = self.tmpdir / "stdout.log"
        path.write_text(
            "BucketedWeightSender stats: "
            "{'bucket_count': 2, 'bucket_bytes': 2048, 'tensor_count': 8, "
            "'sync_ms': 7.0, 'metadata_send_ms': 1.0, 'sender_copy_ms': 3.0}\n"
        )

        values = self.bench._parse_stdout_metrics(path)

        self.assertEqual(values["weight_transfer/sender_bucket_count"], [2.0])
        self.assertEqual(values["weight_transfer/sender_bucket_bytes"], [2048.0])
        self.assertEqual(values["weight_transfer/sender_tensor_count"], [8.0])
        self.assertEqual(values["weight_transfer/sender_sync_ms"], [7.0])
        self.assertEqual(values["weight_transfer/metadata_send_ms"], [1.0])
        self.assertEqual(values["weight_transfer/sender_copy_ms"], [3.0])

    def test_parse_stdout_metrics_collects_receiver_stats(self):
        path = self.tmpdir / "stdout.log"
        path.write_text(
            "BucketedWeightReceiver stats: "
            "{'bucket_count': 4, 'bucket_bytes': 4096, 'tensor_count': 9, "
            "'sync_ms': 8.0, 'metadata_recv_ms': 1.5, 'clone_or_to_device_ms': 4.0}\n"
        )

        values = self.bench._parse_stdout_metrics(path)

        self.assertEqual(values["weight_transfer/receiver_bucket_count"], [4.0])
        self.assertEqual(values["weight_transfer/receiver_bucket_bytes"], [4096.0])
        self.assertEqual(values["weight_transfer/receiver_tensor_count"], [9.0])
        self.assertEqual(values["weight_transfer/receiver_sync_ms"], [8.0])
        self.assertEqual(values["weight_transfer/metadata_recv_ms"], [1.5])
        self.assertEqual(values["weight_transfer/receiver_copy_ms"], [4.0])


class SummaryMathTest(unittest.TestCase):
    def setUp(self):
        self.bench = load_bench_module()

    def test_percentile_empty_returns_zero(self):
        self.assertEqual(self.bench._percentile([], 0.95), 0.0)

    def test_percentile_singleton_returns_only_value(self):
        self.assertEqual(self.bench._percentile([7.0], 0.95), 7.0)

    def test_percentile_interpolates_between_values(self):
        self.assertEqual(self.bench._percentile([1.0, 3.0], 0.50), 2.0)
        self.assertEqual(self.bench._percentile([1.0, 2.0, 3.0, 4.0], 0.50), 2.5)

    def test_add_metric_value_keeps_numeric_values(self):
        values = defaultdict(list)
        self.bench._add_metric_value(values, "timing_s/step", 1.5)

        self.assertEqual(values["timing_s/step"], [1.5])

    def test_add_metric_value_ignores_non_numeric_values(self):
        values = defaultdict(list)
        self.bench._add_metric_value(values, "timing_s/step", "1.5")

        self.assertEqual(values, {})

    def test_add_metric_value_aliases_message_queue_rpc_count(self):
        values = defaultdict(list)
        self.bench._add_metric_value(values, "fully_async/message_queue_get_rpc_count", 8)

        self.assertEqual(values["fully_async/message_queue_get_rpc_count"], [8.0])
        self.assertEqual(values["ray/message_queue_get_rpc_count"], [8.0])

    def test_add_metric_value_aliases_cloudpickle_load_time(self):
        values = defaultdict(list)
        self.bench._add_metric_value(values, "fully_async/cloudpickle_load_time", 0.3)

        self.assertEqual(values["fully_async/cloudpickle_load_time"], [0.3])
        self.assertEqual(values["serialization/cloudpickle_load_s"], [0.3])

    def test_add_metric_value_aliases_total_wait_time(self):
        values = defaultdict(list)
        self.bench._add_metric_value(values, "fully_async/total_wait_time", 1.2)

        self.assertEqual(values["fully_async/total_wait_time"], [1.2])
        self.assertEqual(values["ray/message_queue_get_wait_s"], [1.2])

    def test_summarize_values_computes_mean_p50_p95_min_max_count(self):
        metrics = self.bench._summarize_values({"timing_s/step": [10.0, 12.0, 14.0]})

        self.assertEqual(metrics["timing_s/step"]["mean"], 12.0)
        self.assertEqual(metrics["timing_s/step"]["p50"], 12.0)
        self.assertAlmostEqual(metrics["timing_s/step"]["p95"], 13.8)
        self.assertEqual(metrics["timing_s/step"]["min"], 10.0)
        self.assertEqual(metrics["timing_s/step"]["max"], 14.0)
        self.assertEqual(metrics["timing_s/step"]["count"], 3.0)

    def test_summarize_values_computes_pct_of_step_mean_for_timing_metrics(self):
        metrics = self.bench._summarize_values({"timing_s/step": [10.0], "timing_s/gen": [6.0]})

        self.assertEqual(metrics["timing_s/gen"]["pct_of_step_mean"], 60.0)

    def test_summarize_values_does_not_compute_pct_for_step_itself(self):
        metrics = self.bench._summarize_values({"timing_s/step": [10.0]})

        self.assertNotIn("pct_of_step_mean", metrics["timing_s/step"])


class EndToEndSummarizeTest(unittest.TestCase):
    def setUp(self):
        self.bench = load_bench_module()
        self.tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_summarize_collects_file_logger_stdout_and_alias_metrics(self):
        metrics_path = self.tmpdir / "metrics.jsonl"
        stdout_path = self.tmpdir / "stdout.log"
        summary_path = self.tmpdir / "summary.json"
        csv_path = self.tmpdir / "timing_breakdown.csv"
        rows = [
            {"step": 1, "data": {"timing_s/step": 9.0, "timing_s/gen": 5.0, "perf/throughput": 90.0}},
            {
                "step": 2,
                "data": {
                    "timing_s/step": 10.0,
                    "timing_s/gen": 6.0,
                    "perf/throughput": 100.0,
                    "fully_async/message_queue_get_rpc_count": 8,
                    "fully_async/cloudpickle_load_time": 0.4,
                    "fully_async/total_wait_time": 1.2,
                },
            },
            {
                "step": 3,
                "data": {
                    "timing_s/step": 12.0,
                    "timing_s/gen": 7.0,
                    "perf/throughput": 120.0,
                    "fully_async/message_queue_get_rpc_count": 4,
                    "fully_async/cloudpickle_load_time": 0.2,
                    "fully_async/total_wait_time": 0.8,
                },
            },
        ]
        metrics_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        stdout_path.write_text(
            "\n".join(
                [
                    "CheckpointEngineManager.update_weights timing: "
                    "{'param_sync/send_recv_update_ms': 100.0, 'param_sync/build_pg_ms': 20.0}",
                    "BucketedWeightSender stats: "
                    "{'bucket_count': 2, 'bucket_bytes': 1024, 'sender_copy_ms': 3.0, 'metadata_send_ms': 1.0}",
                    "BucketedWeightReceiver stats: "
                    "{'bucket_count': 2, 'bucket_bytes': 1024, 'clone_or_to_device_ms': 4.0, 'metadata_recv_ms': 1.5}",
                ]
            )
        )

        summary = self.bench.summarize(
            metrics_jsonl=metrics_path,
            stdout_log=stdout_path,
            output_summary=summary_path,
            output_csv=csv_path,
            warmup_steps=1,
            measured_steps=[2, 3],
        )

        self.assertEqual(summary["step_count"], 2)
        self.assertEqual(summary["metrics"]["timing_s/step"]["mean"], 11.0)
        self.assertEqual(summary["metrics"]["timing_s/gen"]["pct_of_step_mean"], 650 / 11)
        self.assertEqual(summary["metrics"]["ray/message_queue_get_rpc_count"]["mean"], 6.0)
        self.assertAlmostEqual(summary["metrics"]["serialization/cloudpickle_load_s"]["mean"], 0.3)
        self.assertEqual(summary["metrics"]["ray/message_queue_get_wait_s"]["mean"], 1.0)
        self.assertEqual(summary["metrics"]["param_sync/send_recv_update_ms"]["mean"], 100.0)
        self.assertEqual(summary["metrics"]["weight_transfer/sender_copy_ms"]["mean"], 3.0)
        self.assertEqual(summary["metrics"]["weight_transfer/receiver_copy_ms"]["mean"], 4.0)
        self.assertTrue(summary_path.exists())
        self.assertIn("metric,mean,p50,p95,min,max,count,pct_of_step_mean", csv_path.read_text())

    def test_summarize_handles_missing_files(self):
        summary = self.bench.summarize(
            metrics_jsonl=self.tmpdir / "missing_metrics.jsonl",
            stdout_log=self.tmpdir / "missing_stdout.log",
            output_summary=self.tmpdir / "summary.json",
            output_csv=self.tmpdir / "timing_breakdown.csv",
            warmup_steps=0,
            measured_steps=None,
        )

        self.assertEqual(summary["step_count"], 0)
        self.assertEqual(summary["metrics"], {})

    def test_write_csv_orders_metrics_by_name(self):
        csv_path = self.tmpdir / "timing.csv"

        self.bench._write_csv(
            csv_path,
            {
                "timing_s/step": {"mean": 2.0, "p50": 2.0, "p95": 2.0, "min": 2.0, "max": 2.0, "count": 1.0},
                "perf/throughput": {"mean": 10.0, "p50": 10.0, "p95": 10.0, "min": 10.0, "max": 10.0, "count": 1.0},
            },
        )

        with csv_path.open() as fp:
            rows = list(csv.DictReader(fp))

        self.assertEqual([row["metric"] for row in rows], ["perf/throughput", "timing_s/step"])


class ComparisonTest(unittest.TestCase):
    def setUp(self):
        self.bench = load_bench_module()
        self.tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write_summary(self, name, metrics):
        path = self.tmpdir / name
        path.write_text(json.dumps({"metrics": metrics}))
        return path

    def test_compare_summaries_reports_lower_and_higher_better_speedups(self):
        baseline_path = self.write_summary(
            "baseline.json",
            {
                "timing_s/step": {"mean": 20.0},
                "param_sync/send_recv_update_ms": {"mean": 1000.0},
                "ray/message_queue_get_rpc_count": {"mean": 512.0},
                "perf/throughput": {"mean": 50.0},
            },
        )
        patched_path = self.write_summary(
            "patched.json",
            {
                "timing_s/step": {"mean": 10.0},
                "param_sync/send_recv_update_ms": {"mean": 800.0},
                "ray/message_queue_get_rpc_count": {"mean": 2.0},
                "perf/throughput": {"mean": 100.0},
            },
        )
        output_path = self.tmpdir / "compare.json"

        comparison = self.bench.compare_summaries(baseline_path, patched_path, output_path)

        self.assertEqual(comparison["metrics"]["timing_s/step"]["speedup"], 2.0)
        self.assertEqual(comparison["metrics"]["perf/throughput"]["speedup"], 2.0)
        self.assertTrue(comparison["verdicts"]["end_to_end step"]["effective"])
        self.assertTrue(comparison["verdicts"]["throughput"]["effective"])
        self.assertTrue(comparison["verdicts"]["param_sync send_recv"]["effective"])
        self.assertTrue(output_path.exists())

    def test_compare_summaries_skips_metrics_missing_from_one_side(self):
        baseline_path = self.write_summary("baseline.json", {"timing_s/step": {"mean": 20.0}})
        patched_path = self.write_summary("patched.json", {"perf/throughput": {"mean": 100.0}})

        comparison = self.bench.compare_summaries(baseline_path, patched_path, self.tmpdir / "compare.json")

        self.assertEqual(comparison["metrics"], {})

    def test_is_higher_better_only_matches_throughput_prefix(self):
        self.assertTrue(self.bench._is_higher_better("perf/throughput"))
        self.assertFalse(self.bench._is_higher_better("timing_s/step"))
        self.assertFalse(self.bench._is_higher_better("param_sync/total_ms"))

    def test_build_verdicts_marks_missing_metrics(self):
        verdicts = self.bench._build_verdicts({})

        self.assertIsNone(verdicts["message_queue batching"]["effective"])
        self.assertEqual(verdicts["message_queue batching"]["reason"], "metric missing")


if __name__ == "__main__":
    unittest.main()
