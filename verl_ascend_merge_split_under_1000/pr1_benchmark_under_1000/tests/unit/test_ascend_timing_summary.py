# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

import importlib.util
import json
import sys
from pathlib import Path


def _load_bench():
    module_path = Path("scripts/bench_ascend_verl_timing.py")
    spec = importlib.util.spec_from_file_location("_bench_ascend_verl_timing", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_bench_ascend_verl_timing"] = module
    spec.loader.exec_module(module)
    return module


def test_summarize_collects_file_logger_stdout_and_alias_metrics(tmp_path):
    bench = _load_bench()
    metrics_path = tmp_path / "metrics.jsonl"
    stdout_path = tmp_path / "stdout.log"
    summary_path = tmp_path / "summary.json"
    csv_path = tmp_path / "timing_breakdown.csv"

    rows = [
        {
            "step": 1,
            "data": {
                "timing_s/step": 9.0,
                "timing_s/gen": 5.0,
                "perf/throughput": 90.0,
            },
        },
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

    summary = bench.summarize(
        metrics_jsonl=metrics_path,
        stdout_log=stdout_path,
        output_summary=summary_path,
        output_csv=csv_path,
        warmup_steps=1,
        measured_steps=[2, 3],
    )

    assert summary["step_count"] == 2
    assert summary["metrics"]["timing_s/step"]["mean"] == 11.0
    assert summary["metrics"]["timing_s/gen"]["pct_of_step_mean"] == 650 / 11
    assert summary["metrics"]["ray/message_queue_get_rpc_count"]["mean"] == 6.0
    assert summary["metrics"]["serialization/cloudpickle_load_s"]["mean"] == 0.3
    assert summary["metrics"]["ray/message_queue_get_wait_s"]["mean"] == 1.0
    assert summary["metrics"]["param_sync/send_recv_update_ms"]["mean"] == 100.0
    assert summary["metrics"]["weight_transfer/sender_copy_ms"]["mean"] == 3.0
    assert summary["metrics"]["weight_transfer/receiver_copy_ms"]["mean"] == 4.0
    assert summary_path.exists()
    assert "metric,mean,p50,p95,min,max,count,pct_of_step_mean" in csv_path.read_text()


def test_summarize_handles_missing_files(tmp_path):
    bench = _load_bench()

    summary = bench.summarize(
        metrics_jsonl=tmp_path / "missing_metrics.jsonl",
        stdout_log=tmp_path / "missing_stdout.log",
        output_summary=tmp_path / "summary.json",
        output_csv=tmp_path / "timing_breakdown.csv",
        warmup_steps=0,
        measured_steps=None,
    )

    assert summary["step_count"] == 0
    assert summary["metrics"] == {}


def test_compare_summaries_reports_lower_and_higher_better_speedups(tmp_path):
    bench = _load_bench()
    baseline_path = tmp_path / "baseline.json"
    patched_path = tmp_path / "patched.json"
    output_path = tmp_path / "compare.json"
    baseline_path.write_text(
        json.dumps(
            {
                "metrics": {
                    "timing_s/step": {"mean": 20.0},
                    "param_sync/send_recv_update_ms": {"mean": 1000.0},
                    "ray/message_queue_get_rpc_count": {"mean": 512.0},
                    "perf/throughput": {"mean": 50.0},
                }
            }
        )
    )
    patched_path.write_text(
        json.dumps(
            {
                "metrics": {
                    "timing_s/step": {"mean": 10.0},
                    "param_sync/send_recv_update_ms": {"mean": 800.0},
                    "ray/message_queue_get_rpc_count": {"mean": 2.0},
                    "perf/throughput": {"mean": 100.0},
                }
            }
        )
    )

    comparison = bench.compare_summaries(baseline_path, patched_path, output_path)

    assert comparison["metrics"]["timing_s/step"]["speedup"] == 2.0
    assert comparison["metrics"]["perf/throughput"]["speedup"] == 2.0
    assert comparison["verdicts"]["end_to_end step"]["effective"] is True
    assert comparison["verdicts"]["throughput"]["effective"] is True
    assert comparison["verdicts"]["param_sync send_recv"]["effective"] is True
    assert output_path.exists()

