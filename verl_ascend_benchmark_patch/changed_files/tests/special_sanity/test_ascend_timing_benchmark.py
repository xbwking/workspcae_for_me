# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def _load_benchmark_module():
    module_path = Path("scripts/bench_ascend_verl_timing.py")
    spec = importlib.util.spec_from_file_location("_bench_ascend_verl_timing", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_bench_ascend_verl_timing"] = module
    spec.loader.exec_module(module)
    return module


def test_summarize_collects_l0_l1_metrics_from_file_logger_and_stdout(tmp_path):
    bench = _load_benchmark_module()
    metrics_path = tmp_path / "metrics.jsonl"
    stdout_path = tmp_path / "stdout.log"
    summary_path = tmp_path / "summary.json"
    csv_path = tmp_path / "timing_breakdown.csv"

    rows = [
        {
            "step": 1,
            "data": {
                "timing_s/step": 10.0,
                "timing_s/gen": 5.0,
                "timing_s/update_actor": 2.0,
                "timing_s/update_weights": 1.0,
                "perf/throughput": 100.0,
                "fully_async/message_queue_get_rpc_count": 8,
                "fully_async/cloudpickle_load_time": 0.4,
            },
        },
        {
            "step": 2,
            "data": {
                "timing_s/step": 12.0,
                "timing_s/gen": 6.0,
                "timing_s/update_actor": 3.0,
                "timing_s/update_weights": 1.5,
                "perf/throughput": 120.0,
                "fully_async/message_queue_get_rpc_count": 4,
                "fully_async/cloudpickle_load_time": 0.2,
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
        warmup_steps=0,
        measured_steps=None,
    )

    assert summary["step_count"] == 2
    assert summary["metrics"]["timing_s/step"]["mean"] == 11.0
    assert summary["metrics"]["timing_s/gen"]["pct_of_step_mean"] == 50.0
    assert summary["metrics"]["fully_async/message_queue_get_rpc_count"]["mean"] == 6.0
    assert summary["metrics"]["param_sync/send_recv_update_ms"]["mean"] == 100.0
    assert summary["metrics"]["weight_transfer/sender_copy_ms"]["mean"] == 3.0
    assert summary["metrics"]["weight_transfer/receiver_copy_ms"]["mean"] == 4.0
    assert summary_path.exists()
    assert "metric,mean,p50,p95,min,max,count,pct_of_step_mean" in csv_path.read_text()


def test_compare_reports_speedups_and_effectiveness(tmp_path):
    bench = _load_benchmark_module()
    baseline = {
        "metrics": {
            "timing_s/step": {"mean": 20.0},
            "ray/message_queue_get_rpc_count": {"mean": 512.0},
            "param_sync/send_recv_update_ms": {"mean": 1000.0},
            "perf/throughput": {"mean": 50.0},
        }
    }
    patched = {
        "metrics": {
            "timing_s/step": {"mean": 10.0},
            "ray/message_queue_get_rpc_count": {"mean": 2.0},
            "param_sync/send_recv_update_ms": {"mean": 800.0},
            "perf/throughput": {"mean": 100.0},
        }
    }
    baseline_path = tmp_path / "baseline.json"
    patched_path = tmp_path / "patched.json"
    output_path = tmp_path / "compare.json"
    baseline_path.write_text(json.dumps(baseline))
    patched_path.write_text(json.dumps(patched))

    comparison = bench.compare_summaries(baseline_path, patched_path, output_path)

    assert comparison["metrics"]["timing_s/step"]["speedup"] == 2.0
    assert comparison["metrics"]["perf/throughput"]["speedup"] == 2.0
    assert comparison["verdicts"]["message_queue batching"]["effective"] is True
    assert comparison["verdicts"]["param_sync send_recv"]["effective"] is True
    assert output_path.exists()


def test_run_dry_run_builds_ascend_benchmark_command(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/bench_ascend_verl_timing.py",
            "run",
            "--output-dir",
            str(tmp_path),
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
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)

    assert payload["env"]["VERL_FILE_LOGGER_PATH"].endswith("metrics.jsonl")
    assert "python3" in payload["command"][0]
    assert "-m" in payload["command"]
    assert "verl.trainer.main_ppo" in payload["command"]
    assert "trainer.device=npu" in payload["command"]
    assert "trainer.logger=['file','console']" in payload["command"]
    assert "global_profiler.tool=npu" in payload["command"]
    assert "global_profiler.steps=[2,3]" in payload["command"]


def test_summarize_cli_writes_report_files(tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    stdout_path = tmp_path / "stdout.log"
    summary_path = tmp_path / "summary.json"
    csv_path = tmp_path / "timing.csv"
    metrics_path.write_text(
        json.dumps({"step": 3, "data": {"timing_s/step": 2.0, "timing_s/gen": 1.0, "perf/throughput": 10.0}})
        + "\n"
    )
    stdout_path.write_text("")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/bench_ascend_verl_timing.py",
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
        check=True,
        capture_output=True,
        text=True,
    )

    assert "timing_s/step" in completed.stdout
    assert json.loads(summary_path.read_text())["metrics"]["timing_s/gen"]["mean"] == 1.0
    assert csv_path.exists()


def test_trainer_logs_checkpoint_manager_param_sync_breakdown():
    ppo_source = Path("verl/trainer/ppo/ray_trainer.py").read_text()
    fully_async_source = Path("verl/experimental/fully_async_policy/fully_async_trainer.py").read_text()

    assert "last_update_weights_timing" in ppo_source
    assert "last_update_weights_timing" in fully_async_source
