# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

import json
import subprocess
import sys


def test_run_dry_run_builds_npu_profiler_command(tmp_path):
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
    assert payload["metrics_jsonl"].endswith("metrics.jsonl")
    assert payload["stdout_log"].endswith("stdout.log")
    assert payload["command"][:3] == [sys.executable, "-m", "verl.trainer.main_ppo"]
    assert "trainer.device=npu" in payload["command"]
    assert "trainer.logger=['file','console']" in payload["command"]
    assert "global_profiler.tool=npu" in payload["command"]
    assert "global_profiler.steps=[2,3]" in payload["command"]
    assert f"global_profiler.save_path={tmp_path / 'npu_profile'}" in payload["command"]


def test_summarize_cli_writes_summary_and_csv(tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    stdout_path = tmp_path / "stdout.log"
    summary_path = tmp_path / "summary.json"
    csv_path = tmp_path / "timing.csv"
    metrics_path.write_text(
        json.dumps({"step": 3, "data": {"timing_s/step": 2.0, "timing_s/gen": 1.0, "perf/throughput": 10.0}})
        + "\n"
    )
    stdout_path.write_text(
        "CheckpointEngineManager.update_weights timing: "
        "{'param_sync/total_ms': 100.0, 'param_sync/send_recv_update_ms': 80.0}\n"
    )

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
    payload = json.loads(summary_path.read_text())
    assert payload["metrics"]["timing_s/gen"]["mean"] == 1.0
    assert payload["metrics"]["param_sync/send_recv_update_ms"]["mean"] == 80.0
    assert csv_path.exists()


def test_shell_wrapper_dry_run_passes_env_and_overrides(tmp_path):
    completed = subprocess.run(
        [
            "bash",
            "tests/special_npu/run_ascend_timing_breakdown_bench.sh",
            "--dry-run",
            "actor_rollout_ref.rollout.checkpoint_engine.backend=hccl",
        ],
        check=True,
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "MODEL_PATH": "/models/qwen",
            "TRAIN_FILES": "/data/train.parquet",
            "VAL_FILES": "/data/test.parquet",
            "OUTPUT_DIR": str(tmp_path),
            "TOTAL_STEPS": "5",
            "PROFILE_STEPS": "2",
        },
    )

    payload_start = completed.stdout.index("{")
    payload_end = completed.stdout.rindex("}") + 1
    payload = json.loads(completed.stdout[payload_start:payload_end])

    assert payload["env"]["VERL_FILE_LOGGER_PATH"] == str(tmp_path / "metrics.jsonl")
    assert "trainer.total_training_steps=5" in payload["command"]
    assert "global_profiler.steps=[2]" in payload["command"]
    assert "actor_rollout_ref.rollout.checkpoint_engine.backend=hccl" in payload["command"]

