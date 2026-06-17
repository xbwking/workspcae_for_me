# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def _load_module():
    module_path = Path("scripts/bench_fully_async_message_queue.py")
    spec = importlib.util.spec_from_file_location("_bench_fully_async_message_queue", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_bench_fully_async_message_queue"] = module
    spec.loader.exec_module(module)
    return module


def test_local_benchmark_reports_expected_call_reduction():
    bench = _load_module()

    result = bench.run_local_benchmark(num_samples=16, batch_size=4, payload_bytes=8)

    assert result["single_consumed"] == 16
    assert result["batched_consumed"] == 16
    assert result["single_get_call_count"] == 16
    assert result["batched_get_call_count"] == 4
    assert result["call_count_reduction"] == 4
    assert result["single_elapsed_ms"] >= 0
    assert result["batched_elapsed_ms"] >= 0


def test_cli_json_mode_is_machine_readable():
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/bench_fully_async_message_queue.py",
            "--mode",
            "local",
            "--num-samples",
            "9",
            "--batch-size",
            "4",
            "--payload-bytes",
            "0",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    result = json.loads(completed.stdout)

    assert result["single_consumed"] == 9
    assert result["batched_consumed"] == 9
    assert result["single_get_call_count"] == 9
    assert result["batched_get_call_count"] == 3
    assert result["call_count_reduction"] == 3


def test_cli_rejects_invalid_arguments():
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/bench_fully_async_message_queue.py",
            "--num-samples",
            "0",
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "--num-samples must be positive" in completed.stderr

