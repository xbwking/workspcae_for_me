# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def _load_report_module():
    module_path = Path("scripts/ascend_verl_timing_report.py")
    spec = importlib.util.spec_from_file_location("_ascend_verl_timing_report", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_ascend_verl_timing_report"] = module
    spec.loader.exec_module(module)
    return module


def _write_run_dir(run_dir: Path) -> None:
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


def test_report_run_rebuilds_summary_and_writes_outputs(tmp_path):
    report_mod = _load_report_module()
    run_dir = tmp_path / "run"
    _write_run_dir(run_dir)
    args = report_mod.parse_args(["--run-dir", str(run_dir), "--top-n", "4"])

    report = report_mod.report_run(args)

    assert Path(report["outputs"]["markdown"]).exists()
    assert Path(report["outputs"]["json"]).exists()
    assert Path(report["outputs"]["csv"]).exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "timing_breakdown.csv").exists()
    assert report["step_count"] == 2
    assert report["key_metrics"]["timing_s/step"]["mean"] == 11.0
    assert report["key_metrics"]["ray/message_queue_get_rpc_count"]["mean"] == 2.0
    assert report["artifacts"]["npu_profile"]["file_count"] == 1


def test_report_markdown_contains_sections_metrics_and_missing_keys(tmp_path):
    run_dir = tmp_path / "run"
    _write_run_dir(run_dir)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/report_ascend_verl_timing.py",
            "--run-dir",
            str(run_dir),
            "--top-n",
            "4",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    outputs = json.loads(completed.stdout)
    markdown = Path(outputs["markdown"]).read_text()
    top_csv = Path(outputs["csv"]).read_text()
    payload = json.loads(Path(outputs["json"]).read_text())

    assert "Ascend verl Benchmark 一页式报告" in markdown
    assert "## Step 耗时主项" in markdown
    assert "`timing_s/gen`" in markdown
    assert "`param_sync/send_recv_update_ms`" in markdown
    assert "`weight_transfer/sender_copy_ms`" in markdown
    assert "## 缺失指标" in markdown
    assert "section,metric,mean,p50,p95" in top_csv
    assert payload["top_param_sync_costs"][0]["metric"] == "param_sync/total_ms"


def test_report_uses_existing_summary_when_present(tmp_path):
    report_mod = _load_report_module()
    run_dir = tmp_path / "run"
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
                    "timing_s/step": {"mean": 1.0, "p50": 1.0, "p95": 1.0, "min": 1.0, "max": 1.0, "count": 1.0}
                },
            }
        )
    )
    args = report_mod.parse_args(["--run-dir", str(run_dir)])

    report = report_mod.report_run(args)

    assert report["step_count"] == 1
    assert report["key_metrics"]["timing_s/step"]["mean"] == 1.0
    assert (run_dir / "timing_breakdown.csv").exists() is False


def test_cli_supports_custom_output_paths(tmp_path):
    run_dir = tmp_path / "run"
    _write_run_dir(run_dir)
    output_md = tmp_path / "custom.md"
    output_json = tmp_path / "custom.json"
    output_csv = tmp_path / "custom.csv"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/report_ascend_verl_timing.py",
            "--run-dir",
            str(run_dir),
            "--output-md",
            str(output_md),
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    outputs = json.loads(completed.stdout)

    assert outputs == {"markdown": str(output_md), "json": str(output_json), "csv": str(output_csv)}
    assert output_md.exists()
    assert output_json.exists()
    assert output_csv.exists()


def test_install_script_copies_report_scripts(tmp_path):
    target = tmp_path / "verl"
    (target / "scripts").mkdir(parents=True)

    completed = subprocess.run(
        ["bash", "install_into_verl.sh", str(target)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Installed report tool" in completed.stdout
    assert (target / "scripts" / "ascend_verl_timing_report.py").exists()
    assert (target / "scripts" / "report_ascend_verl_timing.py").exists()

