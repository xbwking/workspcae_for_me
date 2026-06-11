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
"""Ascend timing-breakdown benchmark harness for verl.

This script has three jobs:
1. Build and run a reproducible Ascend GRPO/PPO benchmark command.
2. Parse verl file-logger metrics and stdout timing logs into a normalized summary.
3. Compare baseline and patched summaries to validate optimization impact.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


LOWER_IS_BETTER_PREFIXES = (
    "timing_s/",
    "timing_per_token_ms/",
    "param_sync/",
    "weight_transfer/",
    "ray/",
    "serialization/",
    "rollout/",
    "fully_async/",
)
HIGHER_IS_BETTER_PREFIXES = ("perf/throughput",)


def _parse_step_list(value: str | None) -> list[int] | None:
    if value is None or value == "":
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _format_omega_list(values: list[int] | None) -> str:
    if values is None:
        return "null"
    return "[" + ",".join(str(value) for value in values) + "]"


def _read_jsonl_metrics(path: Path, warmup_steps: int = 0, measured_steps: set[int] | None = None) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        step = int(row.get("step", 0))
        if step <= warmup_steps:
            continue
        if measured_steps is not None and step not in measured_steps:
            continue
        data = row.get("data", {})
        if isinstance(data, dict):
            rows.append({"step": step, "data": data})
    return rows


def _literal_dict_from_line(line: str) -> dict[str, Any] | None:
    start = line.find("{")
    end = line.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        parsed = ast.literal_eval(line[start : end + 1])
    except (SyntaxError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _parse_stdout_metrics(path: Path) -> dict[str, list[float]]:
    values: dict[str, list[float]] = defaultdict(list)
    if not path.exists():
        return values

    for line in path.read_text(errors="replace").splitlines():
        if "CheckpointEngineManager.update_weights timing:" in line:
            parsed = _literal_dict_from_line(line)
            if parsed:
                for key, value in parsed.items():
                    if isinstance(value, int | float):
                        values[key].append(float(value))
        elif "BucketedWeightSender stats:" in line:
            parsed = _literal_dict_from_line(line)
            if parsed:
                _append_weight_transfer_metrics(values, parsed, role="sender")
        elif "BucketedWeightReceiver stats:" in line:
            parsed = _literal_dict_from_line(line)
            if parsed:
                _append_weight_transfer_metrics(values, parsed, role="receiver")
    return values


def _append_weight_transfer_metrics(values: dict[str, list[float]], stats: dict[str, Any], role: str) -> None:
    direct_keys = {
        "bucket_count": f"weight_transfer/{role}_bucket_count",
        "bucket_bytes": f"weight_transfer/{role}_bucket_bytes",
        "tensor_count": f"weight_transfer/{role}_tensor_count",
        "sync_ms": f"weight_transfer/{role}_sync_ms",
        "metadata_send_ms": "weight_transfer/metadata_send_ms",
        "metadata_recv_ms": "weight_transfer/metadata_recv_ms",
        "sender_copy_ms": "weight_transfer/sender_copy_ms",
        "clone_or_to_device_ms": "weight_transfer/receiver_copy_ms",
    }
    for source_key, metric_key in direct_keys.items():
        value = stats.get(source_key)
        if isinstance(value, int | float):
            values[metric_key].append(float(value))


def _add_metric_value(values: dict[str, list[float]], key: str, value: Any) -> None:
    if not isinstance(value, int | float):
        return
    numeric = float(value)
    values[key].append(numeric)
    if key == "fully_async/message_queue_get_rpc_count":
        values["ray/message_queue_get_rpc_count"].append(numeric)
    elif key == "fully_async/cloudpickle_load_time":
        values["serialization/cloudpickle_load_s"].append(numeric)
    elif key == "fully_async/total_wait_time":
        values["ray/message_queue_get_wait_s"].append(numeric)


def _collect_values(rows: list[dict[str, Any]], stdout_values: dict[str, list[float]]) -> dict[str, list[float]]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for key, value in row["data"].items():
            _add_metric_value(values, key, value)
    for key, items in stdout_values.items():
        values[key].extend(items)
    return values


def _percentile(items: list[float], percentile: float) -> float:
    if not items:
        return 0.0
    ordered = sorted(items)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _summarize_values(values: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    metrics = {}
    step_mean = statistics.fmean(values["timing_s/step"]) if values.get("timing_s/step") else None
    for key in sorted(values):
        items = values[key]
        if not items:
            continue
        mean = statistics.fmean(items)
        metric = {
            "mean": mean,
            "p50": _percentile(items, 0.50),
            "p95": _percentile(items, 0.95),
            "min": min(items),
            "max": max(items),
            "count": float(len(items)),
        }
        if step_mean and key.startswith("timing_s/") and key != "timing_s/step":
            metric["pct_of_step_mean"] = mean / step_mean * 100
        metrics[key] = metric
    return metrics


def summarize(
    metrics_jsonl: Path,
    stdout_log: Path,
    output_summary: Path,
    output_csv: Path,
    warmup_steps: int,
    measured_steps: list[int] | None,
) -> dict[str, Any]:
    measured_set = set(measured_steps) if measured_steps else None
    rows = _read_jsonl_metrics(metrics_jsonl, warmup_steps=warmup_steps, measured_steps=measured_set)
    values = _collect_values(rows, _parse_stdout_metrics(stdout_log))
    metrics = _summarize_values(values)
    summary = {
        "metrics_jsonl": str(metrics_jsonl),
        "stdout_log": str(stdout_log),
        "warmup_steps": warmup_steps,
        "measured_steps": measured_steps,
        "step_count": len(rows),
        "metrics": metrics,
    }
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(summary, indent=2, sort_keys=True))
    _write_csv(output_csv, metrics)
    return summary


def _write_csv(path: Path, metrics: dict[str, dict[str, float]]) -> None:
    columns = ["metric", "mean", "p50", "p95", "min", "max", "count", "pct_of_step_mean"]
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        for key in sorted(metrics):
            row = {"metric": key}
            row.update(metrics[key])
            writer.writerow(row)


def _metric_mean(summary: dict[str, Any], key: str) -> float | None:
    metric = summary.get("metrics", {}).get(key)
    if not metric:
        return None
    value = metric.get("mean")
    return float(value) if isinstance(value, int | float) else None


def _is_higher_better(key: str) -> bool:
    return any(key == prefix or key.startswith(prefix) for prefix in HIGHER_IS_BETTER_PREFIXES)


def _speedup(key: str, baseline: float, patched: float) -> float:
    if patched == 0:
        return float("inf") if baseline > 0 else 1.0
    if baseline == 0:
        return 1.0
    if _is_higher_better(key):
        return patched / baseline
    return baseline / patched


def compare_summaries(baseline_path: Path, patched_path: Path, output_path: Path) -> dict[str, Any]:
    baseline = json.loads(baseline_path.read_text())
    patched = json.loads(patched_path.read_text())
    keys = sorted(set(baseline.get("metrics", {})) | set(patched.get("metrics", {})))
    metrics = {}
    for key in keys:
        baseline_mean = _metric_mean(baseline, key)
        patched_mean = _metric_mean(patched, key)
        if baseline_mean is None or patched_mean is None:
            continue
        delta = patched_mean - baseline_mean
        metrics[key] = {
            "baseline_mean": baseline_mean,
            "patched_mean": patched_mean,
            "delta": delta,
            "delta_pct": delta / baseline_mean * 100 if baseline_mean else 0.0,
            "speedup": _speedup(key, baseline_mean, patched_mean),
            "higher_is_better": _is_higher_better(key),
        }

    comparison = {
        "baseline": str(baseline_path),
        "patched": str(patched_path),
        "metrics": metrics,
        "verdicts": _build_verdicts(metrics),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(comparison, indent=2, sort_keys=True))
    return comparison


def _build_verdicts(metrics: dict[str, dict[str, float]]) -> dict[str, dict[str, Any]]:
    checks = {
        "message_queue batching": "ray/message_queue_get_rpc_count",
        "message_queue wait": "ray/message_queue_get_wait_s",
        "serialization cloudpickle load": "serialization/cloudpickle_load_s",
        "param_sync send_recv": "param_sync/send_recv_update_ms",
        "weight sender copy": "weight_transfer/sender_copy_ms",
        "weight receiver copy": "weight_transfer/receiver_copy_ms",
        "end_to_end step": "timing_s/step",
        "throughput": "perf/throughput",
    }
    verdicts = {}
    for name, key in checks.items():
        metric = metrics.get(key)
        if not metric:
            verdicts[name] = {"metric": key, "effective": None, "reason": "metric missing"}
            continue
        effective = metric["speedup"] > 1.05
        verdicts[name] = {
            "metric": key,
            "effective": effective,
            "speedup": metric["speedup"],
            "baseline_mean": metric["baseline_mean"],
            "patched_mean": metric["patched_mean"],
        }
    return verdicts


def build_run_command(args: argparse.Namespace) -> tuple[dict[str, str], list[str], Path, Path]:
    output_dir = Path(args.output_dir)
    metrics_jsonl = output_dir / "metrics.jsonl"
    stdout_log = output_dir / "stdout.log"
    profile_steps = _parse_step_list(args.profile_steps)

    command = [
        args.python,
        "scripts/run_ppo_with_ascend_benchmark_patches.py",
        "algorithm.adv_estimator=grpo",
        f"data.train_files={args.train_files}",
        f"data.val_files={args.val_files}",
        f"data.train_batch_size={args.train_batch_size}",
        f"data.max_prompt_length={args.max_prompt_length}",
        f"data.max_response_length={args.max_response_length}",
        "data.filter_overlong_prompts=True",
        "data.truncation='error'",
        f"actor_rollout_ref.model.path={args.model_path}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={args.ppo_mini_batch_size}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={args.ppo_micro_batch_size_per_gpu}",
        "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.kl_loss_coef=0.001",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.actor.fsdp_config.param_offload=False",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={args.log_prob_micro_batch_size_per_gpu}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={args.tensor_model_parallel_size}",
        "actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={args.gpu_memory_utilization}",
        f"actor_rollout_ref.rollout.n={args.rollout_n}",
        f"actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes={args.bucket_mb}",
        "actor_rollout_ref.rollout.enable_chunked_prefill=False",
        "actor_rollout_ref.rollout.calculate_log_probs=True",
        f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={args.log_prob_micro_batch_size_per_gpu}",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        "algorithm.use_kl_in_reward=False",
        "trainer.critic_warmup=0",
        "trainer.device=npu",
        "trainer.logger=['file','console']",
        f"trainer.project_name={args.project_name}",
        f"trainer.experiment_name={args.experiment_name}",
        f"trainer.n_gpus_per_node={args.n_gpus_per_node}",
        f"trainer.nnodes={args.nnodes}",
        "trainer.save_freq=-1",
        "trainer.test_freq=-1",
        "trainer.val_before_train=False",
        f"trainer.total_training_steps={args.total_steps}",
        "global_profiler.tool=npu",
        f"global_profiler.steps={_format_omega_list(profile_steps)}",
        f"global_profiler.save_path={output_dir / 'npu_profile'}",
        "actor_rollout_ref.actor.profiler.enable=True",
        "actor_rollout_ref.actor.profiler.all_ranks=False",
        "actor_rollout_ref.actor.profiler.ranks=[0]",
        "actor_rollout_ref.actor.profiler.tool_config.npu.discrete=True",
        "actor_rollout_ref.actor.profiler.tool_config.npu.contents=['npu','cpu']",
        "actor_rollout_ref.actor.profiler.tool_config.npu.level=level0",
        "actor_rollout_ref.actor.profiler.tool_config.npu.analysis=False",
        "actor_rollout_ref.ref.profiler.enable=True",
        "actor_rollout_ref.ref.profiler.all_ranks=False",
        "actor_rollout_ref.ref.profiler.ranks=[0]",
        "actor_rollout_ref.ref.profiler.tool_config.npu.discrete=True",
        "actor_rollout_ref.ref.profiler.tool_config.npu.contents=['npu','cpu']",
        "actor_rollout_ref.ref.profiler.tool_config.npu.level=level0",
        "actor_rollout_ref.ref.profiler.tool_config.npu.analysis=False",
    ]
    command.extend(args.overrides)

    env = os.environ.copy()
    env["VERL_FILE_LOGGER_PATH"] = str(metrics_jsonl)
    return env, command, metrics_jsonl, stdout_log


def run_benchmark(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env, command, metrics_jsonl, stdout_log = build_run_command(args)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "env": {"VERL_FILE_LOGGER_PATH": env["VERL_FILE_LOGGER_PATH"]},
                    "command": command,
                    "metrics_jsonl": str(metrics_jsonl),
                    "stdout_log": str(stdout_log),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    with stdout_log.open("w") as fp:
        process = subprocess.run(command, env=env, stdout=fp, stderr=subprocess.STDOUT, text=True)
    summarize(
        metrics_jsonl=metrics_jsonl,
        stdout_log=stdout_log,
        output_summary=output_dir / "summary.json",
        output_csv=output_dir / "timing_breakdown.csv",
        warmup_steps=args.warmup_steps,
        measured_steps=_parse_step_list(args.measured_steps),
    )
    return process.returncode


def _print_summary_table(summary: dict[str, Any]) -> None:
    print(f"step_count={summary['step_count']}")
    print("metric,mean,p50,p95,pct_of_step_mean")
    for key, metric in summary["metrics"].items():
        if not (
            key.startswith("timing_s/")
            or key.startswith("param_sync/")
            or key.startswith("weight_transfer/")
            or key.startswith("ray/")
            or key.startswith("serialization/")
            or key.startswith("fully_async/")
            or key.startswith("perf/")
        ):
            continue
        pct = metric.get("pct_of_step_mean", "")
        print(f"{key},{metric['mean']:.6f},{metric['p50']:.6f},{metric['p95']:.6f},{pct}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run an Ascend verl timing benchmark.")
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument("--model-path", required=True)
    run_parser.add_argument("--train-files", required=True)
    run_parser.add_argument("--val-files", required=True)
    run_parser.add_argument("--python", default="python3")
    run_parser.add_argument("--total-steps", type=int, default=8)
    run_parser.add_argument("--warmup-steps", type=int, default=2)
    run_parser.add_argument("--measured-steps", default=None)
    run_parser.add_argument("--profile-steps", default=None)
    run_parser.add_argument("--nnodes", type=int, default=1)
    run_parser.add_argument("--n-gpus-per-node", type=int, default=8)
    run_parser.add_argument("--train-batch-size", type=int, default=16)
    run_parser.add_argument("--max-prompt-length", type=int, default=512)
    run_parser.add_argument("--max-response-length", type=int, default=128)
    run_parser.add_argument("--ppo-mini-batch-size", type=int, default=8)
    run_parser.add_argument("--ppo-micro-batch-size-per-gpu", type=int, default=1)
    run_parser.add_argument("--log-prob-micro-batch-size-per-gpu", type=int, default=1)
    run_parser.add_argument("--tensor-model-parallel-size", type=int, default=2)
    run_parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    run_parser.add_argument("--rollout-n", type=int, default=2)
    run_parser.add_argument("--bucket-mb", type=int, default=4096)
    run_parser.add_argument("--project-name", default="verl_ascend_timing_bench")
    run_parser.add_argument("--experiment-name", default="timing_breakdown")
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("overrides", nargs="*")

    summarize_parser = subparsers.add_parser("summarize", help="Summarize benchmark metrics and stdout logs.")
    summarize_parser.add_argument("--metrics-jsonl", type=Path, required=True)
    summarize_parser.add_argument("--stdout-log", type=Path, required=True)
    summarize_parser.add_argument("--output-summary", type=Path, required=True)
    summarize_parser.add_argument("--output-csv", type=Path, required=True)
    summarize_parser.add_argument("--warmup-steps", type=int, default=0)
    summarize_parser.add_argument("--measured-steps", default=None)

    compare_parser = subparsers.add_parser("compare", help="Compare baseline and patched summaries.")
    compare_parser.add_argument("--baseline-summary", type=Path, required=True)
    compare_parser.add_argument("--patched-summary", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path, required=True)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "run":
        return run_benchmark(args)
    if args.command == "summarize":
        summary = summarize(
            metrics_jsonl=args.metrics_jsonl,
            stdout_log=args.stdout_log,
            output_summary=args.output_summary,
            output_csv=args.output_csv,
            warmup_steps=args.warmup_steps,
            measured_steps=_parse_step_list(args.measured_steps),
        )
        _print_summary_table(summary)
        return 0
    if args.command == "compare":
        comparison = compare_summaries(args.baseline_summary, args.patched_summary, args.output)
        print(json.dumps(comparison["verdicts"], indent=2, sort_keys=True))
        return 0
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
