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
"""Human-readable report generator for Ascend verl timing benchmark outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from bench_ascend_verl_timing import summarize


REPORT_GROUPS = (
    ("end_to_end", ("timing_s/step", "perf/throughput", "perf/time_per_step")),
    ("step_breakdown", ("timing_s/",)),
    ("parameter_sync", ("param_sync/", "weight_transfer/")),
    ("ray_and_serialization", ("ray/", "serialization/", "fully_async/")),
    ("rollout", ("rollout/", "timing_per_token_ms/")),
)

KEY_REPORT_METRICS = (
    "timing_s/step",
    "perf/throughput",
    "timing_s/gen",
    "timing_s/update_actor",
    "timing_s/update_weights",
    "param_sync/total_ms",
    "param_sync/send_recv_update_ms",
    "weight_transfer/sender_copy_ms",
    "weight_transfer/receiver_copy_ms",
    "ray/message_queue_get_rpc_count",
    "serialization/cloudpickle_load_s",
)


def _parse_step_list(value: str | None) -> list[int] | None:
    if value is None or value == "":
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _read_summary_or_build(
    run_dir: Path,
    summary_path: Path | None,
    metrics_jsonl: Path | None,
    stdout_log: Path | None,
    warmup_steps: int,
    measured_steps: list[int] | None,
) -> dict[str, Any]:
    summary_path = summary_path or run_dir / "summary.json"
    metrics_jsonl = metrics_jsonl or run_dir / "metrics.jsonl"
    stdout_log = stdout_log or run_dir / "stdout.log"
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    return summarize(
        metrics_jsonl=metrics_jsonl,
        stdout_log=stdout_log,
        output_summary=summary_path,
        output_csv=run_dir / "timing_breakdown.csv",
        warmup_steps=warmup_steps,
        measured_steps=measured_steps,
    )


def _metric_matches_prefixes(metric: str, prefixes: tuple[str, ...]) -> bool:
    return any(metric == prefix or metric.startswith(prefix) for prefix in prefixes)


def _metric_unit(metric: str) -> str:
    if metric.startswith("timing_s/") or metric.startswith("serialization/") or metric.startswith("rollout/"):
        return "s"
    if metric.endswith("_ms") or metric.startswith("param_sync/") or metric.startswith("weight_transfer/"):
        return "ms"
    if "bytes" in metric:
        return "bytes"
    if "throughput" in metric:
        return "samples/s"
    return ""


def _format_number(value: Any, unit: str = "") -> str:
    if value is None:
        return "-"
    if not isinstance(value, int | float):
        return str(value)
    if unit == "bytes":
        return _format_bytes(value)
    if abs(value) >= 100:
        text = f"{value:.2f}"
    elif abs(value) >= 10:
        text = f"{value:.3f}"
    else:
        text = f"{value:.4f}"
    return f"{text} {unit}".strip()


def _format_bytes(value: float) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    current = float(value)
    for unit in units:
        if abs(current) < 1024 or unit == units[-1]:
            return f"{current:.2f} {unit}"
        current /= 1024
    return f"{value:.2f} B"


def _artifact_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    if path.is_file():
        return {"path": str(path), "exists": True, "type": "file", "bytes": path.stat().st_size}
    files = [item for item in path.rglob("*") if item.is_file()]
    total_bytes = sum(item.stat().st_size for item in files)
    return {
        "path": str(path),
        "exists": True,
        "type": "directory",
        "file_count": len(files),
        "bytes": total_bytes,
        "sample_files": [str(item.relative_to(path)) for item in files[:10]],
    }


def _build_report_model(summary: dict[str, Any], run_dir: Path, top_n: int) -> dict[str, Any]:
    metrics = summary.get("metrics", {})
    grouped: dict[str, list[dict[str, Any]]] = {}
    for group, prefixes in REPORT_GROUPS:
        rows = []
        for key, metric in metrics.items():
            if group == "end_to_end":
                if key not in prefixes:
                    continue
            elif not _metric_matches_prefixes(key, prefixes):
                continue
            unit = _metric_unit(key)
            rows.append(
                {
                    "metric": key,
                    "mean": metric.get("mean"),
                    "p50": metric.get("p50"),
                    "p95": metric.get("p95"),
                    "min": metric.get("min"),
                    "max": metric.get("max"),
                    "count": metric.get("count"),
                    "pct_of_step_mean": metric.get("pct_of_step_mean"),
                    "unit": unit,
                }
            )
        rows.sort(key=lambda item: (-(item.get("pct_of_step_mean") or 0.0), item["metric"]))
        grouped[group] = rows

    top_step_costs = [
        item
        for item in grouped.get("step_breakdown", [])
        if item["metric"] != "timing_s/step" and item.get("pct_of_step_mean") is not None
    ][:top_n]
    param_sync_costs = sorted(
        grouped.get("parameter_sync", []),
        key=lambda item: item.get("mean") or 0.0,
        reverse=True,
    )[:top_n]
    ray_costs = sorted(
        grouped.get("ray_and_serialization", []),
        key=lambda item: item.get("mean") or 0.0,
        reverse=True,
    )[:top_n]

    artifacts = {
        "metrics_jsonl": _artifact_info(run_dir / "metrics.jsonl"),
        "stdout_log": _artifact_info(run_dir / "stdout.log"),
        "summary_json": _artifact_info(run_dir / "summary.json"),
        "timing_breakdown_csv": _artifact_info(run_dir / "timing_breakdown.csv"),
        "npu_profile": _artifact_info(run_dir / "npu_profile"),
    }
    return {
        "run_dir": str(run_dir),
        "step_count": summary.get("step_count", 0),
        "warmup_steps": summary.get("warmup_steps"),
        "measured_steps": summary.get("measured_steps"),
        "key_metrics": {key: metrics[key] for key in KEY_REPORT_METRICS if key in metrics},
        "missing_key_metrics": [key for key in KEY_REPORT_METRICS if key not in metrics],
        "groups": grouped,
        "top_step_costs": top_step_costs,
        "top_param_sync_costs": param_sync_costs,
        "top_ray_serialization_costs": ray_costs,
        "artifacts": artifacts,
    }


def _write_top_metrics_csv(path: Path, report: dict[str, Any]) -> None:
    rows = []
    for section in ("top_step_costs", "top_param_sync_costs", "top_ray_serialization_costs"):
        for item in report[section]:
            row = {"section": section}
            row.update(item)
            rows.append(row)
    columns = ["section", "metric", "mean", "p50", "p95", "min", "max", "count", "pct_of_step_mean", "unit"]
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _markdown_metric_table(rows: list[dict[str, Any]], include_pct: bool = True) -> list[str]:
    if not rows:
        return ["无可展示指标。"]
    columns = ["指标", "均值", "P50", "P95", "样本数"]
    if include_pct:
        columns.insert(4, "占 step")
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for item in rows:
        unit = item.get("unit", "")
        cells = [
            f"`{item['metric']}`",
            _format_number(item.get("mean"), unit),
            _format_number(item.get("p50"), unit),
            _format_number(item.get("p95"), unit),
        ]
        if include_pct:
            pct = item.get("pct_of_step_mean")
            cells.append(f"{pct:.2f}%" if isinstance(pct, int | float) else "-")
        cells.append(_format_number(item.get("count")))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    end_to_end_rows = []
    for metric, values in report["key_metrics"].items():
        if metric in ("timing_s/step", "perf/throughput", "perf/time_per_step"):
            unit = _metric_unit(metric)
            item = {"metric": metric, "unit": unit}
            item.update(values)
            end_to_end_rows.append(item)

    lines = [
        "# Ascend verl Benchmark 一页式报告",
        "",
        f"- 运行目录：`{report['run_dir']}`",
        f"- 有效 step 数：`{report['step_count']}`",
        f"- warmup steps：`{report.get('warmup_steps')}`",
        f"- measured steps：`{report.get('measured_steps')}`",
        "",
        "## 结论视图",
        "",
    ]
    lines.extend(_markdown_metric_table(end_to_end_rows, include_pct=False))
    lines.extend(["", "## Step 耗时主项", ""])
    lines.extend(_markdown_metric_table(report["top_step_costs"]))
    lines.extend(["", "## 参数同步 / 权重传输", ""])
    lines.extend(_markdown_metric_table(report["top_param_sync_costs"], include_pct=False))
    lines.extend(["", "## Ray / 序列化 / 异步队列", ""])
    lines.extend(_markdown_metric_table(report["top_ray_serialization_costs"], include_pct=False))
    lines.extend(["", "## 产物索引", ""])
    lines.extend(["| 产物 | 状态 | 大小 | 说明 |", "|---|---|---|---|"])
    for name, info in report["artifacts"].items():
        status = "存在" if info.get("exists") else "缺失"
        size = _format_bytes(info.get("bytes", 0)) if info.get("exists") else "-"
        if info.get("type") == "directory":
            desc = f"{info.get('file_count', 0)} 个文件"
        elif info.get("type") == "file":
            desc = "文件"
        else:
            desc = "-"
        lines.append(f"| `{name}` | {status} | {size} | {desc} |")
    if report["missing_key_metrics"]:
        lines.extend(["", "## 缺失指标", ""])
        lines.append("这些指标没有在本次输出中出现，通常表示对应路径没有触发、补丁未接入、或当前运行模式不涉及：")
        lines.extend(f"- `{key}`" for key in report["missing_key_metrics"])
    lines.extend(
        [
            "",
            "## 使用建议",
            "",
            "- 先看 `timing_s/step` 和 `perf/throughput` 判断端到端是否变好。",
            "- 再看 `Step 耗时主项`，确认瓶颈落在生成、训练更新、reward、log_prob 还是权重同步。",
            "- 如果优化目标是权重同步，重点看 `param_sync/*` 和 `weight_transfer/*`。",
            "- 如果优化目标是异步调度，重点看 `ray/*`、`serialization/*`、`fully_async/*`。",
            "- `npu_profile` 用于解释代表性 step，不建议直接拿 profiler step 当唯一性能结论。",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def report_run(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir)
    summary = _read_summary_or_build(
        run_dir=run_dir,
        summary_path=args.summary,
        metrics_jsonl=args.metrics_jsonl,
        stdout_log=args.stdout_log,
        warmup_steps=args.warmup_steps,
        measured_steps=_parse_step_list(args.measured_steps),
    )
    report = _build_report_model(summary, run_dir=run_dir, top_n=args.top_n)
    output_md = args.output_md or run_dir / "report.md"
    output_json = args.output_json or run_dir / "report.json"
    output_csv = args.output_csv or run_dir / "top_metrics.csv"
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_markdown_report(output_md, report)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True))
    _write_top_metrics_csv(output_csv, report)
    report["outputs"] = {"markdown": str(output_md), "json": str(output_json), "csv": str(output_csv)}
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--metrics-jsonl", type=Path, default=None)
    parser.add_argument("--stdout-log", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--measured-steps", default=None)
    parser.add_argument("--top-n", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    report = report_run(parse_args(argv))
    print(json.dumps(report["outputs"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
