# verl Ascend Benchmark Merge Split

这个目录用于两个人分别提交代码，每次合入不超过 1000 行。

## PR1: Benchmark

目录：`pr1_benchmark_under_1000/`

行数：

```text
147  docs/perf/ascend_timing_breakdown_benchmark.md
503  scripts/bench_ascend_verl_timing.py
269  scripts/bench_fully_async_message_queue.py
 71  tests/special_npu/run_ascend_timing_breakdown_bench.sh
---
990 行
```

特点：

```text
完整 benchmark
不带测试
不修改 verl/ 源码
不包含 report 工具
```

## PR2: Report Tool

目录：`pr2_report_tool_under_1000/`

行数：

```text
168  README.md
 25  install_into_verl.sh
513  scripts/ascend_verl_timing_report.py
 22  scripts/report_ascend_verl_timing.py
---
728 行
```

特点：

```text
独立 report 工具
不修改 verl/ 源码
可单独安装到 verl 仓库
读取 metrics.jsonl/stdout.log，输出 report.md/report.json/top_metrics.csv
```

## 合入顺序

建议先合 PR1，再合 PR2。

PR1 负责产生原始 benchmark 产物：

```text
metrics.jsonl
stdout.log
summary.json
timing_breakdown.csv
npu_profile/
```

PR2 负责整理这些产物：

```text
report.md
report.json
top_metrics.csv
```

## 不包含的内容

这两个 PR 都不包含侵入式源码修改：

```text
verl/checkpoint_engine/base.py
verl/checkpoint_engine/hccl_checkpoint_engine.py
verl/experimental/fully_async_policy/fully_async_trainer.py
verl/experimental/fully_async_policy/message_queue.py
verl/trainer/ppo/ray_trainer.py
verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py
```
