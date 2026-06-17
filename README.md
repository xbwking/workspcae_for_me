# workspcae_for_me

当前用于代码合入的最终目录：`verl_ascend_merge_split_under_1000/`

这个目录把 Ascend benchmark 和 report 工具拆成两个 PR，每次合入都控制在 1000 行以内，并且都不包含 `verl/` 源码侵入式修改。

## PR1 Benchmark

```text
verl_ascend_merge_split_under_1000/pr1_benchmark_under_1000/
990 行
```

包含完整 benchmark：

```text
scripts/bench_ascend_verl_timing.py
scripts/bench_fully_async_message_queue.py
tests/special_npu/run_ascend_timing_breakdown_bench.sh
docs/perf/ascend_timing_breakdown_benchmark.md
```

## PR2 Report Tool

```text
verl_ascend_merge_split_under_1000/pr2_report_tool_under_1000/
728 行
```

包含独立 report 工具：

```text
scripts/ascend_verl_timing_report.py
scripts/report_ascend_verl_timing.py
README.md
install_into_verl.sh
```

压缩包：`verl_ascend_merge_split_under_1000.tar.gz`

说明：旧的可运行覆盖式包仍然保留，但合入代码建议使用这个 split-under-1000 目录。
