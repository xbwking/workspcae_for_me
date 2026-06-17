# workspcae_for_me

当前用于代码合入的最终目录：`verl_ascend_merge_split_under_1000/`

这个目录把 Ascend benchmark 和 report 工具拆成两个 PR。功能代码按每次合入不超过 1000 行拆分，UT 单文件均小于 1000 行，并且都不包含 `verl/` 源码侵入式修改。

## PR1 Benchmark

```text
verl_ascend_merge_split_under_1000/pr1_benchmark_under_1000/
功能代码 990 行
UT 345 行，单文件均小于 1000 行
```

## PR2 Report Tool

```text
verl_ascend_merge_split_under_1000/pr2_report_tool_under_1000/
功能代码 728 行
UT 196 行
总计 924 行
```

压缩包：`verl_ascend_merge_split_under_1000.tar.gz`

说明：旧的可运行覆盖式包仍然保留，但合入代码建议使用这个 split-under-1000 目录。
