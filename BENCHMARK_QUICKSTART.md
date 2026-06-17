# Ascend verl Benchmark Merge Quickstart

代码合入按两个 PR：

```text
PR1: verl_ascend_merge_split_under_1000/pr1_benchmark_under_1000/  # 990 行
PR2: verl_ascend_merge_split_under_1000/pr2_report_tool_under_1000/ # 728 行
```

两个 PR 都不包含 `verl/` 源码侵入式修改。

PR1 运行：

```bash
MODEL_PATH=/path/to/model \
TRAIN_FILES=/path/to/train.parquet \
VAL_FILES=/path/to/test.parquet \
OUTPUT_DIR=outputs/ascend_timing_breakdown/run1 \
bash tests/special_npu/run_ascend_timing_breakdown_bench.sh
```

PR2 生成报告：

```bash
python3 scripts/report_ascend_verl_timing.py \
  --run-dir outputs/ascend_timing_breakdown/run1
```
