# Benchmark Quickstart

完整 verl Ascend 耗时拆解 benchmark：

```text
verl_ascend_benchmark_patch/changed_files/scripts/bench_ascend_verl_timing.py
```

推荐入口：

```text
verl_ascend_benchmark_patch/changed_files/tests/special_npu/run_ascend_timing_breakdown_bench.sh
```

最小运行：

```bash
cd /path/to/verl

MODEL_PATH=/path/to/model TRAIN_FILES=/path/to/train.parquet VAL_FILES=/path/to/test.parquet OUTPUT_DIR=outputs/ascend_timing_breakdown/baseline bash tests/special_npu/run_ascend_timing_breakdown_bench.sh
```

输出重点看：

```text
summary.json
timing_breakdown.csv
compare.json
stdout.log
npu_profile/
```

详细说明见：

- [docs/verl_ascend_timing_breakdown_benchmark_user_manual.md](docs/verl_ascend_timing_breakdown_benchmark_user_manual.md)
- [verl_ascend_benchmark_patch/benchmark_user_manual.md](verl_ascend_benchmark_patch/benchmark_user_manual.md)
