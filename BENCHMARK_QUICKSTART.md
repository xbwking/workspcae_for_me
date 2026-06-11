# Ascend verl Benchmark Quickstart

最终推荐使用：`verl_29ffe753_ascend_benchmark_runtime_package.tar.gz`。

```bash
tar -xzf verl_29ffe753_ascend_benchmark_runtime_package.tar.gz
bash verl_29ffe753_ascend_benchmark_runtime_package/install_into_verl.sh /verl
cd /verl

MODEL_PATH=/path/to/model \
TRAIN_FILES=/path/to/train.parquet \
VAL_FILES=/path/to/test.parquet \
OUTPUT_DIR=outputs/ascend_timing_breakdown/run1 \
bash tests/special_npu/run_ascend_timing_breakdown_bench.sh

python3 scripts/report_ascend_verl_timing.py \
  --run-dir outputs/ascend_timing_breakdown/run1
```

说明：该包使用 monkey patch wrapper，不要求覆盖 verl 核心源码。
