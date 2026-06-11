# workspcae_for_me

当前最终可运行包：`verl_29ffe753_ascend_benchmark_runtime_package/`

适用 verl commit：`29ffe753600ceca3cc5530ee6166be77fb4ecc1c`。

这个包是非侵入式 Ascend timing benchmark runtime package：不覆盖 verl 核心源码，通过 monkey patch wrapper 在 benchmark 运行时注入耗时拆解埋点。

## 直接使用

```bash
tar -xzf verl_29ffe753_ascend_benchmark_runtime_package.tar.gz
bash verl_29ffe753_ascend_benchmark_runtime_package/install_into_verl.sh /verl
cd /verl

MODEL_PATH=/path/to/model \
TRAIN_FILES=/path/to/train.parquet \
VAL_FILES=/path/to/test.parquet \
OUTPUT_DIR=outputs/ascend_timing_breakdown/run1 \
bash tests/special_npu/run_ascend_timing_breakdown_bench.sh
```

生成一页式报告：

```bash
python3 scripts/report_ascend_verl_timing.py \
  --run-dir outputs/ascend_timing_breakdown/run1
```

主要看：

```text
report.md
top_metrics.csv
summary.json
timing_breakdown.csv
npu_profile/
```

旧的覆盖式 `verl_29ffe753_benchmark_patch/` 已删除，避免误用。
