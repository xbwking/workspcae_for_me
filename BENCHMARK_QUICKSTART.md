# Benchmark Quickstart

## 当前推荐版本

你当前镜像里的 verl commit 是：

```text
29ffe753600ceca3cc5530ee6166be77fb4ecc1c
```

请使用：

```text
verl_29ffe753_benchmark_patch/
```

不要把旧的 `verl_ascend_benchmark_patch/` 整份覆盖到该镜像版本。

## 应用 patch

在目标 verl 仓库根目录执行：

```bash
cp -R /path/to/verl_29ffe753_benchmark_patch/changed_files/* .
```

## 最小验证

```bash
python3 -m py_compile   scripts/bench_ascend_verl_timing.py   verl/trainer/ppo/ray_trainer.py   verl/checkpoint_engine/base.py

MODEL_PATH=/models/qwen TRAIN_FILES=/data/train.parquet VAL_FILES=/data/test.parquet OUTPUT_DIR=/tmp/verl_29ffe753_bench_dry_run bash tests/special_npu/run_ascend_timing_breakdown_bench.sh --dry-run
```

## 正式运行

```bash
MODEL_PATH=/path/to/model TRAIN_FILES=/path/to/train.parquet VAL_FILES=/path/to/test.parquet OUTPUT_DIR=outputs/ascend_timing_breakdown/baseline bash tests/special_npu/run_ascend_timing_breakdown_bench.sh
```

输出重点看：

```text
summary.json
timing_breakdown.csv
stdout.log
npu_profile/
```

## A/B 对比

```bash
python3 scripts/bench_ascend_verl_timing.py compare   --baseline-summary outputs/ascend_timing_breakdown/baseline/summary.json   --patched-summary outputs/ascend_timing_breakdown/patched/summary.json   --output outputs/ascend_timing_breakdown/compare.json
```
