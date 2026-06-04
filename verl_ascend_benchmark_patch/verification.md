# Verification

本地机器没有 Ascend NPU，所以只验证 parser、CLI、dry-run、语法和 CPU 可执行开发者测试。

## 已执行

```bash
/tmp/verl-ascend-test-venv/bin/python -m py_compile \
  scripts/bench_ascend_verl_timing.py \
  tests/special_sanity/test_ascend_timing_benchmark.py \
  verl/trainer/ppo/ray_trainer.py \
  verl/experimental/fully_async_policy/fully_async_trainer.py \
  verl/checkpoint_engine/base.py \
  verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py \
  verl/experimental/fully_async_policy/message_queue.py
```

结果：通过。

```bash
bash -n tests/special_npu/run_ascend_timing_breakdown_bench.sh
```

结果：通过。

```bash
/tmp/verl-ascend-test-venv/bin/python -m pytest \
  tests/special_sanity/test_ascend_timing_benchmark.py \
  tests/checkpoint_engine/test_registry_on_cpu.py \
  tests/experimental/fully_async_policy/test_message_queue_on_cpu.py \
  tests/experimental/fully_async_policy/test_message_queue_benchmark_on_cpu.py \
  tests/utils/test_bucketed_weight_transfer.py \
  -q
```

结果：`14 passed, 11 skipped in 0.27s`。

```bash
MODEL_PATH=/models/qwen \
TRAIN_FILES=/data/train.parquet \
VAL_FILES=/data/test.parquet \
OUTPUT_DIR=/tmp/ascend_bench_dry_run \
bash tests/special_npu/run_ascend_timing_breakdown_bench.sh --dry-run
```

结果：成功生成 verl Ascend benchmark 命令，输出目标包括 `metrics.jsonl`、`stdout.log`、`summary.json`、`timing_breakdown.csv`、`npu_profile`。

## 仍需 Ascend 环境验证

```bash
bash tests/special_npu/run_ascend_timing_breakdown_bench.sh
```

真实优化收益需要在同一 Ascend 环境中跑 baseline/patched 两组，再执行 compare。
