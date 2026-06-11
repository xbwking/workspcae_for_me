# verl 29ffe753 Ascend Benchmark Patch

这个目录是针对镜像内 verl commit `29ffe753600ceca3cc5530ee6166be77fb4ecc1c` 适配后的 Ascend 耗时拆解 benchmark patch。

## 适用版本

- verl commit: `29ffe753600ceca3cc5530ee6166be77fb4ecc1c`
- 这是用户当前 Ascend 镜像内确认到的版本。
- 不要再把旧的 `verl_ascend_benchmark_patch` 整份覆盖到该版本上，旧包基线不同，会导致 `verl.experimental.dataset`、`verl.utils.rollout_skip`、`AgentLoopManager(worker_group=...)` 等不兼容问题。

## 主要入口

完整 benchmark 主脚本：

```text
changed_files/scripts/bench_ascend_verl_timing.py
```

推荐一键运行脚本：

```text
changed_files/tests/special_npu/run_ascend_timing_breakdown_bench.sh
```

中文使用手册：

```text
benchmark_user_manual.md
```

## 如何应用

在目标 verl 仓库根目录执行：

```bash
cp -R /path/to/verl_29ffe753_benchmark_patch/changed_files/* .
```

然后做本地结构验证：

```bash
python3 -m py_compile \
  scripts/bench_ascend_verl_timing.py \
  verl/checkpoint_engine/base.py \
  verl/checkpoint_engine/hccl_checkpoint_engine.py \
  verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py \
  verl/experimental/fully_async_policy/message_queue.py \
  verl/experimental/fully_async_policy/fully_async_trainer.py \
  verl/trainer/ppo/ray_trainer.py

MODEL_PATH=/models/qwen \
TRAIN_FILES=/data/train.parquet \
VAL_FILES=/data/test.parquet \
OUTPUT_DIR=/tmp/verl_29ffe753_bench_dry_run \
bash tests/special_npu/run_ascend_timing_breakdown_bench.sh --dry-run
```

## 变更说明

- `ray_trainer.py` 基于 `29ffe753` 原版，只增加 `last_update_weights_timing` 写入 metrics。
- `CheckpointEngineManager.update_weights()` 增加 `param_sync/*` 分段耗时。
- `HCCLCheckpointEngine` 注册 key 修正为 `hccl`。
- `BucketedWeightSender/Receiver` 增加 `weight_transfer/*` 统计日志。
- `MessageQueue` 增加批量 `get_samples(max_n)`。
- `FullyAsyncTrainer` 记录 MessageQueue RPC 次数和 cloudpickle load 时间。
- 新增 Ascend timing breakdown benchmark：`run / summarize / compare`。

## 验证状态

本地无 Ascend NPU，仅验证：

```text
python3 -m py_compile: 通过
run_ascend_timing_breakdown_bench.sh --dry-run: 通过
```

真实性能收益需要在 Ascend 环境里跑 baseline / patched 后确认。
