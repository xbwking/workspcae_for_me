# verl Ascend 耗时拆解 Benchmark 使用手册

## 先说结论

当前真正用于“把 verl Ascend 耗时拆开”的 benchmark 是这一套：

核心脚本：

```text
/Users/xiongbowen/Documents/pink's_project/verl-ascend-supported-4045d670/scripts/bench_ascend_verl_timing.py
```

推荐一键运行入口：

```text
/Users/xiongbowen/Documents/pink's_project/verl-ascend-supported-4045d670/tests/special_npu/run_ascend_timing_breakdown_bench.sh
```

已有英文说明：

```text
/Users/xiongbowen/Documents/pink's_project/verl-ascend-supported-4045d670/docs/perf/ascend_timing_breakdown_benchmark.md
```

不要和下面这个脚本混淆：

```text
scripts/bench_fully_async_message_queue.py
```

`bench_fully_async_message_queue.py` 只用于压测 fully_async MessageQueue，不是完整的 Ascend verl 端到端耗时拆解 benchmark。

---

## 这个 benchmark 能拆哪些耗时

它覆盖三层指标。

### L0：端到端训练阶段耗时

这些指标回答“整体慢在哪里”：

```text
timing_s/step
timing_s/gen
timing_s/reward
timing_s/old_log_prob
timing_s/update_actor
timing_s/update_critic
timing_s/update_weights
perf/time_per_step
perf/throughput
```

重点看：

```text
timing_s/step
timing_s/gen
timing_s/update_weights
perf/throughput
```

### L1：框架链路拆解指标

这些指标回答“框架内部具体慢在哪里”：

```text
ray/message_queue_get_rpc_count
ray/message_queue_get_wait_s
serialization/cloudpickle_load_s

param_sync/abort_ms
param_sync/sleep_ms
param_sync/build_pg_ms
param_sync/send_recv_update_ms
param_sync/finalize_ms
param_sync/wake_ms
param_sync/resume_ms

weight_transfer/sender_copy_ms
weight_transfer/receiver_copy_ms
weight_transfer/metadata_send_ms
weight_transfer/metadata_recv_ms
weight_transfer/sender_bucket_count
weight_transfer/sender_bucket_bytes
weight_transfer/receiver_bucket_count
weight_transfer/receiver_bucket_bytes
```

重点看：

```text
param_sync/send_recv_update_ms
weight_transfer/sender_copy_ms
weight_transfer/receiver_copy_ms
ray/message_queue_get_rpc_count
serialization/cloudpickle_load_s
```

### L2：Ascend NPU profiler 证据

benchmark 会配置 NPU profiler，只采样少量 step：

```text
global_profiler.tool=npu
global_profiler.steps=[...]
actor_rollout_ref.actor.profiler.enable=True
actor_rollout_ref.ref.profiler.enable=True
```

注意：profiler 会改变运行时行为，所以不要用 profiler 全程跑性能对比。建议 L0 / L1 指标用于 A/B 多次对比，L2 profiler 用来解释某个代表性 step。

---

## 环境准备

进入 Ascend-supported verl 目录：

```bash
cd "/Users/xiongbowen/Documents/pink's_project/verl-ascend-supported-4045d670"
```

确认需要的输入：

```text
MODEL_PATH    模型路径，例如 /path/to/Qwen2.5-0.5B-Instruct
TRAIN_FILES   训练 parquet，例如 /data/gsm8k/train.parquet
VAL_FILES     验证 parquet，例如 /data/gsm8k/test.parquet
OUTPUT_DIR    benchmark 输出目录
```

确认环境已经能正常跑 verl Ascend：

```text
CANN / torch_npu / HCCL / Ray / vLLM-Ascend 已安装
NPU 可见
模型和数据路径存在
```

本机没有 Ascend NPU 时，只能跑 dry-run 和 parser 测试，不能验证真实性能收益。

---

## 最推荐的运行方式

直接用包装脚本：

```bash
MODEL_PATH=/path/to/model \
TRAIN_FILES=/path/to/train.parquet \
VAL_FILES=/path/to/test.parquet \
OUTPUT_DIR=outputs/ascend_timing_breakdown/baseline \
bash tests/special_npu/run_ascend_timing_breakdown_bench.sh
```

默认实验形态：

```text
total_steps=8
warmup_steps=2
measured_steps=3,4,5,6,7,8
profile_steps=3
rollout=vllm
device=npu
algorithm=grpo
```

输出目录会生成：

```text
outputs/ascend_timing_breakdown/baseline/metrics.jsonl
outputs/ascend_timing_breakdown/baseline/stdout.log
outputs/ascend_timing_breakdown/baseline/summary.json
outputs/ascend_timing_breakdown/baseline/timing_breakdown.csv
outputs/ascend_timing_breakdown/baseline/npu_profile
```

最常看的两个文件：

```text
summary.json
timing_breakdown.csv
```

排查问题时再看：

```text
stdout.log
npu_profile
```

---

## 先 dry-run，确认命令是否正确

如果不确定模型、数据、参数是否正确，先跑 dry-run：

```bash
MODEL_PATH=/path/to/model \
TRAIN_FILES=/path/to/train.parquet \
VAL_FILES=/path/to/test.parquet \
OUTPUT_DIR=outputs/ascend_timing_breakdown/dry_run \
bash tests/special_npu/run_ascend_timing_breakdown_bench.sh --dry-run
```

dry-run 不会启动训练，只会打印实际执行的 Python 命令、Hydra 参数、metrics 路径和 stdout 路径。

---

## 常用参数怎么改

包装脚本通过环境变量改实验形态。

### 改 step 数

```bash
TOTAL_STEPS=12 \
WARMUP_STEPS=3 \
MEASURED_STEPS=4,5,6,7,8,9,10,11,12 \
PROFILE_STEPS=4 \
MODEL_PATH=/path/to/model \
TRAIN_FILES=/path/to/train.parquet \
VAL_FILES=/path/to/test.parquet \
OUTPUT_DIR=outputs/ascend_timing_breakdown/baseline \
bash tests/special_npu/run_ascend_timing_breakdown_bench.sh
```

建议：

```text
warmup step 不计入最终统计
measured step 用来计算 mean / p50 / p95
profile step 只选一个或少量 step
```

### 改并行和 batch

```bash
N_GPUS_PER_NODE=8 \
TENSOR_MODEL_PARALLEL_SIZE=2 \
TRAIN_BATCH_SIZE=16 \
PPO_MINI_BATCH_SIZE=8 \
PPO_MICRO_BATCH_SIZE_PER_GPU=1 \
ROLLOUT_N=2 \
MODEL_PATH=/path/to/model \
TRAIN_FILES=/path/to/train.parquet \
VAL_FILES=/path/to/test.parquet \
OUTPUT_DIR=outputs/ascend_timing_breakdown/baseline \
bash tests/special_npu/run_ascend_timing_breakdown_bench.sh
```

### 改 rollout 长度

```bash
MAX_PROMPT_LENGTH=512 \
MAX_RESPONSE_LENGTH=128 \
MODEL_PATH=/path/to/model \
TRAIN_FILES=/path/to/train.parquet \
VAL_FILES=/path/to/test.parquet \
OUTPUT_DIR=outputs/ascend_timing_breakdown/baseline \
bash tests/special_npu/run_ascend_timing_breakdown_bench.sh
```

### 改权重同步 bucket 大小

```bash
BUCKET_MB=4096 \
MODEL_PATH=/path/to/model \
TRAIN_FILES=/path/to/train.parquet \
VAL_FILES=/path/to/test.parquet \
OUTPUT_DIR=outputs/ascend_timing_breakdown/baseline \
bash tests/special_npu/run_ascend_timing_breakdown_bench.sh
```

### 追加 Hydra 参数

包装脚本后面可以直接追加 verl Hydra override：

```bash
MODEL_PATH=/path/to/model \
TRAIN_FILES=/path/to/train.parquet \
VAL_FILES=/path/to/test.parquet \
OUTPUT_DIR=outputs/ascend_timing_breakdown/baseline \
bash tests/special_npu/run_ascend_timing_breakdown_bench.sh \
  actor_rollout_ref.rollout.checkpoint_engine.backend=hccl
```

---

## A/B 对比怎么跑

标准流程是同一环境、同一模型、同一数据、同一参数、同样 step 数，分别跑 baseline 和 patched。

### 跑 baseline

```bash
MODEL_PATH=/path/to/model \
TRAIN_FILES=/path/to/train.parquet \
VAL_FILES=/path/to/test.parquet \
OUTPUT_DIR=outputs/ascend_timing_breakdown/baseline \
bash tests/special_npu/run_ascend_timing_breakdown_bench.sh
```

### 跑 patched

切换到包含优化 patch 的代码后，跑同样参数：

```bash
MODEL_PATH=/path/to/model \
TRAIN_FILES=/path/to/train.parquet \
VAL_FILES=/path/to/test.parquet \
OUTPUT_DIR=outputs/ascend_timing_breakdown/patched \
bash tests/special_npu/run_ascend_timing_breakdown_bench.sh
```

### 生成 compare.json

```bash
python3 scripts/bench_ascend_verl_timing.py compare \
  --baseline-summary outputs/ascend_timing_breakdown/baseline/summary.json \
  --patched-summary outputs/ascend_timing_breakdown/patched/summary.json \
  --output outputs/ascend_timing_breakdown/compare.json
```

看结果：

```bash
cat outputs/ascend_timing_breakdown/compare.json
```

compare 会对关键指标给出 verdict。默认判断逻辑是：指标改善超过 5% 时，认为该方向 effective。

---

## 单独重新汇总日志

如果训练已经跑完，只想重新生成 summary 和 csv：

```bash
python3 scripts/bench_ascend_verl_timing.py summarize \
  --metrics-jsonl outputs/ascend_timing_breakdown/baseline/metrics.jsonl \
  --stdout-log outputs/ascend_timing_breakdown/baseline/stdout.log \
  --output-summary outputs/ascend_timing_breakdown/baseline/summary.json \
  --output-csv outputs/ascend_timing_breakdown/baseline/timing_breakdown.csv \
  --warmup-steps 2 \
  --measured-steps 3,4,5,6,7,8
```

---

## 怎么看结果

### 先看 summary.json

确认：

```text
step_count > 0
metrics 里有 timing_s/step
metrics 里有 perf/throughput
```

如果 `step_count=0`，通常说明：

```text
metrics.jsonl 没写入
warmup_steps 或 measured_steps 配错
训练没有真正跑到记录 step
```

### 再看 timing_breakdown.csv

重点看这些列：

```text
metric
mean
p50
p95
pct_of_step_mean
```

`pct_of_step_mean` 主要用于 `timing_s/*` 指标，表示某个阶段占平均 step 时间的比例。

### 怎么判断优化是否有效

不同优化看不同指标。

MessageQueue 批量 get：

```text
ray/message_queue_get_rpc_count 下降
ray/message_queue_get_wait_s 下降
serialization/cloudpickle_load_s 最好也下降
```

权重同步优化：

```text
timing_s/update_weights 下降
param_sync/send_recv_update_ms 下降
param_sync/build_pg_ms 下降
param_sync/resume_ms 下降
```

Bucket transfer 优化：

```text
weight_transfer/sender_copy_ms 下降
weight_transfer/receiver_copy_ms 下降
weight_transfer/metadata_send_ms 下降
weight_transfer/metadata_recv_ms 下降
```

端到端收益：

```text
timing_s/step 下降
perf/throughput 上升
```

注意：如果子指标下降，但 `timing_s/step` 不变，说明这个优化点不是当前主要瓶颈，或者收益被其他阶段抵消。

---

## 和其他文件的关系

### `scripts/bench_ascend_verl_timing.py`

核心 benchmark 程序。负责：

```text
run       构造并运行 verl GRPO benchmark
summarize 解析 metrics.jsonl 和 stdout.log
compare   对比 baseline 和 patched summary
```

### `tests/special_npu/run_ascend_timing_breakdown_bench.sh`

推荐运行入口。负责把常用参数封装成环境变量，避免手写很长的 Python 命令。

日常使用优先跑这个脚本。

### `docs/perf/ascend_timing_breakdown_benchmark.md`

已有英文简版说明。

### `scripts/bench_fully_async_message_queue.py`

只用于 MessageQueue 局部压测，不是完整端到端 benchmark。

---

## 最小可用命令清单

### dry-run

```bash
cd "/Users/xiongbowen/Documents/pink's_project/verl-ascend-supported-4045d670"

MODEL_PATH=/path/to/model \
TRAIN_FILES=/path/to/train.parquet \
VAL_FILES=/path/to/test.parquet \
OUTPUT_DIR=outputs/ascend_timing_breakdown/dry_run \
bash tests/special_npu/run_ascend_timing_breakdown_bench.sh --dry-run
```

### 正式跑一次

```bash
cd "/Users/xiongbowen/Documents/pink's_project/verl-ascend-supported-4045d670"

MODEL_PATH=/path/to/model \
TRAIN_FILES=/path/to/train.parquet \
VAL_FILES=/path/to/test.parquet \
OUTPUT_DIR=outputs/ascend_timing_breakdown/baseline \
bash tests/special_npu/run_ascend_timing_breakdown_bench.sh
```

### A/B 对比

```bash
python3 scripts/bench_ascend_verl_timing.py compare \
  --baseline-summary outputs/ascend_timing_breakdown/baseline/summary.json \
  --patched-summary outputs/ascend_timing_breakdown/patched/summary.json \
  --output outputs/ascend_timing_breakdown/compare.json
```

---

## 建议的汇报口径

如果要用这个 benchmark 说明优化收益，建议按这个顺序讲：

```text
1. 先说明实验环境：模型、NPU 数量、数据集、step 数、warmup step。
2. 再说明 baseline 和 patched 只差一个优化点。
3. 先给端到端指标：timing_s/step 和 perf/throughput。
4. 再给对应子指标：param_sync、weight_transfer、ray、serialization。
5. 最后给 compare.json verdict 和 profiler 片段作为解释。
```

不要只报一个 `timing_s/step`，否则无法证明收益来自哪个优化点。
