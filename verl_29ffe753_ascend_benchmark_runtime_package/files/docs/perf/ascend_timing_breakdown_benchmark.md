# Ascend Timing Breakdown Benchmark

This benchmark is designed to validate framework-level inference and scheduling optimizations in verl on Ascend.
It captures end-to-end step timing, framework timing, parameter-sync timing, Ray/fully-async queue timing, and sampled NPU profiler traces.

## Run One Benchmark

The benchmark runs verl through a non-invasive monkey-patch wrapper:

```text
scripts/run_ppo_with_ascend_benchmark_patches.py
```

The wrapper enables runtime instrumentation for the benchmark process and Ray
workers.  It does not require editing upstream verl source files such as
`ray_trainer.py`, `checkpoint_engine/base.py`, or `message_queue.py`.

```bash
MODEL_PATH=/path/to/model \
TRAIN_FILES=/path/to/train.parquet \
VAL_FILES=/path/to/test.parquet \
OUTPUT_DIR=outputs/ascend_timing_breakdown/baseline \
bash tests/special_npu/run_ascend_timing_breakdown_bench.sh
```

The benchmark writes:

```text
outputs/ascend_timing_breakdown/baseline/metrics.jsonl
outputs/ascend_timing_breakdown/baseline/stdout.log
outputs/ascend_timing_breakdown/baseline/summary.json
outputs/ascend_timing_breakdown/baseline/timing_breakdown.csv
outputs/ascend_timing_breakdown/baseline/npu_profile
```

## Build A One-Page Report

Keep the raw artifacts above, then generate a readable report from the run directory:

```bash
python3 scripts/report_ascend_verl_timing.py \
  --run-dir outputs/ascend_timing_breakdown/baseline
```

This writes:

```text
outputs/ascend_timing_breakdown/baseline/report.md
outputs/ascend_timing_breakdown/baseline/report.json
outputs/ascend_timing_breakdown/baseline/top_metrics.csv
```

The report tool is intentionally separated from `scripts/bench_ascend_verl_timing.py`.
The benchmark script owns collection, summarization, and baseline/patched comparison.
The report script only reads existing artifacts and builds a human-readable view.
This keeps the two code paths easy to review and merge independently.

Use `report.md` for quick reading. It contains:

```text
结论视图：timing_s/step、perf/throughput
Step 耗时主项：生成、reward、log_prob、actor update、update_weights 等占比
参数同步 / 权重传输：param_sync/*、weight_transfer/*
Ray / 序列化 / 异步队列：ray/*、serialization/*、fully_async/*
产物索引：metrics.jsonl、stdout.log、summary.json、timing_breakdown.csv、npu_profile 是否存在和大小
缺失指标：本次运行没有采集到的关键指标
```

If `summary.json` does not exist, the report command rebuilds it from `metrics.jsonl` and `stdout.log`:

```bash
python3 scripts/report_ascend_verl_timing.py \
  --run-dir outputs/ascend_timing_breakdown/baseline \
  --warmup-steps 2 \
  --measured-steps 3,4,5,6,7,8
```

## Compare Baseline and Patched Runs

Run the same benchmark once before the optimization and once after the optimization:

```bash
MODEL_PATH=/path/to/model \
TRAIN_FILES=/path/to/train.parquet \
VAL_FILES=/path/to/test.parquet \
OUTPUT_DIR=outputs/ascend_timing_breakdown/baseline \
bash tests/special_npu/run_ascend_timing_breakdown_bench.sh

MODEL_PATH=/path/to/model \
TRAIN_FILES=/path/to/train.parquet \
VAL_FILES=/path/to/test.parquet \
OUTPUT_DIR=outputs/ascend_timing_breakdown/patched \
bash tests/special_npu/run_ascend_timing_breakdown_bench.sh

python3 scripts/bench_ascend_verl_timing.py compare \
  --baseline-summary outputs/ascend_timing_breakdown/baseline/summary.json \
  --patched-summary outputs/ascend_timing_breakdown/patched/summary.json \
  --output outputs/ascend_timing_breakdown/compare.json
```

## Metrics Covered

L0 end-to-end metrics:

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

L1 framework metrics:

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

L2 Ascend profiler evidence:

```text
global_profiler.tool=npu
global_profiler.steps=[...]
actor_rollout_ref.actor.profiler.enable=True
actor_rollout_ref.ref.profiler.enable=True
```

Profiler traces are sampled on selected steps only because profiling changes runtime behavior. Use L0/L1 metrics for repeated A/B comparisons, then use L2 traces to explain a representative step.

## Default Experiment Shape

The default shell script uses a small GRPO benchmark shape:

```text
total_steps=8
warmup_steps=2
measured_steps=3,4,5,6,7,8
profile_steps=3
model=Qwen/Qwen2.5-0.5B-Instruct
dataset=gsm8k parquet files
rollout=vllm
device=npu
```

Override any shape parameter through environment variables or extra Hydra overrides:

```bash
TOTAL_STEPS=12 \
WARMUP_STEPS=3 \
MEASURED_STEPS=4,5,6,7,8,9,10,11,12 \
PROFILE_STEPS=4 \
TENSOR_MODEL_PARALLEL_SIZE=4 \
bash tests/special_npu/run_ascend_timing_breakdown_bench.sh \
  actor_rollout_ref.rollout.checkpoint_engine.backend=hccl
```

## Interpreting Compare Results

The compare command marks an optimization as effective when the metric improves by more than 5%.

Examples:

```text
message_queue batching:
  ray/message_queue_get_rpc_count decreases

param_sync send_recv:
  param_sync/send_recv_update_ms decreases

weight sender copy:
  weight_transfer/sender_copy_ms decreases

throughput:
  perf/throughput increases
```

Use the verdict as a first-pass signal, not as a replacement for reading `timing_breakdown.csv` and the NPU profiler trace.
