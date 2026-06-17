# Ascend Timing Breakdown Benchmark

This benchmark captures verl Ascend end-to-end timing, framework timing, parameter-sync timing, Ray queue timing, serialization timing, and sampled NPU profiler traces.

It is a non-invasive benchmark package: it adds benchmark scripts and docs only, and does not change files under `verl/`.

## Files

```text
scripts/bench_ascend_verl_timing.py
scripts/bench_fully_async_message_queue.py
tests/special_npu/run_ascend_timing_breakdown_bench.sh
docs/perf/ascend_timing_breakdown_benchmark.md
```

## Run

```bash
MODEL_PATH=/path/to/model \
TRAIN_FILES=/path/to/train.parquet \
VAL_FILES=/path/to/test.parquet \
OUTPUT_DIR=outputs/ascend_timing_breakdown/run1 \
bash tests/special_npu/run_ascend_timing_breakdown_bench.sh
```

The run writes:

```text
metrics.jsonl
stdout.log
summary.json
timing_breakdown.csv
npu_profile/
```

## Default Shape

```text
total_steps=8
warmup_steps=2
measured_steps=3,4,5,6,7,8
profile_steps=3
rollout=vllm
device=npu
```

Override through environment variables:

```bash
TOTAL_STEPS=12 \
WARMUP_STEPS=3 \
MEASURED_STEPS=4,5,6,7,8,9,10,11,12 \
PROFILE_STEPS=4 \
TENSOR_MODEL_PARALLEL_SIZE=4 \
bash tests/special_npu/run_ascend_timing_breakdown_bench.sh
```

Extra Hydra overrides can be appended:

```bash
bash tests/special_npu/run_ascend_timing_breakdown_bench.sh \
  actor_rollout_ref.rollout.checkpoint_engine.backend=hccl
```

## Metrics

End-to-end:

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

Framework and transfer:

```text
ray/message_queue_get_rpc_count
ray/message_queue_get_wait_s
serialization/cloudpickle_load_s
param_sync/abort_ms
param_sync/build_pg_ms
param_sync/send_recv_update_ms
param_sync/finalize_ms
param_sync/resume_ms
weight_transfer/sender_copy_ms
weight_transfer/receiver_copy_ms
weight_transfer/metadata_send_ms
weight_transfer/metadata_recv_ms
```

Profiler evidence:

```text
global_profiler.tool=npu
global_profiler.steps=[...]
actor_rollout_ref.actor.profiler.enable=True
actor_rollout_ref.ref.profiler.enable=True
```

Use L0/L1 metrics for repeated A/B comparison. Use NPU profiler traces to explain one representative step because profiler changes runtime behavior.

## Summarize Existing Logs

```bash
python3 scripts/bench_ascend_verl_timing.py summarize \
  --metrics-jsonl outputs/ascend_timing_breakdown/run1/metrics.jsonl \
  --stdout-log outputs/ascend_timing_breakdown/run1/stdout.log \
  --output-summary outputs/ascend_timing_breakdown/run1/summary.json \
  --output-csv outputs/ascend_timing_breakdown/run1/timing_breakdown.csv \
  --warmup-steps 2 \
  --measured-steps 3,4,5,6,7,8
```

## Compare

```bash
python3 scripts/bench_ascend_verl_timing.py compare \
  --baseline-summary outputs/ascend_timing_breakdown/baseline/summary.json \
  --patched-summary outputs/ascend_timing_breakdown/patched/summary.json \
  --output outputs/ascend_timing_breakdown/compare.json
```

The compare command reports speedup and marks a metric effective when improvement is greater than 5%.

## MessageQueue Micro Benchmark

```bash
python3 scripts/bench_fully_async_message_queue.py \
  --mode local \
  --num-samples 1024 \
  --batch-size 64 \
  --payload-bytes 1024 \
  --json
```

Use Ray mode in a real verl environment to estimate actor RPC reduction:

```bash
python3 scripts/bench_fully_async_message_queue.py --mode ray --json
```
