#!/usr/bin/env bash
# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail
set -x

MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}
MODEL_PATH=${MODEL_PATH:-${HOME}/.cache/models/${MODEL_ID}}
TRAIN_FILES=${TRAIN_FILES:-${HOME}/data/gsm8k/train.parquet}
VAL_FILES=${VAL_FILES:-${HOME}/data/gsm8k/test.parquet}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/ascend_timing_breakdown/$(date +%Y%m%d_%H%M%S)}

TOTAL_STEPS=${TOTAL_STEPS:-8}
WARMUP_STEPS=${WARMUP_STEPS:-2}
MEASURED_STEPS=${MEASURED_STEPS:-3,4,5,6,7,8}
PROFILE_STEPS=${PROFILE_STEPS:-3}

NNODES=${NNODES:-1}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-128}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-2}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.6}
ROLLOUT_N=${ROLLOUT_N:-2}
BUCKET_MB=${BUCKET_MB:-4096}

python3 scripts/bench_ascend_verl_timing.py run \
    --output-dir "${OUTPUT_DIR}" \
    --model-path "${MODEL_PATH}" \
    --train-files "${TRAIN_FILES}" \
    --val-files "${VAL_FILES}" \
    --total-steps "${TOTAL_STEPS}" \
    --warmup-steps "${WARMUP_STEPS}" \
    --measured-steps "${MEASURED_STEPS}" \
    --profile-steps "${PROFILE_STEPS}" \
    --nnodes "${NNODES}" \
    --n-gpus-per-node "${N_GPUS_PER_NODE}" \
    --train-batch-size "${TRAIN_BATCH_SIZE}" \
    --max-prompt-length "${MAX_PROMPT_LENGTH}" \
    --max-response-length "${MAX_RESPONSE_LENGTH}" \
    --ppo-mini-batch-size "${PPO_MINI_BATCH_SIZE}" \
    --ppo-micro-batch-size-per-gpu "${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    --log-prob-micro-batch-size-per-gpu "${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
    --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --rollout-n "${ROLLOUT_N}" \
    --bucket-mb "${BUCKET_MB}" \
    "$@"

echo "Ascend timing benchmark artifacts:"
echo "  ${OUTPUT_DIR}/metrics.jsonl"
echo "  ${OUTPUT_DIR}/stdout.log"
echo "  ${OUTPUT_DIR}/summary.json"
echo "  ${OUTPUT_DIR}/timing_breakdown.csv"
echo "  ${OUTPUT_DIR}/npu_profile"
