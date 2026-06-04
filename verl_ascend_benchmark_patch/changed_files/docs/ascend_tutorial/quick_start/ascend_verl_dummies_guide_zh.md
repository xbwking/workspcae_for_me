# 昇腾上运行 verl 傻瓜教程

> 适用对象：第一次在 Ascend NPU 上跑 verl 的同学。目标是先跑通一个 Qwen2.5-0.5B GRPO 小样例，再跑耗时拆解 benchmark。

## 0. 先看结论

最省事路线：

1. 用官方 Ascend verl 镜像或你们集群已经准备好的 Ascend 容器。
2. 进入容器后确认 `npu-smi info`、`python -c "import torch, torch_npu"` 能跑。
3. 预处理 GSM8K 数据。
4. 跑 `tests/special_npu/run_qwen2_5_05b_grpo.sh`。
5. 如果要验证优化收益，再跑 `tests/special_npu/run_ascend_timing_breakdown_bench.sh`。

截至 2026-06-04，线上最新 verl Ascend 文档里的推荐镜像组件已经更新到 CANN 9.0.0、torch 2.9.0、torch_npu 2.9.0、vLLM/vLLM-Ascend 0.18.0。当前本地仓库里的部分 quick start 文档仍保留 CANN 8.5.0、torch 2.8.0、vLLM 0.13.0 的写法，所以实际安装时以你们目标代码分支和镜像为准。

## 1. 准备机器

你需要一台能访问 Ascend NPU 的机器，常见是：

- Atlas 800T A3
- Atlas 900 A2
- Atlas 200T A2

先登录机器或进入容器：

```bash
npu-smi info
```

能看到 NPU 卡信息才继续。看不到就先找集群管理员确认驱动、容器挂载、权限。

## 2. 推荐方式：用现成 Ascend verl 镜像

如果你们平台已经有 Ascend verl 镜像，直接进平台提供的容器。

如果需要自己拉镜像，参考官方镜像仓库：

```bash
# 示例：具体镜像标签以官方镜像仓库或你们平台镜像为准
export VERL_ASCEND_IMAGE=quay.io/ascend/verl:verl-9.0.0-a3-ubuntu22.04-py3.11-latest
docker pull "$VERL_ASCEND_IMAGE"
```

进入容器后，先激活 Ascend 环境变量：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 如果容器或机器安装了 nnal/atb，再执行这一句
if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
  source /usr/local/Ascend/nnal/atb/set_env.sh
fi
```

检查 Python 依赖：

```bash
python3 - <<'PY'
import torch
import torch_npu
print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("npu available:", torch.npu.is_available())
print("npu count:", torch.npu.device_count())
PY
```

期望看到：

```text
npu available: True
npu count: 大于 0
```

## 3. 裸机源码安装方式

如果不能用现成镜像，再走源码安装。裸机安装更容易踩坑，版本必须匹配。

### 3.1 激活 CANN

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
  source /usr/local/Ascend/nnal/atb/set_env.sh
fi
```

### 3.2 创建 Python 环境

```bash
conda create -n verl-ascend python=3.11 -y
conda activate verl-ascend
python -m pip install --upgrade pip
```

### 3.3 安装 torch / torch_npu

这里不要随便混版本。原则是：

- CANN 9.0.0 通常配 torch 2.9.x / torch_npu 2.9.x。
- CANN 8.5.0 通常配 torch 2.8.x / torch_npu 2.8.x。
- vLLM-Ascend 版本要和 vLLM 版本一致。

如果你们使用官方镜像，这步可以跳过。

### 3.4 安装 verl

```bash
git clone --recursive https://github.com/volcengine/verl.git
cd verl

pip install -r requirements-npu.txt
pip install -v -e .
```

检查：

```bash
python3 - <<'PY'
import verl
print("verl import ok")
PY
```

## 4. 准备数据

先跑 GSM8K 小样例，数据预处理命令：

```bash
python3 examples/data_preprocess/gsm8k.py --local_save_dir "$HOME/data/gsm8k"
```

检查文件：

```bash
ls -lh "$HOME/data/gsm8k"
```

应该至少有：

```text
train.parquet
test.parquet
```

## 5. 准备模型

最小验证建议用：

```text
Qwen/Qwen2.5-0.5B-Instruct
```

如果你的机器不能直接访问 Hugging Face，先把模型下载到本地，例如：

```bash
export MODEL_PATH="$HOME/.cache/models/Qwen/Qwen2.5-0.5B-Instruct"
```

确认目录里有模型文件：

```bash
ls -lh "$MODEL_PATH"
```

## 6. 跑第一个 GRPO 小样例

在仓库根目录执行：

```bash
export VERL_HOME=${VERL_HOME:-$PWD}
cd "$VERL_HOME"

export MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct
export MODEL_PATH=${MODEL_PATH:-$HOME/.cache/models/$MODEL_ID}

bash tests/special_npu/run_qwen2_5_05b_grpo.sh
```

这个脚本会跑一个很小的 GRPO 样例，并开启 NPU profiler 输出检查。第一次启动会比较慢，常见原因是：

- vLLM-Ascend 初始化
- 图编译
- 模型加载
- Ray worker 启动

看到训练 step 正常前进，就说明基础环境基本通了。

## 7. 手写一个最小训练命令

如果你不想用脚本，可以直接跑：

```bash
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=$HOME/data/gsm8k/train.parquet \
  data.val_files=$HOME/data/gsm8k/test.parquet \
  data.train_batch_size=16 \
  data.max_prompt_length=512 \
  data.max_response_length=128 \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.ref.use_torch_compile=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.kl_ctrl.kl_coef=0.001 \
  trainer.critic_warmup=0 \
  trainer.logger=console \
  trainer.project_name='verl_grpo_example_gsm8k' \
  trainer.experiment_name='qwen2_5_0_5b_npu_smoke' \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.total_epochs=1 \
  trainer.total_training_steps=1 \
  trainer.device=npu
```

如果你的 verl 版本已经自动识别 NPU，`trainer.device=npu` 可以不写；但为了傻瓜教程稳定，这里保留。

## 8. 跑耗时拆解 benchmark

如果你要验证框架优化收益，跑这个：

```bash
MODEL_PATH="$MODEL_PATH" \
TRAIN_FILES="$HOME/data/gsm8k/train.parquet" \
VAL_FILES="$HOME/data/gsm8k/test.parquet" \
OUTPUT_DIR=outputs/ascend_timing_breakdown/baseline \
bash tests/special_npu/run_ascend_timing_breakdown_bench.sh
```

跑完后看：

```bash
ls -lh outputs/ascend_timing_breakdown/baseline
```

关键文件：

```text
metrics.jsonl
stdout.log
summary.json
timing_breakdown.csv
npu_profile/
```

如果你要对比优化前后：

```bash
# 1. baseline 分支跑一次
OUTPUT_DIR=outputs/ascend_timing_breakdown/baseline \
bash tests/special_npu/run_ascend_timing_breakdown_bench.sh

# 2. patched 分支跑一次
OUTPUT_DIR=outputs/ascend_timing_breakdown/patched \
bash tests/special_npu/run_ascend_timing_breakdown_bench.sh

# 3. 生成对比报告
python3 scripts/bench_ascend_verl_timing.py compare \
  --baseline-summary outputs/ascend_timing_breakdown/baseline/summary.json \
  --patched-summary outputs/ascend_timing_breakdown/patched/summary.json \
  --output outputs/ascend_timing_breakdown/compare.json
```

## 9. 你应该重点看哪些指标

端到端：

```text
perf/time_per_step
perf/throughput
timing_s/step
timing_s/gen
timing_s/update_actor
timing_s/update_weights
```

权重同步：

```text
param_sync/abort_ms
param_sync/sleep_ms
param_sync/build_pg_ms
param_sync/send_recv_update_ms
param_sync/finalize_ms
param_sync/wake_ms
param_sync/resume_ms
```

队列和序列化：

```text
ray/message_queue_get_rpc_count
ray/message_queue_get_wait_s
serialization/cloudpickle_load_s
```

权重传输：

```text
weight_transfer/sender_copy_ms
weight_transfer/receiver_copy_ms
weight_transfer/metadata_send_ms
weight_transfer/metadata_recv_ms
weight_transfer/sender_bucket_bytes
weight_transfer/receiver_bucket_bytes
```

## 10. 常见错误和处理

### 10.1 `torch.npu.is_available()` 是 False

检查：

```bash
npu-smi info
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 -c "import torch, torch_npu; print(torch.npu.is_available())"
```

如果还是 False，通常是容器没有挂 NPU、驱动/CANN/torch_npu 版本不匹配。

### 10.2 vLLM-Ascend import 或初始化失败

检查：

```bash
python3 - <<'PY'
import vllm
import vllm_ascend
print("vllm import ok")
PY
```

版本要匹配。vLLM `0.18.0` 就应该配 vLLM-Ascend `0.18.0`；vLLM `0.13.0` 就配 vLLM-Ascend `0.13.0`。

### 10.3 Ray 起不来或资源识别不对

先看：

```bash
ray stop --force || true
ray start --head
ray status
```

如果是 SGLang 后端或多进程场景，常见需要：

```bash
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
```

### 10.4 显存不够

先降低：

```bash
data.train_batch_size=8
actor_rollout_ref.actor.ppo_mini_batch_size=4
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
actor_rollout_ref.rollout.n=1
actor_rollout_ref.rollout.gpu_memory_utilization=0.5
```

或者换更小模型。

### 10.5 第一步特别慢

正常。通常是：

- 模型加载
- vLLM-Ascend 初始化
- 图编译
- NPU profiler 启动
- Ray actor 初始化

判断是否卡死，看日志是否持续输出、NPU 是否有占用。

## 11. 最小检查清单

跑训练前确认：

```bash
npu-smi info
python3 -c "import torch, torch_npu; print(torch.npu.is_available(), torch.npu.device_count())"
python3 -c "import verl; print('verl ok')"
python3 -c "import vllm; print('vllm ok')"
ls "$HOME/data/gsm8k/train.parquet"
ls "$MODEL_PATH"
```

以上都没问题，再跑训练脚本。

## 12. 参考资料

- verl Ascend Dockerfile Build Guidance: https://verl.readthedocs.io/en/latest/ascend_tutorial/get_start/dockerfile_build_guidance.html
- verl Ascend Install Guidance: https://verl.readthedocs.io/en/latest/ascend_tutorial/get_start/install_guidance.html
- verl Ascend SGLang Quickstart: https://ascend.github.io/docs/sources/_generated/sources/verl/ascend_sglang_quick_start.html
- vLLM-Ascend Installation: https://docs.vllm.ai/projects/ascend/en/v0.13.0/installation.html
