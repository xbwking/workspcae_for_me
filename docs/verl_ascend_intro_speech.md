# Ascend verl 系统介绍讲稿

大家好，今天这部分我主要讲 Ascend verl 这个系统本身。

我会先把几个问题讲清楚：verl 是什么，Ascend 版本的 verl 和普通 GPU 版本有什么区别，它在强化学习后训练里面怎么工作，训练侧、推理侧、调度侧和数据侧是怎么串起来的。

这部分的目标是建立一个共同的系统视角。因为 Ascend verl 不是一个单独的训练框架，也不是一个单独的推理服务，它是把大模型 RL 后训练里的训练、推理、通信、调度和数据流都组织起来的一套系统工程。

我会按六个部分来讲。

第一，verl 是什么。

第二，Ascend 版本的 verl 和普通 GPU 版本有什么区别。

第三，verl 做 RL 训练时整体流程是什么。

第四，训练侧和推理侧分别由哪些组件组成。

第五，Ascend 版本里的推理链路和权重同步链路怎么工作。

第六，Ascend verl 运行时有哪些典型系统压力点。

---

先讲第一部分，verl 是什么。

verl 可以理解成一个面向大模型后训练的强化学习框架。它主要解决的是大模型 RLHF、RLAIF、GRPO、PPO、DAPO 这类 post-training 场景里的工程问题。

在这类训练里，流程不是简单的“拿一个 batch，前向，反向，更新参数”。

它通常会包含几个角色：

一个是 actor，也就是当前要训练的策略模型。

一个是 rollout，也就是用当前策略模型去生成回答或者轨迹。

一个是 reference model，用来计算 KL 或者作为行为约束。

一个是 critic，在 PPO 这类算法里用于估计 value。

还有一个是 reward model 或 reward function，用来给生成结果打分。

如果对应到第二页这张图，可以把它理解成一个闭环。

最左边是 prompt batch，也就是输入数据，通常会被包装成 verl 内部的 DataProto。

然后进入 rollout。rollout 会调用 vLLM 或 SGLang 这类推理引擎，用当前 actor 策略模型生成 response。

生成结果出来以后，会进入 reward、reference 和 critic 相关计算。reward 负责打分，reference 通常用于 KL 约束，critic 在 PPO 这类算法里用于 value 估计。

这些结果会回到 trainer，trainer 计算 advantage、loss，然后更新 actor 参数。

actor 更新完以后，新权重还要同步回 rollout，这样下一轮生成才会使用新的策略模型。

所以 RL 训练的复杂点在于，它既有训练，又有推理；既有模型参数更新，又有大规模生成；既有 GPU/NPU 计算，又有 Ray 调度、数据流转和模型权重同步。

verl 的价值就是把这些角色组织成一个统一的分布式训练系统。

从使用者角度看，大家可以通过配置指定算法，比如 PPO、GRPO、DAPO；指定训练后端，比如 FSDP、Megatron；指定推理后端，比如 vLLM 或 SGLang；然后由 verl 把这些 worker、resource pool、rollout server、trainer、reward 计算和日志指标串起来。

所以它不是一个单纯的推理框架，也不是一个单纯的训练框架。它更像是一个大模型强化学习 post-training 的编排框架。

---

第二部分，讲 Ascend 版本的 verl 是什么。

Ascend 版本的 verl，就是把 verl 的训练和推理链路适配到昇腾 NPU 环境上。

这里面主要涉及几类适配。

第一类是基础运行时适配。

在 NVIDIA 环境里，我们通常会看到 CUDA、NCCL、torch cuda、vLLM CUDA backend 这些组件。到了昇腾环境，就会变成 CANN、HCCL、torch_npu、vLLM-Ascend 或 SGLang Ascend backend。

也就是说，设备从 GPU 变成 NPU 后，很多底层接口和通信栈都变了。

第二类是训练后端适配。

verl 里 actor、critic、reference model 这些训练角色，可以走 FSDP、FSDP2、Megatron 等不同后端。在 Ascend 环境里，训练侧需要结合 torch_npu、MindSpeed 或 Megatron 相关适配，才能在 NPU 上做并行训练、参数分片和通信。

第三类是推理后端适配。

rollout 侧可以使用 vLLM 或 SGLang。到了 Ascend 环境，vLLM 对应 vLLM-Ascend，SGLang 也需要配置 Ascend attention backend、HCCL 端口范围、NPU 多进程等相关环境。

第四类是通信和权重同步适配。

训练侧模型更新以后，推理侧必须拿到新权重。GPU 环境里可能用 NCCL、CUDA IPC 等机制；Ascend 环境里则需要 HCCL、torch_npu IPC、CPU shared memory fallback，甚至后续可能用 ACL IPC handle。

第五类是性能调优适配。

Ascend 上会有一些 NPU 特有的优化项，比如 fused rotary、fused rmsnorm、fused swiglu、grouped matmul、ACL Graph、chunked prefill、HCCL 通信环境变量等等。这些和 CUDA 环境不完全一样。

所以我们说 Ascend verl，不是简单把设备名从 cuda 改成 npu。它实际是一整套训练、推理、通信、调度、环境变量和性能参数的组合。

---

第三部分，讲 verl 的 RL 训练整体流程。

为了方便理解，我们可以把一次 RL 训练 step 拆成几个阶段。

第一阶段，准备 prompt batch。

trainer 从数据集中取出 prompts，比如数学题、代码题、对话问题，组成一个 batch。这个 batch 会被包装成 verl 内部的 DataProto。

第二阶段，rollout 生成。

actor 当前版本的策略模型会被部署到 rollout engine 里，比如 vLLM 或 SGLang。rollout engine 接收 prompt，生成 response。对于 GRPO 或 DAPO 这类方法，一个 prompt 可能会生成多个 response，也就是 `n` 条采样。

第三阶段，reward 计算。

生成的 response 会送去 reward function 或 reward model。比如数学题可以用规则验证答案，代码题可以跑测试，普通对话可以用 reward model 打分。最终得到每条 response 的 reward 或 token-level reward。

第四阶段，log probability 计算。

训练需要知道当前策略、旧策略、reference model 对这些生成 token 的 log probability。这里通常会计算 old log prob、ref log prob，有些模式下也会利用 rollout log prob。

第五阶段，advantage 和 loss 计算。

根据 reward、KL、value 或 group normalization，计算 advantage。PPO 会有 clipped objective，GRPO 会按组计算归一化 reward 或 advantage。然后 actor 进行反向传播和参数更新。如果有 critic，也会更新 critic。

第六阶段，参数同步。

训练侧 actor 参数更新以后，rollout engine 里的推理模型也要更新，否则后续采样还在用旧权重。这个同步可以是 colocated 的简单同步，也可以通过 checkpoint engine 用 NCCL、HCCL、NIXL、IPC 等方式同步。

第七阶段，进入下一轮 rollout。

新的 rollout 使用更新后的策略模型继续生成样本，循环往复。

所以 verl 的训练流程本质上是一个循环：

取 prompt，生成 response，打 reward，算 log prob 和 advantage，更新 actor，再把新 actor 权重同步给 rollout。

这里面推理侧非常关键。因为 rollout 生成通常是整个 RL 训练里最重、最耗时、最容易成为瓶颈的部分。

---

第四部分，讲 verl 里面几个关键组件。

首先是 Ray。

verl 使用 Ray 做分布式编排。不同角色，比如 actor、critic、reference、reward、rollout server，都会被组织成 Ray worker 或 Ray actor。Ray 负责资源调度、远程调用、worker group 管理和跨进程通信。

所以在 verl 里，我们经常会看到 RayWorkerGroup、ResourcePool、Ray actor 这些概念。

ResourcePool 可以理解成资源池，决定哪些 worker 放在哪些节点、哪些卡上。

WorkerGroup 可以理解成一组执行同一类任务的 worker，比如 actor worker group、critic worker group、rollout worker group。

其次是 trainer。

trainer 是整个训练流程的控制中心。它负责拉数据、触发 rollout、收集 reward、计算 advantage、更新 actor 和 critic，以及调用 checkpoint manager 做权重同步。

在 PPO 或 GRPO 场景里，trainer 会围绕一个训练 step 组织这些阶段。

第三个是 actor。

actor 是正在训练的策略模型。它负责根据训练数据计算 loss，做反向传播，并更新参数。actor 的参数更新后，要同步给 rollout 侧。

第四个是 rollout。

rollout 是推理侧。它负责用当前策略模型生成 response。这个角色通常不直接用训练框架的 forward，而是通过 vLLM 或 SGLang 这样的推理引擎来跑，因为推理引擎有 continuous batching、KV cache、prefix cache、paged attention 或对应 Ascend 后端优化，吞吐会更高。

第五个是 reference model。

reference model 通常是一个固定模型，用于计算 KL。它的作用是约束 actor 不要偏离原模型太远。

第六个是 critic。

critic 用于 PPO 这类 value-based 方法，估计每个 token 或状态的 value。GRPO 这种方法可以不依赖 critic，而是用组内 reward 归一化。

第七个是 reward。

reward 可以是一个模型，也可以是一个函数。比如数学题、代码题经常使用 rule-based reward 或 verifiable reward。

第八个是 checkpoint engine。

这个组件负责把训练侧 actor 的最新权重同步给 rollout 推理侧。不同硬件和部署模式下，它可以有不同 backend，比如 naive、nccl、hccl、nixl 等。

---

第五部分，讲 Ascend verl 里推理和参数同步怎么工作。

在 Ascend 支持版本里，rollout 后端主要可以是 vLLM-Ascend 或 SGLang Ascend。

如果使用 vLLM，verl 会启动 vLLM rollout server。这个 server 是一个长寿命的 Ray actor。它不只是接收一次请求然后退出，而是长期运行，持续接收 prompt，进行 generation。

vLLM server 内部会管理 KV cache、prefix cache、batching、请求队列，以及模型权重。

在训练过程中，当 actor 更新以后，rollout server 里的权重也需要更新。这个时候就会走权重同步链路。

当前 Ascend 支持版本里的权重同步大概是这样：

trainer 侧通过 `CheckpointEngineManager.update_weights()` 发起同步。

如果是非 naive backend，manager 会先让 rollout replica 暂停或 abort 当前请求，然后 sleep rollout server，释放 cache 和权重相关显存。

接着它创建 checkpoint engine worker group，并建立 trainer 和 rollout 之间的通信拓扑。

在 HCCL 路径里，训练侧会把权重按 bucket 打包到 NPU 上的 uint8 buffer 里，然后通过 HCCL broadcast 给 rollout 侧。

rollout 侧收到 bucket 后，再把里面的 tensor view 交给 server adapter，由 adapter 调用 vLLM 或 SGLang 的 update_weights 接口，把新权重加载到推理引擎里。

如果是 vLLM colocated 或 IPC 路径，则还会用 `BucketedWeightSender` 和 `BucketedWeightReceiver`。sender 侧创建一个 bucket buffer，如果设备支持 IPC，就用 torch 的 reduce_tensor 传递 IPC handle；如果不支持，就 fallback 到 CPU shared memory。receiver 侧 rebuild buffer，再把其中的 tensor 交给推理引擎。

这里有几个非常关键的性能点。

第一，权重同步期间 rollout 是否要完全停止。

如果每次同步都 abort 所有请求，再 sleep/wake server，那么同步停顿会很大。

第二，权重传输是走设备 IPC 还是 CPU shared memory。

如果 fallback 到 CPU shared memory，传输链路会更慢，host memory 压力也更大。

第三，bucket size 是否合适。

bucket 太小，通信次数多；bucket 太大，可能占用过多 HBM，甚至遇到大 tensor 超过 bucket 的问题。

第四，metadata 和 tensor 是否还在走 Python pickle。

如果 metadata 或 sample 仍大量依赖 cloudpickle，就会产生 CPU 和内存瓶颈。

第五，权重更新后是否清理 prefix cache。

清理 cache 可以保证一致性，但会造成同步后推理冷启动。

这些点不是某一个单独模块的问题，而是 Ascend verl 运行时自然会出现的系统压力来源。

---

第六部分，讲 Ascend verl 的系统压力点。

在大模型 RL 训练里，推理侧不是一个附属模块，它往往会影响整个训练 step 的节奏。

原因有三个。

第一，rollout 生成本身很重。

训练一个 reasoning model 时，每个 prompt 可能生成很长 response，而且一个 prompt 可能采样多个 response。生成阶段会消耗大量 decode 时间和 KV cache。

第二，训练和推理之间频繁切换。

actor 更新以后，rollout 侧要拿到新权重。这个同步频率越高，参数同步开销越明显。尤其是大模型场景，权重传输本身就非常重。

第三，RL 数据流比 SFT 更复杂。

SFT 的数据相对固定，训练 batch 比较直接；RL 训练里，数据是不断生成出来的。生成结果、reward、log prob、advantage、sample metadata 都要在多个 worker 之间流转。这里很容易出现 Ray 调度、序列化、object store 和队列瓶颈。

所以如果只看单个推理 kernel，可能看不到完整问题。真正影响端到端效率的，是整个系统链路：

rollout server 怎么接请求；

请求怎么 batching；

KV cache 怎么管理；

权重怎么同步；

同步时是否暂停 generation；

sample 怎么从 rollouter 传给 trainer；

Ray actor 是否成为单点；

数据面是否还在走 cloudpickle。

这些共同构成了 Ascend verl 的系统压力点。

第一类压力是 rollout 生成重。长 response、多采样、decode 阶段和 KV cache 都会占用大量 NPU 资源。

第二类压力是权重同步频繁。actor 每次更新后，rollout 都要拿到新权重，大模型权重越大，同步成本越明显。

第三类压力是数据流复杂。sample、reward、logprob、metadata 会在多个 worker 之间流转，数据量和对象数量都不小。

第四类压力是 Ray 控制面压力。短样本、高频 RPC、actor mailbox 和队列都会影响调度稳定性。

所以理解 Ascend verl，不能只看一个模型 forward，也不能只看一个通信算子，而要看训练、推理、权重同步、样本回传和 Ray 调度共同组成的闭环。

---

这里我先做一个小结。

verl 是一个大模型后训练的 RL 编排框架，它把 actor、rollout、reference、critic、reward、Ray 调度和数据流组织在一起。

Ascend 版本的 verl，是这套框架在昇腾 NPU 环境上的适配，它涉及 CANN、HCCL、torch_npu、vLLM-Ascend、SGLang Ascend backend，以及 NPU 上的通信和性能调优。

在 RL 训练流程里，rollout 推理不是边缘模块，而是训练闭环里的核心环节。训练侧更新参数之后，推理侧必须同步权重；rollout 生成的样本也必须回传给 trainer。

因此，Ascend verl 的主线可以概括成这条路径：

训练更新参数；

推理同步权重；

推理生成样本；

样本回到训练；

训练继续更新。

理解了这条路径，就能理解 Ascend verl 为什么是一个系统问题：它既要让训练侧持续更新，也要让推理侧持续生成，还要让权重、样本和调度状态在不同 worker 之间正确流动。

这就是 Ascend verl 的整体工作方式。
