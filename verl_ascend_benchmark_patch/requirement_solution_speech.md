# verl Ascend 推理优化需求与方案设计串讲稿

各位好，今天我主要围绕我们刚刚补充的《verl Ascend 推理优化需求与方案设计》做一次方案串讲。

这次串讲的重点不是单纯介绍某一个 patch，而是把我们为什么要做这件事、要解决什么问题、方案怎么设计、怎么保证可靠可用，以及后续怎么测试验证，完整地串起来。

我会按照五个部分展开：

第一部分是需求分析，包括背景、价值和目标。  
第二部分是场景分析，包括 use case 和约束限制。  
第三部分是方案设计，包括整体方案、影响分析、交互和任务拆解。  
第四部分是可靠可用设计，包括冗余和防呆。  
第五部分是测试建议，也就是后续怎么证明这个方案是有效的。

---

## 一、需求分析

我们先看需求背景。

verl 在大模型强化学习训练里，不是一个单纯的训练框架，它同时要协调训练、推理采样、奖励计算、权重同步和样本回流这几条链路。

尤其在 RLHF、GRPO、PPO 这类场景里，训练侧会不断更新 actor 权重，推理侧又要持续用最新或者较新的 actor 权重去生成 rollout 样本。这个过程天然会有一个矛盾：训练希望尽快更新参数，推理希望持续高吞吐生成样本，而权重同步会打断推理。

在 Ascend NPU 环境下，当前支持基线已经具备了一些基础能力，比如 vLLM-Ascend、HCCL checkpoint engine、bucketed IPC 或 shared memory 权重传输、Ray WorkerGroup，以及 fully_async policy。但是从推理性能优化角度看，现有链路还有几个比较明确的问题。

第一个问题，是权重同步链路对推理采样存在全局暂停。

现在 `CheckpointEngineManager.update_weights()` 的流程比较粗，它会串行做 `abort_all_requests`、`sleep`、`build_process_group`、`send/receive/update`、`finalize`、`wake` 和 `resume_generation`。这个流程的好处是安全，能够保证推理侧更新权重的一致性。但是坏处也很明显，它把请求中断、KV cache 释放、process group 构建、权重通信、推理引擎更新、恢复生成这些成本全部暴露到端到端 step 时间里。

也就是说，我们在日志里看到一个 `timing_s/update_weights` 很慢时，其实不知道到底慢在哪里。可能是 HCCL 传输慢，可能是 sleep/wake 慢，可能是 process group 初始化慢，也可能是 rollout server update 权重慢。

第二个问题，是 Ascend HCCL 后端本身存在一个可用性问题。

我们复核代码时发现，当前基线里的 `HCCLCheckpointEngine` 实际注册成了 `"nccl"`，但是配置和文档里期望的是 `"hccl"` backend。这意味着如果用户配置 `backend=hccl`，可能无法正确命中 HCCL engine。同时它还可能和 NCCL backend 的注册产生覆盖风险。

这个问题首先不是性能优化，而是可用性 bug。所有后续基于 HCCL 的优化，都必须先建立在 backend 能正确命中的基础上。

第三个问题，是现有 IPC 和 shared memory 权重传输缺少可观测性。

vLLM rollout 当前已经会根据环境能力选择 IPC 或 shared memory fallback，但是我们看不到它实际走了哪条路径，也看不到每次权重传输有多少 bucket、多少字节、metadata 花了多久、sender copy 花了多久、receiver clone 或者 to device 花了多久。

这会导致性能分析时只能猜。比如一次权重同步很慢，我们不知道是 IPC 没生效退化到了 SHM，还是 bucket 太多，还是 receiver 端 clone 成本太高。

第四个问题，是 fully_async 样本回流路径存在 Ray RPC 和 cloudpickle 压力。

当前 fully_async 的 MessageQueue 是单 Ray Actor。trainer 端逐个调用 `get_sample_sync()` 拉样本，rollouter 端会把整个 rollout sample 做 `cloudpickle.dumps()`，trainer 端再做 `cloudpickle.loads()`。

如果是长 response、多 rollout replica，或者大量短样本场景，瓶颈就不一定在 NPU 上，反而可能在 Ray actor mailbox、Ray GCS、CPU 序列化，以及 object store 上。

第五个问题，是缺少统一 benchmark 来证明优化收益。

只看训练日志里的 `timing_s/step`，我们只能知道整体变快或者变慢，但无法说明收益来自哪里。比如 step 时间下降了，是因为 MessageQueue RPC 下降，还是因为权重同步下降，还是因为 IPC 传输变快，还是因为 NPU 算子侧变化？如果没有统一的耗时拆解 benchmark，这些问题都回答不了。

所以这次需求的核心价值，就是把 verl Ascend 场景里的推理性能优化，从“凭经验判断”，推进到“可度量、可回退、可持续迭代”的工程体系。

具体价值有五点。

第一，提升推理采样吞吐。我们通过减少权重同步期间的推理暂停、减少 Ray RPC 次数、降低样本序列化成本，提升 rollout 生成链路的有效吞吐。

第二，降低大模型权重同步开销。我们对 HCCL、IPC、SHM 权重传输做分段打点，后续就可以针对 bucket 传输、metadata 编码、receiver copy、process group 构建这些子环节做优化。

第三，降低 Ray 调度和队列压力。MessageQueue 批量化、后续分片化，以及 SampleRef 加 TransferQueue 的设计，可以减少大量短生命周期 RPC 对 Ray GCS 和单 actor mailbox 的压力。

第四，提供可证明的优化收益。通过统一 benchmark 输出 `summary.json`、`timing_breakdown.csv` 和 `compare.json`，我们可以对每个优化点给出 baseline 和 patched 的 mean、p50、p95、speedup 和 verdict。

第五，控制生产风险。我们不是一上来就改高风险链路，而是保留默认保守路径，比如 `abort_all` 权重同步策略、SHM fallback、prefix cache 默认清理、同步失败回退旧权重版本。这样可以避免为了性能破坏训练正确性。

接下来讲需求目标。

近期目标有六个。

第一，修复 HCCL backend 注册问题，保证 Ascend 权重同步链路能正确启用。

第二，为权重同步链路补充分段耗时指标，包括 abort、sleep、build process group、send/receive update、finalize、wake、resume。

第三，为 bucketed weight transfer 增加 sender 和 receiver 两侧指标，包括 bucket 数、字节数、metadata 时间、copy 时间和 sync 时间。

第四，在 fully_async MessageQueue 中实现批量 `get_samples(max_n)`，降低 trainer 端逐样本 Ray RPC 开销。

第五，在 fully_async trainer 中记录 queue get RPC 次数和 cloudpickle load 时间。

第六，构建 Ascend timing breakdown benchmark，支持单次运行、结果汇总和 baseline / patched 对比。

中长期目标主要包括：

权重同步从当前的 `abort_all`，逐步演进到 `drain_then_commit`，再到 `prefetch_then_commit`；MessageQueue 做分片；引入 SampleRef 加 TransferQueue，把大 tensor 字段从 Ray cloudpickle 数据面中移出去；研究 ACL IPC handle 在通信 buffer 上的可行性；最后联合 vLLM-Ascend 或 SGLang 验证 prefix / KV cache version 化策略。

这里要强调一点：中长期目标里有些是研究项，比如 ACL IPC 长期零拷贝权重共享，这个不能在近期直接承诺生产收益。近期最稳妥的方向还是修正链路、做指标、做低风险优化，并用 benchmark 证明收益。

---

## 二、场景分析

接下来我们看场景分析，也就是这个方案到底服务哪些 use case。

第一个 use case，是 Ascend 环境运行常规 GRPO 或 PPO 训练。

用户在 Ascend NPU 集群上跑 verl，使用 vLLM-Ascend 作为 rollout engine，训练侧定期把 actor 权重同步到 rollout 侧。

这个场景下，用户最关心的是每个 training step 的端到端耗时、rollout generation 耗时、actor update 耗时、权重同步耗时，以及参数同步是否导致 rollout 长时间暂停。

我们给这个场景提供的是 L0 和 L1 指标。

L0 是端到端指标，比如 `timing_s/step`、`timing_s/gen`、`timing_s/update_actor`、`timing_s/update_weights`、`perf/throughput`。

L1 是框架链路指标，比如 `param_sync/*`，用于拆解权重同步的每一个阶段。

第二个 use case，是推理性能团队验证某个框架优化是否有效。

比如我们做了 MessageQueue 批量 get、bucket metadata 优化、权重同步策略优化。这个时候不能只说“理论上会更快”，必须能做 A/B 对比。

所以我们提供 `scripts/bench_ascend_verl_timing.py run` 来固定实验参数并落盘，提供 `summarize` 生成 `summary.json` 和 `timing_breakdown.csv`，再提供 `compare` 生成 baseline / patched 对比和优化 verdict。

这就解决了一个关键问题：以后任何优化 patch，都可以用同一套 benchmark 判断收益是否真实存在。

第三个 use case，是 fully_async 短样本高并发采样。

这个场景里 rollouter 持续生成样本并写入 MessageQueue，trainer 异步消费样本并更新 actor。

这里的核心问题不是单个 NPU 算子慢，而是 trainer 等样本、MessageQueue RPC 次数、Ray actor mailbox、GCS 压力，以及 cloudpickle 序列化和反序列化成本。

本方案提供 `get_samples(max_n)` 批量消费接口，并记录 `fully_async/message_queue_get_rpc_count` 和 `fully_async/cloudpickle_load_time`。后续还可以扩展 queue shard 和 SampleRef / TransferQueue。

第四个 use case，是同节点权重传输链路调优。

训练和推理进程部署在同一节点时，rollout 侧可能通过 IPC 或 shared memory fallback 接收权重。这里我们需要知道实际走 IPC 还是 SHM、bucket 数量是多少、总字节数是多少、sender copy 和 receiver clone 成本是多少、metadata 编解码时间是多少。

所以我们给 `BucketedWeightSender` 和 `BucketedWeightReceiver` 都加 stats，然后 benchmark parser 会把这些 stdout 里的统计解析成 `weight_transfer/*` 指标。

再看约束和限制。

第一，Ascend 支持版本必须以实际 recipe 固定 commit 为准。当前设计基线是 `4045d670` 这个 Ascend-supported commit，不能直接把 GitHub main 当成 Ascend 支持版本。

第二，NPU stream 完全重叠不能直接承诺。当前 HCCL engine 虽然有双 buffer，但实际代码里 `BroadcastOperation` 仍然同步执行 ZMQ metadata 和 `pyhccl.broadcast()`，并且有 `torch.npu.synchronize()`。所以能否做到通信流和计算流完全重叠，需要 pyhccl 能力和 profiler 验证。

第三，同节点长期零拷贝权重共享不能作为近期生产承诺。当前已有的是 bucket IPC 传输，不是推理进程长期挂载训练权重。receiver 侧仍可能 clone 或 to device，推理引擎仍有自己的权重副本。

第四，TransferQueue 在当前 commit 中主要是外部集成入口，本地没有完整 in-tree 实现。因此我们可以设计 SampleRef / TransferQueue 方向，但不能说当前代码已经完整具备。

第五，Prefix / KV cache 保留策略正确性敏感。权重版本变化后复用旧 KV cache 可能产生跨权重版本污染，所以默认必须保持保守清理策略。

第六，benchmark 的真实收益必须在 Ascend 环境验证。本地 CPU 测试只能验证 parser、CLI、dry-run 和局部逻辑，不能代替真实 NPU 端到端训练。

---

## 三、方案设计

接下来进入方案设计。

整体方案的原则是：可观测性先行、低风险优化优先、异步化分阶段推进。

这里我们把整体架构分成五层。

第一层是基线修正层。

这一层解决的是“链路要先能正确工作”。最典型的就是修复 HCCL registry，把 `HCCLCheckpointEngine` 的注册 key 修成 `"hccl"`，并增加重复注册告警，避免 NCCL / HCCL backend 静默覆盖。

第二层是指标采集层。

我们在权重同步、bucketed transfer、fully_async queue、cloudpickle load 等关键路径增加 L1 指标。指标来源包括 file logger、stdout log 和 benchmark parser。这样以后不是看一个粗粒度耗时，而是能看每个子环节。

第三层是低风险优化层。

这一层先做不改变训练语义的优化，比如 MessageQueue 批量 get、权重同步分段计时、bucket transfer 统计。这些改动风险低，但能立刻提升可观测性，并且部分场景下能直接降低 Ray RPC 成本。

第四层是权重同步策略层。

当前默认仍然保留 `abort_all`，保证生产回退路径不变。后续我们再引入 `drain_then_commit` 和 `prefetch_then_commit`。

`drain_then_commit` 的思路是：先暂停接收新请求，等待正在执行的请求自然 drain，超时后再 abort 少量尾部请求，然后进行权重 commit。

`prefetch_then_commit` 的思路是：先把新权重预加载到 rollout 侧影子缓存里，不影响当前推理请求；等请求边界或者 stale 阈值触发时，再做短暂停顿，把 READY 状态的新权重 commit 到推理引擎。

第五层是 benchmark 验证层。

通过 Ascend timing breakdown benchmark 对 baseline 和 patched 做 A/B 对比，输出 `summary.json`、`timing_breakdown.csv` 和 `compare.json`。这一层的作用是让每个优化都能被证明，而不是停留在代码解释。

从数据流看，Trainer 更新 actor 权重后，会通过 `CheckpointEngineManager` 进入 HCCL、IPC 或 SHM 权重同步，再进入 Rollout Server 的 `update_weights`。随后 vLLM-Ascend 或 SGLang 继续做采样，产生 RolloutSample 或后续的 SampleRef，再通过 MessageQueue 或 TransferQueue 回流给 Trainer。整个链路中，权重同步产生 `param_sync` 指标，权重传输产生 `weight_transfer` 指标，样本回流产生 `ray` 和 `serialization` 指标，训练主循环产生 `timing_s` 和 `perf` 指标，最后全部进入 benchmark summary 和 compare。

接下来讲影响分析。

首先是对训练正确性的影响。

P0 和 P1 中的 registry 修复、指标打点、MessageQueue 批量 get，原则上不改变训练数学语义。批量 get 只是减少 RPC 次数，不改变样本内容。

但是 `drain_then_commit` 和 `prefetch_then_commit` 会引入权重版本边界，所以必须控制 staleness。样本需要携带 `param_version`，trainer 消费样本时要检查 stale 阈值，commit 失败时不能切换到半更新权重。对于 GRPO，多 response per prompt 的 group normalization 对版本一致性更敏感，所以同一 prompt group 内要尽量保持版本一致。

其次是对推理服务的影响。

当前 `abort_all` 安全但粗暴，会中断请求并清 cache。`drain_then_commit` 可以降低中断概率，但会增加等待 drain 的逻辑。`prefetch_then_commit` 可以缩短最终 commit 停顿，但需要额外缓存和状态机。

第三是对资源消耗的影响。

指标采集成本较低，但 NPU profiler 会影响运行时性能，所以 profiler 只适合少量 step 抽样。prefetch 和 cache 方案可能增加 HBM 或 CPU pinned memory 占用，所以默认应该限制 `max_cached_versions=1`，并且按 bucket 管理生命周期。

第四是对运维和定位的影响。

这是正向影响。以前遇到性能问题时，我们只能知道 step 变慢。现在通过这些指标，可以判断瓶颈在 Ray / MessageQueue、cloudpickle、HCCL / IPC transfer、rollout server update、vLLM-Ascend generation，还是 actor update。

再看 Use Case 设计里的交互。

主要交互对象有五类：

第一类是算法和训练用户。他们通过 Hydra 参数选择 rollout engine、checkpoint backend、同步策略和 benchmark 参数，关注训练是否跑通，吞吐是否提升，结果是否稳定。

第二类是推理性能优化工程师。他们通过 benchmark runner 跑 baseline 和 patched，对比 L0、L1、L2 指标，判断优化是否有效。

第三类是 Rollout Server。它负责接收权重更新、生成样本、处理 pause、resume、abort 和 commit。

第四类是 CheckpointEngineManager。它是权重同步编排的核心入口，也是后续实现 `drain_then_commit` 和 `prefetch_then_commit` 的关键位置。

第五类是 MessageQueue 和 TransferQueue。它们负责样本从 rollouter 到 trainer 的回流，是 fully_async 场景下的数据面核心。

典型流程是：用户配置 benchmark，benchmark runner 启动 `main_ppo`，trainer 调 rollout 生成样本，rollout 写入 MessageQueue，trainer 批量拉取样本并训练，训练后调用 CheckpointEngineManager 更新权重。训练过程中的 file logger 和 stdout 被 benchmark parser 汇总，最后输出 summary、csv 和 compare verdict。

功能原理可以拆成五个点。

第一个是 HCCL backend 修复。

把 `HCCLCheckpointEngine` 注册 key 从 `"nccl"` 修正为 `"hccl"`，这样配置 `backend=hccl` 时能正确实例化 HCCL engine。

第二个是权重同步分段计时。

我们在 `CheckpointEngineManager.update_weights()` 内部围绕每个阶段记录耗时，形成 `param_sync/*` 指标。这样可以判断同步慢到底是因为 abort、sleep、process group、send/recv、finalize，还是 wake/resume。

第三个是 bucketed transfer 指标。

sender 记录 bucket 数、总字节数、copy 时间、metadata send 时间和 sync 时间；receiver 记录 metadata recv 时间、clone 或 to device 时间、bucket bytes 和 sync 时间。

第四个是 MessageQueue 批量 get。

原来 trainer 每次从 Ray actor 拉一个 sample，现在一次最多拉 `max_n` 个 sample。队列为空时最多等待 timeout，队列关闭时返回 None，遇到 None sentinel 时保留原来的终止语义。

第五个是 benchmark 汇总和对比。

benchmark runner 会从 file logger 收集 `timing_s/*`、`perf/*`、`fully_async/*` 等指标，从 stdout 解析 `param_sync/*` 和 `weight_transfer/*`，最后生成 summary 和 compare。

接下来讲 Story 和 Task 分解。

第一个 Story：作为 Ascend verl 用户，我希望 `backend=hccl` 能正确启动，使 HCCL 权重同步链路可用。

对应任务是修正 HCCL registry，增加 registry 覆盖 warning，并增加 CPU 可执行源码测试或 NPU smoke test。

第二个 Story：作为性能工程师，我希望看到权重同步各阶段耗时，定位同步瓶颈。

对应任务是在 `CheckpointEngineManager.update_weights()` 增加分段 timer，把 `last_update_weights_timing` 写入训练 metrics，并在 benchmark parser 中聚合 `param_sync/*`。

第三个 Story：作为性能工程师，我希望知道权重传输走 IPC 还是 SHM，以及每个 bucket 的 copy 和 metadata 成本。

对应任务是给 `BucketedWeightSender` 和 `BucketedWeightReceiver` 增加 stats，并在 stdout parser 中解析这些 stats。

第四个 Story：作为 fully_async 用户，我希望 trainer 批量消费样本，减少 Ray RPC 开销。

对应任务是给 `MessageQueue` 和 `MessageQueueClient` 增加批量接口，修改 `FullyAsyncTrainer._get_samples_from_queue()` 使用批量 get，并记录 RPC 次数和 cloudpickle load 时间。

第五个 Story：作为优化验证负责人，我希望有统一 benchmark 证明每个 patch 的收益。

对应任务是新增 `scripts/bench_ascend_verl_timing.py`，新增 Ascend 一键运行脚本，支持 `run`、`summarize`、`compare`，并输出 summary、csv 和 compare。

第六个 Story：作为后续优化开发者，我希望可以逐步演进到异步权重同步。

对应任务是设计 `sync_policy`，第一阶段实现 `drain_then_commit`，第二阶段实现 `HCCLCachedCheckpointEngine` 和 `prefetch / commit`，并引入 `param_version` 和 stale sample 控制。

---

## 四、可靠可用设计

接下来讲可靠可用设计。

首先是冗余设计。

第一，同步策略冗余。

默认保留当前 `abort_all` 路径。`drain_then_commit` 和 `prefetch_then_commit` 都必须作为可配置策略启用。如果新策略异常，可以直接回退到 `abort_all`。

第二，通信路径冗余。

vLLM rollout 当前已经支持 IPC 和 shared memory fallback。我们优化 IPC 传输时不能移除 SHM fallback。当 Ascend IPC 环境不满足要求时，系统应该自动退回 SHM，并输出 `weight_transfer/path=shm`。

第三，权重版本冗余。

prefetch 阶段不能覆盖当前已 commit 权重。只有新版本完整接收并进入 READY 状态后才允许 commit。如果 commit 失败，继续使用旧版本。

第四，benchmark 数据冗余。

我们同时保留 file logger 和 stdout log。即使部分 metrics 没有进入 file logger，也可以从 stdout 里解析权重同步和 bucket transfer 统计。

第五，profiler 冗余。

NPU profiler 只作为 L2 抽样证据，不依赖 profiler 才能完成 benchmark。即使 profiler 失败，L0 和 L1 指标仍然应该可用。

然后是防呆设计。

第一，backend 防呆。

registry 重复注册时输出 warning，避免 HCCL / NCCL backend 被静默覆盖。

第二，配置防呆。

`sync_policy` 默认使用 `abort_all`。高风险策略，比如 `prefetch_then_commit`、prefix cache version 化、ACL IPC PoC，都必须显式开启。

第三，队列防呆。

`get_samples(max_n)` 要校验 `max_n > 0`。队列关闭且为空时返回 None，保留原终止语义。遇到 None sentinel 时停止本次批量 pop，避免吞掉终止信号后的样本。

第四，版本防呆。

stale threshold 必须可配置，超过阈值的样本不能无条件进入训练。GRPO 多 response per prompt 场景下，要避免同一个 prompt group 内混用过多权重版本。

第五，内存防呆。

cached weights 默认只保留最新一个版本，bucket buffer 必须有生命周期释放。大 tensor 分片不能无限制重组，否则容易导致峰值内存翻倍。

第六，benchmark 防呆。

benchmark runner 支持 dry-run，先打印实际 verl 命令、输出路径和 env。这样用户可以确认参数之后再跑真实训练。summary 对缺失字段应该跳过，而不是因为某个指标缺失直接失败。

第七，profiler 防呆。

默认只 profile 少量 step，避免 profiler 对性能数据产生过大扰动。profile step 也应该避开 warmup step。

---

## 五、测试建议

最后讲测试建议。

测试分六类。

第一类是单元测试。

HCCL registry 测试要验证 HCCL backend 注册为 `"hccl"`，并且不会覆盖 `"nccl"`。

MessageQueue 批量 get 测试要覆盖正常批量返回、timeout 返回空 batch、None sentinel 终止语义、shutdown 后返回 None，以及 `max_n <= 0` 抛错。

Bucketed transfer stats schema 测试要验证 sender 和 receiver 默认 stats 字段完整，避免后续改代码时误删指标 key。

Benchmark parser 测试要构造 fake `metrics.jsonl` 和 `stdout.log`，验证 `timing_s/*` 聚合、`param_sync/*` 解析、`weight_transfer/*` 解析，以及 mean、p50、p95、pct_of_step_mean 计算正确。

Compare 测试要构造 baseline 和 patched summary，验证 speedup、delta 和 effective verdict 正确。

第二类是开发者本地测试。

在没有 Ascend 环境的情况下，至少要跑 Python 语法编译、benchmark parser 测试、MessageQueue CPU 测试、bucket stats schema 测试，以及 `run_ascend_timing_breakdown_bench.sh --dry-run`。

本地测试的目标不是证明 NPU 性能收益，而是证明代码可以运行、parser 正确、命令构造正确、指标字段没有明显问题。

第三类是 Ascend 集成测试。

在真实 Ascend 环境里执行：

```bash
MODEL_PATH=/path/to/model \
TRAIN_FILES=/path/to/train.parquet \
VAL_FILES=/path/to/test.parquet \
OUTPUT_DIR=outputs/ascend_timing_breakdown/baseline \
bash tests/special_npu/run_ascend_timing_breakdown_bench.sh
```

跑完后要检查 `metrics.jsonl`、`stdout.log`、`summary.json`、`timing_breakdown.csv` 和 `npu_profile`。

通过标准是：`summary.json` 里 `step_count > 0`，`timing_breakdown.csv` 包含 `timing_s/step`，stdout 中能看到 checkpoint 或 bucket transfer 统计，配置了 profile step 时 `npu_profile` 有采集文件。

第四类是 A/B 性能测试。

这个是证明优化收益的核心。同一环境、同一模型、同一数据、同一 step 数，分别跑 baseline 和 patched，然后执行 compare。

重点看四类指标：

端到端看 `perf/throughput` 是否提升、`timing_s/step` 是否下降。

MessageQueue 优化看 `ray/message_queue_get_rpc_count` 是否下降。

权重同步优化看 `param_sync/send_recv_update_ms` 和 `timing_s/update_weights` 是否下降。

bucket transfer 优化看 `weight_transfer/sender_copy_ms`、`weight_transfer/receiver_copy_ms`、`metadata_send_ms`、`metadata_recv_ms` 是否下降。

序列化优化看 `serialization/cloudpickle_load_s` 是否下降。

第五类是正确性回归测试。

性能优化不能只看耗时，还要验证训练行为没有被破坏。至少要保证单 step smoke test 能完成，多 step GRPO / PPO 的 loss、reward、KL 没有 NaN 或 Inf，权重同步后 rollout 能继续生成，fully_async 模式下 stale sample 比例符合阈值，启用优化和关闭优化时样本字段完整性一致。

第六类是压力测试。

建议至少覆盖小模型、中模型、长 response、高 rollout_n 和多节点。

小模型比如 Qwen2.5-0.5B，用于快速回归。中模型比如 Qwen2.5-7B 或 Qwen3-8B，用于真实吞吐观察。长 response 用来观察 decode 和 queue 压力。高 rollout_n 用来观察 MessageQueue 和样本序列化压力。多节点用来观察 Ray GCS、HCCL process group 和权重同步稳定性。

---

## 六、总结收口

最后总结一下。

这套方案的核心，不是直接承诺某个高风险优化一定带来多少收益，而是先把 verl Ascend 推理优化做成一个可度量、可回退、可迭代的工程体系。

近期我们优先做四件事：

第一，修正 HCCL backend，保证 Ascend 权重同步链路正确可用。

第二，把权重同步、bucket transfer、MessageQueue 和 cloudpickle 的关键耗时都打出来。

第三，做低风险优化，比如 MessageQueue 批量 get，减少 Ray RPC。

第四，构建 Ascend timing breakdown benchmark，用 baseline / patched 的方式证明优化收益。

中长期再逐步推进 `drain_then_commit`、`prefetch_then_commit`、MessageQueue 分片、SampleRef + TransferQueue，以及 ACL IPC buffer PoC。

这里最重要的工程原则是：默认路径保守，优化路径可配置，收益用 benchmark 证明，风险用回退机制控制。

这样我们既能面向推理性能团队交付有价值的框架层优化，也能保证不会因为过度追求性能而破坏 RL 训练链路的正确性和稳定性。

我的串讲到这里结束。

