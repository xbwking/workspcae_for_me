# verl Ascend 推理优化需求与方案设计串讲稿

各位好，今天我讲的是 verl Ascend 场景下的推理优化需求与方案设计。

这次串讲不是单独介绍某一个 patch，也不是只讲 benchmark。我们要回答的是一个更完整的问题：在 Ascend 上跑 verl 强化学习训练时，推理相关链路为什么会影响整体性能，我们准备做哪些框架层优化，哪些可以近期落地，哪些应该作为后续研究项推进，以及最后怎么证明这些优化确实有效。

我会按一条主线来讲：先讲背景，再讲瓶颈，然后讲方案设计，最后讲可靠性和测试验证。大家可以先记住一个核心判断：这套方案不是直接承诺某个点一定提升多少，而是先把链路变得可观测，再做低风险优化，最后用 Ascend 环境下的 benchmark 做 A/B 验证。

---

## 需求背景

verl 在大模型强化学习训练里，不只是一个普通训练框架。它要同时协调训练、推理采样、奖励计算、权重同步和样本回流。它的关键路径不是单一的 forward、backward 或 optimizer step，而是一整条训练和推理互相耦合的链路。

以 GRPO 或 PPO 为例，训练侧会不断更新 actor 权重；rollout 侧又要使用 actor 生成 response，拿到样本后再回流给 trainer 做 loss 计算和参数更新。这里天然有一个矛盾：训练希望权重尽快更新，推理希望一直持续生成样本，但每次权重同步通常都会让 rollout 侧暂停。

在 Ascend NPU 上，这个问题会更明显，因为链路里还涉及 HCCL、torch_npu、vLLM-Ascend、Ray 调度、IPC 或 shared memory 权重传输等组件。当前 Ascend-supported verl 已经有能跑通的基线，比如 vLLM-Ascend rollout、HCCL checkpoint engine、bucketed IPC / SHM 权重传输、Ray WorkerGroup 和 fully_async policy。

但从推理性能优化角度看，“能跑通”还不够。现在最大的问题是很多关键耗时点没有被拆开。一次 step 慢了，我们可能知道 `timing_s/step` 变大，也可能知道 `timing_s/update_weights` 变大，但很难继续判断到底是 HCCL 传输慢、process group 构建慢、rollout server 更新慢，还是 Ray 样本回流慢。

所以这次方案的出发点是：把 Ascend verl 的推理优化从“凭经验调优”，推进到“可度量、可解释、可回退、可持续迭代”。

---

## 需求价值

这件事的价值主要有几块。

先是提升 rollout 采样的有效吞吐。RL 训练里 rollout 不是辅助环节，它直接影响每个 training step 的节奏。如果权重同步期间推理侧长时间暂停，或者样本回流被 Ray RPC 卡住，NPU 的有效利用率就会下降。

然后是让权重同步可以定位。大模型场景下，一次权重同步里可能包含请求中断、KV cache 清理、通信组构建、权重发送、权重接收、推理引擎更新和恢复生成。以前这些成本混在一个大耗时里，只能猜。补充 `param_sync/*` 后，就能看到每个阶段的占比。

再是降低 Ray 和序列化链路的框架开销。fully_async 场景下，样本持续从 rollouter 流向 trainer。如果 trainer 每次只从 Ray Actor 拉一个 sample，RPC 次数会非常高，再叠加 cloudpickle 序列化和反序列化，CPU 侧也可能成为瓶颈。

最后是形成统一的性能验证方法。后续每个优化 patch 都应该能用 baseline / patched 对比说明收益，而不是只看一次日志截图。benchmark 要能输出 mean、p50、p95、speedup 和 verdict，这样收益才可复现、可解释。

---

## 现有瓶颈

现在主要有四个瓶颈。

先看权重同步。

当前 `CheckpointEngineManager.update_weights()` 是权重同步编排入口。一次 update 会串行做 `abort_all_requests`、sleep、build process group、send / receive / update、finalize、wake 和 `resume_generation`。

这个流程语义安全，但代价是推理侧会暂停，而且所有阶段都被压到一个大的 `update_weights` 里。日志里看到 `timing_s/update_weights` 高，并不能直接判断根因。它可能慢在请求中断，可能慢在 HCCL 传输，也可能慢在 rollout engine 恢复生成。

如果后续想做 `drain_then_commit` 或 `prefetch_then_commit`，也必须先知道当前 `abort_all` 的真实成本分布。否则我们不知道该优化通信、调度、缓存，还是推理引擎更新。

再看 HCCL backend。

代码复核时发现，当前基线里 `HCCLCheckpointEngine` 的注册 key 和配置预期不一致。配置里期望的是 `backend=hccl`，但实现里存在注册成 `"nccl"` 的问题。这个不是性能技巧，但它是 Ascend 权重同步链路的可用性基础。先把 HCCL engine 正确注册到 `"hccl"`，后面 benchmark 和同步策略才站得住。

第三个瓶颈是同节点权重传输缺少统计。

当前 vLLM rollout 侧已有 bucketed weight transfer，可以根据环境使用 IPC 或 shared memory fallback。但我们看不到一次传输实际走 IPC 还是 SHM，也看不到 bucket 数量、总字节数、metadata 时间、sender copy 时间、receiver clone 或 to device 时间。传输慢的时候，就不知道是路径选错了、bucket 设计不合理，还是接收端复制成本太高。

这里也要区分一下：当前已有的是 bucket IPC / SHM 权重传输，不是长期零拷贝权重共享。真正基于 `aclrtIpcOpenMemHandle` 的显存共享，需要处理句柄生命周期、权限、推理引擎权重绑定和一致性问题，适合作为后续 PoC，不适合近期直接承诺生产收益。

最后是 fully_async 样本回流。

fully_async 让 rollout 和 training 尽量解耦，但压力会转移到 MessageQueue 和数据回流上。当前 MessageQueue 是单 Ray Actor，trainer 逐个调用 `get_sample_sync()` 拉样本，样本还要经过 `cloudpickle.dumps()` 和 `cloudpickle.loads()`。在短样本、高并发、多 rollout replica 场景里，RPC 次数和序列化成本会非常明显。

所以我们不仅要看 NPU profiler，也要看 Ray RPC、queue wait 和 cloudpickle 时间。

---

## 需求目标

近期目标要聚焦在能落地、风险低、可验证的范围内。

首先修复 HCCL backend 注册，保证 `backend=hccl` 能正确命中 `HCCLCheckpointEngine`，同时对重复注册给 warning，避免 backend 被静默覆盖。

然后给权重同步补充分段耗时。把 abort、sleep、build process group、send / receive / update、finalize、wake、resume generation 都记录出来。以后同步慢，就能直接定位具体阶段。

同节点权重传输也要补统计。sender 侧记录 bucket 数、总字节数、metadata send、copy、sync；receiver 侧记录 metadata recv、clone 或 to device、sync。这样可以判断 IPC / SHM 传输路径是否符合预期，也能看到接收端是否有额外复制成本。

fully_async 方向先做批量消费。把 trainer 端逐样本 get 改成 `get_samples(max_n)`，减少 Ray RPC 次数，同时记录 queue get RPC count 和 cloudpickle load time。这个优化不改变样本内容，也不改变训练数学语义，适合作为近期低风险优化。

最后补齐 Ascend timing breakdown benchmark。它负责运行、汇总和对比，输出 `summary.json`、`timing_breakdown.csv` 和 `compare.json`。后续每个优化 patch 都可以用它做 baseline / patched 对比。

中长期方向包括 `drain_then_commit`、`prefetch_then_commit`、MessageQueue 分片、SampleRef + TransferQueue、ACL IPC buffer PoC。但这些都涉及权重版本、缓存生命周期、Ray 数据面和 NPU 通信能力，需要真实 Ascend 环境和 profiler 验证，不能在近期直接当成生产能力承诺。

---

## 典型场景

这个方案主要服务几类场景。

最基础的是 Ascend 集群上的常规 GRPO / PPO 训练。用户使用 vLLM-Ascend 做 rollout，训练侧定期把 actor 权重同步到 rollout 侧。这里关注的是 step 是否稳定，rollout 是否被频繁打断，权重同步是否占用过多时间。

这个场景下，端到端看 `timing_s/step`、`timing_s/gen`、`timing_s/update_actor`、`timing_s/update_weights` 和 `perf/throughput`；链路内部看 `param_sync/*`，也就是权重同步各阶段耗时。

另一个场景是推理性能团队验证优化 patch。比如改了 MessageQueue 批量 get，或者后续改了 bucket metadata、权重同步策略。这个时候不能只说理论上会更快，而是要跑同一套 benchmark，拿 baseline 和 patched 做对比。

还有 fully_async 短样本高并发场景。这里瓶颈可能不在 NPU，而在样本回流。批量 get 是近期优化，后续如果单 Actor mailbox 仍然是瓶颈，再考虑 queue shard。

同节点权重传输调优也是重点场景。训练和推理进程在同一台机器上时，我们希望优先利用 IPC 或更低成本的数据路径。但现阶段先把路径和成本打出来，再决定是否做更激进的零拷贝或显存 IPC 方案。

---

## 约束和边界

这里有几个边界必须讲清楚。

Ascend 支持版本要以实际 recipe 固定 commit 为准。这次基线是 Ascend-supported 的 `4045d670`，不能直接把 GitHub main 当成 Ascend 支持版本，因为 main 上的 vLLM 适配、checkpoint engine 和 fully_async 实现可能不一样。

NPU stream 完全重叠也不能直接承诺。当前 HCCL engine 里仍然有 ZMQ metadata、`pyhccl.broadcast()` 和 `torch.npu.synchronize()`。通信和计算能不能真正重叠，需要 pyhccl 能力和 profiler 证明。

同节点长期零拷贝权重共享也不适合作为近期生产承诺。当前是 bucket IPC / SHM 权重传输，不是推理进程直接长期挂载训练权重显存。

TransferQueue 可以作为后续数据面优化方向，但当前 commit 里没有完整 in-tree 实现，所以近期只能作为设计预留。

Prefix / KV cache 也要保守。权重更新后继续复用旧 KV cache，可能导致跨权重版本污染。所以默认先清 cache，后续只有在版本化机制明确后再考虑保留。

最后，真实性能收益必须在 Ascend 环境验证。本地 CPU 测试只能证明 parser、CLI、dry-run 和单元逻辑没问题，不能证明 NPU 端到端收益。

---

## 整体方案

整体方案的原则是：可观测性先行，低风险优化优先，异步化分阶段推进。

先看基线修正。这里主要修 HCCL registry，让 `backend=hccl` 能正确实例化 `HCCLCheckpointEngine`，同时增加重复注册 warning。

再看指标采集。权重同步产生 `param_sync/*`，bucket transfer 产生 `weight_transfer/*`，fully_async queue 产生 RPC 和等待相关指标，cloudpickle 产生序列化指标，训练主循环继续保留 `timing_s/*` 和 `perf/*`。这些指标不是越多越好，而是要能回答定位问题。

近期低风险优化先做 MessageQueue 批量 get。它不改变样本内容，也不改变 trainer 训练逻辑，只是把多次单样本 RPC 合并成一次批量 RPC。在短样本、高并发场景下，这个收益路径很清楚：减少 Ray Actor 调用次数，降低 mailbox 和 GCS 压力。

权重同步策略后续再演进。默认保留 `abort_all`，因为它语义最安全。之后可以新增 `sync_policy`，逐步支持 `drain_then_commit` 和 `prefetch_then_commit`。

`drain_then_commit` 的思路是先暂停接收新请求，让正在执行的请求自然完成，超时后只 abort 尾部请求。`prefetch_then_commit` 则是先把新权重加载到 rollout 侧影子缓存里，当前请求继续用旧权重跑，等请求边界或 stale 阈值满足后，再做一次短 commit。

这个方向收益更大，但需要权重版本状态机，比如 `RECEIVING`、`READY`、`COMMITTED`、`FAILED`；样本也要携带 `param_version`，trainer 要能识别 stale sample。所以它应该分阶段推进。

最后是 benchmark 验证层。Ascend timing breakdown benchmark 负责三件事：运行训练、汇总指标、对比结果。运行阶段固定模型、数据、step 和配置；汇总阶段生成 `summary.json` 和 `timing_breakdown.csv`；对比阶段生成 `compare.json`，输出 speedup、delta 和 verdict。

从数据流看，Trainer 更新 actor 后，通过 `CheckpointEngineManager` 进入 HCCL、IPC 或 SHM 权重同步；Rollout Server 接收权重并更新推理引擎；rollout engine 继续生成样本；样本通过 MessageQueue 或后续 TransferQueue 回流给 Trainer；benchmark parser 最后把 file logger 和 stdout 里的指标汇总成可对比结果。

---

## 关键模块设计

关键模块可以简单拆成四块。

HCCL backend 修复负责保证 Ascend 配置能正确命中 HCCL。验收标准是 `backend=hccl` 能拿到 HCCL engine，`backend=nccl` 不受影响，重复注册有 warning。

权重同步分段计时负责拆开 `update_weights()`。abort 对应请求中断，sleep 对应等待 rollout engine 进入安全状态，build process group 对应通信组构建，send / receive / update 对应权重传输和推理引擎更新，wake 和 resume 对应恢复生成。指标写入 `last_update_weights_timing`，再进入训练 metrics。

Bucketed weight transfer stats 负责记录 sender 和 receiver 两端的传输成本。sender 看路径、bucket、bytes、metadata send、copy、sync；receiver 看 metadata recv、clone 或 to device、sync。两端都要看，因为瓶颈可能在任意一侧。

MessageQueue 批量 get 负责减少 trainer 侧 Ray RPC。新的 `get_samples(max_n)` 一次最多拉一批，队列为空时等待 timeout，队列关闭时返回 None，遇到 None sentinel 时保留终止语义。它必须兼容原来的样本顺序和结束语义。

benchmark 则提供 `run`、`summarize` 和 `compare`。`run` 落盘 stdout、stderr、metrics 和环境信息；`summarize` 生成 summary 和 csv；`compare` 比较 baseline 和 patched，给出关键指标变化。

---

## 影响分析

近期改动对训练正确性的风险比较低。

HCCL registry 修复只是让 backend 正确命中。指标打点只增加观测，不改变原流程。MessageQueue 批量 get 只是把多次 get 合并成一次，不改变样本内容和训练计算。

真正需要谨慎的是后续异步权重同步。一旦从 `abort_all` 走向 `drain_then_commit` 或 `prefetch_then_commit`，系统里就会出现权重版本边界。rollout 样本最好携带 `param_version`，trainer 消费时要判断 stale 程度。尤其是 GRPO 这种一个 prompt 对应多个 response 的场景，版本一致性会更敏感。

从推理服务看，默认保守路径必须保留，新策略必须通过配置显式启用，出现异常时能回退到 `abort_all`。

从资源消耗看，普通指标打点成本很低，可以默认开启；NPU profiler 会影响性能，只适合少量 step 抽样；prefetch 和 cached weights 会增加 HBM 或 CPU pinned memory 占用，所以缓存版本数量要有限制。

从运维定位看，这套方案会明显缩短排查路径。以前只能看到 step 慢，现在可以判断问题是在 Ray / MessageQueue、cloudpickle、HCCL / IPC、rollout server update、generation，还是 actor update。

---

## 可靠可用设计

可靠性主要围绕回退、版本、防呆和观测冗余。

同步策略上，默认保留 `abort_all`。`drain_then_commit` 和 `prefetch_then_commit` 都作为可配置策略启用，新策略异常时直接回退。

通信路径上，IPC 优化不能移除 shared memory fallback。Ascend IPC 环境不满足时自动退回 SHM，并把实际传输路径写到指标里。

权重版本上，prefetch 不能覆盖当前已 commit 权重。新版本必须完整接收并进入 READY 状态后才允许 commit；如果接收、校验或 commit 失败，推理侧继续使用旧版本。

队列接口上，`get_samples(max_n)` 要校验 `max_n > 0`，队列关闭且为空时返回 None，遇到 None sentinel 时保留终止语义。

内存管理上，cached weights 和 bucket buffer 都要有明确生命周期。默认可以限制 `max_cached_versions=1`，避免为了异步化导致峰值内存不可控。

benchmark 上保留 dry-run，用户可以先确认真实命令、输出目录和环境变量。summary 对可选指标缺失要容错，不应该因为某个字段缺失直接失败。

观测上同时保留 file logger 和 stdout log。即使某些指标短期内没有进入统一 metrics，也可以从 stdout 解析权重同步和 bucket transfer 统计。

---

## 测试建议

测试分成四层。

单元测试覆盖局部逻辑。HCCL registry 测试验证 `"hccl"` 注册和重复注册 warning；MessageQueue 测试覆盖批量返回、timeout、None sentinel、shutdown 和非法 `max_n`；bucket stats 测试验证 sender / receiver schema；benchmark parser 测试验证 `timing_s/*`、`param_sync/*`、`weight_transfer/*` 的解析和 p50、p95 计算。

开发者本地测试主要证明代码路径可跑。在没有 Ascend 环境的机器上，可以跑 CPU 单元测试、benchmark parser 测试、MessageQueue 测试、bucket stats 测试，以及一键脚本 dry-run。这里不能证明 NPU 性能收益，只能证明命令、parser 和新增接口没有明显问题。

Ascend 集成测试要在真实环境跑。典型命令是：

```bash
MODEL_PATH=/path/to/model \
TRAIN_FILES=/path/to/train.parquet \
VAL_FILES=/path/to/test.parquet \
OUTPUT_DIR=outputs/ascend_timing_breakdown/baseline \
bash tests/special_npu/run_ascend_timing_breakdown_bench.sh
```

跑完后检查 `metrics.jsonl`、`stdout.log`、`summary.json`、`timing_breakdown.csv` 和可选的 `npu_profile`。基本通过标准是 summary 里有 step，csv 里有 `timing_s/step`，stdout 能看到 checkpoint 或 bucket transfer 统计。

A/B 性能测试是证明收益的关键。同一模型、同一数据、同一参数、同样 step 数，分别跑 baseline 和 patched。端到端看 `perf/throughput` 和 `timing_s/step`；MessageQueue 优化看 RPC count；权重同步优化看 `param_sync/send_recv_update_ms` 和 `timing_s/update_weights`；bucket transfer 优化看 sender / receiver copy 和 metadata 时间；序列化优化看 cloudpickle load 时间。

除了性能，还要做正确性回归。至少确认单 step smoke test 能完成，多 step GRPO / PPO 的 loss、reward、KL 没有 NaN 或 Inf，权重同步后 rollout 能继续生成，fully_async 模式下 stale sample 比例符合阈值。

压力测试建议覆盖小模型、中模型、长 response、高 rollout_n 和多节点。小模型用于快速回归，中模型用于真实吞吐观察，长 response 看 decode 和 queue 压力，高 rollout_n 看 MessageQueue 和序列化压力，多节点看 Ray GCS、HCCL process group 和权重同步稳定性。

---

## 总结

最后总结一下。

这套方案不是把所有高风险想法一次性塞进生产链路，而是先建立一个可观测、可回退、可验证的优化闭环。

近期最值得做的是四件事：修 HCCL backend，补权重同步和 bucket transfer 指标，做 MessageQueue 批量 get，再用 Ascend timing breakdown benchmark 做 baseline / patched 对比。

这些事情风险可控，而且能直接提升可观测性。即使某些优化没有立刻带来明显吞吐提升，也能告诉我们真正瓶颈在哪里。

中长期再推进更激进的方向，包括 `drain_then_commit`、`prefetch_then_commit`、MessageQueue 分片、SampleRef / TransferQueue，以及 ACL IPC buffer PoC。但这些方向必须建立在权重版本、缓存生命周期、失败回退和 profiler 验证都清楚的基础上。

所以落地原则很明确：默认路径保守，优化路径可配置，收益用 benchmark 证明，风险用回退机制控制。

我的串讲到这里结束。
