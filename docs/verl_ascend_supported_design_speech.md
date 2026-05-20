# verl Ascend 推理优化设计详细讲稿

大家好，我今天汇报的主题是：基于当前 Ascend 支持版本的 verl，做一套面向推理性能优化的框架层设计。

这次汇报我想先强调一个定位：我们不是从算法侧去改 PPO、GRPO 或 reward 计算，也不是直接下沉到 vLLM-Ascend 或 CANN kernel 里做算子优化。我们关注的是 verl 框架层，也就是训练和推理之间的参数同步、rollout 调度、Ray 控制面、样本数据面、IPC 权重传输，以及 KV cache 策略这些位置。

这些位置有两个特点。

第一，它们足够靠近推理性能瓶颈。RL 训练里的 rollout 侧吞吐、参数同步停顿、短请求调度开销，都会直接影响端到端训练效率。

第二，它们又足够工程化。我们可以通过补配置、改 adapter、改 checkpoint engine、改 queue、改 transfer path 来做，不需要等底层推理引擎大改，也不需要承担算法正确性的大风险。

所以今天这份设计的目标，不是罗列一堆理论上可能的优化，而是回答三个问题：

第一，当前 Ascend 支持的 verl 到底是哪一个版本？

第二，这个版本里推理侧真正可优化的链路在哪里？

第三，如果我们团队要做框架层推理性能优化，哪些方向可以落地，哪些方向需要降级，哪些方向只适合做研究验证？

我会按照五个部分来讲。

第一部分，讲版本基线。

第二部分，讲当前代码里的关键链路和瓶颈。

第三部分，逐个分析我们最初提出的四个优化方案。

第四部分，给出我建议团队主打的可落地优化包。

第五部分，给出分阶段路线和验收指标。

---

先讲第一部分，版本基线。

这里最重要的结论是：我们不能直接把 GitHub main 分支当成 Ascend 支持的最新版 verl。

我一开始也拉了 GitHub main，但是后面进一步看 Ascend 支持逻辑的时候，发现真正应该看的入口不是 main 的最新代码，而是 main 里面的 Ascend Docker recipe。

当前 Ascend Dockerfile 里最新的一组配置是面向 CANN 8.5.2、vLLM 0.18.0、vLLM-Ascend 固定 commit、torch 2.9.0 和 torch_npu 2.9.0 的环境。这个 recipe 在安装 verl 的时候，不是直接安装 main，而是明确执行了一个 checkout：

`4045d67063052dcb800c918c107b8d5a87046006`。

所以我们这次设计的代码基线，就是这个 Ascend recipe 实际 pin 住的 verl commit。

这个判断非常关键。因为 main 分支会继续变化，里面可能有更新的抽象、更完善的实现，甚至已经修了一些问题；但这些东西不一定已经被 Ascend 环境验证。我们如果基于 main 做方案，很容易写出“代码上看起来先进，但当前 Ascend 支持链路里不一定能跑”的设计。

反过来，如果基于这个 pin 版本做分析，就能更真实地回答“现在能不能做”“要改哪些文件”“风险在哪里”。

所以本文后面所有代码判断，都是围绕本地拉下来的 `verl-ascend-supported-4045d670` 这个版本。

---

第二部分，讲当前代码里的关键链路。

对于推理性能团队来说，我认为这个版本里最重要的链路有三条。

第一条，是训练到推理的权重同步链路。

第二条，是 rollout server 的请求暂停、恢复和 cache 管理链路。

第三条，是 fully_async 场景下 rollout sample 从 rollouter 到 trainer 的数据传输链路。

先看权重同步链路。

当前核心入口是 `CheckpointEngineManager.update_weights()`。它的流程可以概括为八步。

第一步，判断 backend。如果是 naive，就走 colocated 的简单更新。

第二步，如果不是 naive，就先对所有 rollout replica 调用 `abort_all_requests()`。

第三步，创建临时的 RayWorkerGroup，把所有 rollout workers 包装成 checkpoint engine worker。

第四步，调用 `sleep_replicas()`，释放 rollout 侧的 KV cache 和权重显存。

第五步，build process group，也就是建立 trainer 和 rollout 之间的通信拓扑。

第六步，trainer 侧发送权重，rollout 侧接收权重，并通过 server adapter 更新到推理引擎。

第七步，finalize checkpoint engine。

第八步，wake up rollout replica，然后 resume generation。

这个流程很清楚，但问题也很明显：它是一个大暂停式同步。

也就是说，每次参数更新的时候，rollout 侧要中断请求、释放资源、同步权重、恢复资源，然后再继续生成。对于小模型或者低频同步，这可能还可以接受。但如果模型规模很大，或者 RL 训练里 rollout 和 update 交替很频繁，这个停顿会直接影响端到端吞吐。

这条链路还有一个必须先修的问题：`HCCLCheckpointEngine` 当前注册名是 `"nccl"`。但是 rollout config 里注释写的 backend 包括 `naive、nccl、nixl、hccl`。这意味着如果用户配置 `backend=hccl`，可能找不到 HCCL engine；如果配置 `nccl`，又可能出现 NCCL 和 HCCL 注册冲突。

这个问题本身不复杂，但优先级很高。因为如果 backend 注册都不准，后面所有 HCCL 参数同步优化都可能根本没跑到目标实现。

因此，第一步应该把 `HCCLCheckpointEngine` 的 registry 从 `"nccl"` 修正为 `"hccl"`，再补一个 registry 级别的 smoke test。

接下来是 rollout server 的暂停和恢复链路。

vLLM server 侧有 `abort_all_requests()`，在较新的 vLLM 版本里，它会调用 `pause_generation(wait_for_inflight_requests=False, clear_cache=True)`。这个接口会暂停接收新请求，abort 当前 in-flight 请求，等待请求 drain，并清理 prefix cache 或多模态 cache。

这个行为从一致性角度是安全的，但从性能角度比较粗。因为它会把正在跑的请求直接中断，而且每次权重更新后清 prefix cache，会造成同步后的 cache 冷启动。

第三条链路是 fully_async 的样本传输。

当前 fully_async 里有一个单独的 `MessageQueue` Ray Actor。rollouter 每生成一个 sample，就把整个 `RolloutSample` 做 `ray.cloudpickle.dumps`，然后塞进队列。trainer 端再用一个同步循环，一个一个 `get_sample_sync()` 拉出来，然后 `ray.cloudpickle.loads`。

这个路径的问题是：控制面和数据面混在一起了。Ray Actor 不只是传 metadata，而是在传整个样本对象。样本里又包含 DataProto、tensor dict、non_tensor_batch 和状态信息，这会把 Python 序列化、Ray object store、actor mailbox 全部卷进来。

所以如果 rollout 很短、sample 很多，这条链路非常容易成为瓶颈。

---

第三部分，逐个分析四个原始方案。

先看第一个方案：异步权重同步与流水线预加载。

这个方案的核心想法是，引入双缓冲或者影子参数缓冲区，让推理采样器可以先用上一轮旧权重继续跑，训练器在后台异步广播新权重。等新权重准备好之后，再在合适的边界切过去。

我对这个方案的判断是：方向正确，可落地，而且应该作为主线推进。但实现方式需要调整。

为什么说方向正确？

因为当前最大的问题就是参数同步期间 rollout 大停顿。如果我们能把权重同步拆成“后台预加载”和“短暂停机提交”，就能把原来的长暂停压缩成短暂停。

当前代码里也已经有可复用基础。

第一，`CheckpointEngineWithCache` 这个抽象已经存在。它的注释里明确提到，可以把权重同步到本地缓存，不中断正在进行的 rollout 请求，等请求耗尽后再从缓存里取权重。

第二，fully_async_policy 已经有 `staleness_threshold`，也就是新鲜度控制。它允许一定数量使用旧参数生成的样本存在。这正好可以和影子权重机制结合。

第三，vLLM rollout 的 `update_weights()` 已经是 server adapter 风格，它会先 non-block 调用 server 侧的 `update_weights_from_ipc`，再启动 bucket sender 传权重。这说明当前框架已经具备异步协作的形态。

但为什么又说不能简单按原方案直接做？

因为当前 HCCL 的实现还不是严格异步。`BroadcastOperation` 的注释写的是 async broadcast，但实际代码是在 `__init__` 里直接执行 `_run()`，而 `_run()` 会直接调用 `pyhccl.broadcast()`。同时 sender 在 bucket 切换前会调用 `torch.npu.synchronize()`。

所以如果我们只是加一个 `torch_npu.npu.Stream()`，但底层 `pyhccl.broadcast()` 并不跟随当前 stream，或者内部仍然阻塞，那并不能实现真正的通信计算重叠。

因此，我建议把这个方案拆成两个阶段落地。

第一阶段做 `drain_then_commit`。

也就是参数同步前，不再一上来就 abort 所有请求，而是先暂停接收新请求，然后等待当前 in-flight 请求自然 drain。设置一个超时时间，比如 200 毫秒或按实际场景配置。如果超时还有尾部请求，再只 abort 这部分尾部请求。

这样可以减少无意义的中断，降低 cache 清理和请求重试带来的浪费。

第二阶段做 `prefetch_then_commit`。

也就是新增一个 `HCCLCachedCheckpointEngine`，继承 `CheckpointEngineWithCache`。训练侧把新权重按 bucket 传到 rollout 侧缓存里；rollout 侧不立即加载到推理引擎，而是把它标记为 READY。等到请求边界、或者 staleness 达到阈值时，再执行 commit。

这里可以设计一个 `WeightCacheStore`。它按 `param_version` 管理缓存，状态从 `RECEIVING` 到 `READY`，再到 `COMMITTED`，最后 `RELEASED`。为了控制 HBM，默认只保留一个最新版本，并且尽量按 bucket 或 layer 分段，不做全量双份权重常驻。

这个方案的验收指标也比较明确。

第一，`param_sync/paused_ms` 要下降。

第二，aborted request 数量要下降。

第三，rollout tokens/s 要上升，至少不能因为 cache 管理引入明显回退。

第四，stale sample 比例要可控，不能超过配置阈值。

所以总结一下，3.1 是可做的，但落地路径应该是“先小停顿 commit，再 cache prefetch”，而不是一上来承诺 NPU stream 级完全重叠。

---

第二个方案，是同节点零拷贝权重共享与显存重映射。

原始想法是：训练和推理进程在同一台机器时，不通过 HCCL 复制多份权重，而是通过 NPU 显存 IPC，比如 `aclrtIpcOpenMemHandle`，把训练进程里的物理显存句柄传给推理进程，让推理进程直接挂载共享显存。

这个方案听起来收益很大，但我建议拆成两个层次看。

第一层，是 IPC 加速权重传输。

这一层当前代码已经有基础，而且可落地。

在 vLLM rollout 初始化时，会调用 `is_support_ipc()`。对于 NPU，它会检查 HDK 和 CANN 版本，要求大致是 HDK `>=25.3.rc1`，CANN `>=8.3.RC1`。如果支持 IPC，就走设备 IPC；如果不支持，就 fallback 到 CPU shared memory。

`BucketedWeightSender` 在 IPC 路径里会创建一个 device uint8 buffer，然后通过 `torch.multiprocessing.reductions.reduce_tensor` 生成 handle；receiver 侧通过 `rebuild_ipc` 重建这个 buffer。

所以，“IPC 作为权重传输加速手段”是现实存在的。

第二层，是长期零拷贝共享权重。

这一层当前代码没有现成支持，风险非常高。

原因有几个。

第一，当前 receiver 在 rebuild IPC buffer 后，会对 tensor 做 clone。也就是说，它只是借 IPC buffer 传输，最终推理引擎仍然拥有自己的权重副本。

第二，推理引擎通常假设自己拥有权重 tensor 的生命周期和 allocator 语义。如果我们让它直接引用训练进程的参数显存，就会遇到 ownership 问题。

第三，训练侧 optimizer step 会原地更新参数。推理侧如果同时读取这块显存，就会出现读写一致性问题。除非我们引入版本化只读快照，或者 copy-on-write，否则不能保证推理看到的是一个稳定版本。

第四，vLLM-Ascend 或 SGLang 是否支持外部 storage/view 形式的权重，也需要单独验证。

所以我的建议是：3.2 不要作为第一阶段生产目标来承诺“降低 50% 显存占用”。这个收益在当前代码里没有直接证据。

更稳妥的落地路径是：

第一阶段，增强现有 IPC bucket 传输。包括记录 `ipc_supported`、`use_shm`、bucket copy 时间、receiver clone 时间、fallback 次数，并做大 tensor 分片和 bucket size 自适应。

第二阶段，做 ACL IPC buffer PoC。注意这里 PoC 的对象只是通信 buffer，不是模型权重。我们先验证 `aclrtIpcGetMemHandle`、`aclrtIpcOpenMemHandle`、close、异常退出、跨 device、进程清理、torch_npu allocator 兼容性。

第三阶段，如果 PoC 成功，再评估是否能替代当前 `reduce_tensor/rebuild_ipc`，以及是否有机会减少 receiver clone。

至于长期共享模型权重，我建议作为研究方向保留，但不要进入第一批交付承诺。

---

第三个方案，是分布式调度引擎优化，也就是 Ray 相关优化。

这个方向我认为可落地，而且对 RL 场景很关键。

当前代码已经做对了一部分事情：主干 rollout server、replica、trainer worker group 基本都是长寿命 actor，不是每个样本都创建一个短生命周期 task。

但 fully_async 里还有明显问题。

`MessageQueue` 是一个单 Ray Actor，内部是一个 deque。rollouter 每生成一个 sample，就调用一次 `put_sample`。trainer 端在 `_get_samples_from_queue` 里，用 while 循环不断调用 `get_sample_sync()`，直到收集到 required samples。

这会带来三个瓶颈。

第一，Ray RPC 次数太多。每个 sample 一次 put，一次 get，短样本场景下 RPC 开销很明显。

第二，单 actor mailbox 成为中心瓶颈。所有 rollouter 和 trainer 都围绕一个 MessageQueue Actor 转。

第三，queue 里放的是 cloudpickle 后的整样本 bytes，actor 不只是调度 metadata，还承载了大量数据流。

所以我建议分三步优化。

第一步，做批量 get 和批量 put。

把当前 `get_sample()` 扩成 `get_samples(max_n, timeout_ms)`，trainer 一次拉一批，而不是一个一个拉。rollouter 侧也可以提供 `put_samples()`，尤其是流式场景可以把若干样本合并提交。

第二步，做 MessageQueue 分片。

比如按 rollout replica 数或者节点数创建多个 queue shard。rollouter 按 sample_id hash 到不同 shard，trainer 通过一个简单的 QueueSampler 轮询或者按 queue size 加权拉取。

这样可以避免所有样本都打到一个 Ray Actor 上。

第三步，做背压策略。

当前 queue 满了会丢 oldest sample。对于 RL 训练来说，更合理的是按 param_version 丢最旧版本样本，或者在特定场景下 reject new，避免 stale sample 比例失控。

这个方案的验收指标包括：

Ray RPC 数下降；

MessageQueue Actor CPU 下降；

queue 等待时间下降；

stale sample 比例可控；

rollout 到 trainer 的端到端 sample 延迟下降。

所以 3.3 是很适合框架层团队切入的方向，改动范围相对清楚，风险也可控。

---

第四个方案，是状态字典序列化开销压降。

这里要先澄清一个点：在这个 Ascend 支持基线里，权重同步路径已经不是简单地 pickle 一个完整 state_dict 了。

HCCL checkpoint engine、vLLM bucket IPC、SGLang update_weights，都已经有 bucket 化处理。

所以权重侧的优化重点，不是“从 pickle state_dict 改成 tensor 传输”，因为这件事已经部分完成了。我们要做的是继续打磨 bucket 热路径。

比如：

第一，当前如果单个 tensor 大于 bucket size，会直接 assert。代码里也有 TODO，提到 embedding layer weight 需要切片。所以大 tensor 分片是明确可做的优化。

第二，当前 bucket metadata 使用 `send_pyobj`，还是 Python pickle。可以改成 manifest，也就是把 name、shape、dtype、offset、is_last 这些信息编码成 JSON 或 msgpack。这样 metadata 更稳定，也减少 Python 对象序列化成本。

第三，当前 sender 在 bucket 边界会 synchronize，receiver 在 IPC 路径会 clone。我们应该把这些同步点和 clone 时间打出来，再决定是否能用 event 或 stream 做更细粒度 overlap。

第四，对于非连续 tensor，可以按需 contiguous，而不是无脑复制。每次 contiguous 都应该打指标，因为它本身就是一次额外拷贝。

真正更大的问题，是 rollout sample 的序列化。

fully_async 现在传的是整个 `RolloutSample`。这个对象里包含 DataProto、tensor batch、non_tensor_batch、rollout status 等。cloudpickle 这种复杂对象，会带来 CPU 开销和内存峰值。

所以这里建议引入 `SampleRef`。

`SampleRef` 可以理解成一个轻量引用对象。它里面有 sample_id、param_version、storage_backend、tensor_refs 和 non_tensor_meta。

rollouter 生成样本后，不把大 tensor 直接塞进 Ray queue，而是把 tensor 字段写入 TransferQueue 或其他外部 storage。Ray queue 里只传 `SampleRef`。

trainer 拿到 SampleRef 后，再根据 tensor_refs 去拉取具体字段，组装 batch。

第一阶段不用迁移所有字段，可以先迁移最大的几个 tensor，比如 input_ids、responses、attention_mask、position_ids、old_log_probs。

这样收益会比较直接：

cloudpickle dumps 和 loads 时间下降；

Ray object store 压力下降；

MessageQueue Actor 内存下降；

trainer assemble batch 的路径更清晰。

所以 3.4 的结论是：权重侧继续做 bucket 热路径优化，样本侧做 SampleRef 和 TransferQueue 化，这两个都可落地。

---

接下来讲第四部分，也就是我建议推理性能团队主打的优化包。

我建议把团队方向收敛成五个优化点。

第一个，参数同步的小停顿化，也就是 `drain_then_commit` 和 `prefetch_then_commit`。

这是最核心的一条，因为它直接解决 rollout 在参数同步时被大幅暂停的问题。

第一步先做 `drain_then_commit`，改动相对小。它不需要马上引入影子缓存，只需要把“立即 abort 全部请求”改成“先暂停接收新请求，等待 in-flight 请求 drain，超时后再 abort 尾部请求”。

这个优化可以先保持权重更新逻辑不变，只改变停顿策略。风险比较低。

第二步再做 `prefetch_then_commit`，需要新增 cached checkpoint engine 和 WeightCacheStore。这个阶段收益更大，但也需要处理 HBM 占用和 staleness。

第二个，Bucketed IPC 权重传输增强。

这个方向的优势是非常工程化，和 Ascend 通信、IPC、HBM 都相关。我们可以先不碰长期共享权重，只把现有传输链路做扎实。

具体包括：IPC/SHM fallback 指标、大 tensor 分片、bucket size auto-tune、manifest metadata、clone time 统计、非连续 tensor 统计。

第三个，Rollout queue 批量化和分片。

这个方向解决的是短请求和大量 sample 下 Ray 控制面的压力。

它的优点是改动边界清楚：主要在 `MessageQueue`、`FullyAsyncTrainer._get_samples_from_queue()` 和 `FullyAsyncRollouter._process_single_sample_streaming()` 附近。

第一步加 batch get，第二步加 queue shard，第三步加 param_version-aware 的 drop policy。

第四个，SampleRef 加 TransferQueue。

这个方向的核心是“控制面和数据面解耦”。

Ray 更适合调度和传 metadata，不适合在这个场景里持续传复杂的大样本对象。大 tensor 数据应该交给 TransferQueue 这类数据通道。

这也是比较容易形成团队技术亮点的方向，因为它不只是调一个参数，而是在框架层把数据流重新分层。

第五个，Prefix/KV cache version 化。

这个方向是推理框架味道最重的优化。因为权重更新后是否必须清 prefix cache，直接影响同步后的冷启动损失。

但这也是正确性最敏感的方向。不同权重版本不能错误复用同一份 KV cache。比较稳妥的做法，是把 cache key 和 weight version 绑定。旧权重请求可以命中旧 version cache，新权重请求只能命中新 version cache。

这个方向建议做成实验开关，不作为第一批默认开启功能。

---

最后讲阶段路线。

我建议分四个 sprint 或阶段推进。

Sprint 1，一周内可交付，目标是修正基线和建立可观测性。

具体包括：

第一，修 `HCCLCheckpointEngine` registry，把它从 `"nccl"` 改成 `"hccl"`。

第二，给 `CheckpointEngineManager.update_weights()` 加分段 timer，至少拆出 abort、sleep、build process group、send/update、finalize、wake/resume。

第三，给 IPC/SHM 路径加指标，让我们知道当前到底走设备 IPC 还是 CPU shared memory。

第四，给 cloudpickle dumps 和 loads 加计时。

第五，给 MessageQueue 加 `get_samples(max_n)`，trainer 端先从逐个 get 改成批量 get。

这一阶段的目标是：先知道瓶颈在哪里，并且用很小的改动减少一部分 Ray RPC。

Sprint 2，两到三周可交付，目标是降低同步停顿和短请求调度压力。

具体包括：

第一，MessageQueue 分片。

第二，Bucket 大 tensor 分片，避免 embedding 或 lm_head 超过 bucket 时直接 assert。

第三，Bucket metadata 从 `send_pyobj` 改成 manifest。

第四，实现 `drain_then_commit` 权重同步策略。

第五，把 prefix cache 清理策略做成配置，默认保持保守，后续再验证是否能关闭或 version 化。

这一阶段的目标是：让权重同步暂停时间下降，让 rollout 调度路径更稳，让 bucket 传输不再依赖用户手动调大 bucket。

Sprint 3，四到六周可交付，目标是做真正的异步预加载和数据面解耦。

具体包括：

第一，实现 `HCCLCachedCheckpointEngine`。

第二，实现 `prefetch_then_commit` 两阶段参数同步。

第三，接入 TransferQueue 的 SampleRef，先迁移最大、最重的几个 tensor 字段。

第四，实现 rollout runtime controller，在 verl 上游动态调提交并发，不直接动态改 vLLM engine 的静态参数。

这一阶段的目标是：让参数同步和 rollout 生成开始重叠，同时让样本数据从 Ray cloudpickle 路径里解耦出来。

Sprint 4，是研究线。

这里做三件事：

第一，ACL IPC buffer PoC。

第二，vLLM-Ascend prefix cache version key 验证。

第三，长期共享权重 view 的可行性验证。

这个阶段的目标不是承诺上线，而是形成技术判断。尤其是长期零拷贝共享权重，如果没有版本化快照和推理引擎外部 storage 支持，不建议贸然进入主链路。

---

最后总结一下。

这份设计的核心结论有三个。

第一，当前 Ascend 支持的 verl 基线不是 GitHub main，而是 Ascend Docker recipe pin 住的 `4045d670` commit。我们必须基于这个真实支持版本做设计。

第二，当前最值得优化的推理侧瓶颈，不是单个算子，而是框架层链路：参数同步大停顿、IPC bucket 热路径、Ray queue 单点、整样本 cloudpickle，以及权重更新后的 cache 冷启动。

第三，最适合我们团队落地的方向，是“推理侧异步参数同步 + 数据面解耦”。具体包括 prefetch/commit、bucketed IPC 增强、MessageQueue 批量化和分片、SampleRef 加 TransferQueue，以及 Prefix/KV cache version 化。

如果要给这个方向一个完整的名字，我建议叫：

“面向 Ascend verl 的推理侧异步参数同步与数据面解耦优化”。

它的价值在于：不改算法，不依赖底层 kernel 大改，主要通过 verl 框架层的 checkpoint engine、rollout adapter、Ray queue、TransferQueue 和 cache policy 来拿收益。

最终我们可以用几个指标来验证效果：

参数同步暂停时间是否下降；

rollout tokens/s 是否提升；

Ray RPC 次数和 MessageQueue CPU 是否下降；

cloudpickle dumps/loads 时间是否下降；

Ray object store 和 host memory 峰值是否下降；

以及 HBM 峰值和 stale sample 比例是否可控。

我认为这条路线既能体现推理性能团队的技术深度，也能拆成可交付的工程任务，并且风险边界比较清楚。

我的汇报就到这里。
