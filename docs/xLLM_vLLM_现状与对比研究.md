# xLLM 现状与 vLLM 对比研究

> 调研日期：2026-06-12  
> 主题：xLLM 的当前状态、它与 vLLM 的关系、主要差异、技术特点和适用场景。

## 1. 结论摘要

xLLM 和 vLLM 都属于大模型推理与服务框架，核心目标都是提升大模型在线推理的吞吐、延迟、资源利用率和部署效率。但二者的出发点不同：

- vLLM 是当前更通用、更成熟、社区生态更大的 LLM 推理框架，适合绝大多数基于 NVIDIA GPU、AMD GPU、CPU 以及云原生生态的通用推理服务场景。
- xLLM 是京东开源的企业级大模型推理框架，更强调国产/异构 AI 加速器适配、服务层与引擎层解耦、在线离线混合调度、PD/EPD 分离、全局 KV Cache 和企业级集群调度。

如果只是要快速部署通用 LLM 服务，vLLM 通常仍是默认首选。如果目标环境涉及国产 AI 芯片、异构加速器、大规模企业内部推理集群、在线离线混部、多模态 EPD 分离或复杂 KV Cache 调度，xLLM 值得重点评估。

## 2. xLLM 是什么

xLLM 是京东开源的大模型推理框架，官方定位是一个高效、易用的 LLM 智能推理框架，为国产 AI 加速器上的模型推理提供企业级服务保障和高性能引擎计算能力。

官方文档将 xLLM 的背景归纳为几个核心矛盾：

- 大模型正在被部署到智能客服、实时推荐、内容生成等核心业务场景；
- 企业需要低成本、高性能、可规模化的推理部署；
- 国产专用加速器的架构特性和通用推理框架之间存在适配难度；
- MoE 模型、长序列、KV Cache 管理、通信瓶颈、负载不均衡等问题限制了推理系统效率。

从项目形态看，xLLM 主要由两层组成：

- `xLLM-Engine`：推理执行引擎，负责模型加载、调度、图执行、算子优化、KV Cache、推理加速等。
- `xLLM-Service`：服务与集群调度层，负责请求接入、实例管理、全局调度、PD/EPD 分离、故障恢复、在线离线混部等。

这种服务层与引擎层解耦的设计，是 xLLM 和很多单体式推理 server 的重要区别。

## 3. xLLM 当前状态

截至 2026-06-12，公开信息显示：

- 主仓库是 [`jd-opensource/xllm`](https://github.com/jd-opensource/xllm)。
- 服务层仓库是 [`jd-opensource/xllm-service`](https://github.com/jd-opensource/xllm-service)。
- xLLM technical report 已发布在 arXiv，首次提交时间为 2025-10-16，v2 修订时间为 2026-03-03。
- GitHub release 页面显示，`xllm` 在 2026-04-14 发布过 `v0.9.0` 和 `v0.9.1`，其中 `v0.9.0` 标为 Latest。
- `xllm-service` 在 2026-04-15 发布 `v0.9.0`，标为 Latest。

从 release 内容看，xLLM 仍处于快速演进阶段，近期重点包括：

- 增加 GLM、Qwen、DeepSeek、Kimi、MiniMax、OneRec、LongCat-Image 等模型支持；
- 适配 NPU、CUDA、MLU、ILU 等设备；
- 支持或增强图模式、context parallelism、prefix cache、chunked prefill、KV cache 远端传输、batch offload；
- 增强动态多模型 serving、Anthropic Messages API、embedding interface、worker health check 和统一请求统计；
- 修复 DeepSeek、Qwen、GLM、多模态、PD disaggregation、graph mode、KV cache 等稳定性问题。

这说明 xLLM 已不是单纯的研究原型，而是在向生产级系统演进；但相比 vLLM，其社区规模、生态成熟度和第三方集成丰富度仍明显更小。

## 4. vLLM 是什么

vLLM 是一个高吞吐、易用的大模型推理与服务框架，最初因 PagedAttention 技术而受到广泛关注。它的核心目标是让 LLM serving 更快、更便宜、更容易部署。

vLLM 的典型能力包括：

- PagedAttention，用于高效管理 attention KV memory；
- continuous batching，持续合批在线请求，提高吞吐；
- OpenAI-compatible API server；
- offline batch inference；
- streaming outputs；
- tensor/pipeline/data/expert/context parallelism；
- prefix caching；
- chunked prefill；
- speculative decoding；
- 多种量化方案；
- 多 LoRA 支持；
- 结构化输出、tool calling、reasoning parser 等服务能力；
- 支持大量 Hugging Face 模型架构和多硬件后端。

vLLM 当前生态非常成熟，文档、示例、社区讨论、云原生部署、Ray/KServe/LangChain/LlamaIndex 等集成更丰富，因此它通常是通用 LLM serving 的基准选型。

## 5. xLLM 和 vLLM 的关系

xLLM 和 vLLM 不是简单的上下游关系。更准确地说，它们是同一问题域中的两个推理系统：

- 二者都面向 LLM/VLM 推理；
- 二者都关注 KV Cache、batching、prefill/decode、长序列、MoE、speculative decoding、图执行、算子优化等推理性能问题；
- 二者都提供在线服务能力；
- 二者都在向多模态、多模型、多硬件和分布式服务方向演进。

但 xLLM 不是 vLLM 的普通插件，也不是 vLLM 的简单 fork。xLLM 的设计更偏向企业级集群服务体系，尤其是将 service layer 和 engine layer 分开，围绕企业实际 workload 做全局调度、故障恢复、在线离线混合调度、PD/EPD 分离和全局 KV Cache。

在 Ascend 场景中，vLLM 主要通过 [`vllm-ascend`](https://github.com/vllm-project/vllm-ascend) 等硬件插件支持华为 Ascend；xLLM 则把国产 AI 加速器适配作为自身核心目标之一。因此，如果讨论国产加速器，xLLM 和 vLLM-Ascend 更像是可对比的方案。

## 6. 核心架构对比

| 维度 | xLLM | vLLM |
| --- | --- | --- |
| 基本定位 | 企业级 LLM/VLM 推理框架，强调国产/异构加速器和集群调度 | 通用高性能 LLM 推理与服务框架 |
| 架构形态 | `xLLM-Service` + `xLLM-Engine` 解耦 | 以 vLLM engine/server 为核心，配合外部调度、云原生或 serving 组件 |
| 服务层能力 | 内置集群调度、在线离线混部、PD/EPD 调度、故障恢复、KV-aware routing | 内置 OpenAI-compatible server，生产级集群调度通常结合外部系统 |
| 引擎层能力 | 图模式、xTensor 内存管理、PageAttention、MoE 优化、speculative decoding、国产加速器算子适配 | PagedAttention、continuous batching、CUDA/HIP graph、FlashAttention/FlashInfer、speculative decoding、chunked prefill |
| 硬件重点 | NPU、MLU、ILU、MUSA、DCU、CUDA 等，特别强调国产 AI 加速器 | NVIDIA、AMD、CPU，以及 TPU、Gaudi、Ascend、Apple Silicon 等插件生态 |
| 生态成熟度 | 较新，京东主导，适合特定生产场景深度评估 | 成熟度高，社区活跃，模型与框架集成广 |
| 默认适用场景 | 企业私有大规模推理、国产芯片、异构集群、多模态复杂 workload | 通用 LLM serving、快速落地、标准 OpenAI API、Hugging Face 模型生态 |

## 7. xLLM 的主要技术特点

### 7.1 服务-引擎解耦

xLLM 把服务调度和模型执行拆开：

- `xLLM-Service` 处理请求接入、路由、实例状态、故障恢复、PD/EPD 资源调度；
- `xLLM-Engine` 处理实际推理，包括模型图执行、算子、内存、KV Cache、并行策略等。

这种设计的价值在于，企业生产环境中的瓶颈不只在单次模型执行，还在多实例、多节点、多 workload 的全局调度。服务层独立出来后，可以更方便地实现跨实例调度、故障转移和资源动态重分配。

### 7.2 在线离线统一调度

xLLM-Service 支持在线请求和离线请求的统一调度：

- 在线请求通常有严格 SLA，需要优先执行；
- 离线请求可以 best-effort 执行，用于填充空闲算力；
- 当在线压力升高时，系统可以抢占或调整离线任务资源。

这对企业集群很重要，因为推理资源成本高，如果只为在线峰值预留机器，低峰期利用率会很低。在线离线混部可以提高整体资源利用率。

### 7.3 动态 PD 分离

LLM 推理通常可以拆成两个阶段：

- Prefill：处理 prompt，计算初始 KV Cache，通常计算密集；
- Decode：逐 token 生成，通常访存和调度压力更明显。

PD disaggregation 的思想是把 Prefill 和 Decode 分配到不同实例或资源池中，以便分别优化。

xLLM 的重点是 workload-adaptive dynamic PD disaggregation，即根据负载动态调整 Prefill/Decode 资源比例，而不是静态固定。例如：

- 长 prompt 多、首 token 压力大时，需要更多 Prefill 能力；
- 输出长、并发 decode 多时，需要更多 Decode 能力；
- workload 变化时，实例角色可以调整。

### 7.4 EPD 三阶段分离

对于多模态请求，仅有 Prefill/Decode 还不够。多模态模型通常先要处理图像、视频、音频等输入，形成 embedding 或视觉 token。

xLLM-Service 提到 EPD three-stage disaggregation：

- Encode：多模态编码阶段；
- Prefill：文本/多模态上下文预填充阶段；
- Decode：自回归生成阶段。

这种拆分可以让不同阶段使用不同资源池，并根据多模态请求比例动态调度资源。对于 VLM 服务，这比只做 PD 分离更细。

### 7.5 全局多级 KV Cache

KV Cache 是 LLM serving 的核心资源。xLLM 强调 global multi-level KV cache management，包括：

- 分层缓存；
- KV offloading；
- KV prefetching；
- KV cache-centric distributed storage；
- KV-aware routing；
- 跨计算节点的 KV 路由；
- 远端主机与本地设备之间的 KV cache 传输。

这类设计适用于多轮对话、长上下文、prefix 复用、多实例服务和集群级请求路由。如果请求可以被路由到已有相关 KV Cache 的实例，系统可以减少重复计算。

### 7.6 图模式和动态 shape 优化

xLLM 文档强调 graph optimization for dynamic shapes，主要包括：

- 基于参数化和多图缓存适配动态 shape；
- 通过受控 tensor memory pool 提高地址安全性和复用；
- 集成 PageAttention、AllReduce 等性能关键算子；
- 通过 full graph pipeline execution orchestration 减少计算气泡。

大模型推理的输入长度、batch size、生成长度动态变化明显。图模式可以降低调度开销，但动态 shape 会让静态图复用变复杂。xLLM 的方向是尽量在保持图执行收益的同时适配动态 workload。

### 7.7 MoE 优化和 EPLB

MoE 模型的核心问题是 expert 负载不均衡和通信开销。xLLM 在 MoE 上强调：

- GroupMatmul 优化；
- Chunked Prefill 支持长序列；
- dynamic EPLB，也就是动态 expert parallel load balancing；
- MoE expert 分布的动态调整。

这对 DeepSeek、Qwen-MoE、Mixtral 类模型比较关键。MoE 模型推理成本低于同规模 dense 模型，但系统实现复杂度更高，尤其需要处理 expert routing、all-to-all 通信和负载倾斜。

### 7.8 算法驱动加速

xLLM 还集成 speculative decoding、MTP speculative inference 等推理加速方式。其目的和 vLLM 中的 speculative decoding 类似：通过草稿模型、额外预测头或并行候选 token 减少主模型解码步数，从而提高吞吐或降低延迟。

## 8. vLLM 的主要技术特点

### 8.1 PagedAttention

vLLM 最核心的技术之一是 PagedAttention。它借鉴操作系统虚拟内存分页思想，将 KV Cache 划分成 block，减少内存碎片，并提升 KV Cache 管理效率。

在传统 serving 中，不同请求的 prompt 和输出长度不同，KV Cache 很容易造成显存浪费。PagedAttention 通过分页式管理，让系统能更细粒度地分配和复用 KV Cache。

### 8.2 Continuous Batching

vLLM 支持 continuous batching，即不断把新请求加入正在执行的 batch，并在请求完成后释放位置。这比静态 batch 更适合在线 serving，因为线上请求是连续到达的，输入输出长度也不一致。

continuous batching 的收益是：

- 提高 GPU 利用率；
- 减少空等；
- 支持更多并发；
- 在一定延迟约束下提升吞吐。

### 8.3 OpenAI-Compatible API

vLLM 提供 OpenAI-compatible API server，这是它被广泛采用的重要原因。应用层可以用类似 OpenAI SDK 的方式调用本地或私有部署模型，迁移成本低。

随着版本演进，vLLM 还支持 Anthropic Messages API、gRPC、tool calling、reasoning parser、structured outputs 等上层能力。

### 8.4 模型和生态支持

vLLM 支持大量 Hugging Face 模型架构，包括：

- decoder-only LLM，如 Llama、Qwen、Gemma；
- MoE LLM，如 Mixtral、DeepSeek、Qwen-MoE；
- hybrid attention 和 state-space 模型；
- 多模态模型，如 LLaVA、Qwen-VL、Pixtral；
- embedding、retrieval、reward、classification 模型。

这类生态优势使 vLLM 更适合作为通用推理底座。

## 9. 关键差异分析

### 9.1 设计重心不同

vLLM 的设计重心是通用 LLM inference engine 和 serving API。它尽量让用户快速把 Hugging Face 模型部署成高性能服务。

xLLM 的设计重心更偏企业级系统工程。它关注的不只是单个推理引擎快不快，还包括：

- 多节点多实例如何调度；
- 在线离线 workload 如何混部；
- 多模态不同阶段如何拆分；
- KV Cache 如何跨实例复用和路由；
- 国产 AI 加速器如何深度适配；
- 故障后请求如何恢复。

### 9.2 硬件适配思路不同

vLLM 的主流使用场景仍以 NVIDIA GPU 为中心，虽然它也支持 AMD、CPU，以及通过插件支持 TPU、Gaudi、Ascend 等硬件。

xLLM 从文档和论文看，国产 AI 加速器是核心方向之一。官方文档列出的硬件平台包括 NPU、MLU、ILU、MUSA、DCU 等，也有 NVIDIA GPU 支持。

因此，在国产芯片环境中，xLLM 可能更贴近目标硬件；在通用 GPU 云环境中，vLLM 的成熟度通常更高。

### 9.3 服务层能力不同

vLLM 自带 API server，但更复杂的生产调度通常依赖外部系统，例如 Kubernetes、KServe、Ray Serve、AIBrix、Dynamo 或企业自研网关。

xLLM-Service 则直接把部分企业级调度能力放进项目中，包括：

- PD 分离；
- EPD 分离；
- KV-aware routing；
- KV Cache Pool；
- instance lease/failover；
- 多 xLLM-Service 与 xLLM instance 的连接关系；
- 请求取消、实例状态、监控指标等。

这意味着 xLLM 更像“推理引擎 + 服务调度系统”，而 vLLM 更像“强推理引擎 + 标准 API server + 生态集成”。

### 9.4 生态和风险不同

vLLM 的优势：

- 社区大；
- 文档丰富；
- 模型支持广；
- third-party integration 多；
- 问题更容易搜索；
- 生产案例和经验更多。

xLLM 的优势：

- 对特定国产硬件和企业 workload 更有针对性；
- 内建复杂服务调度能力；
- 对 PD/EPD、全局 KV Cache、在线离线混部等问题有系统设计；
- 京东业务场景落地经验强。

xLLM 的风险：

- 社区和生态仍较小；
- 与 vLLM 相比，第三方教程、issue 经验、工具链集成较少；
- 生产接入需要更强系统理解和压测验证；
- 论文性能结果主要来自特定模型、硬件和 workload，不应直接外推到所有场景。

## 10. 性能结论如何理解

xLLM technical report 中提到，在相同 TPOT 约束下：

- Qwen 系列模型上，xLLM 吞吐最高可达到 MindIE 的 1.7 倍、vLLM-Ascend 的 2.2 倍；
- DeepSeek 系列模型上，xLLM 平均吞吐达到 MindIE 的 1.7 倍。

这些数据有参考价值，但需要谨慎理解：

- 对比对象主要是 MindIE 和 vLLM-Ascend，不是所有 vLLM 后端；
- 性能结果依赖硬件、模型、输入输出长度分布、并发数、batch 策略、SLA 指标和量化配置；
- 企业生产环境中，p50/p95/p99 latency、TTFT、TPOT、吞吐、成本、稳定性、故障恢复都需要一起看；
- 真正选型应以本地 benchmark 为准。

建议测试维度包括：

- TTFT：首 token 延迟；
- TPOT：每输出 token 延迟；
- QPS/RPS；
- tokens/s；
- p50/p95/p99 latency；
- 显存/内存占用；
- KV Cache 命中率；
- 长上下文稳定性；
- 多轮对话复用收益；
- 模型加载和扩缩容耗时；
- 实例故障后的恢复行为；
- 在线离线混部时的 SLA 影响。

## 11. xLLM 适合用在哪里

xLLM 更适合以下场景：

1. 国产 AI 加速器集群

   如果基础设施主要是 Ascend NPU、Cambricon MLU、Hygon DCU、Iluvatar ILU、Moore Threads MUSA 等硬件，xLLM 的定位和优化方向更契合。

2. 大规模企业内部推理服务

   如果企业有大量在线推理请求，并且需要服务治理、故障恢复、全局调度、资源池管理，xLLM-Service 的设计更有吸引力。

3. 在线离线混合推理

   如果同一套集群既要承载在线客服、推荐、问答，又要运行离线批处理、数据生成、评测任务，xLLM 的在线离线统一调度值得评估。

4. 多模态服务

   如果业务中 VLM 请求占比较高，EPD 三阶段分离可以更细粒度地调度 Encode、Prefill、Decode 资源。

5. 长上下文和多轮对话

   全局 KV Cache、KV-aware routing、prefix cache、KV offload/prefetch 等能力对多轮对话和长上下文场景有潜在收益。

6. MoE 大模型服务

   对 DeepSeek、Qwen-MoE、Mixtral 类 MoE 模型，xLLM 的 MoE kernel、EPLB、通信优化方向较有价值。

## 12. vLLM 适合用在哪里

vLLM 更适合以下场景：

1. 快速搭建 OpenAI-compatible 私有模型服务

   vLLM 的 API server 成熟，应用迁移成本低。

2. Hugging Face 模型生态

   如果模型来自 Hugging Face，并且需要尽快验证或上线，vLLM 通常支持更广、资料更多。

3. NVIDIA GPU 云环境

   在 A100、H100、H200、L40S 等常见 GPU 环境中，vLLM 的使用经验和优化路径更成熟。

4. 与现有框架集成

   如果需要结合 Ray、KServe、LangChain、LlamaIndex、Kubernetes、observability 工具，vLLM 的生态会更方便。

5. 研发验证和模型实验

   vLLM 安装、调试和社区资料更丰富，更适合快速实验。

## 13. 选型建议

### 13.1 默认建议

如果没有明确的国产芯片或复杂企业调度需求，优先从 vLLM 开始。原因是：

- 成熟度更高；
- 社区更大；
- 模型支持更广；
- 文档和案例更多；
- API 兼容和生态集成更完善；
- 出问题时更容易定位。

### 13.2 什么时候优先评估 xLLM

出现以下条件时，应认真评估 xLLM：

- 目标部署环境是国产 AI 加速器；
- 需要企业级多实例调度而不只是单模型 API server；
- 在线和离线任务需要混部；
- 多模态请求比例高，需要 EPD 分离；
- 长上下文、多轮对话、KV Cache 复用收益明显；
- MoE 模型是核心 workload；
- 希望在服务层做 KV-aware routing、failover、PD 动态调度。

### 13.3 建议的 PoC 路线

如果要在企业内部评估 xLLM 和 vLLM，建议按以下路线做 PoC：

1. 固定模型

   选择 1-2 个真实业务模型，例如 Qwen、DeepSeek、GLM、Qwen-VL 或内部微调模型。

2. 固定硬件

   在目标硬件上测试，不要用开发机结果替代生产集群结果。

3. 使用真实流量分布

   构造真实 prompt 长度、输出长度、并发模式、多轮比例、多模态比例。

4. 同时测试延迟和吞吐

   不只看 tokens/s，也要看 TTFT、TPOT、p95/p99、错误率和超时率。

5. 测试稳定性

   包括长时间压测、实例重启、请求取消、网络抖动、KV Cache 压力、显存压力。

6. 测试运维复杂度

   包括部署脚本、镜像、日志、监控、故障定位、版本升级和回滚。

## 14. 与其他相关系统的关系

除 xLLM 和 vLLM 外，大模型推理生态中还有一些相关系统：

- TensorRT-LLM：NVIDIA 生态中高性能推理方案，深度绑定 NVIDIA GPU 和 TensorRT。
- SGLang：强调前端语言、RadixAttention、复杂 LLM program serving 和 agentic workload。
- TGI：Hugging Face Text Generation Inference，适合 Hugging Face 生态部署。
- MindIE：华为昇腾生态中的推理服务方案。
- vLLM-Ascend：vLLM 对 Ascend 的硬件插件路径。
- LMDeploy、LightLLM、DeepSpeed-MII 等：也覆盖不同推理部署场景。

xLLM 的差异化点主要不在“又一个单机推理引擎”，而在“企业级服务调度 + 国产/异构加速器优化 + 全局 KV Cache + PD/EPD 分离”。

## 15. 常见误区

### 15.1 误区：xLLM 是 vLLM 的替代品

不完全准确。xLLM 和 vLLM 有重叠，但目标重心不同。vLLM 更适合通用生态，xLLM 更适合特定企业级和国产硬件场景。

### 15.2 误区：xLLM 一定比 vLLM 快

不准确。性能依赖硬件、模型、batch、prompt 长度、输出长度、并发和优化配置。xLLM 论文中的优势主要是在特定评测条件下相对 MindIE 和 vLLM-Ascend 的结果。

### 15.3 误区：vLLM 不支持国产硬件

不准确。vLLM 可以通过插件支持 Ascend 等硬件。但插件成熟度、性能和功能覆盖需要结合具体版本验证。

### 15.4 误区：只要支持 OpenAI API 就是生产可用

不准确。生产系统还需要关注限流、鉴权、监控、熔断、重试、实例健康、故障恢复、扩缩容、日志、成本和 SLA。

## 16. 简明对比表

| 问题 | 更倾向 xLLM | 更倾向 vLLM |
| --- | --- | --- |
| 国产 AI 加速器深度优化 | 是 | 视插件而定 |
| 通用 NVIDIA GPU 部署 | 可用，但不是默认优先 | 是 |
| OpenAI-compatible 快速上线 | 支持相关接口演进 | 更成熟 |
| 企业级服务调度 | 内建 xLLM-Service | 通常依赖外部系统 |
| 在线离线混部 | 强项 | 需外部调度 |
| 动态 PD 分离 | 强项 | 有相关能力，但体系不同 |
| 多模态 EPD 分离 | 强项 | 有多模态和 disaggregated 能力，但服务体系不同 |
| 全局 KV Cache 和 KV-aware routing | 强项 | 需看具体部署方案 |
| Hugging Face 模型生态 | 正在扩展 | 强项 |
| 社区成熟度 | 较新 | 强项 |
| 运维资料和案例 | 较少 | 较多 |

## 17. 推荐结论

对于一般团队，建议：

- 用 vLLM 做默认推理底座；
- 在模型、API 和业务链路稳定后，再根据成本和硬件约束做更深优化；
- 如果部署环境是国产 AI 加速器，或者业务已经需要 PD/EPD、全局 KV Cache、在线离线混部，则并行评估 xLLM；
- 不要直接依据论文数字做最终选型，应使用真实业务 workload 做 benchmark。

对于平台团队，建议：

- 将 xLLM 视为企业级推理平台候选，而不是只看作一个单机推理库；
- 重点验证 xLLM-Service 的调度能力、故障恢复能力和多实例运维复杂度；
- 结合业务的 TTFT、TPOT、吞吐、p95/p99、成本和稳定性指标建立统一评测框架；
- 对 vLLM、xLLM、MindIE、TensorRT-LLM、SGLang 等方案做同条件横向比较。

## 18. 参考来源

- [xLLM GitHub 主仓库](https://github.com/jd-opensource/xllm)
- [xLLM-Service GitHub 仓库](https://github.com/jd-opensource/xllm-service)
- [xLLM 官方文档](https://docs.xllm-ai.com/en/)
- [xLLM Releases](https://github.com/jd-opensource/xllm/releases)
- [xLLM-Service Releases](https://github.com/jd-opensource/xllm-service/releases)
- [xLLM Technical Report, arXiv:2510.14686](https://arxiv.org/abs/2510.14686)
- [vLLM 官方文档](https://docs.vllm.ai/)
- [vLLM OpenAI-Compatible Server 文档](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)
- [vLLM-Ascend GitHub 仓库](https://github.com/vllm-project/vllm-ascend)

