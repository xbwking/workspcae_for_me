# verl Ascend 推理优化与耗时拆解交付包

这个仓库用于保存 Ascend-supported verl 推理性能优化相关交付物，包括设计文档、串讲稿、benchmark 使用手册，以及可以应用到 verl 仓库的 patch 文件。

## 最重要的入口

如果你只想找最终材料，优先看下面几份：

| 目的 | 文件 |
| --- | --- |
| 最终代码级设计文档 | [docs/verl_ascend_supported_code_level_design.md](docs/verl_ascend_supported_code_level_design.md) |
| 需求/方案结构化文档 | [docs/verl_ascend_supported_requirement_solution_sections.md](docs/verl_ascend_supported_requirement_solution_sections.md) |
| 方案串讲稿 | [docs/verl_ascend_requirement_solution_speech.md](docs/verl_ascend_requirement_solution_speech.md) |
| Ascend verl 前置介绍讲稿 | [docs/verl_ascend_intro_speech.md](docs/verl_ascend_intro_speech.md) |
| Ascend verl 前置介绍 PPT | [docs/ascend_verl_intro_deck.pptx](docs/ascend_verl_intro_deck.pptx) |
| 耗时拆解 benchmark 使用手册 | [docs/verl_ascend_timing_breakdown_benchmark_user_manual.md](docs/verl_ascend_timing_breakdown_benchmark_user_manual.md) |
| 可应用到 verl 的 patch 包 | [verl_ascend_benchmark_patch/](verl_ascend_benchmark_patch/) |

## Benchmark 到底用哪份

完整的 verl Ascend 耗时拆解 benchmark 是：

```text
verl_ascend_benchmark_patch/changed_files/scripts/bench_ascend_verl_timing.py
```

推荐一键运行入口是：

```text
verl_ascend_benchmark_patch/changed_files/tests/special_npu/run_ascend_timing_breakdown_bench.sh
```

不要和下面这个脚本混淆：

```text
verl_ascend_benchmark_patch/changed_files/scripts/bench_fully_async_message_queue.py
```

`bench_fully_async_message_queue.py` 只用于 fully_async MessageQueue 局部压测，不是完整端到端 benchmark。

详细使用方式看：

- [docs/verl_ascend_timing_breakdown_benchmark_user_manual.md](docs/verl_ascend_timing_breakdown_benchmark_user_manual.md)
- [verl_ascend_benchmark_patch/benchmark_user_manual.md](verl_ascend_benchmark_patch/benchmark_user_manual.md)

## Ascend-supported 基线

- verl Ascend-supported source baseline: `verl-project/verl@4045d67063052dcb800c918c107b8d5a87046006`
- Ascend recipe evidence: `docker/ascend/Dockerfile.ascend_8.5.2_a3_qwen3-5`
- Runtime stack from the Ascend recipe: CANN 8.5.2, vLLM v0.18.0, vLLM-Ascend `54879467c41784a446aa5b486a391d9bfbf488fa`, torch 2.9.0, torch_npu 2.9.0

## 目录说明

```text
docs/
  设计文档、讲稿、PPT、benchmark 使用手册

verl_ascend_benchmark_patch/
  可拷贝到 Ascend-supported verl 仓库的 patch 交付包

verl_ascend_benchmark_patch/changed_files/
  按 verl 仓库相对路径保存的变更文件
```

## 当前方案覆盖的优化方向

- HCCL checkpoint engine backend 注册修复
- 权重同步链路分段耗时指标
- bucketed IPC / SHM 权重传输统计
- fully_async MessageQueue 批量消费与序列化指标
- Ascend timing breakdown benchmark
- baseline / patched A/B compare

## 使用建议

先读设计文档了解方案，再看 benchmark 手册上机验证。真实性能收益必须在 Ascend 环境中跑 baseline / patched 两组结果后确认，本地 CPU 测试只能证明 parser、CLI、dry-run 和局部逻辑可用。
