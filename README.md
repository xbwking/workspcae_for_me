# verl Ascend 推理优化与耗时拆解交付包

这个仓库保存 Ascend verl 推理性能优化相关交付物，包括设计文档、串讲稿、benchmark 使用手册，以及按 verl 版本整理的 patch 包。

## 当前应该用哪份 patch

你当前 Ascend 镜像里的 verl commit 是：

```text
29ffe753600ceca3cc5530ee6166be77fb4ecc1c
```

因此优先使用新版 patch 包：

```text
verl_29ffe753_benchmark_patch/
```

旧目录 `verl_ascend_benchmark_patch/` 是早期基于 Ascend-supported `4045d670` 做的包，不要整份覆盖到 `29ffe753` 镜像里，否则会出现 `verl.experimental.dataset`、`verl.utils.rollout_skip`、`AgentLoopManager(worker_group=...)` 等版本不兼容问题。

## 最重要的入口

| 目的 | 文件 |
| --- | --- |
| 29ffe753 benchmark patch 包 | [verl_29ffe753_benchmark_patch/](verl_29ffe753_benchmark_patch/) |
| 29ffe753 patch 使用说明 | [verl_29ffe753_benchmark_patch/README.md](verl_29ffe753_benchmark_patch/README.md) |
| benchmark 快速入口 | [BENCHMARK_QUICKSTART.md](BENCHMARK_QUICKSTART.md) |
| 中文 benchmark 使用手册 | [docs/verl_ascend_timing_breakdown_benchmark_user_manual.md](docs/verl_ascend_timing_breakdown_benchmark_user_manual.md) |
| 最终代码级设计文档 | [docs/verl_ascend_supported_code_level_design.md](docs/verl_ascend_supported_code_level_design.md) |
| 需求/方案结构化文档 | [docs/verl_ascend_supported_requirement_solution_sections.md](docs/verl_ascend_supported_requirement_solution_sections.md) |
| 方案串讲稿 | [docs/verl_ascend_requirement_solution_speech.md](docs/verl_ascend_requirement_solution_speech.md) |
| Ascend verl 前置介绍 PPT | [docs/ascend_verl_intro_deck.pptx](docs/ascend_verl_intro_deck.pptx) |

## 29ffe753 benchmark 入口

完整 benchmark 主脚本：

```text
verl_29ffe753_benchmark_patch/changed_files/scripts/bench_ascend_verl_timing.py
```

推荐一键运行脚本：

```text
verl_29ffe753_benchmark_patch/changed_files/tests/special_npu/run_ascend_timing_breakdown_bench.sh
```

局部 MessageQueue 压测脚本：

```text
verl_29ffe753_benchmark_patch/changed_files/scripts/bench_fully_async_message_queue.py
```

注意：`bench_fully_async_message_queue.py` 只用于 fully_async MessageQueue 局部压测，不是完整端到端 benchmark。

## 当前方案覆盖的优化方向

- HCCL checkpoint engine backend 注册修复
- 权重同步链路分段耗时指标 `param_sync/*`
- bucketed IPC / SHM 权重传输统计 `weight_transfer/*`
- fully_async MessageQueue 批量消费与序列化指标
- Ascend timing breakdown benchmark
- baseline / patched A/B compare

真实性能收益必须在 Ascend 环境中跑 baseline / patched 两组结果后确认，本地 CPU 测试只能证明 parser、CLI、dry-run 和局部逻辑可用。
