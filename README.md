# verl Ascend Inference Optimization Design

This repository contains a code-level feasibility and design document for inference performance optimization on the Ascend-supported verl baseline.

## Baseline

- verl Ascend-supported source baseline: `verl-project/verl@4045d67063052dcb800c918c107b8d5a87046006`
- Ascend recipe evidence: `docker/ascend/Dockerfile.ascend_8.5.2_a3_qwen3-5`
- Runtime stack from the Ascend recipe: CANN 8.5.2, vLLM v0.18.0, vLLM-Ascend `54879467c41784a446aa5b486a391d9bfbf488fa`, torch 2.9.0, torch_npu 2.9.0

## Document

- [Ascend verl 前置介绍讲稿](docs/verl_ascend_intro_speech.md)
- [Ascend verl 前置介绍 PPT](docs/ascend_verl_intro_deck.pptx)
- [Ascend verl 前置介绍 PPT 预览图](docs/ascend_verl_intro_deck_preview.png)
- [verl Ascend 支持版本推理优化代码级设计文档](docs/verl_ascend_supported_code_level_design.md)
- [verl Ascend 推理优化设计详细讲稿](docs/verl_ascend_supported_design_speech.md)

## Main Topics

- Async weight synchronization and prefetch/commit pipeline
- Ascend HCCL and IPC weight transfer optimization
- Ray and rollout scheduling optimization
- TransferQueue/SampleRef-based rollout sample transfer
- Prefix/KV cache policy and rollout runtime tuning
