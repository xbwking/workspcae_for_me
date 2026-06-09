# verl Ascend Benchmark Patch

这个目录是可以应用到 Ascend-supported verl 仓库的 patch 交付包，重点是“耗时拆解 benchmark + 低风险框架指标补丁”。

## 先看哪份

| 目的 | 文件 |
| --- | --- |
| 中文使用手册 | [benchmark_user_manual.md](benchmark_user_manual.md) |
| 本地验证记录 | [verification.md](verification.md) |
| 需求/方案结构化文档 | [requirement_solution_sections.md](requirement_solution_sections.md) |
| 方案串讲稿 | [requirement_solution_speech.md](requirement_solution_speech.md) |
| 变更文件目录 | [changed_files/](changed_files/) |

## 真正的 benchmark 文件

完整端到端耗时拆解 benchmark：

```text
changed_files/scripts/bench_ascend_verl_timing.py
```

推荐一键运行入口：

```text
changed_files/tests/special_npu/run_ascend_timing_breakdown_bench.sh
```

局部 MessageQueue 压测脚本：

```text
changed_files/scripts/bench_fully_async_message_queue.py
```

注意：`bench_fully_async_message_queue.py` 不是完整端到端 benchmark，只用于 MessageQueue 局部压测。

## 内容结构

```text
changed_files/
  docs/
    ascend_tutorial/quick_start/ascend_verl_dummies_guide_zh.md
    perf/ascend_timing_breakdown_benchmark.md
  scripts/
    bench_ascend_verl_timing.py
    bench_fully_async_message_queue.py
  tests/
    special_npu/run_ascend_timing_breakdown_bench.sh
    special_sanity/test_ascend_timing_benchmark.py
    ...
  verl/
    checkpoint_engine/
    experimental/fully_async_policy/
    trainer/ppo/
    workers/rollout/vllm_rollout/
```

## 如何应用到 verl 仓库

在目标 Ascend-supported verl 仓库根目录执行：

```bash
cp -R /path/to/verl_ascend_benchmark_patch/changed_files/* .
```

然后先做本地结构验证：

```bash
bash -n tests/special_npu/run_ascend_timing_breakdown_bench.sh
python3 -m py_compile scripts/bench_ascend_verl_timing.py
```

有 pytest 环境时再跑：

```bash
python3 -m pytest tests/special_sanity/test_ascend_timing_benchmark.py -q
```

## Ascend 环境运行 benchmark

```bash
MODEL_PATH=/path/to/model TRAIN_FILES=/path/to/train.parquet VAL_FILES=/path/to/test.parquet OUTPUT_DIR=outputs/ascend_timing_breakdown/baseline bash tests/special_npu/run_ascend_timing_breakdown_bench.sh
```

输出：

```text
metrics.jsonl
stdout.log
summary.json
timing_breakdown.csv
npu_profile/
```

## A/B 对比

```bash
python3 scripts/bench_ascend_verl_timing.py compare   --baseline-summary outputs/ascend_timing_breakdown/baseline/summary.json   --patched-summary outputs/ascend_timing_breakdown/patched/summary.json   --output outputs/ascend_timing_breakdown/compare.json
```

真实性能收益必须在 Ascend 环境里跑 baseline / patched 后确认。
