# verl Ascend Benchmark Patch

这个目录保存本次针对 Ascend-supported verl 的推理/框架耗时拆解 benchmark 与低风险指标补丁。

## 内容结构

- `changed_files/`: 按 verl 仓库相对路径保存的源码、测试、脚本和文档。
- `verification.md`: 本地已执行的验证命令和结果。

## 主要能力

1. Ascend 端到端耗时拆解 benchmark：`scripts/bench_ascend_verl_timing.py`。
2. Ascend 一键运行脚本：`tests/special_npu/run_ascend_timing_breakdown_bench.sh`。
3. baseline/patched A/B compare：输出 `summary.json`、`timing_breakdown.csv`、`compare.json`。
4. 参数同步、权重传输、MessageQueue、cloudpickle 相关 L1 指标采集。
5. 昇腾上运行 verl 的傻瓜教程。

## 如何应用到 verl 仓库

在目标 verl 仓库根目录执行：

```bash
cp -R verl_ascend_benchmark_patch/changed_files/* .
```

然后运行：

```bash
python -m pytest tests/special_sanity/test_ascend_timing_benchmark.py -q
```

Ascend 环境上运行 benchmark：

```bash
MODEL_PATH=/path/to/model \
TRAIN_FILES=/path/to/train.parquet \
VAL_FILES=/path/to/test.parquet \
OUTPUT_DIR=outputs/ascend_timing_breakdown/baseline \
bash tests/special_npu/run_ascend_timing_breakdown_bench.sh
```
