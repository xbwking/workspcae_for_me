# verl 29ffe753 Ascend Benchmark Runtime Package

这是一个可以直接放进 verl 镜像/verl 仓库根目录运行的 Ascend 耗时拆解 benchmark 包。

## 适用版本

- verl commit: `29ffe753600ceca3cc5530ee6166be77fb4ecc1c`
- 设计目标：不侵入修改 verl 开源源码，运行 benchmark 时通过 monkey patch 注入埋点。

## 目录说明

```text
files/
  scripts/
    bench_ascend_verl_timing.py                 # 主 benchmark：run / summarize / compare
    run_ppo_with_ascend_benchmark_patches.py    # PPO 启动 wrapper，负责启用 monkey patch
    ascend_benchmark_monkey_patch/              # monkey patch 主体
    ascend_benchmark_monkey_patch_bootstrap/    # Ray worker 自动加载 patch 的 sitecustomize
    ascend_verl_timing_report.py                # 报告生成模块
    report_ascend_verl_timing.py                # 报告生成命令入口
    bench_fully_async_message_queue.py          # 可选：MessageQueue 批量拉取微基准
  tests/special_npu/
    run_ascend_timing_breakdown_bench.sh        # 推荐一键运行入口
  docs/
    perf/ascend_timing_breakdown_benchmark.md   # benchmark 使用说明
    ascend_tutorial/quick_start/...             # 昇腾运行 verl 傻瓜教程
  tests/
    ...                                         # 可选开发者测试
```

## 真正有用的文件

运行必需：

```text
scripts/bench_ascend_verl_timing.py
scripts/run_ppo_with_ascend_benchmark_patches.py
scripts/ascend_benchmark_monkey_patch/__init__.py
scripts/ascend_benchmark_monkey_patch_bootstrap/sitecustomize.py
tests/special_npu/run_ascend_timing_breakdown_bench.sh
```

看结果必需：

```text
scripts/ascend_verl_timing_report.py
scripts/report_ascend_verl_timing.py
```

建议保留：

```text
docs/perf/ascend_timing_breakdown_benchmark.md
docs/ascend_tutorial/quick_start/ascend_verl_dummies_guide_zh.md
scripts/bench_fully_async_message_queue.py
tests/
```

不需要放进镜像的旧内容：

```text
旧 changed_files 覆盖包
旧 4045d670 版本补丁
旧 ray_trainer.py / checkpoint_engine/base.py / message_queue.py 覆盖文件
PPT、讲稿、设计文档
```

## 安装到 verl 镜像

把本目录上传到容器后执行：

```bash
bash install_into_verl.sh /verl
```

如果你已经在 verl 仓库根目录，也可以直接：

```bash
cp -R files/. .
```

## 运行 benchmark

在 verl 仓库根目录执行：

```bash
MODEL_PATH=/path/to/model \
TRAIN_FILES=/path/to/train.parquet \
VAL_FILES=/path/to/test.parquet \
OUTPUT_DIR=outputs/ascend_timing_breakdown/run1 \
bash tests/special_npu/run_ascend_timing_breakdown_bench.sh
```

输出目录会包含：

```text
metrics.jsonl
stdout.log
summary.json
timing_breakdown.csv
npu_profile/
```

## 生成一页式报告

```bash
python3 scripts/report_ascend_verl_timing.py \
  --run-dir outputs/ascend_timing_breakdown/run1
```

会生成：

```text
report.md
report.json
top_metrics.csv
```

## 本地快速校验

在 verl 仓库根目录执行：

```bash
python3 -m py_compile \
  scripts/bench_ascend_verl_timing.py \
  scripts/run_ppo_with_ascend_benchmark_patches.py \
  scripts/ascend_benchmark_monkey_patch/__init__.py \
  scripts/ascend_benchmark_monkey_patch_bootstrap/sitecustomize.py \
  scripts/ascend_verl_timing_report.py \
  scripts/report_ascend_verl_timing.py

MODEL_PATH=/models/qwen \
TRAIN_FILES=/data/train.parquet \
VAL_FILES=/data/test.parquet \
OUTPUT_DIR=/tmp/verl_ascend_bench_dry_run \
bash tests/special_npu/run_ascend_timing_breakdown_bench.sh --dry-run
```

## 非侵入式说明

这个包不会要求你覆盖 verl 的核心源码文件。benchmark 运行时实际入口是：

```text
scripts/run_ppo_with_ascend_benchmark_patches.py
```

它会在 driver 和 Ray worker 进程里启用 monkey patch，然后再调用原始 verl PPO 主流程。
