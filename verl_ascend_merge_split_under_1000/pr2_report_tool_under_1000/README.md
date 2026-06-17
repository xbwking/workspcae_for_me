# verl Ascend Benchmark Report Tool

这个目录是独立的一页式报告工具，只负责把 benchmark 的原始输出整理成容易读的报告。

它不负责跑训练，不修改 verl 源码，也不要求你使用 monkey patch。只要你的 benchmark 输出目录里有 `metrics.jsonl` 和 `stdout.log`，就可以用。

## 包含文件

```text
scripts/ascend_verl_timing_report.py   # 报告生成主逻辑
scripts/report_ascend_verl_timing.py   # 命令行入口
```

## 输入文件

推荐输入目录结构：

```text
outputs/ascend_timing_breakdown/run1/
  metrics.jsonl
  stdout.log
  summary.json              # 可选，没有会自动重建
  timing_breakdown.csv      # 可选，没有会自动重建
  npu_profile/              # 可选，报告里会显示是否存在
```

最低要求：

```text
metrics.jsonl
stdout.log
```

## 安装方式

把本工具的 `scripts/` 目录复制到 verl 仓库根目录：

```bash
cd /path/to/verl
cp -R /path/to/verl_ascend_benchmark_report_tool/scripts/. scripts/
```

如果你只是临时分析，也可以不复制，直接从工具目录运行：

```bash
cd /path/to/verl
python3 /path/to/verl_ascend_benchmark_report_tool/scripts/report_ascend_verl_timing.py \
  --run-dir outputs/ascend_timing_breakdown/run1
```

## 生成报告

在 verl 仓库根目录执行：

```bash
python3 scripts/report_ascend_verl_timing.py \
  --run-dir outputs/ascend_timing_breakdown/run1
```

生成：

```text
outputs/ascend_timing_breakdown/run1/report.md
outputs/ascend_timing_breakdown/run1/report.json
outputs/ascend_timing_breakdown/run1/top_metrics.csv
```

## 看哪个文件

优先看：

```text
report.md
```

它会汇总：

```text
结论视图：timing_s/step、perf/throughput
Step 耗时主项：gen、reward、old_log_prob、update_actor、update_weights 等
参数同步 / 权重传输：param_sync/*、weight_transfer/*
Ray / 序列化 / 异步队列：ray/*、serialization/*、fully_async/*
产物索引：metrics.jsonl、stdout.log、summary.json、timing_breakdown.csv、npu_profile 是否存在
缺失指标：哪些关键指标没有采到
```

程序处理看：

```text
report.json
```

表格分析看：

```text
top_metrics.csv
```

## 常用参数

指定输出路径：

```bash
python3 scripts/report_ascend_verl_timing.py \
  --run-dir outputs/ascend_timing_breakdown/run1 \
  --output-md /tmp/report.md \
  --output-json /tmp/report.json \
  --output-csv /tmp/top_metrics.csv
```

如果没有 `summary.json`，并且需要指定 warmup / measured steps：

```bash
python3 scripts/report_ascend_verl_timing.py \
  --run-dir outputs/ascend_timing_breakdown/run1 \
  --warmup-steps 2 \
  --measured-steps 3,4,5,6,7,8
```

控制每组展示的 Top N：

```bash
python3 scripts/report_ascend_verl_timing.py \
  --run-dir outputs/ascend_timing_breakdown/run1 \
  --top-n 12
```

## 常见问题

### 1. report.md 里很多“缺失指标”

这不一定是错误。通常原因：

```text
当前运行路径没有触发对应逻辑
benchmark 没有打开相关埋点
stdout.log 没有包含对应统计日志
运行的是普通同步 PPO，不涉及 fully_async 队列
```

### 2. 没有 summary.json 可以跑吗

可以。工具会用 `metrics.jsonl` 和 `stdout.log` 自动生成：

```text
summary.json
timing_breakdown.csv
```

然后再生成报告。

### 3. npu_profile 会被解析吗

当前报告只检查 `npu_profile/` 是否存在、文件数和大小，不解析 profiler trace。原因是 profiler 目录结构和 CANN 版本相关，直接解析容易不稳定。性能结论优先看 `metrics.jsonl / summary.json / timing_breakdown.csv`。

### 4. 能不能单独用在别的 run 上

可以。只要 run 目录里有兼容的 `metrics.jsonl` 和 `stdout.log`。

## 快速自检

```bash
python3 -m py_compile \
  scripts/ascend_verl_timing_report.py \
  scripts/report_ascend_verl_timing.py

python3 scripts/report_ascend_verl_timing.py --help
```
