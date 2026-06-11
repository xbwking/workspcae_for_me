"""Bootstrap Ascend benchmark monkey patches for Python worker processes."""

from __future__ import annotations

import os


if os.getenv("VERL_ASCEND_BENCHMARK_MONKEY_PATCH") == "1":
    try:
        from ascend_benchmark_monkey_patch import apply_all

        apply_all()
    except Exception as exc:  # pragma: no cover - visible in worker logs
        print(f"[ascend_benchmark_monkey_patch] failed to apply patches: {exc}", flush=True)
