# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run verl PPO with Ascend benchmark monkey patches enabled.

This wrapper keeps upstream verl source files unchanged.  It enables the
benchmark patch package in the driver process and injects the same bootstrap
into Ray runtime env so worker processes see the same patches.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from omegaconf import OmegaConf


def _prepend_env_path(name: str, paths: list[Path]) -> None:
    existing = os.environ.get(name, "")
    values = [str(path) for path in paths]
    if existing:
        values.append(existing)
    os.environ[name] = os.pathsep.join(values)


def _enable_patch_environment() -> list[str]:
    scripts_dir = Path(__file__).resolve().parent
    repo_root = scripts_dir.parent
    bootstrap_dir = scripts_dir / "ascend_benchmark_monkey_patch_bootstrap"
    paths = [bootstrap_dir, scripts_dir, repo_root]
    os.environ["VERL_ASCEND_BENCHMARK_MONKEY_PATCH"] = "1"
    _prepend_env_path("PYTHONPATH", paths)
    for path in reversed(paths):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    return [str(path) for path in paths]


def _patch_ray_runtime_env(main_ppo_module, patch_paths: list[str]) -> None:
    original_get_runtime_env = main_ppo_module.get_ppo_ray_runtime_env

    def patched_get_runtime_env():
        runtime_env = original_get_runtime_env()
        runtime_env = OmegaConf.create(runtime_env)
        env_vars = runtime_env.get("env_vars", {})
        env_vars["VERL_ASCEND_BENCHMARK_MONKEY_PATCH"] = "1"
        current_pythonpath = env_vars.get("PYTHONPATH") or os.environ.get("PYTHONPATH", "")
        merged_pythonpath = os.pathsep.join([*patch_paths, current_pythonpath]) if current_pythonpath else os.pathsep.join(patch_paths)
        env_vars["PYTHONPATH"] = merged_pythonpath
        runtime_env["env_vars"] = env_vars
        return runtime_env

    main_ppo_module.get_ppo_ray_runtime_env = patched_get_runtime_env


def main() -> int:
    patch_paths = _enable_patch_environment()
    from ascend_benchmark_monkey_patch import apply_all

    apply_all()

    from verl.trainer import main_ppo

    _patch_ray_runtime_env(main_ppo, patch_paths)
    main_ppo.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
