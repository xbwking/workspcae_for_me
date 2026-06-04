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

from pathlib import Path


def test_checkpoint_engine_registry_has_overwrite_warning():
    source = Path("verl/checkpoint_engine/base.py").read_text()

    assert "if backend in CheckpointEngineRegistry._registry:" in source
    assert "Checkpoint engine backend %s is overwritten" in source


def test_hccl_checkpoint_engine_registers_hccl_backend():
    source = Path("verl/checkpoint_engine/hccl_checkpoint_engine.py").read_text()

    assert '@CheckpointEngineRegistry.register("hccl")' in source
    assert '@CheckpointEngineRegistry.register("nccl")' not in source
