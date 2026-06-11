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

import json
import subprocess
import sys


def test_message_queue_benchmark_local_mode_reports_call_reduction():
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/bench_fully_async_message_queue.py",
            "--mode",
            "local",
            "--num-samples",
            "16",
            "--batch-size",
            "4",
            "--payload-bytes",
            "8",
            "--json",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    result = json.loads(completed.stdout)

    assert result["single_consumed"] == 16
    assert result["batched_consumed"] == 16
    assert result["single_get_call_count"] == 16
    assert result["batched_get_call_count"] == 4
    assert result["call_count_reduction"] == 4
