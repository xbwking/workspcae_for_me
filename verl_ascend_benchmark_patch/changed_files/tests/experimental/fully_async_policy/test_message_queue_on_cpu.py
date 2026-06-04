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

import asyncio
import importlib.util
import sys
import types
from pathlib import Path

import pytest


class _FakeRay(types.SimpleNamespace):
    def remote(self, *args, **kwargs):
        def decorator(cls):
            return cls

        return decorator


def _load_message_queue(monkeypatch):
    fake_omegaconf = types.SimpleNamespace(DictConfig=dict)
    monkeypatch.setitem(sys.modules, "ray", _FakeRay())
    monkeypatch.setitem(sys.modules, "omegaconf", fake_omegaconf)
    module_name = "_test_message_queue_module"
    sys.modules.pop(module_name, None)
    module_path = Path("verl/experimental/fully_async_policy/message_queue.py")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.MessageQueue


def test_get_samples_returns_batch_and_updates_consumed_count(monkeypatch):
    MessageQueue = _load_message_queue(monkeypatch)
    queue = MessageQueue({}, max_queue_size=16)

    assert asyncio.run(queue.put_sample("sample-0"))
    assert asyncio.run(queue.put_sample("sample-1"))
    assert asyncio.run(queue.put_sample("sample-2"))

    samples, remaining = asyncio.run(queue.get_samples(max_n=2, timeout_ms=100))

    assert samples == ["sample-0", "sample-1"]
    assert remaining == 1
    assert asyncio.run(queue.get_statistics())["total_consumed"] == 2


def test_get_samples_returns_empty_batch_on_timeout(monkeypatch):
    MessageQueue = _load_message_queue(monkeypatch)
    queue = MessageQueue({}, max_queue_size=16)

    samples, remaining = asyncio.run(queue.get_samples(max_n=4, timeout_ms=10))

    assert samples == []
    assert remaining == 0
    assert asyncio.run(queue.get_statistics())["total_consumed"] == 0


def test_get_samples_stops_at_none_sentinel(monkeypatch):
    MessageQueue = _load_message_queue(monkeypatch)
    queue = MessageQueue({}, max_queue_size=16)

    assert asyncio.run(queue.put_sample("sample-0"))
    assert asyncio.run(queue.put_sample(None))
    assert asyncio.run(queue.put_sample("sample-after-sentinel"))

    samples, remaining = asyncio.run(queue.get_samples(max_n=8, timeout_ms=100))

    assert samples == ["sample-0", None]
    assert remaining == 1
    assert asyncio.run(queue.get_statistics())["total_consumed"] == 2


def test_get_samples_returns_none_when_queue_is_shutdown_and_empty(monkeypatch):
    MessageQueue = _load_message_queue(monkeypatch)
    queue = MessageQueue({}, max_queue_size=16)

    asyncio.run(queue.shutdown())

    assert asyncio.run(queue.get_samples(max_n=1, timeout_ms=100)) is None


def test_get_samples_rejects_non_positive_batch_size(monkeypatch):
    MessageQueue = _load_message_queue(monkeypatch)
    queue = MessageQueue({}, max_queue_size=16)

    with pytest.raises(ValueError, match="max_n must be positive"):
        asyncio.run(queue.get_samples(max_n=0, timeout_ms=100))
