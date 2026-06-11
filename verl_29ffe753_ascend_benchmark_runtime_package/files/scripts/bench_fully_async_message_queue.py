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
"""Benchmark fully_async MessageQueue single-sample vs batched get paths.

The local mode runs without Ray and validates batching behavior on developer
machines. The ray mode should be used in a real verl environment to measure
actor RPC overhead reduction.
"""

import argparse
import asyncio
import contextlib
import importlib.util
import io
import json
import sys
import time
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MESSAGE_QUEUE_PATH = REPO_ROOT / "verl" / "experimental" / "fully_async_policy" / "message_queue.py"


class _FakeRay(types.SimpleNamespace):
    def remote(self, *args, **kwargs):
        def decorator(cls):
            return cls

        return decorator


def _load_message_queue_module(module_name: str):
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, MESSAGE_QUEUE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _install_local_get_samples(MessageQueue):
    if hasattr(MessageQueue, "get_samples"):
        return MessageQueue

    async def get_samples(self, max_n: int, timeout_ms: int | None = None):
        if max_n <= 0:
            raise ValueError(f"max_n must be positive, got {max_n}")
        async with self._lock:
            if timeout_ms is None:
                while len(self.queue) == 0 and self.running:
                    await self._consumer_condition.wait()
            elif len(self.queue) == 0 and self.running:
                try:
                    await asyncio.wait_for(self._consumer_condition.wait(), timeout=timeout_ms / 1000)
                except TimeoutError:
                    return [], 0
            if not self.running and len(self.queue) == 0:
                return None
            samples = []
            while self.queue and len(samples) < max_n:
                data = self.queue.popleft()
                self.total_consumed += 1
                samples.append(data)
                if data is None:
                    break
            return samples, len(self.queue)

    MessageQueue.get_samples = get_samples
    return MessageQueue


def _samples(num_samples: int, payload_bytes: int):
    payload = b"x" * payload_bytes
    return [payload for _ in range(num_samples)]


async def _local_fill_queue(queue, samples):
    for sample in samples:
        await queue.put_sample(sample)


async def _local_single_get(queue, num_samples: int):
    consumed = 0
    call_count = 0
    while consumed < num_samples:
        result = await queue.get_sample()
        call_count += 1
        if result is None:
            break
        consumed += 1
    return consumed, call_count


async def _local_batched_get(queue, num_samples: int, batch_size: int):
    consumed = 0
    call_count = 0
    while consumed < num_samples:
        result = await queue.get_samples(max_n=batch_size, timeout_ms=1000)
        call_count += 1
        if result is None:
            break
        batch, _ = result
        consumed += len(batch)
    return consumed, call_count


def run_local_benchmark(num_samples: int, batch_size: int, payload_bytes: int):
    fake_omegaconf = types.SimpleNamespace(DictConfig=dict)
    sys.modules["ray"] = _FakeRay()
    sys.modules["omegaconf"] = fake_omegaconf
    module = _load_message_queue_module("_bench_message_queue_local")
    MessageQueue = _install_local_get_samples(module.MessageQueue)
    samples = _samples(num_samples, payload_bytes)

    single_queue = MessageQueue({}, max_queue_size=max(num_samples + 1, 1))
    asyncio.run(_local_fill_queue(single_queue, samples))
    start = time.perf_counter()
    single_consumed, single_calls = asyncio.run(_local_single_get(single_queue, num_samples))
    single_elapsed_ms = (time.perf_counter() - start) * 1000

    batched_queue = MessageQueue({}, max_queue_size=max(num_samples + 1, 1))
    asyncio.run(_local_fill_queue(batched_queue, samples))
    start = time.perf_counter()
    batched_consumed, batched_calls = asyncio.run(_local_batched_get(batched_queue, num_samples, batch_size))
    batched_elapsed_ms = (time.perf_counter() - start) * 1000

    return _result(
        mode="local",
        num_samples=num_samples,
        batch_size=batch_size,
        payload_bytes=payload_bytes,
        single_consumed=single_consumed,
        single_calls=single_calls,
        single_elapsed_ms=single_elapsed_ms,
        batched_consumed=batched_consumed,
        batched_calls=batched_calls,
        batched_elapsed_ms=batched_elapsed_ms,
    )


def _ray_fill_queue(ray, queue, samples):
    refs = [queue.put_sample.remote(sample) for sample in samples]
    ray.get(refs)


def _ray_single_get(ray, queue, num_samples: int):
    consumed = 0
    call_count = 0
    while consumed < num_samples:
        result = ray.get(queue.get_sample.remote())
        call_count += 1
        if result is None:
            break
        consumed += 1
    return consumed, call_count


def _ray_batched_get(ray, queue, num_samples: int, batch_size: int):
    consumed = 0
    call_count = 0
    while consumed < num_samples:
        result = ray.get(queue.get_samples.remote(batch_size, 1000))
        call_count += 1
        if result is None:
            break
        batch, _ = result
        consumed += len(batch)
    return consumed, call_count


def run_ray_benchmark(num_samples: int, batch_size: int, payload_bytes: int):
    import ray
    from omegaconf import OmegaConf

    from ascend_benchmark_monkey_patch import apply_all

    apply_all()
    from verl.experimental.fully_async_policy import message_queue as module

    MessageQueue = module.MessageQueue
    samples = _samples(num_samples, payload_bytes)

    started_ray = not ray.is_initialized()
    if started_ray:
        ray.init(ignore_reinit_error=True, include_dashboard=False)

    try:
        single_queue = MessageQueue.remote(OmegaConf.create({}), max_queue_size=max(num_samples + 1, 1))
        _ray_fill_queue(ray, single_queue, samples)
        start = time.perf_counter()
        single_consumed, single_calls = _ray_single_get(ray, single_queue, num_samples)
        single_elapsed_ms = (time.perf_counter() - start) * 1000

        batched_queue = MessageQueue.remote(OmegaConf.create({}), max_queue_size=max(num_samples + 1, 1))
        _ray_fill_queue(ray, batched_queue, samples)
        start = time.perf_counter()
        batched_consumed, batched_calls = _ray_batched_get(ray, batched_queue, num_samples, batch_size)
        batched_elapsed_ms = (time.perf_counter() - start) * 1000
    finally:
        if started_ray:
            ray.shutdown()

    return _result(
        mode="ray",
        num_samples=num_samples,
        batch_size=batch_size,
        payload_bytes=payload_bytes,
        single_consumed=single_consumed,
        single_calls=single_calls,
        single_elapsed_ms=single_elapsed_ms,
        batched_consumed=batched_consumed,
        batched_calls=batched_calls,
        batched_elapsed_ms=batched_elapsed_ms,
    )


def _result(
    mode: str,
    num_samples: int,
    batch_size: int,
    payload_bytes: int,
    single_consumed: int,
    single_calls: int,
    single_elapsed_ms: float,
    batched_consumed: int,
    batched_calls: int,
    batched_elapsed_ms: float,
):
    return {
        "mode": mode,
        "num_samples": num_samples,
        "batch_size": batch_size,
        "payload_bytes": payload_bytes,
        "single_consumed": single_consumed,
        "single_get_call_count": single_calls,
        "single_elapsed_ms": single_elapsed_ms,
        "batched_consumed": batched_consumed,
        "batched_get_call_count": batched_calls,
        "batched_elapsed_ms": batched_elapsed_ms,
        "call_count_reduction": single_calls / max(batched_calls, 1),
        "elapsed_speedup": single_elapsed_ms / max(batched_elapsed_ms, 1e-9),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("local", "ray"), default="local")
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--payload-bytes", type=int, default=128)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON only.")
    args = parser.parse_args()
    if args.num_samples <= 0:
        parser.error("--num-samples must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.payload_bytes < 0:
        parser.error("--payload-bytes must be non-negative")
    return args


def main():
    args = parse_args()
    output_buffer = io.StringIO()
    output_context = contextlib.redirect_stdout(output_buffer) if args.json else contextlib.nullcontext()
    with output_context:
        if args.mode == "local":
            result = run_local_benchmark(args.num_samples, args.batch_size, args.payload_bytes)
        else:
            result = run_ray_benchmark(args.num_samples, args.batch_size, args.payload_bytes)

    if args.json:
        logs = output_buffer.getvalue()
        if logs:
            print(logs, file=sys.stderr, end="")
        print(json.dumps(result, sort_keys=True))
        return

    print(f"mode={result['mode']}")
    print(f"num_samples={result['num_samples']} batch_size={result['batch_size']} payload_bytes={result['payload_bytes']}")
    print(
        "single_get: "
        f"calls={result['single_get_call_count']} elapsed_ms={result['single_elapsed_ms']:.3f} "
        f"consumed={result['single_consumed']}"
    )
    print(
        "batched_get: "
        f"calls={result['batched_get_call_count']} elapsed_ms={result['batched_elapsed_ms']:.3f} "
        f"consumed={result['batched_consumed']}"
    )
    print(
        f"call_count_reduction={result['call_count_reduction']:.3f} "
        f"elapsed_speedup={result['elapsed_speedup']:.3f}"
    )


if __name__ == "__main__":
    main()
