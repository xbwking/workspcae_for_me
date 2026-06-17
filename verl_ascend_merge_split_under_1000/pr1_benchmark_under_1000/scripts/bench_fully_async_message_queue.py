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
"""Benchmark single-sample queue get versus batched queue get.

This script is self-contained so it can be merged without modifying verl's
fully_async MessageQueue implementation.  Use it to estimate the scheduling
benefit of reducing many queue get RPCs into fewer batched get RPCs.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import sys
import time
from collections import deque
from typing import Any


class LocalMessageQueue:
    def __init__(self, max_queue_size: int = 1000):
        self.queue = deque(maxlen=max_queue_size)
        self.running = True
        self.total_consumed = 0
        self._lock = asyncio.Lock()
        self._consumer_condition = asyncio.Condition(self._lock)

    async def put_sample(self, sample: Any) -> bool:
        async with self._lock:
            self.queue.append(sample)
            self._consumer_condition.notify_all()
            return True

    async def get_sample(self) -> Any | None:
        async with self._lock:
            while not self.queue and self.running:
                await self._consumer_condition.wait()
            if not self.running and not self.queue:
                return None
            self.total_consumed += 1
            return self.queue.popleft(), len(self.queue)

    async def get_samples(self, max_n: int) -> tuple[list[Any], int] | None:
        if max_n <= 0:
            raise ValueError(f"max_n must be positive, got {max_n}")
        async with self._lock:
            while not self.queue and self.running:
                await self._consumer_condition.wait()
            if not self.running and not self.queue:
                return None
            samples = []
            while self.queue and len(samples) < max_n:
                samples.append(self.queue.popleft())
                self.total_consumed += 1
            return samples, len(self.queue)


def _samples(num_samples: int, payload_bytes: int) -> list[bytes]:
    payload = b"x" * payload_bytes
    return [payload for _ in range(num_samples)]


async def _fill_queue(queue, samples: list[Any]) -> None:
    for sample in samples:
        await queue.put_sample(sample)


async def _local_single_get(queue, num_samples: int) -> tuple[int, int]:
    consumed = 0
    calls = 0
    while consumed < num_samples:
        result = await queue.get_sample()
        calls += 1
        if result is None:
            break
        consumed += 1
    return consumed, calls


async def _local_batched_get(queue, num_samples: int, batch_size: int) -> tuple[int, int]:
    consumed = 0
    calls = 0
    while consumed < num_samples:
        result = await queue.get_samples(batch_size)
        calls += 1
        if result is None:
            break
        batch, _ = result
        consumed += len(batch)
    return consumed, calls


def run_local_benchmark(num_samples: int, batch_size: int, payload_bytes: int) -> dict[str, Any]:
    samples = _samples(num_samples, payload_bytes)

    single_queue = LocalMessageQueue(max_queue_size=max(num_samples + 1, 1))
    asyncio.run(_fill_queue(single_queue, samples))
    start = time.perf_counter()
    single_consumed, single_calls = asyncio.run(_local_single_get(single_queue, num_samples))
    single_elapsed_ms = (time.perf_counter() - start) * 1000

    batched_queue = LocalMessageQueue(max_queue_size=max(num_samples + 1, 1))
    asyncio.run(_fill_queue(batched_queue, samples))
    start = time.perf_counter()
    batched_consumed, batched_calls = asyncio.run(_local_batched_get(batched_queue, num_samples, batch_size))
    batched_elapsed_ms = (time.perf_counter() - start) * 1000

    return _result(
        "local",
        num_samples,
        batch_size,
        payload_bytes,
        single_consumed,
        single_calls,
        single_elapsed_ms,
        batched_consumed,
        batched_calls,
        batched_elapsed_ms,
    )


def run_ray_benchmark(num_samples: int, batch_size: int, payload_bytes: int) -> dict[str, Any]:
    import ray

    RemoteQueue = ray.remote(num_cpus=1, max_concurrency=20)(LocalMessageQueue)
    samples = _samples(num_samples, payload_bytes)
    started_ray = not ray.is_initialized()
    if started_ray:
        ray.init(ignore_reinit_error=True, include_dashboard=False)
    try:
        single_queue = RemoteQueue.remote(max_queue_size=max(num_samples + 1, 1))
        ray.get([single_queue.put_sample.remote(sample) for sample in samples])
        start = time.perf_counter()
        single_consumed, single_calls = _ray_single_get(ray, single_queue, num_samples)
        single_elapsed_ms = (time.perf_counter() - start) * 1000

        batched_queue = RemoteQueue.remote(max_queue_size=max(num_samples + 1, 1))
        ray.get([batched_queue.put_sample.remote(sample) for sample in samples])
        start = time.perf_counter()
        batched_consumed, batched_calls = _ray_batched_get(ray, batched_queue, num_samples, batch_size)
        batched_elapsed_ms = (time.perf_counter() - start) * 1000
    finally:
        if started_ray:
            ray.shutdown()

    return _result(
        "ray",
        num_samples,
        batch_size,
        payload_bytes,
        single_consumed,
        single_calls,
        single_elapsed_ms,
        batched_consumed,
        batched_calls,
        batched_elapsed_ms,
    )


def _ray_single_get(ray, queue, num_samples: int) -> tuple[int, int]:
    consumed = 0
    calls = 0
    while consumed < num_samples:
        result = ray.get(queue.get_sample.remote())
        calls += 1
        if result is None:
            break
        consumed += 1
    return consumed, calls


def _ray_batched_get(ray, queue, num_samples: int, batch_size: int) -> tuple[int, int]:
    consumed = 0
    calls = 0
    while consumed < num_samples:
        result = ray.get(queue.get_samples.remote(batch_size))
        calls += 1
        if result is None:
            break
        batch, _ = result
        consumed += len(batch)
    return consumed, calls


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
) -> dict[str, Any]:
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


def parse_args() -> argparse.Namespace:
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


def main() -> None:
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
    print(f"single_get: calls={result['single_get_call_count']} elapsed_ms={result['single_elapsed_ms']:.3f}")
    print(f"batched_get: calls={result['batched_get_call_count']} elapsed_ms={result['batched_elapsed_ms']:.3f}")
    print(f"call_count_reduction={result['call_count_reduction']:.3f} elapsed_speedup={result['elapsed_speedup']:.3f}")


if __name__ == "__main__":
    main()
