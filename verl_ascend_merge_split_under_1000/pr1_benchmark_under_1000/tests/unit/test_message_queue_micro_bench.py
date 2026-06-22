# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

import asyncio
import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "bench_fully_async_message_queue.py"


def load_module():
    spec = importlib.util.spec_from_file_location("_bench_fully_async_message_queue_ut", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_bench_fully_async_message_queue_ut"] = module
    spec.loader.exec_module(module)
    return module


class LocalMessageQueueTest(unittest.TestCase):
    def setUp(self):
        self.bench = load_module()

    def test_put_sample_accepts_payload_and_notifies_consumer(self):
        async def scenario():
            queue = self.bench.LocalMessageQueue()
            accepted = await queue.put_sample(b"abc")
            result = await queue.get_sample()
            return accepted, result

        accepted, result = asyncio.run(scenario())

        self.assertTrue(accepted)
        self.assertEqual(result, (b"abc", 0))

    def test_get_sample_returns_none_when_stopped_and_empty(self):
        async def scenario():
            queue = self.bench.LocalMessageQueue()
            queue.running = False
            return await queue.get_sample()

        self.assertIsNone(asyncio.run(scenario()))

    def test_get_sample_consumes_in_fifo_order(self):
        async def scenario():
            queue = self.bench.LocalMessageQueue()
            await queue.put_sample("a")
            await queue.put_sample("b")
            first = await queue.get_sample()
            second = await queue.get_sample()
            return first, second, queue.total_consumed

        first, second, consumed = asyncio.run(scenario())

        self.assertEqual(first, ("a", 1))
        self.assertEqual(second, ("b", 0))
        self.assertEqual(consumed, 2)

    def test_get_samples_rejects_non_positive_batch_size(self):
        async def scenario():
            queue = self.bench.LocalMessageQueue()
            await queue.get_samples(0)

        with self.assertRaises(ValueError):
            asyncio.run(scenario())

    def test_get_samples_returns_none_when_stopped_and_empty(self):
        async def scenario():
            queue = self.bench.LocalMessageQueue()
            queue.running = False
            return await queue.get_samples(4)

        self.assertIsNone(asyncio.run(scenario()))

    def test_get_samples_consumes_up_to_batch_size(self):
        async def scenario():
            queue = self.bench.LocalMessageQueue()
            for item in ["a", "b", "c"]:
                await queue.put_sample(item)
            batch, remaining = await queue.get_samples(2)
            return batch, remaining, queue.total_consumed

        batch, remaining, consumed = asyncio.run(scenario())

        self.assertEqual(batch, ["a", "b"])
        self.assertEqual(remaining, 1)
        self.assertEqual(consumed, 2)

    def test_get_samples_can_return_less_than_batch_size(self):
        async def scenario():
            queue = self.bench.LocalMessageQueue()
            await queue.put_sample("only")
            batch, remaining = await queue.get_samples(8)
            return batch, remaining

        batch, remaining = asyncio.run(scenario())

        self.assertEqual(batch, ["only"])
        self.assertEqual(remaining, 0)


class LocalBenchmarkHelpersTest(unittest.TestCase):
    def setUp(self):
        self.bench = load_module()

    def test_samples_builds_payloads_with_requested_size(self):
        samples = self.bench._samples(num_samples=3, payload_bytes=5)

        self.assertEqual(len(samples), 3)
        self.assertEqual(samples, [b"xxxxx", b"xxxxx", b"xxxxx"])

    def test_fill_queue_places_all_samples_in_queue(self):
        async def scenario():
            queue = self.bench.LocalMessageQueue(max_queue_size=10)
            await self.bench._fill_queue(queue, ["a", "b", "c"])
            return list(queue.queue)

        self.assertEqual(asyncio.run(scenario()), ["a", "b", "c"])

    def test_local_single_get_consumes_one_rpc_per_sample(self):
        async def scenario():
            queue = self.bench.LocalMessageQueue()
            await self.bench._fill_queue(queue, ["a", "b", "c"])
            return await self.bench._local_single_get(queue, 3)

        consumed, calls = asyncio.run(scenario())

        self.assertEqual(consumed, 3)
        self.assertEqual(calls, 3)

    def test_local_batched_get_consumes_ceil_batches(self):
        async def scenario():
            queue = self.bench.LocalMessageQueue()
            await self.bench._fill_queue(queue, list(range(9)))
            return await self.bench._local_batched_get(queue, 9, 4)

        consumed, calls = asyncio.run(scenario())

        self.assertEqual(consumed, 9)
        self.assertEqual(calls, 3)

    def test_result_computes_call_count_reduction_and_elapsed_speedup(self):
        result = self.bench._result(
            mode="local",
            num_samples=8,
            batch_size=4,
            payload_bytes=0,
            single_consumed=8,
            single_calls=8,
            single_elapsed_ms=16.0,
            batched_consumed=8,
            batched_calls=2,
            batched_elapsed_ms=4.0,
        )

        self.assertEqual(result["call_count_reduction"], 4.0)
        self.assertEqual(result["elapsed_speedup"], 4.0)

    def test_result_guards_against_zero_batched_calls(self):
        result = self.bench._result(
            mode="local",
            num_samples=1,
            batch_size=1,
            payload_bytes=0,
            single_consumed=1,
            single_calls=1,
            single_elapsed_ms=1.0,
            batched_consumed=0,
            batched_calls=0,
            batched_elapsed_ms=0.0,
        )

        self.assertEqual(result["call_count_reduction"], 1.0)
        self.assertGreater(result["elapsed_speedup"], 0)


class LocalBenchmarkEndToEndTest(unittest.TestCase):
    def setUp(self):
        self.bench = load_module()

    def test_local_benchmark_reports_expected_call_reduction(self):
        result = self.bench.run_local_benchmark(num_samples=16, batch_size=4, payload_bytes=8)

        self.assertEqual(result["single_consumed"], 16)
        self.assertEqual(result["batched_consumed"], 16)
        self.assertEqual(result["single_get_call_count"], 16)
        self.assertEqual(result["batched_get_call_count"], 4)
        self.assertEqual(result["call_count_reduction"], 4)
        self.assertGreaterEqual(result["single_elapsed_ms"], 0)
        self.assertGreaterEqual(result["batched_elapsed_ms"], 0)

    def test_local_benchmark_uses_ceiling_for_uneven_batches(self):
        result = self.bench.run_local_benchmark(num_samples=10, batch_size=4, payload_bytes=0)

        self.assertEqual(result["single_get_call_count"], 10)
        self.assertEqual(result["batched_get_call_count"], 3)
        self.assertEqual(result["batched_consumed"], 10)

    def test_local_benchmark_handles_batch_larger_than_sample_count(self):
        result = self.bench.run_local_benchmark(num_samples=3, batch_size=8, payload_bytes=0)

        self.assertEqual(result["single_get_call_count"], 3)
        self.assertEqual(result["batched_get_call_count"], 1)
        self.assertEqual(result["call_count_reduction"], 3)


class MessageQueueCliTest(unittest.TestCase):
    def test_cli_json_mode_is_machine_readable(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--mode",
                "local",
                "--num-samples",
                "9",
                "--batch-size",
                "4",
                "--payload-bytes",
                "0",
                "--json",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        result = json.loads(completed.stdout)

        self.assertEqual(result["single_consumed"], 9)
        self.assertEqual(result["batched_consumed"], 9)
        self.assertEqual(result["single_get_call_count"], 9)
        self.assertEqual(result["batched_get_call_count"], 3)
        self.assertEqual(result["call_count_reduction"], 3)

    def test_cli_text_mode_prints_human_readable_lines(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--mode",
                "local",
                "--num-samples",
                "4",
                "--batch-size",
                "2",
                "--payload-bytes",
                "0",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("mode=local", completed.stdout)
        self.assertIn("single_get: calls=4", completed.stdout)
        self.assertIn("batched_get: calls=2", completed.stdout)
        self.assertIn("call_count_reduction=2.000", completed.stdout)

    def test_cli_rejects_zero_num_samples(self):
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), "--num-samples", "0"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("--num-samples must be positive", completed.stderr)

    def test_cli_rejects_zero_batch_size(self):
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), "--batch-size", "0"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("--batch-size must be positive", completed.stderr)

    def test_cli_rejects_negative_payload_size(self):
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), "--payload-bytes", "-1"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("--payload-bytes must be non-negative", completed.stderr)

    def test_main_routes_to_local_benchmark(self):
        bench = load_module()
        with mock.patch.object(sys, "argv", [str(SCRIPT), "--mode", "local", "--num-samples", "2", "--batch-size", "1"]):
            with mock.patch.object(bench, "run_local_benchmark", return_value=bench._result("local", 2, 1, 0, 2, 2, 1, 2, 2, 1)) as local_mock:
                with mock.patch("builtins.print"):
                    bench.main()

        local_mock.assert_called_once_with(2, 1, 128)


if __name__ == "__main__":
    unittest.main()
