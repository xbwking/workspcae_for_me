"""Runtime monkey patches for Ascend timing benchmark instrumentation.

The patches in this package are intentionally kept outside the verl package so
the upstream source tree can stay unchanged.  They are enabled by the benchmark
wrapper and by the optional sitecustomize bootstrap used by Ray workers.
"""

from __future__ import annotations

import importlib
import os
import time
from typing import Any


_APPLIED = False


def apply_all() -> None:
    """Apply all benchmark monkey patches once per Python process."""
    global _APPLIED
    if _APPLIED:
        return
    _APPLIED = True

    _patch_hccl_registry()
    _patch_checkpoint_manager_timing()
    _patch_bucketed_weight_transfer_stats()
    _patch_fully_async_message_queue_batching()


def _patch_hccl_registry() -> None:
    """Expose the Ascend HCCL checkpoint engine under the `hccl` backend key."""
    try:
        base = importlib.import_module("verl.checkpoint_engine.base")
        hccl = importlib.import_module("verl.checkpoint_engine.hccl_checkpoint_engine")
    except Exception:
        return

    engine_cls = getattr(hccl, "HCCLCheckpointEngine", None)
    registry = getattr(base, "CheckpointEngineRegistry", None)
    if engine_cls is None or registry is None:
        return
    registry._registry["hccl"] = engine_cls


def _patch_checkpoint_manager_timing() -> None:
    """Record parameter-sync timing without editing CheckpointEngineManager source."""
    base = importlib.import_module("verl.checkpoint_engine.base")
    manager_cls = base.CheckpointEngineManager
    if getattr(manager_cls, "_ascend_benchmark_timing_patch", False):
        return

    original_init = manager_cls.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.last_update_weights_timing = {}

    async def timed_update_weights(self, global_steps: int = None):
        if self.backend == "naive":
            start = time.perf_counter()
            base.ray.get(self.trainer.update_weights(global_steps=global_steps, mode=self.backend))
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.last_update_weights_timing = {
                "param_sync/naive_update_ms": elapsed_ms,
                "param_sync/total_ms": elapsed_ms,
            }
            print(f"CheckpointEngineManager.update_weights timing: {self.last_update_weights_timing}", flush=True)
            return

        timing: dict[str, float] = {}

        start = time.perf_counter()
        await self.abort_replicas()
        timing["param_sync/abort_ms"] = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        workers = []
        for replica in self.replicas:
            workers.extend(replica.workers)
        rollout = base.RayWorkerGroup(
            worker_handles=workers,
            ray_cls_with_init=base.RayClassWithInitArgs(cls=base._worker_cls),
        )
        trainer = self.trainer
        timing["param_sync/create_rollout_wg_ms"] = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        await self.release_kv_cache_replicas()
        timing["param_sync/release_kv_cache_ms"] = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        self.build_process_group(rollout)
        timing["param_sync/build_pg_ms"] = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        base.ray.get(
            trainer.update_weights(global_steps=global_steps, mode=self.backend)
            + rollout.update_weights(global_steps=global_steps)
        )
        timing["param_sync/send_recv_update_ms"] = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        base.ray.get(
            trainer.execute_checkpoint_engine(["finalize"] * trainer.world_size)
            + rollout.execute_checkpoint_engine(["finalize"] * rollout.world_size)
        )
        timing["param_sync/finalize_ms"] = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        await self.resume_kv_cache_replicas()
        timing["param_sync/resume_kv_cache_ms"] = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        await self.resume_generation_replicas()
        timing["param_sync/resume_ms"] = (time.perf_counter() - start) * 1000

        timing["param_sync/total_ms"] = sum(timing.values())
        self.last_update_weights_timing = timing
        print(f"CheckpointEngineManager.update_weights timing: {timing}", flush=True)

    manager_cls.__init__ = patched_init
    manager_cls.update_weights = base.auto_await(timed_update_weights)
    manager_cls._ascend_benchmark_timing_patch = True


def _patch_bucketed_weight_transfer_stats() -> None:
    """Emit coarse sender/receiver stats while keeping the original transfer logic."""
    module = importlib.import_module("verl.workers.rollout.vllm_rollout.bucketed_weight_transfer")
    sender_cls = module.BucketedWeightSender
    receiver_cls = module.BucketedWeightReceiver
    if getattr(sender_cls, "_ascend_benchmark_stats_patch", False):
        return

    sender_original = sender_cls.async_send_weights
    receiver_original = receiver_cls.receive_weights

    async def sender_async_send_weights(self, weights):
        self.stats = _new_transfer_stats("shm" if self.use_shm else "ipc")
        original_init_buffer = self._init_buffer
        original_direct_send = self._direct_send_large_weight

        def timed_init_buffer():
            original_init_buffer()
            if self.buffer is not None:
                self.stats["bucket_bytes"] = max(self.stats["bucket_bytes"], getattr(self.buffer, "nbytes", 0))

        def counted_direct_send(name, weight):
            self.stats["tensor_count"] += 1
            self.stats["bucket_count"] += 1
            self.stats["bucket_bytes"] += getattr(weight, "nbytes", 0)
            return original_direct_send(name, weight)

        self._init_buffer = timed_init_buffer
        self._direct_send_large_weight = counted_direct_send
        start = time.perf_counter()
        try:
            return await sender_original(self, _counting_weight_iterator(weights, self.stats))
        finally:
            self.stats["sync_ms"] += (time.perf_counter() - start) * 1000
            module.logger.info("BucketedWeightSender stats: %s", self.stats)
            self._init_buffer = original_init_buffer
            self._direct_send_large_weight = original_direct_send

    def receiver_receive_weights(self, on_bucket_received):
        self.stats = _new_transfer_stats("shm" if self.use_shm else "ipc")

        def counted_on_bucket_received(items):
            self.stats["bucket_count"] += 1
            for _, tensor in items:
                self.stats["tensor_count"] += 1
                self.stats["bucket_bytes"] += getattr(tensor, "nbytes", 0)
            return on_bucket_received(items)

        start = time.perf_counter()
        try:
            return receiver_original(self, counted_on_bucket_received)
        finally:
            self.stats["sync_ms"] += (time.perf_counter() - start) * 1000
            module.logger.info("BucketedWeightReceiver stats: %s", self.stats)

    sender_cls.async_send_weights = sender_async_send_weights
    receiver_cls.receive_weights = receiver_receive_weights
    sender_cls._ascend_benchmark_stats_patch = True
    receiver_cls._ascend_benchmark_stats_patch = True


def _new_transfer_stats(path: str) -> dict[str, int | float | str]:
    return {
        "path": path,
        "bucket_count": 0,
        "bucket_bytes": 0,
        "tensor_count": 0,
        "sender_copy_ms": 0.0,
        "metadata_send_ms": 0.0,
        "metadata_recv_ms": 0.0,
        "clone_or_to_device_ms": 0.0,
        "sync_ms": 0.0,
    }


async def _counting_weight_iterator(weights, stats: dict[str, Any]):
    from verl.workers.rollout.utils import ensure_async_iterator

    async for name, weight in ensure_async_iterator(weights):
        stats["tensor_count"] += 1
        stats["bucket_bytes"] += getattr(weight, "nbytes", 0)
        yield name, weight


def _patch_fully_async_message_queue_batching() -> None:
    try:
        queue_module = importlib.import_module("verl.experimental.fully_async_policy.message_queue")
        trainer_module = importlib.import_module("verl.experimental.fully_async_policy.fully_async_trainer")
    except Exception:
        return

    _patch_message_queue_actor(queue_module)
    _patch_message_queue_client(queue_module)
    _patch_fully_async_trainer_get_samples(trainer_module)


def _patch_message_queue_actor(queue_module) -> None:
    if getattr(queue_module, "_ascend_benchmark_queue_patch", False):
        return

    ray = queue_module.ray
    original_actor = queue_module.MessageQueue
    metadata = getattr(original_actor, "__ray_metadata__", None)
    original_cls = getattr(metadata, "modified_class", None)
    if original_cls is None:
        return

    class PatchedMessageQueue(original_cls):
        async def get_samples(self, max_n: int, timeout_ms: int | None = None):
            if max_n <= 0:
                raise ValueError(f"max_n must be positive, got {max_n}")
            async with self._lock:
                while len(self.queue) == 0 and self.running:
                    await self._consumer_condition.wait()
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

    queue_module.MessageQueue = ray.remote(num_cpus=2, max_concurrency=20)(PatchedMessageQueue)
    queue_module._ascend_benchmark_queue_patch = True


def _patch_message_queue_client(queue_module) -> None:
    client_cls = queue_module.MessageQueueClient
    if getattr(client_cls, "_ascend_benchmark_queue_patch", False):
        return

    async def get_samples(self, max_n: int, timeout_ms: int | None = None):
        future = self.queue_actor.get_samples.remote(max_n, timeout_ms)
        return await queue_module.asyncio.wrap_future(future.future())

    def get_samples_sync(self, max_n: int, timeout_ms: int | None = None):
        return queue_module.ray.get(self.queue_actor.get_samples.remote(max_n, timeout_ms))

    client_cls.get_samples = get_samples
    client_cls.get_samples_sync = get_samples_sync
    client_cls._ascend_benchmark_queue_patch = True


def _patch_fully_async_trainer_get_samples(trainer_module) -> None:
    trainer_cls = trainer_module.FullyAsyncTrainer
    metadata = getattr(trainer_cls, "__ray_metadata__", None)
    target_cls = getattr(metadata, "modified_class", trainer_cls)
    if getattr(target_cls, "_ascend_benchmark_get_samples_patch", False):
        return

    async def get_samples_from_queue(self):
        print(f"[FullyAsyncTrainer] Requesting {self.required_samples} samples from queue", flush=True)
        consumer_start = time.time()
        queue_samples = []
        queue_len = 0
        queue_get_rpc_count = 0
        while len(queue_samples) < self.required_samples:
            batch_size = self.required_samples - len(queue_samples)
            result = await self.message_queue_client.get_samples(batch_size)
            queue_get_rpc_count += 1
            if result is None:
                print(
                    f"[FullyAsyncTrainer] Detected termination signal (None), stopping sample collection. "
                    f"Collected {len(queue_samples)}/{self.required_samples} samples"
                )
                break

            samples, queue_len = result
            if not samples:
                continue
            for sample in samples:
                if sample is None:
                    print(
                        f"[FullyAsyncTrainer] Detected termination signal (None), stopping sample collection. "
                        f"Collected {len(queue_samples)}/{self.required_samples} samples"
                    )
                    break
                queue_samples.append(sample)
            if samples[-1] is None:
                break

            if len(queue_samples) % 64 == 0:
                print(
                    f"[FullyAsyncTrainer] Collected {len(queue_samples)}/{self.required_samples} samples. "
                    f"mq_len: {queue_len}"
                )

        consumer_end = time.time()
        if not queue_samples or len(queue_samples) < self.required_samples:
            print("[FullyAsyncTrainer] not enough samples collected after loop")
            return None, None

        total_wait_time = consumer_end - consumer_start
        print(
            f"[FullyAsyncTrainer] Loop collection completed: {len(queue_samples)}/{self.required_samples} samples, "
            f"total wait time: {total_wait_time:.2f} seconds. "
            f"mq_len: {queue_len}"
        )

        load_start = time.time()
        queue_samples = [trainer_module.ray.cloudpickle.loads(x) for x in queue_samples]
        cloudpickle_load_time = time.time() - load_start
        if self.config.trainer.balance_batch:
            batch = trainer_module.assemble_batch_from_rollout_samples(
                queue_samples, self.tokenizer, self.config, self._balance_batch
            )
        else:
            batch = trainer_module.assemble_batch_from_rollout_samples(queue_samples, self.tokenizer, self.config, None)

        batch.meta_info["fully_async/total_wait_time"] = total_wait_time
        batch.meta_info["fully_async/message_queue_get_rpc_count"] = queue_get_rpc_count
        batch.meta_info["fully_async/cloudpickle_load_time"] = cloudpickle_load_time
        return 0, batch

    target_cls._get_samples_from_queue = get_samples_from_queue
    target_cls._ascend_benchmark_get_samples_patch = True


if os.getenv("VERL_ASCEND_BENCHMARK_MONKEY_PATCH") == "1":
    apply_all()
