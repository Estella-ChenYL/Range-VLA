"""Bounded, single-threaded inference scheduler; no CUDA work on the event loop."""
import asyncio
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor


class DynamicBatcher:
    def __init__(self, policy, batch_size, wait_ms):
        if batch_size < 1 or wait_ms < 0:
            raise ValueError('batch_size must be positive and wait_ms nonnegative')
        self.policy, self.limit, self.wait = policy, batch_size, wait_ms / 1000
        self.queue = asyncio.Queue(maxsize=batch_size * 4)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='policy')
        self.task = None
        self.closed = False
        self.pending = set()

    async def infer(self, payload):
        if self.closed:
            raise RuntimeError('Inference scheduler closed')
        examples = payload.get('examples')
        if not isinstance(examples, list) or not 0 < len(examples) <= self.limit:
            raise ValueError(f'examples must contain 1..{self.limit} items')
        options = {k: v for k, v in payload.items() if k != 'examples'}
        # Strict serialization prevents unsafe equality of array-valued options.
        key = json.dumps(options, sort_keys=True, allow_nan=False)
        future = asyncio.get_running_loop().create_future()
        self.pending.add(future)
        try:
            try:
                self.queue.put_nowait((time.monotonic(), key, payload, future))
            except asyncio.QueueFull:
                raise RuntimeError("Inference request queue is full") from None
            if self.task is None:
                self.task = asyncio.create_task(self._run())
            return await future
        finally:
            self.pending.discard(future)

    async def _run(self):
        carry = None
        while not self.closed:
            first = carry if carry is not None else await self.queue.get()
            carry = None
            batch = [first]
            size = len(first[2]['examples'])
            deadline = first[0] + self.wait
            while size < self.limit and not self.closed:
                try:
                    item = self.queue.get_nowait()
                except asyncio.QueueEmpty:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        item = await asyncio.wait_for(self.queue.get(), remaining)
                    except asyncio.TimeoutError:
                        break
                if item[3].cancelled():
                    continue
                count = len(item[2]['examples'])
                if item[1] != first[1] or size + count > self.limit:
                    carry = item
                    break
                batch.append(item)
                size += count
            batch = [item for item in batch if not item[3].cancelled()]
            if not batch:
                continue
            payload = dict(batch[0][2])
            payload['examples'] = [ex for item in batch for ex in item[2]['examples']]
            started = time.monotonic()
            try:
                result = await asyncio.get_running_loop().run_in_executor(
                    self.executor, lambda: self._predict(payload))
                count = len(payload['examples'])
                if any(len(value) != count for value in result.values()):
                    raise ValueError('Policy output batch dimension does not match input')
                offset = 0
                for _, _, request, future in batch:
                    end = offset + len(request['examples'])
                    if not future.done():
                        future.set_result({k: v[offset:end] for k, v in result.items()})
                    offset = end
                logging.info('inference batch_size=%d batch_limit=%d inference_ms=%.2f',
                             count, self.limit, (time.monotonic() - started) * 1000)
            except Exception as error:
                logging.exception('Batched inference failed')
                for *_, future in batch:
                    if not future.done():
                        future.set_exception(error)

    def _predict(self, payload):
        result = self.policy.predict_action(**payload)
        log_cuda_peak()
        return result

    async def close(self):
        self.closed = True
        if self.task:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        for future in list(self.pending):
            if not future.done():
                future.set_exception(RuntimeError('Inference scheduler closed'))
        while not self.queue.empty():
            self.queue.get_nowait()
        # Shutdown is terminal: wait for any already-running CUDA forward.
        self.executor.shutdown(wait=True, cancel_futures=True)


def log_cuda_peak():
    torch = sys.modules.get('torch')
    if torch is not None and torch.cuda.is_available():
        logging.info('inference cuda_peak_allocated_bytes=%d cuda_peak_reserved_bytes=%d',
                     torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved())
