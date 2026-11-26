"""Process-local single-writer coordination."""

import asyncio
from weakref import WeakValueDictionary


class InProcessWriteCoordinator:
    def __init__(self, *, serialize_all_stores=True):
        self._serialize_all = serialize_all_stores
        self._locks = WeakValueDictionary()

    async def run(self, store_id, operation):
        key = None if self._serialize_all else store_id
        lock = self._locks.get(key)
        if lock is None:
            lock = self._locks[key] = asyncio.Lock()
        async with lock:
            task = asyncio.ensure_future(operation())
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                # A blocking SDK operation can finish after cancellation. Keep
                # the writer lock until the operation has actually returned.
                while not task.done():
                    try:
                        await asyncio.shield(task)
                    except asyncio.CancelledError:
                        continue
                    except Exception:
                        break
                if not task.cancelled():
                    task.exception()
                raise


def create_in_process_write_coordinator(**options):
    return InProcessWriteCoordinator(**options)
