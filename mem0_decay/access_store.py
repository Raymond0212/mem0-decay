"""Optional usage sidecar for recency ranking."""

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from .contracts import MS_PER_DAY, call, finite, nonempty
from .metadata import stable_json

DEFAULT_MAX_AGE_REDUCTION_DAYS = 14
DEFAULT_ACCESS_READ_TIMEOUT_MS = 100


@dataclass(frozen=True, slots=True)
class AccessKey:
    tenant: str
    store_id: str
    memory_id: str


@dataclass(frozen=True, slots=True)
class AccessRecord:
    last_used_at: int
    access_count: int


class AccessStore(Protocol):
    async def get_many(self, keys: list[AccessKey]) -> Mapping[str, AccessRecord]: ...
    async def record_use(
        self, key: AccessKey, event_id: str, used_at_ms: int
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class AccessSidecar:
    store: AccessStore
    max_age_reduction_days: float = DEFAULT_MAX_AGE_REDUCTION_DAYS
    read_timeout_ms: float = DEFAULT_ACCESS_READ_TIMEOUT_MS


def serialize_access_key(key):
    return stable_json([key.tenant, key.store_id, key.memory_id])


class InMemoryAccessStore:
    def __init__(self):
        self._records = {}
        self._events = set()
        self._closed = False

    async def get_many(self, keys):
        self._assert_open()
        serialized = map(serialize_access_key, keys)
        return {key: self._records[key] for key in serialized if key in self._records}

    async def record_use(self, key, event_id, used_at_ms):
        self._assert_open()
        event_id = nonempty(event_id, "event_id")
        if not finite(used_at_ms):
            raise ValueError("used_at_ms must be finite")
        key = serialize_access_key(key)
        if (key, event_id) in self._events:
            return
        self._events.add((key, event_id))
        previous = self._records.get(key)
        self._records[key] = (
            AccessRecord(
                max(previous.last_used_at, int(used_at_ms)), previous.access_count + 1
            )
            if previous
            else AccessRecord(int(used_at_ms), 1)
        )

    async def close(self):
        self._closed = True

    def _assert_open(self):
        if self._closed:
            raise RuntimeError("AccessStore is closed")


def compute_effective_age_days(
    raw_age_days,
    reference_now_ms,
    usage=None,
    max_age_reduction_days=DEFAULT_MAX_AGE_REDUCTION_DAYS,
):
    if (
        not usage
        or not finite(usage.last_used_at)
        or max_age_reduction_days <= 0
        or raw_age_days <= 0
    ):
        return max(0, raw_age_days)
    since_use = max(0, (reference_now_ms - usage.last_used_at) / MS_PER_DAY)
    return raw_age_days - min(max_age_reduction_days, max(0, raw_age_days - since_use))


async def read_access_batch_safe(sidecar, keys):
    if sidecar is None or not keys:
        return {}
    try:
        result = await asyncio.wait_for(
            call(sidecar.store.get_many, keys), sidecar.read_timeout_ms / 1000
        )
        return result if isinstance(result, Mapping) else {}
    except Exception:
        return {}
