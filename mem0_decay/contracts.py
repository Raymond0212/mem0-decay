"""Shared Python API types and the Mem0 SDK boundary."""

import asyncio
import inspect
import math
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Protocol

DateInput = str | int | float | date | datetime
MS_PER_HOUR = 3_600_000
MS_PER_DAY = 24 * MS_PER_HOUR
IDENTITY_FIELDS = ("user_id", "agent_id", "run_id")


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def unit_interval(value, name):
    if not finite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


def nonempty(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def positive_int(value, name, *, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def now_ms():
    return time.time_ns() // 1_000_000


async def call(method, *args, **kwargs):
    """Run async adapters directly and blocking Python SDK calls off the event loop."""
    if inspect.iscoroutinefunction(method):
        return await method(*args, **kwargs)
    result = await asyncio.to_thread(method, *args, **kwargs)
    return await result if inspect.isawaitable(result) else result


class Mem0Memory(Protocol):
    def add(self, messages: Any, **kwargs: Any) -> Any: ...
    def search(self, query: str, **kwargs: Any) -> Any: ...
    def get(self, memory_id: str) -> Any: ...
    def delete(self, memory_id: str) -> Any: ...


@dataclass(frozen=True, slots=True)
class ResolvedScope:
    tenant: str
    user_id: str | None = None
    agent_id: str | None = None
    run_id: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "tenant", nonempty(self.tenant, "tenant"))
        for key, value in self.identity().items():
            object.__setattr__(self, key, nonempty(value, key))
        if not self.identity():
            raise ValueError(
                "scope requires at least one of user_id, agent_id, or run_id"
            )

    def identity(self) -> dict[str, str]:
        return {
            key: getattr(self, key)
            for key in IDENTITY_FIELDS
            if getattr(self, key) is not None
        }


@dataclass(frozen=True, slots=True)
class TenantBinding:
    mode: str
    tenant: str | None = None
    field: str = "decay_tenant"

    def __post_init__(self):
        if self.mode == "collection":
            object.__setattr__(self, "tenant", nonempty(self.tenant, "tenant"))
        elif self.mode == "metadata":
            nonempty(self.field, "field")
            if self.field in IDENTITY_FIELDS or (
                self.field.startswith("decay_") and self.field != "decay_tenant"
            ):
                raise ValueError(
                    "tenant field collides with identity or decay metadata"
                )
        else:
            raise ValueError("tenant binding mode must be collection or metadata")


def validate_scope(scope, binding=None):
    if not isinstance(scope, ResolvedScope):
        raise TypeError("scope must be a ResolvedScope")
    if binding and binding.mode == "collection" and scope.tenant != binding.tenant:
        raise ValueError(
            f"Scope tenant {scope.tenant!r} does not match the collection tenant"
        )
    return scope


@dataclass(frozen=True, slots=True)
class TimeSpan:
    start: int | float
    end: int | float


@dataclass(frozen=True, slots=True)
class RangeQuery:
    start: DateInput
    end: DateInput
    strict: bool = True
    half_life_days: float | None = None
    curve: Any = None
    time_zone: str | None = None


@dataclass(frozen=True, slots=True)
class AnchorQuery:
    around: DateInput
    half_life_days: float | None = None
    curve: Any = None
    time_zone: str | None = None


@dataclass(frozen=True, slots=True)
class DecaySearchItem:
    id: str
    memory: str
    store_id: str
    base_score: float
    adjusted_score: float
    time_weight: float
    time_span: TimeSpan | None
    tier: int | None
    policy_id: str | None
    ingested_at: int | float | None
    expires_at: int | float | None
    metadata: dict[str, Any]
    created_at: str | None = None
    updated_at: str | None = None
    native_result: dict[str, Any] = field(
        default_factory=dict, repr=False, compare=False
    )


@dataclass(frozen=True, slots=True)
class DecaySearchResult:
    results: list[DecaySearchItem]
    reference_now_ms: int | float
    candidate_count: int
    mode: str
