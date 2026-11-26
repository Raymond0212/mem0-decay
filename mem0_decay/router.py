"""An authorized multi-store router with a shared write coordinator."""

import asyncio
from dataclasses import dataclass, replace

from .contracts import (
    DecaySearchResult,
    ResolvedScope,
    call,
    finite,
    positive_int,
    unit_interval,
    validate_scope,
)
from .contracts import now_ms as system_now_ms
from .coordinator import InProcessWriteCoordinator
from .curves import resolve_decay_curve
from .search import DEFAULT_TOP_K, query_mode, rank_key


@dataclass(frozen=True, slots=True)
class RouterAuthorizationContext:
    scope: ResolvedScope
    operation: str
    store_id: str
    principal: str | None


class DecayRouter:
    def __init__(
        self,
        *,
        now_ms=None,
        write_coordinator=None,
        authorize=None,
        access_store=None,
        alpha=None,
        default_query_curve=None,
        curve_registry=None,
    ):
        self._clock = system_now_ms if now_ms is None else now_ms
        self.write_coordinator = (
            InProcessWriteCoordinator()
            if write_coordinator is None
            else write_coordinator
        )
        self._authorize = authorize
        self._access_store = access_store
        self._alpha = None if alpha is None else unit_interval(alpha, "alpha")
        self._curve_registry = curve_registry
        self._query_curve = (
            None
            if default_query_curve is None
            else resolve_decay_curve(default_query_curve, curve_registry)
        )
        self._stores = {}
        self._active_writer = None

    def register_store(self, store, *, active_writer=False):
        if store.store_id in self._stores:
            raise ValueError(f"Store {store.store_id!r} already registered")
        store.write_coordinator = self.write_coordinator
        self._stores[store.store_id] = store
        if active_writer or self._active_writer is None:
            self._active_writer = store.store_id

    def _get_store(self, store_id):
        if store_id not in self._stores:
            raise ValueError(f"Unknown store {store_id!r}")
        return self._stores[store_id]

    def set_active_write_store(self, store_id):
        self._get_store(store_id)
        self._active_writer = store_id

    def get_active_write_store_id(self):
        return self._active_writer

    def list_store_ids(self):
        return list(self._stores)

    async def _allowed(self, scope, operation, store_id, principal):
        if self._authorize is None:
            return True
        return bool(
            await call(
                self._authorize,
                RouterAuthorizationContext(scope, operation, store_id, principal),
            )
        )

    async def _require(self, scope, operation, store, principal):
        if not await self._allowed(scope, operation, store.store_id, principal):
            raise PermissionError(
                f"Unauthorized {operation} for tenant {scope.tenant!r} on {store.store_id!r}"
            )

    async def add(self, messages, *, scope, principal=None, **options):
        scope = validate_scope(scope)
        if self._active_writer is None:
            raise RuntimeError("no active write store")
        store = self._get_store(self._active_writer)
        await self._require(scope, "add", store, principal)
        return {
            **await store.add(messages, scope=scope, **options),
            "store_id": store.store_id,
        }

    async def search(
        self,
        query,
        *,
        scope,
        principal=None,
        store_ids=None,
        reference_now_ms=None,
        top_k=DEFAULT_TOP_K,
        alpha=None,
        curve=None,
        range=None,
        anchor=None,
        **options,
    ):
        scope = validate_scope(scope)
        mode = query_mode(range, anchor)
        positive_int(top_k, "top_k")
        now = self._clock() if reference_now_ms is None else reference_now_ms
        if not finite(now):
            raise ValueError("reference_now_ms must be finite")
        alpha = self._alpha if alpha is None else unit_interval(alpha, "alpha")
        if self._curve_registry is not None and curve is not None:
            curve = resolve_decay_curve(curve, self._curve_registry)

        def query_options(value, soft):
            if value is None:
                return None
            if value.curve is not None and self._curve_registry is not None:
                return replace(
                    value, curve=resolve_decay_curve(value.curve, self._curve_registry)
                )
            if (
                soft
                and value.curve is None
                and value.half_life_days is None
                and curve is None
            ):
                return replace(value, curve=self._query_curve)
            return value

        range = query_options(range, mode == "soft_range")
        anchor = query_options(anchor, True)
        stores = []
        for store_id in self._stores if store_ids is None else store_ids:
            store = self._get_store(store_id)
            if await self._allowed(scope, "search", store_id, principal):
                stores.append(store)
        responses = await asyncio.gather(
            *(
                store.search(
                    query,
                    scope=scope,
                    top_k=top_k,
                    reference_now_ms=now,
                    alpha=alpha,
                    curve=curve,
                    range=range,
                    anchor=anchor,
                    **options,
                )
                for store in stores
            )
        )
        items = sorted(
            (item for result in responses for item in result.results), key=rank_key
        )
        return DecaySearchResult(
            items[:top_k],
            now,
            sum(result.candidate_count for result in responses),
            mode,
        )

    async def sweep(self, store_id, *, scope, principal=None, **options):
        scope = validate_scope(scope)
        store = self._get_store(store_id)
        await self._require(scope, "sweep", store, principal)
        return await store.sweep(scope=scope, **options)

    async def record_use(
        self, store_id, scope, memory_id, event_id, *, principal=None, used_at_ms=None
    ):
        scope = validate_scope(scope)
        store = self._get_store(store_id)
        await self._require(scope, "record_use", store, principal)
        await store.record_use(scope, memory_id, event_id, used_at_ms)

    async def shutdown(self):
        close = getattr(self._access_store, "close", None)
        if close is not None:
            await call(close)


def create_decay_router(**options):
    return DecayRouter(**options)
