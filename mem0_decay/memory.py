"""Async wrapper for Python Mem0 OSS Memory and AsyncMemory."""

from .access_store import AccessKey, AccessSidecar
from .compaction import InMemoryCompactionWorkStore
from .contracts import (
    DateInput,
    Mem0Memory,
    ResolvedScope,
    TenantBinding,
    call,
    finite,
    nonempty,
    positive_int,
    unit_interval,
    validate_scope,
)
from .contracts import now_ms as system_now_ms
from .coordinator import InProcessWriteCoordinator
from .curves import exponential, resolve_decay_curve
from .metadata import (
    EXPIRES_AT,
    INGESTED_AT,
    POLICY,
    TENANT,
    TIER,
    TIME_END,
    TIME_START,
    assert_no_reserved_metadata_keys,
    parse_timestamp_ms,
)
from .policy import DEFAULT_POLICY_IDS, default_policy_registry
from .search import (
    DEFAULT_ALPHA,
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_OVERFETCH_FACTOR,
    DEFAULT_QUERY_HALF_LIFE_DAYS,
    execute_decay_search,
)
from .sweep import execute_sweep


class DecayMemory:
    def __init__(
        self,
        store_id: str,
        memory: Mem0Memory,
        *,
        now_ms=None,
        policy_resolver=None,
        write_coordinator=None,
        default_policy_id=DEFAULT_POLICY_IDS["soft"],
        tenant_binding: TenantBinding | None = None,
        default_time_zone="UTC",
        overfetch_factor=DEFAULT_OVERFETCH_FACTOR,
        max_candidates=DEFAULT_MAX_CANDIDATES,
        default_query_half_life_days=None,
        default_query_curve=None,
        curve_registry=None,
        alpha=DEFAULT_ALPHA,
        qdrant_reader=None,
        scope_sweep_policy=None,
        archive_sink=None,
        summarizer=None,
        compaction_work_store=None,
        access_sidecar: AccessSidecar | None = None,
    ):
        self.store_id = nonempty(store_id, "store_id")
        if memory is None:
            raise TypeError("memory is required")
        if default_query_curve is not None and default_query_half_life_days is not None:
            raise ValueError(
                "provide either default_query_curve or default_query_half_life_days"
            )
        self.memory = memory
        self.clock = system_now_ms if now_ms is None else now_ms
        self.policy_resolver = (
            default_policy_registry if policy_resolver is None else policy_resolver
        )
        self.write_coordinator = (
            InProcessWriteCoordinator()
            if write_coordinator is None
            else write_coordinator
        )
        self.default_policy_id = default_policy_id
        self.tenant_binding = tenant_binding
        self.metadata_binding = (
            tenant_binding is not None and tenant_binding.mode == "metadata"
        )
        self.tenant_field = tenant_binding.field if self.metadata_binding else TENANT
        self.default_time_zone = default_time_zone
        self.overfetch_factor = positive_int(overfetch_factor, "overfetch_factor")
        self.max_candidates = positive_int(max_candidates, "max_candidates")
        self.curve_registry = curve_registry
        if default_query_curve is not None:
            self.default_query_curve = resolve_decay_curve(
                default_query_curve, curve_registry
            )
        else:
            self.default_query_curve = exponential(
                DEFAULT_QUERY_HALF_LIFE_DAYS
                if default_query_half_life_days is None
                else default_query_half_life_days
            )
        self.alpha = unit_interval(alpha, "alpha")
        self.qdrant_reader = qdrant_reader
        self.scope_sweep_policy = scope_sweep_policy
        self.archive_sink = archive_sink
        self.summarizer = summarizer
        self.compaction_work_store = (
            InMemoryCompactionWorkStore()
            if compaction_work_store is None
            else compaction_work_store
        )
        self.access_sidecar = access_sidecar

    async def add(
        self,
        messages,
        *,
        scope: ResolvedScope,
        event_time: DateInput | None = None,
        event_span: tuple[DateInput, DateInput] | None = None,
        policy_id=None,
        expires_at: DateInput | None = None,
        metadata=None,
        infer=True,
        expiration_date=None,
    ):
        scope = validate_scope(scope, self.tenant_binding)
        assert_no_reserved_metadata_keys(metadata)
        if event_time is not None and event_span is not None:
            raise ValueError("provide either event_time or event_span")
        policy = self.policy_resolver.resolve_or_raise(
            self.default_policy_id if policy_id is None else policy_id
        )
        now = self.clock()
        if not finite(now):
            raise ValueError("clock must return a finite epoch millisecond timestamp")
        if event_span is not None:
            start, end = (
                parse_timestamp_ms(value, time_zone=self.default_time_zone)
                for value in event_span
            )
            if start > end:
                raise ValueError("event_span start must be <= end")
        elif event_time is not None:
            start = end = parse_timestamp_ms(
                event_time, time_zone=self.default_time_zone
            )
        else:
            start = end = now

        stamped = {
            **(metadata or {}),
            TIME_START: start,
            TIME_END: end,
            INGESTED_AT: now,
            POLICY: policy.id,
            TIER: 0,
            TENANT: scope.tenant,
            self.tenant_field: scope.tenant,
        }
        if expires_at is not None:
            stamped[EXPIRES_AT] = parse_timestamp_ms(
                expires_at, time_zone=self.default_time_zone
            )
        elif policy.default_ttl_ms is not None:
            stamped[EXPIRES_AT] = now + policy.default_ttl_ms
        extra = (
            {"expiration_date": expiration_date} if expiration_date is not None else {}
        )
        return await self.write_coordinator.run(
            self.store_id,
            lambda: call(
                self.memory.add,
                messages,
                **scope.identity(),
                metadata=stamped,
                infer=infer,
                **extra,
            ),
        )

    async def search(self, query, *, scope: ResolvedScope, **options):
        return await execute_decay_search(
            self, query, scope=validate_scope(scope, self.tenant_binding), **options
        )

    async def sweep(self, *, scope: ResolvedScope, **options):
        return await execute_sweep(
            self, scope=validate_scope(scope, self.tenant_binding), **options
        )

    async def record_use(
        self, scope: ResolvedScope, memory_id, event_id, used_at_ms=None
    ):
        scope = validate_scope(scope, self.tenant_binding)
        memory_id = nonempty(memory_id, "memory_id")
        if self.access_sidecar is not None:
            key = AccessKey(scope.tenant, self.store_id, memory_id)
            timestamp = self.clock() if used_at_ms is None else used_at_ms
            await call(self.access_sidecar.store.record_use, key, event_id, timestamp)


def create_decay_memory(store_id, memory, **options):
    return DecayMemory(store_id, memory, **options)
