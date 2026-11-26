"""Deterministic compaction groups and resumable summary/source mutations."""

import math
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Protocol

from .contracts import MS_PER_DAY, ResolvedScope, TimeSpan, call, finite, positive_int
from .metadata import (
    COMPACTION_KEY,
    EXPIRES_AT,
    INGESTED_AT,
    POLICY,
    SOURCES,
    TENANT,
    TIER,
    TIME_END,
    TIME_START,
    derive_compaction_key,
    item_payload,
    js_order,
    parse_managed_metadata,
    payload_matches_scope,
    sorted_source_refs,
    stable_json,
)
from .policy import DEFAULT_POLICY_IDS, DecayPolicy
from .qdrant_reader import QdrantMemoryRecord

DEFAULT_COMPACTION_TIME_BUCKET_MS = 7 * MS_PER_DAY
DEFAULT_COMPACTION_SIMILARITY_THRESHOLD = 0.75
DEFAULT_COMPACTION_MIN_GROUP_SIZE = 2
DEFAULT_COMPACTION_MAX_GROUP_SIZE = 8


@dataclass(frozen=True, slots=True)
class CompactionConfig:
    time_bucket_ms: int = DEFAULT_COMPACTION_TIME_BUCKET_MS
    similarity_threshold: float = DEFAULT_COMPACTION_SIMILARITY_THRESHOLD
    min_group_size: int = DEFAULT_COMPACTION_MIN_GROUP_SIZE
    max_group_size: int = DEFAULT_COMPACTION_MAX_GROUP_SIZE
    target_tier_policy_id: str = DEFAULT_POLICY_IDS["tier1_summary"]

    def __post_init__(self):
        positive_int(self.time_bucket_ms, "time_bucket_ms")
        positive_int(self.min_group_size, "min_group_size", minimum=2)
        positive_int(self.max_group_size, "max_group_size", minimum=self.min_group_size)
        if (
            not finite(self.similarity_threshold)
            or not -1 <= self.similarity_threshold <= 1
        ):
            raise ValueError("similarity_threshold must be in [-1, 1]")


@dataclass(frozen=True, slots=True)
class CompactionCandidate:
    record: QdrantMemoryRecord
    scope: ResolvedScope
    metadata: dict
    policy: DecayPolicy
    revision: str

    def source_ref(self):
        return {"id": self.record.id, "revision": self.revision}


@dataclass(frozen=True, slots=True)
class PlannedCompactionGroup:
    key: str
    scope: ResolvedScope
    sources: list[CompactionCandidate]
    time_span: TimeSpan
    inherited_expires_at: int | float | None

    def refs(self):
        return sorted_source_refs([source.source_ref() for source in self.sources])


@dataclass(slots=True)
class CompactionWorkRecord:
    compaction_key: str
    store_id: str
    scope: ResolvedScope
    sources: list[dict[str, str]]
    created_at_ms: int | float
    updated_at_ms: int | float
    target_tier: int = 1
    completed_source_ids: list[str] = field(default_factory=list)
    summary_id: str | None = None
    status: str = "pending"
    error: str | None = None


class CompactionWorkStore(Protocol):
    async def list_pending(
        self, store_id: str, scope: ResolvedScope
    ) -> list[CompactionWorkRecord]: ...
    async def get(self, compaction_key: str) -> CompactionWorkRecord | None: ...
    async def save(self, record: CompactionWorkRecord) -> None: ...


class InMemoryCompactionWorkStore:
    def __init__(self):
        self._records = {}

    async def list_pending(self, store_id, scope):
        return [
            deepcopy(record)
            for record in self._records.values()
            if record.status != "completed"
            and record.store_id == store_id
            and record.scope.tenant == scope.tenant
            and all(
                record.scope.identity().get(key) == value
                for key, value in scope.identity().items()
            )
        ]

    async def get(self, compaction_key):
        return deepcopy(self._records.get(compaction_key))

    async def save(self, record):
        self._records[record.compaction_key] = deepcopy(record)


def cosine_similarity(a, b):
    if not a or len(a) != len(b) or not all(map(finite, a)) or not all(map(finite, b)):
        return None
    x, y = math.hypot(*a), math.hypot(*b)
    if not x or not y or not math.isfinite(x) or not math.isfinite(y):
        return None
    return max(
        -1.0, min(1.0, math.fsum((left / x) * (right / y) for left, right in zip(a, b)))
    )


def plan_compaction_groups(store_id, candidates, config=None):
    config = config or CompactionConfig()
    buckets = defaultdict(list)
    unclustered = []
    for candidate in candidates:
        vector = candidate.record.vector
        if candidate.metadata[TIER] != 0 or not vector:
            unclustered.append(candidate)
            continue
        scope = candidate.scope
        key = stable_json(
            [
                scope.tenant,
                store_id,
                scope.user_id,
                scope.agent_id,
                scope.run_id,
                candidate.metadata[TIER],
                math.floor(candidate.metadata[TIME_END] / config.time_bucket_ms),
                candidate.policy.retention_compatibility_key or candidate.policy.id,
                len(vector),
            ]
        )
        buckets[key].append(candidate)

    groups = []
    for key in sorted(buckets, key=js_order):
        remaining = sorted(
            buckets[key],
            key=lambda item: (
                item.metadata[TIME_START],
                item.metadata[TIME_END],
                js_order(item.record.id),
            ),
        )
        while remaining:
            cluster, unused = [remaining[0]], []
            for candidate in remaining[1:]:
                if len(cluster) < config.max_group_size and all(
                    (
                        score := cosine_similarity(
                            member.record.vector, candidate.record.vector
                        )
                    )
                    is not None
                    and score >= config.similarity_threshold
                    for member in cluster
                ):
                    cluster.append(candidate)
                else:
                    unused.append(candidate)
            remaining = unused
            if len(cluster) < config.min_group_size:
                unclustered.extend(cluster)
                continue
            scope = cluster[0].scope
            span = TimeSpan(
                min(item.metadata[TIME_START] for item in cluster),
                max(item.metadata[TIME_END] for item in cluster),
            )
            expires = min(
                (
                    item.metadata[EXPIRES_AT]
                    for item in cluster
                    if EXPIRES_AT in item.metadata
                ),
                default=None,
            )
            key = derive_compaction_key(
                store_id, scope, 1, [item.source_ref() for item in cluster]
            )
            groups.append(PlannedCompactionGroup(key, scope, cluster, span, expires))
    return groups, unclustered


def verify_summary_lineage(
    payload,
    scope,
    compaction_key,
    sources,
    *,
    tenant_field=TENANT,
    require_tenant=False,
):
    if not payload_matches_scope(
        payload,
        scope,
        exact=True,
        tenant_field=tenant_field,
        require_tenant=require_tenant,
    ):
        return False
    status, metadata = parse_managed_metadata(payload)
    return (
        status == "valid"
        and metadata[TIER] == 1
        and metadata.get(COMPACTION_KEY) == compaction_key
        and SOURCES in metadata
        and sorted_source_refs(metadata[SOURCES]) == sorted_source_refs(sources)
    )


class CompactionExecutor:
    def __init__(self, run):
        self.run = run
        self.store = run.store
        self.work_store = self.store.compaction_work_store

    async def _save(self, work):
        work.updated_at_ms = self.store.clock()
        if self.work_store is not None:
            await call(self.work_store.save, work)

    def _matches(self, payload, work):
        return verify_summary_lineage(
            payload,
            work.scope,
            work.compaction_key,
            work.sources,
            tenant_field=self.store.tenant_field,
            require_tenant=self.store.metadata_binding,
        )

    async def _existing_summary(self, work):
        summary = await call(
            self.store.qdrant_reader.find_summary, work.scope, work.compaction_key
        )
        if summary is not None and not self._matches(summary.payload, work):
            raise RuntimeError(
                "Existing summary lineage does not match source revisions"
            )
        return summary

    async def resume(self):
        if self.work_store is None:
            return set()
        try:
            pending = await call(
                self.work_store.list_pending, self.store.store_id, self.run.scope
            )
        except Exception as exc:
            self.run.error("resume", exc)
            self.run.report.partial = True
            return None
        reserved = set()
        for work in pending:
            reserved.update(source["id"] for source in work.sources)
            if self.run.report.dry_run:
                if work.status == "uncertain":
                    self.run.unresolved(work)
                else:
                    count = sum(
                        source["id"] not in work.completed_source_ids
                        for source in work.sources
                    )
                    self.run.report.planned_mutations += count
                    self.run.report.pending_work |= count > 0
                continue
            try:
                expired = await self.store.write_coordinator.run(
                    self.store.store_id, lambda: self._resume(work)
                )
                for source_id in expired:
                    await self.run.handle_expired(source_id)
            except Exception as exc:
                self.run.error("resume", exc, compaction_key=work.compaction_key)
                self.run.unresolved(work)
        return reserved

    async def _resume(self, work):
        current = await call(self.work_store.get, work.compaction_key)
        if current is not None and current.status == "completed":
            return []
        work = current or work
        summary = await self._existing_summary(work)
        if summary is None:
            work.status = "uncertain"
            await self._save(work)
            self.run.unresolved(work)
            return []
        work.summary_id = summary.id
        work.status = "confirmed"
        await self._save(work)
        return await self._retire(work)

    async def _retire(self, work):
        expired = []
        for source in work.sources:
            source_id = source["id"]
            if source_id in work.completed_source_ids:
                continue
            if self.run.remaining <= 0:
                self.run.report.pending_work = True
                break
            try:
                item = await call(self.store.memory.get, source_id)
                if item is not None:
                    item_status = self.run.check_compaction_source(
                        item, source, work.scope
                    )
                    if item_status == "eligible":
                        await call(self.store.memory.delete, source_id)
                        self.run.mutated()
                    elif item_status == "expired":
                        expired.append(source_id)
                    else:
                        self.run.report.skipped["revision_conflict"] += 1
                work.completed_source_ids.append(source_id)
                await self._save(work)
            except Exception as exc:
                self.run.error(
                    "compact",
                    exc,
                    source_id=source_id,
                    compaction_key=work.compaction_key,
                )
        if len(work.completed_source_ids) == len(work.sources):
            work.status = "completed"
            await self._save(work)
        else:
            self.run.report.pending_work = True
        return expired

    async def execute(self, group):
        if self.store.summarizer is None:
            self.run.error(
                "compact", "Compaction requires a summarizer", compaction_key=group.key
            )
            return
        try:
            summarize = getattr(
                self.store.summarizer, "summarize", self.store.summarizer
            )
            result = await call(
                summarize,
                {
                    "store_id": self.store.store_id,
                    "scope": group.scope,
                    "sources": [
                        {
                            **source.source_ref(),
                            "memory": source.record.memory,
                            "time_span": TimeSpan(
                                source.metadata[TIME_START], source.metadata[TIME_END]
                            ),
                            "metadata": source.record.payload,
                        }
                        for source in group.sources
                    ],
                },
            )
            text = result.get("summary") if isinstance(result, dict) else result
            if not isinstance(text, str) or not text.strip():
                raise ValueError("summarizer returned an empty summary")
            expired = await self.store.write_coordinator.run(
                self.store.store_id,
                lambda: self._commit(group, text.strip()),
            )
            for source_id in expired:
                await self.run.handle_expired(source_id)
        except Exception as exc:
            self.run.error("compact", exc, compaction_key=group.key)

    async def _commit(self, group, text):
        if self.work_store is not None:
            existing = await call(self.work_store.get, group.key)
            if existing is not None:
                self.run.report.pending_work |= existing.status != "completed"
                return []

        expired = []
        for source in group.sources:
            item = await call(self.store.memory.get, source.record.id)
            status = self.run.check_compaction_source(
                item, source.source_ref(), group.scope
            )
            if status == "expired":
                expired.append(source.record.id)
            elif status != "eligible":
                self.run.report.skipped["revision_conflict"] += 1
                return expired
        if expired:
            return expired

        now = self.store.clock()
        work = CompactionWorkRecord(
            group.key, self.store.store_id, group.scope, group.refs(), now, now
        )
        target = self.store.policy_resolver.resolve_or_raise(
            self.run.policy.compaction.target_tier_policy_id
        )
        await self._save(work)
        try:
            existing = await self._existing_summary(work)
            if existing is None:
                metadata = {
                    TIME_START: group.time_span.start,
                    TIME_END: group.time_span.end,
                    INGESTED_AT: now,
                    POLICY: target.id,
                    TIER: 1,
                    COMPACTION_KEY: group.key,
                    SOURCES: work.sources,
                    TENANT: group.scope.tenant,
                    self.store.tenant_field: group.scope.tenant,
                }
                if group.inherited_expires_at is not None:
                    metadata[EXPIRES_AT] = group.inherited_expires_at
                # An exception during add can mean the write landed. Reserve its
                # budget before issuing the call and reconcile via the work record.
                self.run.remaining -= 1
                result = await call(
                    self.store.memory.add,
                    text,
                    **group.scope.identity(),
                    metadata=metadata,
                    infer=False,
                )
                created = (result or {}).get("results") or []
                if not created or not created[0].get("id"):
                    raise RuntimeError("Summary add did not return a memory ID")
                work.summary_id = str(created[0]["id"])
                self.run.report.applied_mutations += 1
                read_back = await call(self.store.memory.get, work.summary_id)
                if read_back is None or not self._matches(
                    item_payload(read_back), work
                ):
                    raise RuntimeError("Written summary failed lineage verification")
            else:
                work.summary_id = existing.id
        except Exception as exc:
            work.status, work.error = "uncertain", str(exc)
            await self._save(work)
            self.run.error("compact", exc, compaction_key=group.key)
            self.run.unresolved(work)
            return []

        work.status = "confirmed"
        await self._save(work)
        return await self._retire(work)
