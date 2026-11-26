"""Scoped maintenance with dry-run previews and a shared mutation budget."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import inf

from .compaction import (
    CompactionCandidate,
    CompactionConfig,
    CompactionExecutor,
    CompactionWorkRecord,
    plan_compaction_groups,
)
from .contracts import MS_PER_DAY, ResolvedScope, call, finite, positive_int
from .metadata import (
    EXPIRES_AT,
    INGESTED_AT,
    POLICY,
    TIER,
    TIME_END,
    build_archive_key,
    compute_revision_fingerprint,
    item_payload,
    parse_managed_metadata,
    payload_matches_scope,
    scope_from_payload,
)
from .qdrant_reader import QdrantMemoryRecord, ScanCursor

DEFAULT_SWEEP_PAGE_SIZE = 100
DEFAULT_MAX_MUTATIONS_PER_RUN = 50
_SKIP_REASONS = (
    "unmanaged",
    "malformed_metadata",
    "unknown_policy",
    "exempt_policy",
    "ranking_only",
    "grace_period",
    "above_threshold",
    "tier_exempt",
    "action_disabled",
    "budget_exhausted",
    "revision_conflict",
    "scope_mismatch",
    "unclustered",
)


@dataclass(frozen=True, slots=True)
class ScopeSweepPolicy:
    action: str = "none"
    expiry_action: str | None = None
    compaction: CompactionConfig = field(default_factory=CompactionConfig)

    def __post_init__(self):
        if self.action not in ("none", "archive", "delete", "compact"):
            raise ValueError("unknown sweep action")
        if self.expiry_action not in (None, "archive", "delete"):
            raise ValueError("expiry_action must be archive or delete")


@dataclass(frozen=True, slots=True)
class PlannedMutation:
    action: str
    reason: str
    source_ids: list[str]
    estimated_mutations: int
    compaction_key: str | None = None


@dataclass(frozen=True, slots=True)
class SweepError:
    stage: str
    message: str
    source_id: str | None = None
    compaction_key: str | None = None


@dataclass(slots=True)
class SweepReport:
    store_id: str
    scope: ResolvedScope
    dry_run: bool
    reference_now_ms: float
    scanned: int = 0
    eligible: int = 0
    skipped: dict[str, int] = field(
        default_factory=lambda: dict.fromkeys(_SKIP_REASONS, 0)
    )
    planned_mutations: int = 0
    applied_mutations: int = 0
    planned_actions: list[PlannedMutation] = field(default_factory=list)
    errors: list[SweepError] = field(default_factory=list)
    unresolved_compactions: list[CompactionWorkRecord] = field(default_factory=list)
    next_cursor: ScanCursor | None = None
    has_more: bool = False
    pending_work: bool = False
    partial: bool = False

    def plan(self, action):
        self.planned_actions.append(action)
        self.planned_mutations += action.estimated_mutations


@dataclass(frozen=True, slots=True)
class _Eligible:
    source: CompactionCandidate
    action: str
    trigger: str


class _SweepRun:
    def __init__(self, store, scope, policy, report, budget):
        self.store = store
        self.scope = scope
        self.policy = policy
        self.report = report
        self.remaining = budget

    def error(self, stage, exc, **details):
        self.report.errors.append(SweepError(stage, str(exc), **details))
        self.report.pending_work = True

    def unresolved(self, work):
        self.report.unresolved_compactions.append(work)
        self.report.pending_work = True

    def mutated(self):
        self.remaining -= 1
        self.report.applied_mutations += 1

    def _matches(self, payload, scope, *, exact=False):
        return payload_matches_scope(
            payload,
            scope,
            exact=exact,
            tenant_field=self.store.tenant_field,
            require_tenant=self.store.metadata_binding,
        )

    def evaluate(self, record, scope, now, *, exact=False):
        if not self._matches(record.payload, scope, exact=exact):
            return None, "scope_mismatch"
        status, metadata = parse_managed_metadata(record.payload)
        if status != "valid":
            return None, "unmanaged" if status == "unmanaged" else "malformed_metadata"
        policy = self.store.policy_resolver.resolve(metadata[POLICY])
        if policy is None:
            return None, "unknown_policy"
        if policy.exempt_from_sweep:
            return None, "exempt_policy"
        try:
            record_scope = scope_from_payload(record.payload, scope.tenant)
        except ValueError:
            return None, "scope_mismatch"

        action = self.policy.action
        trigger = "decay"
        expires = metadata.get(EXPIRES_AT)
        if expires is not None and expires <= now:
            action = self.policy.expiry_action or (
                action if action in ("archive", "delete") else None
            )
            if not action:
                return None, "action_disabled"
            trigger = "expired"
        else:
            if action == "none":
                return None, "action_disabled"
            if action == "compact" and metadata[TIER] != 0:
                return None, "tier_exempt"
            if not policy.sweep_eligible:
                return None, "ranking_only"
            if now - metadata[INGESTED_AT] < policy.ingestion_grace_ms:
                return None, "grace_period"
            if not policy.criterion(
                max(0, (now - metadata[TIME_END]) / MS_PER_DAY)
            ).below_threshold:
                return None, "above_threshold"

        revision = compute_revision_fingerprint(record.memory, record_scope, metadata)
        return _Eligible(
            CompactionCandidate(record, record_scope, metadata, policy, revision),
            action,
            trigger,
        ), None

    def check_compaction_source(self, item, ref, scope):
        if item is None:
            return "conflict"
        payload = item_payload(item)
        if not self._matches(payload, scope, exact=True):
            return "conflict"
        status, metadata = parse_managed_metadata(payload)
        if status != "valid":
            return "conflict"
        policy = self.store.policy_resolver.resolve(metadata[POLICY])
        if policy is None or policy.exempt_from_sweep:
            return "conflict"
        now = self.report.reference_now_ms
        if metadata.get(EXPIRES_AT, inf) <= max(now, self.store.clock()):
            return "expired"
        if (
            not policy.sweep_eligible
            or metadata[TIER] != 0
            or now - metadata[INGESTED_AT] < policy.ingestion_grace_ms
        ):
            return "conflict"
        if (
            compute_revision_fingerprint(item["memory"], scope, metadata)
            != ref["revision"]
        ):
            return "conflict"
        age = max(0, (now - metadata[TIME_END]) / MS_PER_DAY)
        return "eligible" if policy.criterion(age).below_threshold else "conflict"

    async def handle_expired(self, source_id):
        if self.policy.expiry_action is None or self.remaining <= 0:
            self.report.skipped["action_disabled"] += 1
            return
        item = await call(self.store.memory.get, source_id)
        if item is None:
            return
        now = max(self.report.reference_now_ms, self.store.clock())
        record = QdrantMemoryRecord(source_id, item["memory"], item_payload(item))
        eligible, _ = self.evaluate(record, self.scope, now)
        if eligible and eligible.trigger == "expired":
            await self.apply_archive_or_delete(eligible, now)

    async def apply_archive_or_delete(self, item, timestamp):
        source = item.source
        source_id = source.record.id
        if item.action == "archive":
            try:
                sink = self.store.archive_sink
                if sink is None:
                    raise RuntimeError("Archive action requires an archive sink")
                key = build_archive_key(self.store.store_id, source_id, source.revision)
                archive = getattr(sink, "archive", sink)
                ack = await call(
                    archive,
                    {
                        "key": key,
                        "store_id": self.store.store_id,
                        "scope": source.scope,
                        "source_id": source_id,
                        "revision": source.revision,
                        "memory": source.record.memory,
                        "metadata": source.record.payload,
                        "archived_at_ms": self.store.clock(),
                    },
                )
                if ack is not None and (
                    ack.get("confirmed") is not True or ack.get("key") != key
                ):
                    raise RuntimeError(f"Archive was not confirmed for {key!r}")
            except Exception as exc:
                self.error("archive", exc, source_id=source_id)
                return

        async def rechecked_delete():
            current = await call(self.store.memory.get, source_id)
            if current is None:
                return
            record = QdrantMemoryRecord(
                source_id, current["memory"], item_payload(current)
            )
            eligible, _ = self.evaluate(record, source.scope, timestamp, exact=True)
            if (
                not eligible
                or eligible.source.revision != source.revision
                or eligible.action != item.action
            ):
                self.report.skipped["revision_conflict"] += 1
                return
            await call(self.store.memory.delete, source_id)
            self.mutated()

        try:
            await self.store.write_coordinator.run(
                self.store.store_id, rechecked_delete
            )
        except Exception as exc:
            self.error("delete", exc, source_id=source_id)


async def execute_sweep(
    store,
    *,
    scope,
    cursor=None,
    page_size=DEFAULT_SWEEP_PAGE_SIZE,
    max_scanned=None,
    max_mutations_per_run=DEFAULT_MAX_MUTATIONS_PER_RUN,
    dry_run=True,
):
    if store.qdrant_reader is None:
        raise RuntimeError("sweep requires a Qdrant reader")
    positive_int(page_size, "page_size")
    max_scanned = (
        inf if max_scanned is None else positive_int(max_scanned, "max_scanned")
    )
    positive_int(max_mutations_per_run, "max_mutations_per_run", minimum=0)
    timestamp = store.clock()
    if not finite(timestamp):
        raise ValueError("clock must return a finite epoch millisecond timestamp")
    policy = store.scope_sweep_policy
    if callable(policy):
        policy = await call(policy, scope, store.store_id)
    if isinstance(policy, Mapping):
        policy = ScopeSweepPolicy(**policy)
    policy = ScopeSweepPolicy() if policy is None else policy
    report = SweepReport(store.store_id, scope, dry_run, timestamp)
    run = _SweepRun(store, scope, policy, report, max_mutations_per_run)
    compaction = CompactionExecutor(run)
    reserved = await compaction.resume()
    if reserved is None:
        report.has_more, report.next_cursor = True, cursor
        return report

    ordinary, compactable = [], []
    with_vector = policy.action == "compact"
    while report.scanned < max_scanned:
        limit = min(page_size, max_scanned - report.scanned)
        try:
            page = await call(
                store.qdrant_reader.scan,
                scope,
                cursor=cursor,
                limit=limit,
                with_vector=with_vector,
            )
            if page.next_cursor is not None and page.next_cursor == cursor:
                raise RuntimeError("Qdrant scan cursor did not advance")
        except Exception as exc:
            run.error("scan", exc)
            report.partial, report.has_more = True, True
            break
        for record in page.records:
            report.scanned += 1
            if record.id in reserved:
                continue
            candidate, reason = run.evaluate(record, scope, timestamp)
            if not candidate:
                report.skipped[reason] += 1
                continue
            report.eligible += 1
            if candidate.action == "compact":
                compactable.append(candidate.source)
            else:
                ordinary.append(candidate)
        cursor = page.next_cursor
        if cursor is None:
            break
        if report.scanned >= max_scanned:
            report.partial, report.has_more, report.pending_work = True, True, True
    report.next_cursor = cursor if report.has_more else None

    for item in ordinary:
        if run.remaining <= 0:
            report.skipped["budget_exhausted"] += 1
            report.pending_work = True
            continue
        report.plan(
            PlannedMutation(item.action, item.trigger, [item.source.record.id], 1)
        )
        if dry_run:
            run.remaining -= 1
        else:
            await run.apply_archive_or_delete(item, timestamp)

    if compactable:
        groups, unclustered = plan_compaction_groups(
            store.store_id, compactable, policy.compaction
        )
        report.skipped["unclustered"] = len(unclustered)
        for group in groups:
            count = len(group.sources) + 1
            if run.remaining < count:
                report.skipped["budget_exhausted"] += len(group.sources)
                report.pending_work = True
                continue
            report.plan(
                PlannedMutation(
                    "compact",
                    "decay",
                    [source.record.id for source in group.sources],
                    count,
                    group.key,
                )
            )
            if dry_run:
                run.remaining -= count
            else:
                await compaction.execute(group)
    return report
