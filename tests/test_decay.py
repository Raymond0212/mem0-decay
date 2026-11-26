import unittest

from mem0_decay import (
    AccessKey,
    AccessSidecar,
    AnchorQuery,
    CompactionConfig,
    DecayMemory,
    DecayRouter,
    InMemoryAccessStore,
    InMemoryCompactionWorkStore,
    PolicyRegistry,
    RangeQuery,
    ResolvedScope,
    ScopeSweepPolicy,
    TenantBinding,
    create_decay_criterion,
    exponential,
    sigmoid,
)
from mem0_decay.metadata import compute_revision_fingerprint
from mem0_decay.qdrant_reader import QdrantMemoryRecord, ScanPage

DAY = 86_400_000
SCOPE = ResolvedScope("tenant", user_id="alice")


class MemoryFixture:
    def __init__(self):
        self.items = {}
        self.now = 200 * DAY
        self.fail_delete = None
        self.fail_after_write = False
        self.calls = []
        self.next_id = 1

    async def add(self, messages, *, metadata, infer=True, **ids):
        item_id = str(self.next_id)
        self.next_id += 1
        self.items[item_id] = {
            "id": item_id,
            "memory": messages,
            "metadata": metadata,
            **ids,
            "score": 1.0,
        }
        if self.fail_after_write:
            self.fail_after_write = False
            raise RuntimeError("connection lost after write")
        return {"results": [{"id": item_id, "memory": messages, "event": "ADD"}]}

    async def search(self, query, *, filters, top_k, **kwargs):
        self.calls.append((query, filters, top_k))
        matches = []
        for item in self.items.values():
            payload = self.payload(item)
            if all(
                self.matches(payload.get(key), value) for key, value in filters.items()
            ):
                matches.append(item)
        return {"results": matches[:top_k]}

    @staticmethod
    def matches(actual, expected):
        if isinstance(expected, dict):
            return actual is not None and all(
                actual >= value if op == "gte" else actual < value
                for op, value in expected.items()
            )
        return actual == expected

    async def get(self, item_id):
        return self.items.get(item_id)

    async def delete(self, item_id):
        if self.fail_delete == item_id:
            raise RuntimeError("connection lost before delete")
        self.items.pop(item_id, None)

    @staticmethod
    def payload(item):
        return {
            **item["metadata"],
            **{
                key: item[key]
                for key in ("user_id", "agent_id", "run_id")
                if key in item
            },
        }

    async def scan(self, scope, *, limit=100, cursor=None, with_vector=False):
        records = [
            QdrantMemoryRecord(
                key, item["memory"], self.payload(item), [1, 0] if with_vector else None
            )
            for key, item in self.items.items()
            if (cursor is None or int(key) >= int(cursor))
            and all(item.get(key) == value for key, value in scope.identity().items())
        ]
        return ScanPage(
            records[:limit], records[limit].id if len(records) > limit else None
        )

    async def find_summary(self, scope, compaction_key):
        page = await self.scan(scope)
        return next(
            (
                r
                for r in page.records
                if r.payload.get("decay_compaction_key") == compaction_key
            ),
            None,
        )

    def wrap(self, **options):
        return DecayMemory(
            "mem", self, now_ms=lambda: self.now, qdrant_reader=self, **options
        )


class CurveTest(unittest.TestCase):
    def test_raw_threshold_is_strict_and_floor_is_separate(self):
        curve = exponential(30)
        self.assertEqual(curve(30), 0.5)
        self.assertEqual(curve.with_floor(0.95)(30), 0.975)
        self.assertFalse(curve.to_criterion(0.5)(30).below_threshold)
        self.assertTrue(curve.to_criterion(0.5)(31).below_threshold)
        self.assertEqual(sigmoid(midpoint=60, asymmetry=2)(60), 0.25)
        self.assertEqual(
            create_decay_criterion(curve={"kind": "constant", "strength": 0.3})(
                100
            ).strength,
            0.3,
        )

    def test_bad_custom_output_raises(self):
        with self.assertRaises(ValueError):
            create_decay_criterion(curve=lambda _: float("nan"))(1)

    def test_revision_hash_matches_existing_store_keys(self):
        self.assertEqual(
            compute_revision_fingerprint(
                "café",
                SCOPE,
                {
                    "decay_time_start": 0,
                    "decay_time_end": 1,
                    "decay_ingested_at": 2,
                    "decay_policy": "soft@v1",
                    "decay_tier": 0,
                },
            ),
            "e0d15bfad104273f5b1e662a6c2c594ef651c520f2908f017ce845d4e23c691b",
        )


class MemoryTest(unittest.IsolatedAsyncioTestCase):
    async def test_weighted_search_uses_mem0_candidates_and_original_policy(self):
        backend = MemoryFixture()
        backend.now = 30 * DAY
        memory = backend.wrap()
        await memory.add("old", scope=SCOPE, event_time=0)
        await memory.add("recent", scope=SCOPE, event_time=30 * DAY)
        result = await memory.search("q", scope=SCOPE, top_k=1)
        self.assertEqual(backend.calls[-1], ("q", {"user_id": "alice"}, 4))
        self.assertEqual([item.memory for item in result.results], ["recent"])
        two = await memory.search("q", scope=SCOPE, top_k=2)
        self.assertAlmostEqual(two.results[1].adjusted_score, 0.9825)
        self.assertEqual(two.results[1].metadata["decay_policy"], "soft@v1")

    async def test_date_only_range_includes_local_day_across_dst(self):
        backend = MemoryFixture()
        memory = backend.wrap()
        await memory.add("before", scope=SCOPE, event_time="2024-03-10T04:59:59Z")
        await memory.add("inside", scope=SCOPE, event_time="2024-03-11T03:59:59Z")
        await memory.add("after", scope=SCOPE, event_time="2024-03-11T04:00:00Z")
        result = await memory.search(
            "q",
            scope=SCOPE,
            range=RangeQuery(
                "2024-03-10",
                "2024-03-10",
                time_zone="America/New_York",
            ),
        )
        self.assertEqual([item.memory for item in result.results], ["inside"])
        bounds = backend.calls[-1][1]
        self.assertEqual(
            bounds["decay_time_end"]["lt"] - bounds["decay_time_start"]["gte"],
            23 * 3_600_000,
        )

    async def test_anchor_curve_can_override_recency_for_temporal_queries(self):
        memory = MemoryFixture().wrap()
        await memory.add("a", scope=SCOPE, event_time=0, policy_id="pinned")
        await memory.add("b", scope=SCOPE, event_time=DAY)
        result = await memory.search(
            "q", scope=SCOPE, alpha=1, anchor=AnchorQuery(DAY, half_life_days=1)
        )
        self.assertEqual(
            [(item.memory, item.time_weight) for item in result.results],
            [("b", 1), ("a", 0.5)],
        )

    async def test_reserved_metadata_and_tenant_scope(self):
        backend = MemoryFixture()
        memory = backend.wrap(
            tenant_binding=TenantBinding("metadata", field="workspace")
        )
        with self.assertRaises(ValueError):
            await memory.add("bad", scope=SCOPE, metadata={"decay_tier": 1})
        await memory.add("private", scope=ResolvedScope("other", user_id="alice"))
        await memory.add("ours", scope=SCOPE)
        result = await memory.search("q", scope=SCOPE, filters={"workspace": "other"})
        self.assertEqual([item.memory for item in result.results], ["ours"])
        with self.assertRaises(ValueError):
            await memory.search(
                "q", scope=SCOPE, filters={"AND": [{"workspace": "other"}]}
            )
        with self.assertRaises(ValueError):
            await memory.search(
                "q", scope=SCOPE, filters={"OR": [{"run_id": "another-run"}]}
            )

    async def test_visibility_expiry_can_be_overridden(self):
        backend = MemoryFixture()
        policies = PolicyRegistry(
            [
                {
                    "id": "expires@v1",
                    "default_ttl_ms": DAY,
                    "enforce_exact_visibility_expiry": True,
                }
            ]
        )
        memory = backend.wrap(policy_resolver=policies)
        await memory.add("expired", scope=SCOPE, policy_id="expires@v1")
        backend.now += DAY
        self.assertFalse((await memory.search("q", scope=SCOPE)).results)
        self.assertEqual(
            len((await memory.search("q", scope=SCOPE, show_expired=True)).results), 1
        )

    async def test_access_events_are_idempotent_and_bounded(self):
        backend = MemoryFixture()
        store = InMemoryAccessStore()
        memory = backend.wrap(
            access_sidecar=AccessSidecar(store, max_age_reduction_days=10)
        )
        item = (await memory.add("a", scope=SCOPE, event_time=0))["results"][0]["id"]
        before = (await memory.search("q", scope=SCOPE)).results[0].time_weight
        await memory.record_use(SCOPE, item, "event")
        await memory.record_use(SCOPE, item, "event")
        after = (await memory.search("q", scope=SCOPE)).results[0].time_weight
        record = next(
            iter((await store.get_many([AccessKey("tenant", "mem", item)])).values())
        )
        self.assertEqual(record.access_count, 1)
        self.assertGreater(after, before)
        self.assertAlmostEqual(after, 0.3 + 0.7 * 0.5 ** (190 / 30))

    async def test_router_freezes_clock_and_authorizes_each_store(self):
        visible, denied = MemoryFixture().wrap(), MemoryFixture().wrap()
        visible.store_id, denied.store_id = "a", "b"
        router = DecayRouter(
            now_ms=lambda: 400 * DAY, authorize=lambda ctx: ctx.store_id != "b"
        )
        router.register_store(visible)
        router.register_store(denied)
        await router.add("visible", scope=SCOPE)
        await denied.add("hidden", scope=SCOPE)
        result = await router.search("q", scope=SCOPE)
        self.assertEqual([item.memory for item in result.results], ["visible"])
        self.assertEqual(result.reference_now_ms, 400 * DAY)
        with self.assertRaises(PermissionError):
            await router.sweep("b", scope=SCOPE)


class SweepTest(unittest.IsolatedAsyncioTestCase):
    async def test_preview_and_budget_only_delete_eligible_scope(self):
        backend = MemoryFixture()
        memory = backend.wrap(scope_sweep_policy=ScopeSweepPolicy("delete"))
        for text in ("a", "b"):
            await memory.add(text, scope=SCOPE, policy_id="ephemeral", event_time=0)
        await memory.add(
            "other",
            scope=ResolvedScope("tenant", user_id="bob"),
            policy_id="ephemeral",
            event_time=0,
        )
        backend.now += 2 * DAY
        preview = await memory.sweep(scope=SCOPE, page_size=1)
        self.assertEqual(preview.planned_mutations, 2)
        self.assertEqual(len(backend.items), 3)
        report = await memory.sweep(scope=SCOPE, dry_run=False, max_mutations_per_run=1)
        self.assertEqual(report.applied_mutations, 1)
        self.assertTrue(report.pending_work)
        self.assertEqual(len(backend.items), 2)

    async def test_archive_rechecks_revision_after_acknowledgment(self):
        backend = MemoryFixture()

        async def archive(request):
            backend.items[request["source_id"]]["memory"] = "edited while archiving"
            return {"confirmed": True, "key": request["key"]}

        memory = backend.wrap(
            scope_sweep_policy=ScopeSweepPolicy("archive"), archive_sink=archive
        )
        await memory.add("source", scope=SCOPE, policy_id="ephemeral", event_time=0)
        backend.now += 2 * DAY
        report = await memory.sweep(scope=SCOPE, dry_run=False)
        self.assertEqual(report.skipped["revision_conflict"], 1)
        self.assertEqual(len(backend.items), 1)

    async def test_recover_summary_written_before_timeout(self):
        backend = MemoryFixture()
        work = InMemoryCompactionWorkStore()
        summary_calls = []

        async def summarize(request):
            summary_calls.append(request)
            return {"summary": "merged"}

        memory = backend.wrap(
            scope_sweep_policy=ScopeSweepPolicy(
                "compact", compaction=CompactionConfig()
            ),
            summarizer=summarize,
            compaction_work_store=work,
        )
        await memory.add("one", scope=SCOPE, policy_id="ephemeral", event_time=0)
        await memory.add("two", scope=SCOPE, policy_id="ephemeral", event_time=1)
        backend.now += 2 * DAY
        backend.fail_after_write = True
        first = await memory.sweep(scope=SCOPE, dry_run=False)
        self.assertTrue(first.pending_work)
        self.assertEqual(len(backend.items), 3)
        second = await memory.sweep(scope=SCOPE, dry_run=False)
        self.assertEqual(second.applied_mutations, 2)
        self.assertFalse(second.pending_work)
        self.assertEqual(len(summary_calls), 1)
        self.assertEqual(len(backend.items), 1)
        summary = next(iter(backend.items.values()))
        self.assertEqual(summary["memory"], "merged")
        self.assertEqual(summary["metadata"]["decay_tier"], 1)
        self.assertEqual(len(summary["metadata"]["decay_sources"]), 2)

    async def test_compaction_refuses_unconfirmed_lineage(self):
        backend = MemoryFixture()
        work = InMemoryCompactionWorkStore()

        async def summarize(request):
            return {"summary": "merged"}

        memory = backend.wrap(
            scope_sweep_policy=ScopeSweepPolicy("compact"),
            summarizer=summarize,
            compaction_work_store=work,
        )
        await memory.add("one", scope=SCOPE, policy_id="ephemeral", event_time=0)
        await memory.add("two", scope=SCOPE, policy_id="ephemeral", event_time=1)
        backend.now += 2 * DAY
        backend.fail_after_write = True
        await memory.sweep(scope=SCOPE, dry_run=False)
        backend.items["3"]["metadata"]["decay_sources"] = [
            {"id": "1", "revision": "bogus"}
        ]
        report = await memory.sweep(scope=SCOPE, dry_run=False)
        self.assertTrue(report.pending_work)
        self.assertEqual(set(backend.items), {"1", "2", "3"})

    async def test_compaction_retries_only_unfinished_deletes(self):
        backend = MemoryFixture()
        memory = backend.wrap(
            scope_sweep_policy=ScopeSweepPolicy("compact"),
            summarizer=lambda _: "merged",
        )
        await memory.add(
            "one",
            scope=SCOPE,
            policy_id="ephemeral",
            event_time=1,
            expires_at=215 * DAY,
        )
        await memory.add(
            "two",
            scope=SCOPE,
            policy_id="ephemeral",
            event_time=2,
            expires_at=214 * DAY,
        )
        backend.now += 2 * DAY
        backend.fail_delete = "2"
        first = await memory.sweep(scope=SCOPE, dry_run=False)
        self.assertEqual(first.applied_mutations, 2)
        self.assertTrue(first.pending_work)
        self.assertEqual(set(backend.items), {"2", "3"})
        metadata = backend.items["3"]["metadata"]
        self.assertEqual(
            (
                metadata["decay_time_start"],
                metadata["decay_time_end"],
                metadata["decay_expires_at"],
            ),
            (1, 2, 214 * DAY),
        )
        backend.fail_delete = None
        second = await memory.sweep(scope=SCOPE, dry_run=False, max_mutations_per_run=1)
        self.assertEqual(second.applied_mutations, 1)
        self.assertFalse(second.pending_work)
        self.assertEqual(set(backend.items), {"3"})

    async def test_uncertain_summary_write_reserves_part_of_mutation_budget(self):
        backend = MemoryFixture()
        memory = backend.wrap(
            scope_sweep_policy=ScopeSweepPolicy("compact"),
            summarizer=lambda _: "merged",
        )
        for timestamp in (1, 2, 8 * DAY, 8 * DAY + 1):
            await memory.add(
                "source", scope=SCOPE, policy_id="ephemeral", event_time=timestamp
            )
        backend.now += 2 * DAY
        backend.fail_after_write = True
        report = await memory.sweep(scope=SCOPE, dry_run=False, max_mutations_per_run=3)
        self.assertTrue(report.pending_work)
        self.assertEqual(
            sum(item["metadata"]["decay_tier"] == 1 for item in backend.items.values()),
            1,
        )
        self.assertEqual(report.skipped["budget_exhausted"], 2)


if __name__ == "__main__":
    unittest.main()
