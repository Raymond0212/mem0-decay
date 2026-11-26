"""Exercise the real Python SDKs with local Qdrant and deterministic embeddings."""

import importlib.util
import os
import tempfile
import unittest
import warnings
from contextlib import asynccontextmanager
from uuid import uuid4

from mem0_decay import (
    DecayMemory,
    QdrantReader,
    RangeQuery,
    ResolvedScope,
    ScopeSweepPolicy,
    TenantBinding,
    ensure_qdrant_decay_indexes,
)

DAY = 86_400_000
HAS_SDKS = all(
    importlib.util.find_spec(name) is not None for name in ("mem0", "qdrant_client")
)


class FixtureEmbedding:
    def embed(self, text, memory_action=None):
        return [1.0, 0.0, 0.0]


@unittest.skipUnless(HAS_SDKS, "install the oss extra")
class SDKTest(unittest.IsolatedAsyncioTestCase):
    async def test_python_mem0_with_local_qdrant(self):
        from unittest.mock import patch

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(
                os.environ,
                {
                    "MEM0_TELEMETRY": "false",
                    "MEM0_DIR": directory,
                },
            ),
        ):
            from mem0 import AsyncMemory, Memory

            for memory_class in (AsyncMemory, Memory):
                with self.subTest(sdk=memory_class.__name__):
                    collection = memory_class.__name__.lower()
                    config = {
                        "vector_store": {
                            "provider": "qdrant",
                            "config": {
                                "collection_name": collection,
                                "path": f"{directory}/{collection}",
                                "embedding_model_dims": 3,
                            },
                        },
                        "embedder": {
                            "provider": "openai",
                            "config": {"api_key": "fixture"},
                        },
                        "llm": {"provider": "openai", "config": {"api_key": "fixture"}},
                        "history_db_path": f"{directory}/{collection}.db",
                    }
                    native = memory_class.from_config(config)
                    native.embedding_model = FixtureEmbedding()
                    native.vector_store._bm25_encoder = False
                    client = native.vector_store.client
                    try:
                        reader = QdrantReader(client, collection)
                        scope = ResolvedScope("test", user_id="alice")
                        clock = [200 * DAY]
                        wrapper = DecayMemory(
                            collection,
                            native,
                            qdrant_reader=reader,
                            now_ms=lambda: clock[0],
                            scope_sweep_policy=ScopeSweepPolicy("delete"),
                        )
                        await wrapper.add(
                            "early event",
                            scope=scope,
                            event_time=0,
                            policy_id="ephemeral",
                            infer=False,
                        )
                        await wrapper.add(
                            "later event", scope=scope, event_time=DAY, infer=False
                        )
                        result = await wrapper.search(
                            "event", scope=scope, range=RangeQuery(DAY, 2 * DAY)
                        )
                        self.assertEqual(
                            [record.memory for record in result.results],
                            ["later event"],
                        )
                        clock[0] += DAY
                        report = await wrapper.sweep(scope=scope, dry_run=False)
                        self.assertEqual(report.errors, [])
                        self.assertEqual(report.applied_mutations, 1)
                        self.assertEqual(
                            [
                                point.memory
                                for point in (await reader.scan(scope)).records
                            ],
                            ["later event"],
                        )

                        compacting = DecayMemory(
                            collection,
                            native,
                            qdrant_reader=reader,
                            now_ms=lambda: clock[0],
                            scope_sweep_policy=ScopeSweepPolicy("compact"),
                            summarizer=lambda _: "combined",
                        )
                        await compacting.add(
                            "first fact",
                            scope=scope,
                            event_time=0,
                            policy_id="ephemeral",
                            infer=False,
                        )
                        await compacting.add(
                            "second fact",
                            scope=scope,
                            event_time=1,
                            policy_id="ephemeral",
                            infer=False,
                        )
                        clock[0] += DAY
                        report = await compacting.sweep(scope=scope, dry_run=False)
                        self.assertEqual(report.errors, [])
                        self.assertEqual(report.applied_mutations, 3)
                        facts = {
                            point.memory: point
                            for point in (await reader.scan(scope)).records
                        }
                        self.assertEqual(set(facts), {"later event", "combined"})
                        self.assertEqual(facts["combined"].payload["decay_tier"], 1)
                    finally:
                        client.close()

    async def test_qdrant_reader_accepts_named_dense_vectors_and_filters_scope(self):
        from qdrant_client import AsyncQdrantClient, models

        ids = sorted(str(uuid4()) for _ in range(3))
        async with _client_context(AsyncQdrantClient(":memory:")) as client:
            await client.create_collection(
                "facts",
                vectors_config={
                    "dense": models.VectorParams(
                        size=2,
                        distance=models.Distance.COSINE,
                    )
                },
            )
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Payload indexes have no effect in the local Qdrant",
                )
                await ensure_qdrant_decay_indexes(
                    client, "facts", tenant_field="workspace"
                )
            await client.upsert(
                "facts",
                [
                    models.PointStruct(
                        id=ids[0],
                        vector={"dense": [1.0, 0.0]},
                        payload={
                            "user_id": "alice",
                            "workspace": "other",
                            "data": "excluded",
                        },
                    ),
                    models.PointStruct(
                        id=ids[1],
                        vector={"dense": [0.0, 1.0]},
                        payload={
                            "user_id": "alice",
                            "workspace": "test",
                            "data": "first",
                            "decay_compaction_key": "key",
                        },
                    ),
                    models.PointStruct(
                        id=ids[2],
                        vector={"dense": [0.0, 1.0]},
                        payload={
                            "user_id": "alice",
                            "workspace": "test",
                            "data": "second",
                        },
                    ),
                ],
            )
            reader = QdrantReader(
                client,
                "facts",
                tenant_binding=TenantBinding("metadata", field="workspace"),
                dense_vector_name="dense",
            )
            scope = ResolvedScope("test", user_id="alice")
            page = await reader.scan(scope, limit=1, with_vector=True)
            self.assertEqual(
                [(record.memory, record.vector) for record in page.records],
                [("first", [0.0, 1.0])],
            )
            self.assertIsNotNone(page.next_cursor)
            tail = await reader.scan(scope, cursor=page.next_cursor)
            self.assertEqual([record.memory for record in tail.records], ["second"])
            self.assertEqual((await reader.find_summary(scope, "key")).id, ids[1])


@asynccontextmanager
async def _client_context(client):
    try:
        yield client
    finally:
        await client.close()
