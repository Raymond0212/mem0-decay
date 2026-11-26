"""Read and index Mem0 payloads through the Python Qdrant SDK."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from .contracts import (
    ResolvedScope,
    call,
    finite,
    nonempty,
    positive_int,
    validate_scope,
)
from .metadata import (
    COMPACTION_KEY,
    EXPIRES_AT,
    INGESTED_AT,
    POLICY,
    TENANT,
    TIER,
    TIME_END,
    TIME_START,
    payload_matches_scope,
)

ScanCursor = str | int


@dataclass(frozen=True, slots=True)
class QdrantMemoryRecord:
    id: str
    memory: str
    payload: dict[str, Any]
    vector: list[float] | None = None
    created_at: str | None = None
    updated_at: str | None = None


@dataclass(frozen=True, slots=True)
class ScanPage:
    records: list[QdrantMemoryRecord]
    next_cursor: ScanCursor | None


class QdrantReaderPort(Protocol):
    async def scan(
        self, scope: ResolvedScope, *, cursor=None, limit=None, with_vector=False
    ) -> ScanPage: ...
    async def find_summary(
        self, scope: ResolvedScope, compaction_key: str
    ) -> QdrantMemoryRecord | None: ...


class QdrantReader:
    def __init__(
        self,
        client,
        collection_name,
        *,
        tenant_binding=None,
        default_page_size=100,
        dense_vector_name="",
    ):
        from qdrant_client import models

        self.client = client
        self._models = models
        self.collection_name = nonempty(collection_name, "collection_name")
        self.tenant_binding = tenant_binding
        self._metadata_binding = (
            tenant_binding is not None and tenant_binding.mode == "metadata"
        )
        self.tenant_field = tenant_binding.field if self._metadata_binding else TENANT
        self.default_page_size = positive_int(default_page_size, "default_page_size")
        self.dense_vector_name = dense_vector_name

    def _filter(self, scope, **extra):
        filters = {**scope.identity(), **extra}
        if self._metadata_binding:
            filters[self.tenant_field] = scope.tenant
        return self._models.Filter(
            must=[
                self._models.FieldCondition(
                    key=key, match=self._models.MatchValue(value=value)
                )
                for key, value in filters.items()
            ]
        )

    def _record(self, point, with_vector):
        payload = dict(point.payload or {})
        raw = point.vector if with_vector else None
        if isinstance(raw, Mapping):
            raw = raw.get(self.dense_vector_name)
        vector = (
            list(raw)
            if isinstance(raw, list) and raw and all(map(finite, raw))
            else None
        )
        return QdrantMemoryRecord(
            str(point.id),
            payload.get("data") or "",
            payload,
            vector,
            payload.get("created_at", payload.get("createdAt")),
            payload.get("updated_at", payload.get("updatedAt")),
        )

    async def scan(self, scope, *, cursor=None, limit=None, with_vector=False):
        validate_scope(scope, self.tenant_binding)
        limit = (
            self.default_page_size if limit is None else positive_int(limit, "limit")
        )
        points, offset = await call(
            self.client.scroll,
            self.collection_name,
            scroll_filter=self._filter(scope),
            offset=cursor,
            limit=limit,
            with_payload=True,
            with_vectors=with_vector,
        )
        return ScanPage([self._record(point, with_vector) for point in points], offset)

    async def find_summary(self, scope, compaction_key):
        validate_scope(scope, self.tenant_binding)
        scroll_filter = self._filter(
            scope, **{COMPACTION_KEY: nonempty(compaction_key, "compaction_key")}
        )
        offset = None
        while True:
            points, offset = await call(
                self.client.scroll,
                self.collection_name,
                scroll_filter=scroll_filter,
                offset=offset,
                limit=100,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                record = self._record(point, False)
                if payload_matches_scope(
                    record.payload,
                    scope,
                    exact=True,
                    tenant_field=self.tenant_field,
                    require_tenant=self._metadata_binding,
                ):
                    return record
            if offset is None:
                return None


def create_qdrant_reader(client, collection_name, **options):
    return QdrantReader(client, collection_name, **options)


async def ensure_qdrant_decay_indexes(client, collection_name, *, tenant_field=TENANT):
    for schema, fields in (
        ("integer", (TIME_START, TIME_END, INGESTED_AT, EXPIRES_AT, TIER)),
        (
            "keyword",
            (COMPACTION_KEY, POLICY, tenant_field, "user_id", "agent_id", "run_id"),
        ),
    ):
        for field in dict.fromkeys(fields):
            await call(
                client.create_payload_index,
                collection_name,
                field_name=field,
                field_schema=schema,
                wait=True,
            )
