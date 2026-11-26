"""Self-hosted Mem0 decay endpoints consumed by the Option B benchmark client."""

import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .compat import Mem0Decay
from .metadata import TIME_END, parse_timestamp_ms
from .policy import PolicyRegistry


def _open_sdk():
    from mem0 import AsyncMemory

    config = {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "host": os.getenv("QDRANT_HOST", "localhost"),
                "port": int(os.getenv("QDRANT_PORT", "6333")),
                "collection_name": os.getenv("COLLECTION_NAME", "memories"),
            },
        },
        "llm": {
            "provider": "openai",
            "config": {
                "model": os.getenv("LLM_MODEL", "gpt-4o-mini"),
                "temperature": 0.1,
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {"model": os.getenv("EMBEDDER_MODEL", "text-embedding-3-small")},
        },
    }
    if path := os.getenv("HISTORY_DB_PATH"):
        config["history_db_path"] = path
    return AsyncMemory.from_config(config)


@asynccontextmanager
async def lifespan(app):
    # Option B sends the session timestamp when adding, but no clock when searching.
    reference = os.getenv("MEM0_DECAY_REFERENCE_DATE")
    app.state.reference_ms = parse_timestamp_ms(reference) if reference else None
    policies = PolicyRegistry()
    policy = policies.register("bounded_30d@v1", half_life_days=30, floor=0.95)
    async with _open_sdk() as native:
        app.state.memory = Mem0Decay(
            native,
            default_policy_id=policy.id,
            policy_resolver=policies,
            alpha=1,
            overfetch_factor=1,
            max_candidates=1000,
        )
        yield


app = FastAPI(title="Mem0 Decay OSS Server", lifespan=lifespan)


class AddRequest(BaseModel):
    messages: list[dict[str, Any]]
    user_id: str = Field(min_length=1)
    timestamp: int | None = None
    custom_instructions: str | None = None
    metadata: dict[str, Any] | None = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    limit: int = Field(default=100, ge=1, le=1000)
    rerank: bool = False


@app.get("/health")
async def health():
    memory = app.state.memory.decay
    return {
        "status": "ok",
        "memory": "mem0-decay",
        "policy": memory.default_policy_id,
        "reference_ms": app.state.reference_ms,
    }


@app.post("/memories")
async def add(request: AddRequest):
    try:
        return await app.state.memory.add(
            request.messages,
            user_id=request.user_id,
            timestamp=request.timestamp,
            metadata=request.metadata,
            prompt=request.custom_instructions,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/search")
async def search(request: SearchRequest):
    result = await app.state.memory.search(
        request.query,
        filters={"user_id": request.user_id},
        top_k=request.limit,
        rerank=request.rerank,
        reference_date=app.state.reference_ms,
    )
    # The stock answer prompt reads created_at and ignores metadata.
    for item in result["results"]:
        event_ms = item["metadata"].get(TIME_END)
        if event_ms is not None:
            event = datetime.fromtimestamp(event_ms / 1000, tz=UTC)
            item["created_at"] = event.isoformat(timespec="milliseconds").replace(
                "+00:00", "Z"
            )
    return result


@app.delete("/memories")
async def delete_user(user_id: str = Query(min_length=1)):
    await app.state.memory.delete_all(user_id=user_id)
    return {"message": "All memories deleted"}
