# `mem0-decay`

Time-aware ranking and scoped lifecycle management for Mem0 OSS and Qdrant. The Python package wraps Mem0's native retrieval and uses the existing `decay_*` Qdrant payload format.

## Install

```bash
python -m pip install '.[oss]'
```

The core package has no third-party dependencies. Install the `oss` extra to use the Python Mem0 and Qdrant SDKs. The integration tests exercise Mem0 `2.2.1` and Qdrant client `1.20.0`. Python 3.11 or later is required.

## Add and search

Configure Mem0 with the model and Qdrant collection for your deployment. Pass either `mem0.AsyncMemory` or `mem0.Memory` to the async wrapper.

```python
import asyncio

from mem0 import AsyncMemory
from mem0_decay import DecayMemory, RangeQuery, ResolvedScope, TenantBinding


async def main():
    native = AsyncMemory.from_config({
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "workspace_memory",
                "url": "http://127.0.0.1:6333",
            },
        },
    })
    binding = TenantBinding("collection", tenant="workspace")
    memory = DecayMemory("workspace_memory", native, tenant_binding=binding)
    scope = ResolvedScope("workspace", user_id="alice")

    await memory.add(
        "Alice started a new job in January.",
        scope=scope,
        event_time="2024-01-17",
        infer=False,
    )

    result = await memory.search(
        "When did Alice start her job?",
        scope=scope,
        range=RangeQuery("2024-01-01", "2024-01-31"),
    )
    for item in result.results:
        print(item.memory, item.adjusted_score)


asyncio.run(main())
```

Mem0 retrieves candidates with the authorized scope and optional filters. In recency mode, the wrapper reranks them using `base_score - base_score * (1 - time_weight) * alpha`. The default `alpha` is `0.05`. Policy floors bound `time_weight` and the sweep threshold uses raw curve strength. Range and anchor queries accept custom curves and IANA time zones. A date-only range end includes the entire local day.

## Sweep and compact

Use a `QdrantReader` pointed at the same collection and configured with the same tenant binding. The reader accepts both Python Qdrant clients and supports named dense vectors via `dense_vector_name`. Provision filter indexes with `await ensure_qdrant_decay_indexes(client, collection_name)` when the collection is created.

- Call `await memory.sweep(scope=scope)` to preview eligible actions. Pass `dry_run=False` to apply them and use `max_mutations_per_run` to cap writes and deletions.
- Configure `ScopeSweepPolicy("delete")`, `ScopeSweepPolicy("archive")`, or `ScopeSweepPolicy("compact")`. The default action is `"none"`. The `soft@v1` policy is ranking-only by default. `ephemeral@v1` can become eligible after its ingestion grace period.
- For archive, supply an `archive_sink` callback that durably stores the record before returning. It can return an acknowledgment with `confirmed=True` and the supplied `key`.
- For compaction, supply a `summarizer` callback that accepts the group and returns its summary text. The sweep writes and verifies lineage on a Tier 1 summary before retiring Tier 0 sources. The default in-memory work store supports retries within one process. Supply a durable `CompactionWorkStore` and a shared write coordinator for recovery across process restarts or concurrent workers.

`DecayRouter` provides scope authorization, a shared writer, and multi-store search using a common reference clock. `AccessSidecar` optionally lets explicit, idempotent `record_use()` events reduce a memory's effective ranking age.

## Migrate a caller

Use `create_decay_memory()` or `DecayMemory` with snake_case keyword arguments and a `ResolvedScope`. Query operations return Python dataclasses. The Mem0 SDK response from `add()` remains a dictionary. The Python Mem0 SDK accepts search filters, while add scoping uses `user_id`, `agent_id`, and `run_id`. Use a collection per tenant or tenant-unique Mem0 identities for inferred writes, since Mem0 itself scopes extraction by those IDs. Metadata binding can additionally scope wrapper searches and sweeps.

The port keeps existing decay metadata, policy IDs, and revision and compaction keys. The Mem0 SDK owns the underlying vector and entity indexes. Point the Python wrapper at a collection supported by your Python Mem0 configuration.

For callers that use the Mem0 OSS SDK interface, `Mem0Decay.from_config(config)` exposes async `add()` and `search()` with Mem0 arguments and `{"results": [...]}` responses. Add takes Mem0 identity keywords. Search and listing take identity filters. Search returns the adjusted value in `score` and preserves the upstream relevance in `base_score`. Historical adds can supply Unix seconds via `timestamp`. Search can supply `reference_date` as an ISO timestamp or epoch milliseconds.

## Tests and benchmarks

```bash
python -m unittest discover -s tests -v
```

With the `oss` extra installed, this also exercises Mem0 with local Qdrant and deterministic embeddings. That fixture requires no OpenAI calls.

## Run LoCoMo with benchmark main's Option B

Option B talks to an OSS server over HTTP. The Compose override in this repository supplies the Python decay server for the `mem0` service. The benchmark's own Compose file supplies Qdrant, the history volume, the health check and port mappings. Use a separate checkout of the benchmark's `main` branch. The commands below can run with that checkout mounted read-only.

Start in the root of `mem0-decay-py`, with `OPENAI_API_KEY` available to Compose and the benchmark process.

```bash
export MEM0_DECAY_ROOT="$PWD"
export BENCHMARK_MAIN=/absolute/path/to/mem0-decay-benchmark-main
export MEM0_DECAY_REFERENCE_DATE=2024-01-12T13:41:00Z

docker compose -p locomo-decay \
  -f "$BENCHMARK_MAIN/docker-compose.yml" \
  -f "$MEM0_DECAY_ROOT/compose.option-b.yaml" up -d --build --wait
curl -fsS http://localhost:8888/health
```

The health response identifies `mem0-decay`, policy `bounded_30d@v1`, and the reference time in epoch milliseconds. The example uses the last session in LoCoMo-10 as the search clock across all conversations. Omitting the variable selects wall-clock time. The separate Compose project gives this run its own database volumes.

Run the upstream CLI from a writable directory outside both repositories. It writes its dataset cache, logs, checkpoints and results there. Use a fresh directory for a new experiment.

```bash
python -m pip install -r "$BENCHMARK_MAIN/requirements.txt"
mkdir -p "$HOME/locomo-decay-run"
cd "$HOME/locomo-decay-run"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$BENCHMARK_MAIN" \
  python -m benchmarks.locomo.run --project-name locomo-decay-py \
    --backend oss --mem0-host http://localhost:8888 --top-k 200 \
    --top-k-cutoffs 50,200 --answerer-model gpt-4o --judge-model gpt-4o
```

Each `POST /memories` carries the session timestamp in Unix seconds. On `POST /search`, the server retrieves the requested number of native candidates for that user and applies the 30-day exponential policy with a 0.95 floor and `alpha=1`. It returns the adjusted value as `score` and the event date as `created_at`, the two fields the stock harness uses for ordering and for dates in the answer prompt. The harness handles answering, judging and retrieval timing. This configuration measures search-time decay. A scheduled compaction run requires a separate sweep driver.

The image includes fastembed and the spaCy English model. On first use, fastembed downloads `Qdrant/bm25` from Hugging Face. If that download fails, Mem0 logs a warning and proceeds with BM25 disabled. The CLI command above uses OpenAI for extraction, embeddings, answering and judging.
