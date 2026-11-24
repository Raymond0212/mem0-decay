# `mem0-decay`

Time-aware ranking and scoped lifecycle management wrapper for **Mem0 OSS** and **Qdrant**.

- **Tested Mem0 OSS compatibility target:** `mem0ai@3.3.1`
- **Vector store target:** Qdrant (`@qdrant/js-client-rest` `>=1.13.0 <1.19.0`; `1.18.x` tested with `mem0ai@3.3.1` because `@qdrant/js-client-rest@1.19.0` removed `QdrantClient.search()`)
- **Zero runtime dependencies** in the core library; curve helpers (`mem0-decay/curves`) work standalone without initializing Mem0 or Qdrant.

## Scripts

```bash
npm run typecheck         # Type-check src/ and test/
npm run build             # Emit dist/ ESM + .d.ts declarations
npm test                  # Run unit test suite (independent of live Qdrant)
npm run test:integration  # Run live Qdrant + Mem0 3.3.1 integration fixture
```

For the live integration suite, start a local Qdrant container (e.g. `docker run -p 6333:6333 qdrant/qdrant`) and run:

```bash
QDRANT_URL=http://127.0.0.1:6333 REQUIRE_LIVE_QDRANT=1 npm run test:integration
```

## Module layout

| File | Responsibility |
| --- | --- |
| `src/curves.ts` | Standalone pure decay curves (`exponential`, reversed `sigmoid`, custom) and callable `createDecayCriterion` / `evaluateDecay` |
| `src/contracts.ts` | Public interfaces (`ResolvedScope`, `Mem0Memory`, `WriteCoordinator`, `QdrantReader`, `AccessStore`, etc.) |
| `src/policy.ts` | Revisioned policy registry and defaults (`soft@v1`, `ephemeral@v1`, `pinned@v1`, `evergreen@v1`, `tier1_summary@v1`) |
| `src/metadata.ts` | Reserved `decay_*` metadata namespace protection, timezone-aware date normalization, revision fingerprints, and compaction keys |
| `src/memory.ts` | `createDecayMemory()` wrapper composing `add()`, `search()`, `sweep()`, and `recordUse()` |
| `src/search.ts` | No-date recency ranking (`adjustedScore = baseScore × timeWeight`), strict `[from, to)` Qdrant pre-filtering, soft range, and anchor fade |
| `src/qdrant-reader.ts` | Cursor-based scoped `scan()`, `findSummary()`, and `ensureQdrantDecayIndexes()` |
| `src/sweep.ts` | Scheduled `sweep()` dry-run preview, budget enforcement, durable archive confirmation, and coordinator-rechecked delete |
| `src/compaction.ts` | Resumable Tier 0 → Tier 1 compaction with embedding clustering, lineage verification, and crash recovery |
| `src/access-store.ts` | Optional `AccessStore` port, `InMemoryAccessStore`, effective-age reduction, and timeout fallback |
| `src/router.ts` | `createDecayRouter()` and `createInProcessWriteCoordinator()` enforcing scope authorization, frozen-clock fan-out, and the single-writer boundary |
