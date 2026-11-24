import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { randomUUID } from "node:crypto";
import { QdrantClient } from "@qdrant/js-client-rest";
import { Memory } from "mem0ai/oss";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import {
  DECAY_METADATA_KEYS,
  DEFAULT_POLICY_IDS,
  InMemoryCompactionWorkStore,
  MS_PER_DAY,
  createDecayMemory,
  createDecayRouter,
  createInProcessWriteCoordinator,
  createQdrantReader,
  ensureQdrantDecayIndexes,
  type ArchiveRequest,
} from "../src/index.js";

/**
 * Live Qdrant + Mem0 OSS 3.3.1 integration fixture.
 *
 * Runs automatically when a Qdrant instance is reachable at QDRANT_URL
 * (defaults to http://127.0.0.1:6333). Uses a local deterministic OpenAI-compatible
 * embedding mock server and `infer: false` so no external API keys or LLM calls are required.
 *
 * Set REQUIRE_LIVE_QDRANT=1 in CI to fail instead of skipping when Qdrant is unavailable.
 */

const QDRANT_URL = process.env.QDRANT_URL ?? "http://127.0.0.1:6333";
const REQUIRE_LIVE_QDRANT = process.env.REQUIRE_LIVE_QDRANT === "1";
const EMBEDDING_DIM = 8;
const DAY = MS_PER_DAY;
const NOW = Date.UTC(2026, 5, 1, 12, 0, 0);

function deterministicEmbed(text: string): number[] {
  const lower = text.toLowerCase();
  const vec = new Array<number>(EMBEDDING_DIM).fill(0.05);
  if (lower.includes("deployment") || lower.includes("release") || lower.includes("incident")) {
    vec[0] = 1.0;
    vec[1] = 0.8;
  }
  if (lower.includes("quarterly") || lower.includes("budget")) {
    vec[2] = 1.0;
    vec[3] = 0.7;
  }
  if (lower.includes("ephemeral") || lower.includes("sprint")) {
    vec[4] = 1.0;
    vec[5] = 0.9;
  }
  // Small hash-based tie-breaker
  let hash = 0;
  for (let i = 0; i < lower.length; i++) {
    hash = (hash * 31 + lower.charCodeAt(i)) >>> 0;
  }
  vec[6] = ((hash % 100) + 1) / 1000;
  vec[7] = (((hash >>> 8) % 100) + 1) / 1000;

  const norm = Math.hypot(...vec);
  return vec.map((v) => v / norm);
}

async function startMockEmbedderServer(): Promise<{ server: Server; baseURL: string }> {
  const server = createServer((req, res) => {
    if (req.method !== "POST" || !req.url?.includes("/embeddings")) {
      res.statusCode = 404;
      res.end("Not found");
      return;
    }
    const chunks: Buffer[] = [];
    req.on("data", (chunk: Buffer) => chunks.push(chunk));
    req.on("end", () => {
      const body = JSON.parse(Buffer.concat(chunks).toString("utf8")) as {
        input: string | string[];
        model?: string;
      };
      const inputs = Array.isArray(body.input) ? body.input : [body.input];
      const data = inputs.map((text, index) => ({
        object: "embedding",
        index,
        embedding: deterministicEmbed(String(text)),
      }));
      res.setHeader("content-type", "application/json");
      res.end(
        JSON.stringify({
          object: "list",
          data,
          model: body.model ?? "text-embedding-3-small",
          usage: { prompt_tokens: inputs.length, total_tokens: inputs.length },
        }),
      );
    });
  });

  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address() as AddressInfo;
  return { server, baseURL: `http://127.0.0.1:${address.port}/v1` };
}

async function isQdrantReachable(client: QdrantClient): Promise<boolean> {
  try {
    await client.getCollections();
    return true;
  } catch {
    return false;
  }
}

describe("live Qdrant + Mem0 OSS 3.3.1 integration", () => {
  const collectionName = `mem0_decay_it_${randomUUID().replace(/-/g, "").slice(0, 12)}`;
  const qdrantClient = new QdrantClient({ url: QDRANT_URL });
  let qdrantAvailable = false;
  let embedderServer: Server | undefined;
  let memory: Memory;

  beforeAll(async () => {
    qdrantAvailable = await isQdrantReachable(qdrantClient);
    if (!qdrantAvailable) {
      if (REQUIRE_LIVE_QDRANT) {
        throw new Error(`Qdrant is not reachable at ${QDRANT_URL} (REQUIRE_LIVE_QDRANT=1)`);
      }
      return;
    }

    const mock = await startMockEmbedderServer();
    embedderServer = mock.server;

    await qdrantClient.createCollection(collectionName, {
      vectors: { size: EMBEDDING_DIM, distance: "Cosine" },
    });
    await ensureQdrantDecayIndexes(qdrantClient, collectionName);

    memory = new Memory({
      disableHistory: true,
      embedder: {
        provider: "openai",
        config: {
          apiKey: "test-local-key",
          baseURL: mock.baseURL,
          model: "text-embedding-3-small",
          embeddingDims: EMBEDDING_DIM,
        },
      },
      vectorStore: {
        provider: "qdrant",
        config: {
          client: qdrantClient,
          collectionName,
          dimension: EMBEDDING_DIM,
          embeddingModelDims: EMBEDDING_DIM,
        },
      },
      llm: {
        provider: "openai",
        config: {
          apiKey: "test-local-key",
          baseURL: mock.baseURL,
          model: "gpt-4o-mini",
        },
      },
    });
  });

  afterAll(async () => {
    if (qdrantAvailable) {
      try {
        await qdrantClient.deleteCollection(collectionName);
      } catch {
        // Ignore cleanup errors on disposable collection
      }
    }
    if (embedderServer) {
      await new Promise<void>((resolve) => embedderServer!.close(() => resolve()));
    }
  });

  it("verifies metadata round trip, strict range pre-filtering on Qdrant, multi-page sweep, archive/delete, and compacted summary search", async (ctx) => {
    if (!qdrantAvailable) {
      ctx.skip();
      return;
    }

    const scopeAlice = { tenant: "acme", userId: "alice" };
    const scopeBob = { tenant: "acme", userId: "bob" };
    const archived: ArchiveRequest[] = [];
    const workStore = new InMemoryCompactionWorkStore();
    let activeSweepAction: "archive" | "compact" = "archive";

    const coordinator = createInProcessWriteCoordinator({ serializeAllStores: true });
    const router = createDecayRouter({
      nowMs: () => NOW,
      writeCoordinator: coordinator,
    });

    const wrapper = createDecayMemory({
      storeId: "live-qdrant-store",
      memory,
      nowMs: () => NOW,
      writeCoordinator: coordinator,
      tenantBinding: { mode: "metadata", field: DECAY_METADATA_KEYS.tenant },
      qdrantReader: createQdrantReader({
        client: qdrantClient,
        collectionName,
        tenantBinding: { mode: "metadata", field: DECAY_METADATA_KEYS.tenant },
      }),
      scopeSweepPolicy: () => ({
        action: activeSweepAction,
        expiryAction: "delete",
        compaction: {
          similarityThreshold: 0.7,
          minGroupSize: 2,
          maxGroupSize: 4,
        },
      }),
      archiveSink: {
        async archive(req) {
          archived.push(req);
          return { confirmed: true, key: req.key };
        },
      },
      compactionWorkStore: workStore,
      summarizer: {
        async summarize({ sources }) {
          return {
            summary: `Sprint ephemeral summary covering ${sources.length} items`,
          };
        },
      },
    });

    router.registerStore(wrapper, { activeWriter: true });

    // 1. Seed 8 newer deployment hits + 1 older historical hit in Jan 2025 + 1 out-of-scope Bob hit
    const historicalJan = Date.UTC(2025, 0, 15, 10, 0, 0);
    await router.add("Deployment incident root cause in January 2025", {
      scope: scopeAlice,
      eventTime: historicalJan,
      policyId: DEFAULT_POLICY_IDS.soft,
      infer: false,
    });

    for (let i = 0; i < 8; i++) {
      await router.add(`Deployment release note number ${i}`, {
        scope: scopeAlice,
        eventTime: NOW - i * DAY,
        policyId: DEFAULT_POLICY_IDS.soft,
        infer: false,
      });
    }

    await router.add("Deployment incident in January 2025 for Bob", {
      scope: scopeBob,
      eventTime: historicalJan,
      policyId: DEFAULT_POLICY_IDS.soft,
      infer: false,
    });

    // Strict range search on Qdrant finds the older Jan 2025 record within topK = 2,
    // excluding out-of-window and out-of-scope points
    const strictSearch = await router.search("Deployment incident release", {
      scope: scopeAlice,
      topK: 2,
      range: { from: "2025-01-10", to: "2025-01-20", strict: true },
    });
    expect(strictSearch.results).toHaveLength(1);
    expect(strictSearch.results[0]!.memory).toContain("January 2025");
    expect(strictSearch.results[0]!.timeSpan).toEqual({
      start: historicalJan,
      end: historicalJan,
    });

    // 2. Seed stale ephemeral records for archive sweep + compaction sweep
    await router.add("Old ephemeral sprint standup note A", {
      scope: scopeAlice,
      eventTime: NOW - 60 * DAY,
      policyId: DEFAULT_POLICY_IDS.ephemeral,
      infer: false,
    });

    // Dry run over multiple pages finds 0 eligible until ingestion grace elapses
    const dryWithinGrace = await router.sweep({
      storeId: "live-qdrant-store",
      options: { scope: scopeAlice, pageSize: 3, dryRun: true },
    });
    expect(dryWithinGrace.scanned).toBeGreaterThanOrEqual(10);
    expect(dryWithinGrace.skipped.grace_period).toBe(1);
    expect(dryWithinGrace.eligible).toBe(0);

    // Advance reference clock past 24h ingestion grace via a new wrapper pointing to same collection
    const futureNow = NOW + 2 * DAY;
    const futureWrapper = createDecayMemory({
      storeId: "live-qdrant-store",
      memory,
      nowMs: () => futureNow,
      writeCoordinator: coordinator,
      tenantBinding: { mode: "metadata", field: DECAY_METADATA_KEYS.tenant },
      qdrantReader: createQdrantReader({
        client: qdrantClient,
        collectionName,
        tenantBinding: { mode: "metadata", field: DECAY_METADATA_KEYS.tenant },
      }),
      scopeSweepPolicy: () => ({
        action: activeSweepAction,
        expiryAction: "delete",
        compaction: { similarityThreshold: 0.7, minGroupSize: 2 },
      }),
      archiveSink: {
        async archive(req) {
          archived.push(req);
          return { confirmed: true, key: req.key };
        },
      },
      compactionWorkStore: workStore,
      summarizer: {
        async summarize({ sources }) {
          return {
            summary: `Sprint ephemeral summary covering ${sources.length} items`,
          };
        },
      },
    });

    // Archive execution retires the stale ephemeral record after durable confirmation
    const archiveSweep = await futureWrapper.sweep({
      scope: scopeAlice,
      pageSize: 3,
      dryRun: false,
    });
    expect(archiveSweep.appliedMutations).toBe(1);
    expect(archived).toHaveLength(1);

    // 3. Seed two similar stale ephemeral records and run Tier 0 -> Tier 1 compaction
    activeSweepAction = "compact";
    await router.add("Ephemeral sprint task one", {
      scope: scopeAlice,
      eventTime: NOW - 60 * DAY,
      policyId: DEFAULT_POLICY_IDS.ephemeral,
      infer: false,
    });
    await router.add("Ephemeral sprint task two", {
      scope: scopeAlice,
      eventTime: NOW - 59 * DAY,
      policyId: DEFAULT_POLICY_IDS.ephemeral,
      infer: false,
    });

    const compactSweep = await futureWrapper.sweep({
      scope: scopeAlice,
      pageSize: 3,
      dryRun: false,
    });
    expect(compactSweep.errors).toHaveLength(0);
    expect(compactSweep.appliedMutations).toBe(3);

    // Compacted Tier 1 summary is searchable through the normal entry point
    const summarySearch = await futureWrapper.search("Sprint ephemeral summary", {
      scope: scopeAlice,
      topK: 5,
    });
    const foundSummary = summarySearch.results.find((r) => r.tier === 1);
    expect(foundSummary).toBeDefined();
    expect(foundSummary!.memory).toContain("Sprint ephemeral summary");
  });
});
