import { randomUUID } from "node:crypto";
import type {
  AddMemoryOptions,
  Mem0Memory,
  MemoryItem,
  Message,
  SearchFilters,
  SearchMemoryOptions,
  SearchResult,
} from "../../src/contracts.js";
import type {
  QdrantClientLike,
  QdrantFilterCondition,
  QdrantScrollPoint,
  QdrantScrollResponse,
} from "../../src/qdrant-reader.js";

export interface StoredFakeRecord {
  id: string;
  memory: string;
  payload: Record<string, unknown>;
  vector: number[];
  baseScore?: number;
  createdAt?: string;
  updatedAt?: string;
}

function matchesFilterValue(actual: unknown, expected: unknown): boolean {
  if (expected === undefined) return true;
  if (typeof expected !== "object" || expected === null) {
    if (expected === "*") return true;
    return actual === expected;
  }
  if (Array.isArray(expected)) {
    return expected.includes(actual);
  }

  const cond = expected as Record<string, unknown>;
  if ("gte" in cond || "gt" in cond || "lte" in cond || "lt" in cond) {
    if (typeof actual !== "number" || !Number.isFinite(actual)) return false;
    if ("gte" in cond && typeof cond.gte === "number" && !(actual >= cond.gte)) return false;
    if ("gt" in cond && typeof cond.gt === "number" && !(actual > cond.gt)) return false;
    if ("lte" in cond && typeof cond.lte === "number" && !(actual <= cond.lte)) return false;
    if ("lt" in cond && typeof cond.lt === "number" && !(actual < cond.lt)) return false;
    return true;
  }
  if ("eq" in cond) return actual === cond.eq;
  if ("ne" in cond) return actual !== cond.ne;
  if ("in" in cond && Array.isArray(cond.in)) return cond.in.includes(actual);
  if ("nin" in cond && Array.isArray(cond.nin)) return !cond.nin.includes(actual);
  return true;
}

function matchesFilters(payload: Record<string, unknown>, filters?: SearchFilters): boolean {
  if (!filters) return true;
  for (const [key, expected] of Object.entries(filters)) {
    if (!matchesFilterValue(payload[key], expected)) {
      return false;
    }
  }
  return true;
}

function matchesQdrantMust(
  payload: Record<string, unknown>,
  must?: QdrantFilterCondition[],
): boolean {
  if (!must || must.length === 0) return true;
  for (const cond of must) {
    if (cond.match !== undefined && payload[cond.key] !== cond.match.value) {
      return false;
    }
  }
  return true;
}

export class FakeMem0QdrantStore {
  readonly records = new Map<string, StoredFakeRecord>();
  readonly addCalls: Array<{ messages: string | Message[]; config: AddMemoryOptions }> = [];
  readonly searchCalls: Array<{ query: string; config: SearchMemoryOptions }> = [];
  readonly deleteCalls: string[] = [];
  readonly scrollCalls: Array<{
    collectionName: string;
    limit?: number;
    offset?: string | number | null;
    withVector?: boolean;
  }> = [];

  hooks: {
    beforeAdd?: (messages: string | Message[], config: AddMemoryOptions) => Promise<void> | void;
    afterAdd?: (created: StoredFakeRecord) => Promise<void> | void;
    beforeDelete?: (id: string) => Promise<void> | void;
    afterDelete?: (id: string) => Promise<void> | void;
    beforeGet?: (id: string) => Promise<void> | void;
  } = {};

  seed(record: Omit<StoredFakeRecord, "vector"> & { vector?: number[] }): StoredFakeRecord {
    const full: StoredFakeRecord = {
      ...record,
      vector: record.vector ?? [1, 0, 0, 0],
      payload: {
        ...record.payload,
        data: record.memory,
        ...(record.createdAt !== undefined ? { createdAt: record.createdAt } : {}),
        ...(record.updatedAt !== undefined ? { updatedAt: record.updatedAt } : {}),
      },
    };
    this.records.set(full.id, full);
    return full;
  }

  asMem0(): Mem0Memory {
    const store = this;
    return {
      async add(messages: string | Message[], config: AddMemoryOptions): Promise<SearchResult> {
        store.addCalls.push({ messages, config });
        if (store.hooks.beforeAdd) {
          await store.hooks.beforeAdd(messages, config);
        }

        const text =
          typeof messages === "string"
            ? messages
            : messages
                .filter((m) => m.role !== "system")
                .map((m) => (typeof m.content === "string" ? m.content : ""))
                .join(" ");

        const id = randomUUID();
        const nowIso = new Date().toISOString();
        const payload: Record<string, unknown> = {
          ...(config.metadata ?? {}),
          data: text,
          createdAt: nowIso,
          updatedAt: nowIso,
        };
        if (config.userId) payload.user_id = config.userId;
        if (config.agentId) payload.agent_id = config.agentId;
        if (config.runId) payload.run_id = config.runId;
        if (config.expirationDate) payload.expiration_date = config.expirationDate;

        const rec: StoredFakeRecord = {
          id,
          memory: text,
          payload,
          vector: [1, 0, 0, 0],
          baseScore: 0.9,
          createdAt: nowIso,
          updatedAt: nowIso,
        };
        store.records.set(id, rec);

        if (store.hooks.afterAdd) {
          await store.hooks.afterAdd(rec);
        }

        return {
          results: [
            {
              id,
              memory: text,
              metadata: { event: "ADD" },
            },
          ],
        };
      },

      async search(query: string, config: SearchMemoryOptions): Promise<SearchResult> {
        store.searchCalls.push({ query, config });
        const filters = config.filters;
        const topK = config.topK ?? 20;

        const matching: Array<{ rec: StoredFakeRecord; score: number }> = [];
        for (const rec of store.records.values()) {
          if (!matchesFilters(rec.payload, filters)) {
            continue;
          }
          const score = rec.baseScore ?? 0.5;
          matching.push({ rec, score });
        }

        matching.sort((a, b) => b.score - a.score || (a.rec.id < b.rec.id ? -1 : 1));
        const sliced = matching.slice(0, topK);

        const results: MemoryItem[] = sliced.map(({ rec, score }) => {
          const metadata: Record<string, unknown> = {};
          for (const [k, v] of Object.entries(rec.payload)) {
            if (k !== "data" && k !== "createdAt" && k !== "updatedAt" && k !== "hash") {
              metadata[k] = v;
            }
          }
          return {
            id: rec.id,
            memory: rec.memory,
            score,
            createdAt: rec.createdAt,
            updatedAt: rec.updatedAt,
            metadata,
          };
        });

        return { results };
      },

      async get(memoryId: string): Promise<MemoryItem | null> {
        if (store.hooks.beforeGet) {
          await store.hooks.beforeGet(memoryId);
        }
        const rec = store.records.get(memoryId);
        if (!rec) return null;

        const metadata: Record<string, unknown> = {};
        for (const [k, v] of Object.entries(rec.payload)) {
          if (
            k !== "data" &&
            k !== "createdAt" &&
            k !== "updatedAt" &&
            k !== "hash" &&
            k !== "user_id" &&
            k !== "agent_id" &&
            k !== "run_id"
          ) {
            metadata[k] = v;
          }
        }

        return {
          id: rec.id,
          memory: rec.memory,
          createdAt: rec.createdAt,
          updatedAt: rec.updatedAt,
          metadata,
          ...(typeof rec.payload.user_id === "string" ? { user_id: rec.payload.user_id } : {}),
          ...(typeof rec.payload.agent_id === "string" ? { agent_id: rec.payload.agent_id } : {}),
          ...(typeof rec.payload.run_id === "string" ? { run_id: rec.payload.run_id } : {}),
        } as MemoryItem;
      },

      async delete(memoryId: string): Promise<{ message: string }> {
        store.deleteCalls.push(memoryId);
        if (store.hooks.beforeDelete) {
          await store.hooks.beforeDelete(memoryId);
        }
        store.records.delete(memoryId);
        if (store.hooks.afterDelete) {
          await store.hooks.afterDelete(memoryId);
        }
        return { message: "Memory deleted successfully!" };
      },
    };
  }

  asQdrantClient(): QdrantClientLike {
    const store = this;
    return {
      async scroll(collectionName, params): Promise<QdrantScrollResponse> {
        store.scrollCalls.push({
          collectionName,
          limit: params.limit,
          offset: params.offset,
          withVector: params.with_vector,
        });

        const allMatching = Array.from(store.records.values()).filter((rec) =>
          matchesQdrantMust(rec.payload, params.filter?.must),
        );

        const startIdx = typeof params.offset === "number" ? params.offset : 0;
        const limit = params.limit ?? 100;
        const page = allMatching.slice(startIdx, startIdx + limit);
        const nextOffset = startIdx + page.length < allMatching.length ? startIdx + page.length : null;

        const points: QdrantScrollPoint[] = page.map((rec) => ({
          id: rec.id,
          payload: { ...rec.payload },
          ...(params.with_vector ? { vector: [...rec.vector] } : {}),
        }));

        return {
          points,
          next_page_offset: nextOffset,
        };
      },
    };
  }
}
