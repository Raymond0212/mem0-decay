import { describe, expect, it, vi } from "vitest";
import {
  DECAY_METADATA_KEYS,
  DEFAULT_POLICY_IDS,
  InMemoryAccessStore,
  MS_PER_DAY,
  createDecayMemory,
  createQdrantReader,
  serializeAccessKey,
  type AccessStore,
} from "../src/index.js";
import { FakeMem0QdrantStore } from "./helpers/fake-store.js";

const DAY = MS_PER_DAY;
const NOW = Date.UTC(2026, 5, 1, 12, 0, 0);
const SCOPE = { tenant: "acme", userId: "alice" };

describe("optional access sidecar", () => {
  it("deduplicates eventId deliveries, keeps lastUsedAt monotonic, and isolates tenant/store keys", async () => {
    const store = new InMemoryAccessStore();
    const keyA = { tenant: "acme", storeId: "store-1", memoryId: "m1" };
    const keyB = { tenant: "acme", storeId: "store-2", memoryId: "m1" };
    const keyC = { tenant: "other", storeId: "store-1", memoryId: "m1" };

    // Duplicate delivery of evt-1 counts once
    await store.recordUse(keyA, "evt-1", NOW - 2 * DAY);
    await store.recordUse(keyA, "evt-1", NOW - 2 * DAY);

    // Older event (evt-2) increments count but CANNOT move lastUsedAt backward
    await store.recordUse(keyA, "evt-2", NOW - 10 * DAY);

    // Distinct store and tenant with same memoryId remain isolated
    await store.recordUse(keyB, "evt-1", NOW - 5 * DAY);

    const map = await store.getMany([keyA, keyB, keyC]);
    expect(map.get(serializeAccessKey(keyA))).toEqual({
      lastUsedAt: NOW - 2 * DAY,
      accessCount: 2,
    });
    expect(map.get(serializeAccessKey(keyB))).toEqual({
      lastUsedAt: NOW - 5 * DAY,
      accessCount: 1,
    });
    expect(map.has(serializeAccessKey(keyC))).toBe(false);
  });

  it("reduces effective ranking age up to maxAgeReductionDays and falls back cleanly on sidecar failure/timeout", async () => {
    const fake = new FakeMem0QdrantStore();
    fake.seed({
      id: "m-old",
      memory: "Frequently referenced note",
      baseScore: 0.8,
      payload: {
        user_id: "alice",
        [DECAY_METADATA_KEYS.tenant]: "acme",
        [DECAY_METADATA_KEYS.timeStart]: NOW - 30 * DAY,
        [DECAY_METADATA_KEYS.timeEnd]: NOW - 30 * DAY,
        [DECAY_METADATA_KEYS.ingestedAt]: NOW - 30 * DAY,
        [DECAY_METADATA_KEYS.policy]: DEFAULT_POLICY_IDS.soft,
        [DECAY_METADATA_KEYS.tier]: 0,
      },
    });

    const accessStore = new InMemoryAccessStore();
    const wrapperAgeOnly = createDecayMemory({
      storeId: "store-1",
      memory: fake.asMem0(),
      nowMs: () => NOW,
    });
    const wrapperReinforced = createDecayMemory({
      storeId: "store-1",
      memory: fake.asMem0(),
      nowMs: () => NOW,
      accessSidecar: {
        store: accessStore,
        maxAgeReductionDays: 10,
      },
    });

    const ageOnly = await wrapperAgeOnly.search("note", { scope: SCOPE, topK: 5 });
    expect(ageOnly.results[0]!.adjustedScore).toBeCloseTo(0.52, 12);

    // Record confirmed use today -> reduces effective age from 30 days to 20 days (capped at 10 days)
    await wrapperReinforced.recordUse(SCOPE, "m-old", "use-1", NOW);
    const boosted = await wrapperReinforced.search("note", { scope: SCOPE, topK: 5 });
    const expectedStrength = Math.pow(0.5, 20 / 30);
    const expectedScore = 0.8 * (0.3 + 0.7 * expectedStrength);
    expect(boosted.results[0]!.adjustedScore).toBeCloseTo(expectedScore, 12);
    expect(boosted.results[0]!.adjustedScore).toBeGreaterThan(ageOnly.results[0]!.adjustedScore);

    // Sidecar failure falls back to exact same score as age-only search
    const failingStore: AccessStore = {
      async getMany() {
        throw new Error("Redis connection refused");
      },
      async recordUse() {},
    };
    const wrapperFailing = createDecayMemory({
      storeId: "store-1",
      memory: fake.asMem0(),
      nowMs: () => NOW,
      accessSidecar: { store: failingStore },
    });
    const fallbackError = await wrapperFailing.search("note", { scope: SCOPE, topK: 5 });
    expect(fallbackError.results[0]!.adjustedScore).toBeCloseTo(ageOnly.results[0]!.adjustedScore, 12);

    // Sidecar timeout falls back to exact same score as age-only search
    const slowStore: AccessStore = {
      async getMany() {
        await new Promise((r) => setTimeout(r, 80));
        return new Map();
      },
      async recordUse() {},
    };
    const wrapperTimeout = createDecayMemory({
      storeId: "store-1",
      memory: fake.asMem0(),
      nowMs: () => NOW,
      accessSidecar: { store: slowStore, readTimeoutMs: 10 },
    });
    const fallbackTimeout = await wrapperTimeout.search("note", { scope: SCOPE, topK: 5 });
    expect(fallbackTimeout.results[0]!.adjustedScore).toBeCloseTo(
      ageOnly.results[0]!.adjustedScore,
      12,
    );
  });

  it("issues ZERO sidecar requests for strict range, soft range, anchor queries, and sweeps", async () => {
    const fake = new FakeMem0QdrantStore();
    fake.seed({
      id: "m1",
      memory: "Historical note",
      baseScore: 0.8,
      payload: {
        user_id: "alice",
        [DECAY_METADATA_KEYS.tenant]: "acme",
        [DECAY_METADATA_KEYS.timeStart]: NOW - 60 * DAY,
        [DECAY_METADATA_KEYS.timeEnd]: NOW - 60 * DAY,
        [DECAY_METADATA_KEYS.ingestedAt]: NOW - 60 * DAY,
        [DECAY_METADATA_KEYS.policy]: DEFAULT_POLICY_IDS.ephemeral,
        [DECAY_METADATA_KEYS.tier]: 0,
      },
    });

    const getManySpy = vi.fn(async () => new Map());
    const recordUseSpy = vi.fn(async () => {});

    const wrapper = createDecayMemory({
      storeId: "store-1",
      memory: fake.asMem0(),
      nowMs: () => NOW,
      qdrantReader: createQdrantReader({
        client: fake.asQdrantClient(),
        collectionName: "memories",
      }),
      scopeSweepPolicy: { action: "delete" },
      accessSidecar: {
        store: {
          getMany: getManySpy,
          recordUse: recordUseSpy,
        },
      },
    });

    await wrapper.search("note", {
      scope: SCOPE,
      range: { from: NOW - 90 * DAY, to: NOW - 30 * DAY, strict: true },
    });
    await wrapper.search("note", {
      scope: SCOPE,
      range: { from: NOW - 90 * DAY, to: NOW - 30 * DAY, strict: false },
    });
    await wrapper.search("note", {
      scope: SCOPE,
      anchor: { around: NOW - 60 * DAY },
    });
    await wrapper.sweep({ scope: SCOPE, dryRun: true });
    await wrapper.sweep({ scope: SCOPE, dryRun: false });

    expect(getManySpy).not.toHaveBeenCalled();
    expect(recordUseSpy).not.toHaveBeenCalled();
  });
});
