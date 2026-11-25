import { describe, expect, it, vi } from "vitest";
import {
  DECAY_METADATA_KEYS,
  DEFAULT_POLICY_IDS,
  InMemoryAccessStore,
  MS_PER_DAY,
  createDecayMemory,
  createDecayRouter,
  createInProcessWriteCoordinator,
  createQdrantReader,
} from "../src/index.js";
import { FakeMem0QdrantStore } from "./helpers/fake-store.js";

const DAY = MS_PER_DAY;
const NOW = Date.UTC(2026, 5, 1, 12, 0, 0);
const SCOPE = { tenant: "acme", userId: "alice" };

describe("host router and single-writer coordinator", () => {
  it("fans out searches with one frozen clock, merges by adjustedScore, and enforces scope authorization", async () => {
    const fakeA = new FakeMem0QdrantStore();
    const fakeB = new FakeMem0QdrantStore();

    fakeA.seed({
      id: "a-old",
      memory: "Store A old hit",
      baseScore: 0.85,
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
    fakeB.seed({
      id: "b-fresh",
      memory: "Store B fresh hit",
      baseScore: 0.84,
      payload: {
        user_id: "alice",
        [DECAY_METADATA_KEYS.tenant]: "acme",
        [DECAY_METADATA_KEYS.timeStart]: NOW,
        [DECAY_METADATA_KEYS.timeEnd]: NOW,
        [DECAY_METADATA_KEYS.ingestedAt]: NOW,
        [DECAY_METADATA_KEYS.policy]: DEFAULT_POLICY_IDS.soft,
        [DECAY_METADATA_KEYS.tier]: 0,
      },
    });

    let tick = NOW;
    const router = createDecayRouter({
      nowMs: () => {
        const val = tick;
        tick += 5000; // would drift if not frozen once per search
        return val;
      },
      authorize: ({ scope, storeId }) => {
        if (scope.tenant === "blocked") return false;
        if (scope.userId === "readonly-a" && storeId === "store-b") return false;
        return true;
      },
    });

    router.registerStore(
      createDecayMemory({
        storeId: "store-a",
        memory: fakeA.asMem0(),
        writeCoordinator: router.writeCoordinator,
      }),
      { activeWriter: true },
    );
    router.registerStore(
      createDecayMemory({
        storeId: "store-b",
        memory: fakeB.asMem0(),
        writeCoordinator: router.writeCoordinator,
      }),
    );

    const res = await router.search("hit", { scope: SCOPE, topK: 2 });
    expect(res.referenceNowMs).toBe(NOW);
    expect(res.results.map((r) => [r.storeId, r.id])).toEqual([
      ["store-b", "b-fresh"], // 0.84
      ["store-a", "a-old"], // 0.85 - 0.85 * (1 - 0.65) * 0.05 = 0.835125
    ]);
    expect(res.results[1]!.adjustedScore).toBeCloseTo(0.835125, 12);

    // Adjustable alpha forwarded across router fan-out
    const unpenalized = await router.search("hit", { scope: SCOPE, topK: 2, alpha: 0 });
    expect(unpenalized.results.map((r) => [r.storeId, r.id])).toEqual([
      ["store-a", "a-old"], // 0.85
      ["store-b", "b-fresh"], // 0.84
    ]);

    await expect(
      router.add("blocked write", { scope: { tenant: "blocked", userId: "alice" } }),
    ).rejects.toThrow(/Unauthorized/);
  });

  it("enforces the single-writer boundary between a concurrent agent write and a sweep targeting a different store", async () => {
    const fakeA = new FakeMem0QdrantStore();
    const fakeB = new FakeMem0QdrantStore();

    fakeB.seed({
      id: "b-stale",
      memory: "Stale item in store B",
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

    const timeline: string[] = [];
    let activeWriters = 0;
    let maxConcurrentWriters = 0;

    const enterCritical = (label: string) => {
      activeWriters += 1;
      maxConcurrentWriters = Math.max(maxConcurrentWriters, activeWriters);
      timeline.push(`start:${label}`);
    };
    const exitCritical = (label: string) => {
      timeline.push(`end:${label}`);
      activeWriters -= 1;
    };

    fakeA.hooks.beforeAdd = async () => {
      enterCritical("add:store-a");
      await new Promise((r) => setTimeout(r, 30));
    };
    fakeA.hooks.afterAdd = async () => {
      exitCritical("add:store-a");
    };

    fakeB.hooks.beforeDelete = async () => {
      enterCritical("delete:store-b");
      await new Promise((r) => setTimeout(r, 30));
    };
    fakeB.hooks.afterDelete = async () => {
      exitCritical("delete:store-b");
    };

    const accessStore = new InMemoryAccessStore();
    const closeSpy = vi.spyOn(accessStore, "close");

    const coordinator = createInProcessWriteCoordinator({ serializeAllStores: true });
    const router = createDecayRouter({
      nowMs: () => NOW,
      writeCoordinator: coordinator,
      accessStore,
    });

    router.registerStore(
      createDecayMemory({
        storeId: "store-a",
        memory: fakeA.asMem0(),
        nowMs: () => NOW,
        writeCoordinator: coordinator,
        accessSidecar: { store: accessStore },
      }),
      { activeWriter: true },
    );

    router.registerStore(
      createDecayMemory({
        storeId: "store-b",
        memory: fakeB.asMem0(),
        nowMs: () => NOW,
        writeCoordinator: coordinator,
        qdrantReader: createQdrantReader({
          client: fakeB.asQdrantClient(),
          collectionName: "store_b_memories",
        }),
        scopeSweepPolicy: { action: "delete" },
        accessSidecar: { store: accessStore },
      }),
    );

    // Fire agent write (to active writer store-a) and scheduled sweep (fixed to store-b) concurrently
    const [addRes, sweepRes] = await Promise.all([
      router.add("Agent write to store A", { scope: SCOPE, infer: false }),
      router.sweep({ storeId: "store-b", options: { scope: SCOPE, dryRun: false } }),
    ]);

    expect(addRes.storeId).toBe("store-a");
    expect(sweepRes.storeId).toBe("store-b");
    expect(sweepRes.appliedMutations).toBe(1);
    expect(fakeA.records.size).toBe(1);
    expect(fakeB.records.has("b-stale")).toBe(false);

    // Single-writer boundary strictly honored: never more than 1 concurrent writer across stores
    expect(maxConcurrentWriters).toBe(1);
    expect(timeline).toHaveLength(4);
    expect(timeline[0]!.startsWith("start:")).toBe(true);
    expect(timeline[1]!.startsWith("end:")).toBe(true);
    expect(timeline[2]!.startsWith("start:")).toBe(true);
    expect(timeline[3]!.startsWith("end:")).toBe(true);

    await router.shutdown();
    expect(closeSpy).toHaveBeenCalledTimes(1);
  });
});
