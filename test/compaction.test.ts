import { describe, expect, it, vi } from "vitest";
import {
  DECAY_METADATA_KEYS,
  DEFAULT_POLICY_IDS,
  InMemoryCompactionWorkStore,
  MS_PER_DAY,
  createDecayMemory,
  createQdrantReader,
} from "../src/index.js";
import { FakeMem0QdrantStore } from "./helpers/fake-store.js";

const DAY = MS_PER_DAY;
const NOW = Date.UTC(2026, 5, 1, 12, 0, 0);
const SCOPE = { tenant: "acme", userId: "alice" };

function seedEphemeral(
  fake: FakeMem0QdrantStore,
  id: string,
  opts: {
    userId?: string;
    tenant?: string;
    eventMs?: number;
    vector?: number[];
    expiresAt?: number;
    memory?: string;
  } = {},
) {
  const eventMs = opts.eventMs ?? NOW - 60 * DAY;
  return fake.seed({
    id,
    memory: opts.memory ?? `Fact ${id}`,
    vector: opts.vector ?? [0.9, 0.1, 0, 0],
    payload: {
      user_id: opts.userId ?? "alice",
      [DECAY_METADATA_KEYS.tenant]: opts.tenant ?? "acme",
      [DECAY_METADATA_KEYS.timeStart]: eventMs,
      [DECAY_METADATA_KEYS.timeEnd]: eventMs,
      [DECAY_METADATA_KEYS.ingestedAt]: NOW - 60 * DAY,
      [DECAY_METADATA_KEYS.policy]: DEFAULT_POLICY_IDS.ephemeral,
      [DECAY_METADATA_KEYS.tier]: 0,
      ...(opts.expiresAt !== undefined ? { [DECAY_METADATA_KEYS.expiresAt]: opts.expiresAt } : {}),
    },
  });
}

describe("Tier 0 -> Tier 1 resumable compaction", () => {
  it("compacts similar Tier 0 records into one Tier 1 summary with min/max span and inherited earliest expiry", async () => {
    const fake = new FakeMem0QdrantStore();
    const workStore = new InMemoryCompactionWorkStore();

    // Choose two timestamps within the same 7-day time bucket
    const t1 = NOW - 60 * DAY;
    const t2 = NOW - 58 * DAY;
    const exp1 = NOW + 90 * DAY;
    const exp2 = NOW + 30 * DAY;

    seedEphemeral(fake, "s1", { eventMs: t1, expiresAt: exp1, vector: [1, 0, 0, 0] });
    seedEphemeral(fake, "s2", { eventMs: t2, expiresAt: exp2, vector: [0.99, 0.1, 0, 0] });
    // Different scope must NEVER be grouped with alice
    seedEphemeral(fake, "s-bob", { userId: "bob", eventMs: t1, vector: [1, 0, 0, 0] });

    const wrapper = createDecayMemory({
      storeId: "store-1",
      memory: fake.asMem0(),
      nowMs: () => NOW,
      qdrantReader: createQdrantReader({
        client: fake.asQdrantClient(),
        collectionName: "memories",
      }),
      scopeSweepPolicy: { action: "compact", expiryAction: "delete" },
      compactionWorkStore: workStore,
      summarizer: {
        async summarize({ sources }) {
          return { summary: `Summary of ${sources.map((s) => s.id).join(",")}` };
        },
      },
    });

    const report = await wrapper.sweep({ scope: SCOPE, dryRun: false });
    expect(report.errors).toHaveLength(0);
    expect(report.appliedMutations).toBe(3); // 1 summary write + 2 source deletions
    expect(fake.records.has("s1")).toBe(false);
    expect(fake.records.has("s2")).toBe(false);
    expect(fake.records.has("s-bob")).toBe(true);

    const summaries = Array.from(fake.records.values()).filter(
      (r) => r.payload[DECAY_METADATA_KEYS.tier] === 1,
    );
    expect(summaries).toHaveLength(1);
    const summary = summaries[0]!;
    expect(summary.memory).toBe("Summary of s1,s2");
    expect(summary.payload[DECAY_METADATA_KEYS.timeStart]).toBe(t1);
    expect(summary.payload[DECAY_METADATA_KEYS.timeEnd]).toBe(t2);
    expect(summary.payload[DECAY_METADATA_KEYS.ingestedAt]).toBe(NOW);
    expect(summary.payload[DECAY_METADATA_KEYS.expiresAt]).toBe(exp2); // earliest expiry inherited
    expect(summary.payload[DECAY_METADATA_KEYS.policy]).toBe(DEFAULT_POLICY_IDS.tier1Summary);
    expect(Array.isArray(summary.payload[DECAY_METADATA_KEYS.sources])).toBe(true);
    expect(fake.addCalls[0]!.config.infer).toBe(false);
  });

  it("recovers cleanly after a crash right after summary creation and after the first source deletion", async () => {
    const fake = new FakeMem0QdrantStore();
    const workStore = new InMemoryCompactionWorkStore();

    seedEphemeral(fake, "c1", { eventMs: NOW - 60 * DAY, vector: [1, 0, 0, 0] });
    seedEphemeral(fake, "c2", { eventMs: NOW - 59 * DAY, vector: [1, 0, 0, 0] });
    seedEphemeral(fake, "c3", { eventMs: NOW - 58 * DAY, vector: [1, 0, 0, 0] });

    let failOnDelete: string | null = "all";
    fake.hooks.beforeDelete = async (id) => {
      if (failOnDelete === "all") {
        throw new Error(`Simulated crash right after summary creation before deleting ${id}`);
      }
      if (failOnDelete === "after-c1" && id !== "c1") {
        throw new Error(`Simulated crash after first source deletion before deleting ${id}`);
      }
    };

    const summarizeSpy = vi.fn(async ({ sources }: { sources: Array<{ id: string }> }) => ({
      summary: `Consolidated: ${sources.map((s) => s.id).join("+")}`,
    }));

    const wrapper = createDecayMemory({
      storeId: "store-1",
      memory: fake.asMem0(),
      nowMs: () => NOW,
      qdrantReader: createQdrantReader({
        client: fake.asQdrantClient(),
        collectionName: "memories",
      }),
      scopeSweepPolicy: { action: "compact", expiryAction: "delete" },
      compactionWorkStore: workStore,
      summarizer: { summarize: summarizeSpy },
    });

    // Run 1: summary created, crashes before any source deletion completes
    const run1 = await wrapper.sweep({ scope: SCOPE, dryRun: false });
    expect(run1.pendingWork).toBe(true);
    expect(fake.addCalls).toHaveLength(1);
    expect(fake.records.has("c1")).toBe(true);
    expect(fake.records.has("c2")).toBe(true);
    expect(fake.records.has("c3")).toBe(true);

    // Run 2: resumes existing summary (zero new add calls!), deletes c1, crashes before c2/c3
    failOnDelete = "after-c1";
    const run2 = await wrapper.sweep({ scope: SCOPE, dryRun: false });
    expect(run2.pendingWork).toBe(true);
    expect(fake.addCalls).toHaveLength(1);
    expect(summarizeSpy).toHaveBeenCalledTimes(1);
    expect(fake.records.has("c1")).toBe(false);
    expect(fake.records.has("c2")).toBe(true);
    expect(fake.records.has("c3")).toBe(true);

    // Run 3: resumes again, finishes c2 and c3 -> converges on ONE summary
    failOnDelete = null;
    const run3 = await wrapper.sweep({ scope: SCOPE, dryRun: false });
    expect(run3.errors).toHaveLength(0);
    expect(run3.pendingWork).toBe(false);
    expect(fake.addCalls).toHaveLength(1);
    expect(fake.records.has("c1")).toBe(false);
    expect(fake.records.has("c2")).toBe(false);
    expect(fake.records.has("c3")).toBe(false);

    const summaries = Array.from(fake.records.values()).filter(
      (r) => r.payload[DECAY_METADATA_KEYS.tier] === 1,
    );
    expect(summaries).toHaveLength(1);
  });

  it("preserves originals on ambiguous write outcome, changed source, or insufficient budget (< 1 + sourceCount)", async () => {
    const fake = new FakeMem0QdrantStore();
    const workStore = new InMemoryCompactionWorkStore();

    seedEphemeral(fake, "u1", { vector: [1, 0, 0, 0] });
    seedEphemeral(fake, "u2", { vector: [1, 0, 0, 0] });

    const summarizeSpy = vi.fn(async () => ({ summary: "Summary text" }));

    const wrapper = createDecayMemory({
      storeId: "store-1",
      memory: fake.asMem0(),
      nowMs: () => NOW,
      qdrantReader: createQdrantReader({
        client: fake.asQdrantClient(),
        collectionName: "memories",
      }),
      scopeSweepPolicy: { action: "compact", expiryAction: "delete" },
      compactionWorkStore: workStore,
      summarizer: { summarize: summarizeSpy },
    });

    // 1. Budget smaller than 1 + sourceCount (need 3, give 2) leaves group intact without calling summarizer
    const lowBudget = await wrapper.sweep({
      scope: SCOPE,
      dryRun: false,
      maxMutationsPerRun: 2,
    });
    expect(summarizeSpy).not.toHaveBeenCalled();
    expect(lowBudget.skipped.budget_exhausted).toBe(2);
    expect(lowBudget.pendingWork).toBe(true);
    expect(fake.records.has("u1")).toBe(true);
    expect(fake.records.has("u2")).toBe(true);

    // 2. Ambiguous write outcome (add throws) preserves originals and surfaces unresolved compaction
    fake.hooks.beforeAdd = async () => {
      throw new Error("Network timeout during summary write");
    };
    const ambiguous = await wrapper.sweep({ scope: SCOPE, dryRun: false, maxMutationsPerRun: 10 });
    expect(ambiguous.unresolvedCompactions).toHaveLength(1);
    expect(ambiguous.unresolvedCompactions[0]!.status).toBe("uncertain");
    expect(fake.records.has("u1")).toBe(true);
    expect(fake.records.has("u2")).toBe(true);

    // Next sweep keeps surfacing unresolved work without blindly retrying add
    fake.hooks.beforeAdd = undefined;
    const afterAmbiguous = await wrapper.sweep({ scope: SCOPE, dryRun: false, maxMutationsPerRun: 10 });
    expect(afterAmbiguous.unresolvedCompactions).toHaveLength(1);
    expect(fake.addCalls).toHaveLength(1); // still only the original failed attempt
    expect(fake.records.has("u1")).toBe(true);
    expect(fake.records.has("u2")).toBe(true);
  });

  it("preserves a source modified mid-compaction and routes a source expiring mid-prep through expiry policy", async () => {
    const fake = new FakeMem0QdrantStore();
    const workStore = new InMemoryCompactionWorkStore();

    seedEphemeral(fake, "e1", { vector: [1, 0, 0, 0] });
    seedEphemeral(fake, "e2", { vector: [1, 0, 0, 0] });

    const wrapper = createDecayMemory({
      storeId: "store-1",
      memory: fake.asMem0(),
      nowMs: () => NOW,
      qdrantReader: createQdrantReader({
        client: fake.asQdrantClient(),
        collectionName: "memories",
      }),
      scopeSweepPolicy: { action: "compact", expiryAction: "delete" },
      compactionWorkStore: workStore,
      summarizer: {
        async summarize() {
          // Simulate e1 expiring while summary is being generated
          fake.records.get("e1")!.payload[DECAY_METADATA_KEYS.expiresAt] = NOW - 500;
          return { summary: "Summary" };
        },
      },
    });

    await wrapper.sweep({ scope: SCOPE, dryRun: false });
    // e1 was routed through expiry policy (deleted), e2 preserved, no summary written
    expect(fake.records.has("e1")).toBe(false);
    expect(fake.records.has("e2")).toBe(true);
    expect(fake.addCalls).toHaveLength(0);

    // Now test mid-flight re-pin conflict
    const fake2 = new FakeMem0QdrantStore();
    seedEphemeral(fake2, "r1", { vector: [1, 0, 0, 0] });
    seedEphemeral(fake2, "r2", { vector: [1, 0, 0, 0] });

    const wrapper2 = createDecayMemory({
      storeId: "store-1",
      memory: fake2.asMem0(),
      nowMs: () => NOW,
      qdrantReader: createQdrantReader({
        client: fake2.asQdrantClient(),
        collectionName: "memories",
      }),
      scopeSweepPolicy: { action: "compact", expiryAction: "delete" },
      summarizer: {
        async summarize() {
          fake2.records.get("r1")!.payload[DECAY_METADATA_KEYS.policy] = DEFAULT_POLICY_IDS.pinned;
          return { summary: "Should abort" };
        },
      },
    });

    const res2 = await wrapper2.sweep({ scope: SCOPE, dryRun: false });
    expect(res2.skipped.revision_conflict).toBe(1);
    expect(fake2.records.has("r1")).toBe(true);
    expect(fake2.records.has("r2")).toBe(true);
    expect(fake2.addCalls).toHaveLength(0);
  });
});
