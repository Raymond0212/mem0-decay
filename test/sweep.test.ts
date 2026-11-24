import { describe, expect, it, vi } from "vitest";
import {
  DECAY_METADATA_KEYS,
  DEFAULT_POLICY_IDS,
  MS_PER_DAY,
  MS_PER_HOUR,
  createDecayMemory,
  createQdrantReader,
  type ArchiveRequest,
} from "../src/index.js";
import { FakeMem0QdrantStore } from "./helpers/fake-store.js";

const DAY = MS_PER_DAY;
const NOW = Date.UTC(2026, 5, 1, 12, 0, 0);
const SCOPE = { tenant: "acme", userId: "alice" };
const OTHER_SCOPE = { tenant: "acme", userId: "bob" };

describe("sweep() preview, archive, and delete", () => {
  it("runs a multi-page dry run, finds page-2 eligible record, skips page-1 backfill/pinned/unmanaged, and mutates nothing", async () => {
    const fake = new FakeMem0QdrantStore();

    // Page 1 (pageSize = 3): recent backfill (in grace), pinned old, unmanaged legacy
    fake.seed({
      id: "p1-backfill",
      memory: "Old event ingested 1 hour ago",
      payload: {
        user_id: "alice",
        [DECAY_METADATA_KEYS.tenant]: "acme",
        [DECAY_METADATA_KEYS.timeStart]: NOW - 60 * DAY,
        [DECAY_METADATA_KEYS.timeEnd]: NOW - 60 * DAY,
        [DECAY_METADATA_KEYS.ingestedAt]: NOW - MS_PER_HOUR, // inside 24h grace
        [DECAY_METADATA_KEYS.policy]: DEFAULT_POLICY_IDS.ephemeral,
        [DECAY_METADATA_KEYS.tier]: 0,
      },
    });
    fake.seed({
      id: "p1-pinned",
      memory: "Very old pinned fact",
      payload: {
        user_id: "alice",
        [DECAY_METADATA_KEYS.tenant]: "acme",
        [DECAY_METADATA_KEYS.timeStart]: NOW - 200 * DAY,
        [DECAY_METADATA_KEYS.timeEnd]: NOW - 200 * DAY,
        [DECAY_METADATA_KEYS.ingestedAt]: NOW - 200 * DAY,
        [DECAY_METADATA_KEYS.policy]: DEFAULT_POLICY_IDS.pinned,
        [DECAY_METADATA_KEYS.tier]: 0,
      },
    });
    fake.seed({
      id: "p1-unmanaged",
      memory: "Legacy unmanaged fact",
      payload: {
        user_id: "alice",
      },
    });

    // Page 2: eligible decayed ephemeral, unknown policy revision, malformed metadata
    fake.seed({
      id: "p2-eligible",
      memory: "Stale ephemeral note",
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
    fake.seed({
      id: "p2-unknown-rev",
      memory: "Future policy revision",
      payload: {
        user_id: "alice",
        [DECAY_METADATA_KEYS.tenant]: "acme",
        [DECAY_METADATA_KEYS.timeStart]: NOW - 60 * DAY,
        [DECAY_METADATA_KEYS.timeEnd]: NOW - 60 * DAY,
        [DECAY_METADATA_KEYS.ingestedAt]: NOW - 60 * DAY,
        [DECAY_METADATA_KEYS.policy]: "ephemeral@v99",
        [DECAY_METADATA_KEYS.tier]: 0,
      },
    });
    fake.seed({
      id: "p2-malformed",
      memory: "Corrupted decay fields",
      payload: {
        user_id: "alice",
        [DECAY_METADATA_KEYS.tenant]: "acme",
        [DECAY_METADATA_KEYS.timeStart]: NOW,
        [DECAY_METADATA_KEYS.timeEnd]: NOW - DAY, // invalid start > end
        [DECAY_METADATA_KEYS.ingestedAt]: NOW - 60 * DAY,
        [DECAY_METADATA_KEYS.policy]: DEFAULT_POLICY_IDS.ephemeral,
        [DECAY_METADATA_KEYS.tier]: 0,
      },
    });

    const archiveFn = vi.fn();
    const summarizeFn = vi.fn();

    const wrapper = createDecayMemory({
      storeId: "store-1",
      memory: fake.asMem0(),
      nowMs: () => NOW,
      qdrantReader: createQdrantReader({
        client: fake.asQdrantClient(),
        collectionName: "memories",
      }),
      scopeSweepPolicy: { action: "archive", expiryAction: "delete" },
      archiveSink: { archive: archiveFn },
      summarizer: { summarize: summarizeFn },
    });

    // Default dryRun = true
    const report = await wrapper.sweep({ scope: SCOPE, pageSize: 3 });
    expect(report.dryRun).toBe(true);
    expect(report.scanned).toBe(6);
    expect(report.eligible).toBe(1);
    expect(report.plannedMutations).toBe(1);
    expect(report.appliedMutations).toBe(0);
    expect(report.plannedActions).toEqual([
      {
        action: "archive",
        reason: "decay",
        sourceIds: ["p2-eligible"],
        estimatedMutations: 1,
      },
    ]);
    expect(report.skipped.grace_period).toBe(1);
    expect(report.skipped.exempt_policy).toBe(1);
    expect(report.skipped.unmanaged).toBe(1);
    expect(report.skipped.unknown_policy).toBe(1);
    expect(report.skipped.malformed_metadata).toBe(1);
    expect(report.hasMore).toBe(false);
    expect(report.partial).toBe(false);

    // Zero side effects in dry run
    expect(fake.addCalls).toHaveLength(0);
    expect(fake.deleteCalls).toHaveLength(0);
    expect(archiveFn).not.toHaveBeenCalled();
    expect(summarizeFn).not.toHaveBeenCalled();
    // Should not request vectors when action is not compact
    expect(fake.scrollCalls.every((c) => c.withVector === false)).toBe(true);

    // Bounded scan returns cursor and marks pending work
    const bounded = await wrapper.sweep({ scope: SCOPE, pageSize: 2, maxScanned: 2 });
    expect(bounded.scanned).toBe(2);
    expect(bounded.hasMore).toBe(true);
    expect(bounded.partial).toBe(true);
    expect(bounded.pendingWork).toBe(true);
    expect(bounded.nextCursor).not.toBeNull();

    // Resuming from nextCursor scans the rest and finds p2-eligible
    const resumed = await wrapper.sweep({
      scope: SCOPE,
      pageSize: 2,
      cursor: bounded.nextCursor,
    });
    expect(resumed.scanned).toBe(4);
    expect(resumed.eligible).toBe(1);
    expect(resumed.hasMore).toBe(false);
  });

  it("handles archive failure, delete failure with identical retry archive key, post-plan conflict, budget, and scope isolation", async () => {
    const fake = new FakeMem0QdrantStore();

    const makeStale = (id: string, userId: string, overrides: Record<string, unknown> = {}) =>
      fake.seed({
        id,
        memory: `Memory ${id}`,
        payload: {
          user_id: userId,
          [DECAY_METADATA_KEYS.tenant]: "acme",
          [DECAY_METADATA_KEYS.timeStart]: NOW - 60 * DAY,
          [DECAY_METADATA_KEYS.timeEnd]: NOW - 60 * DAY,
          [DECAY_METADATA_KEYS.ingestedAt]: NOW - 60 * DAY,
          [DECAY_METADATA_KEYS.policy]: DEFAULT_POLICY_IDS.ephemeral,
          [DECAY_METADATA_KEYS.tier]: 0,
          ...overrides,
        },
      });

    makeStale("m-archive-fail", "alice");
    makeStale("m-delete-fail-once", "alice");
    makeStale("m-pinned-midflight", "alice");
    makeStale("m-other-scope", "bob");

    let failArchiveFor = new Set<string>(["m-archive-fail"]);
    let failDeleteOnce = true;
    const archivedRequests: ArchiveRequest[] = [];

    fake.hooks.beforeDelete = async (id) => {
      if (id === "m-delete-fail-once" && failDeleteOnce) {
        failDeleteOnce = false;
        throw new Error("Simulated Qdrant delete timeout");
      }
    };

    const wrapper = createDecayMemory({
      storeId: "store-1",
      memory: fake.asMem0(),
      nowMs: () => NOW,
      qdrantReader: createQdrantReader({
        client: fake.asQdrantClient(),
        collectionName: "memories",
      }),
      scopeSweepPolicy: { action: "archive" },
      archiveSink: {
        async archive(req) {
          archivedRequests.push(req);
          if (failArchiveFor.has(req.sourceId)) {
            throw new Error("Archive S3 unavailable");
          }
          // Simulate concurrent re-pin right after archive confirmation, before coordinator recheck
          if (req.sourceId === "m-pinned-midflight") {
            const rec = fake.records.get("m-pinned-midflight")!;
            rec.payload[DECAY_METADATA_KEYS.policy] = DEFAULT_POLICY_IDS.pinned;
          }
          return { confirmed: true, key: req.key };
        },
      },
    });

    const firstRun = await wrapper.sweep({ scope: SCOPE, dryRun: false, pageSize: 2 });
    // m-archive-fail: archive throws -> source preserved
    expect(fake.records.has("m-archive-fail")).toBe(true);
    // m-delete-fail-once: archive succeeded, delete threw -> source preserved
    expect(fake.records.has("m-delete-fail-once")).toBe(true);
    // m-pinned-midflight: pinned before coordinator recheck -> preserved as revision_conflict
    expect(fake.records.has("m-pinned-midflight")).toBe(true);
    expect(firstRun.skipped.revision_conflict).toBe(1);
    expect(firstRun.errors).toHaveLength(2);
    // Unrelated scope untouched
    expect(fake.records.has("m-other-scope")).toBe(true);

    const firstDeleteFailArchiveKey = archivedRequests.find(
      (r) => r.sourceId === "m-delete-fail-once",
    )!.key;

    // Second run: archive succeeds for m-archive-fail, retry for m-delete-fail-once uses the SAME archive key
    failArchiveFor.clear();
    const secondRun = await wrapper.sweep({
      scope: SCOPE,
      dryRun: false,
      pageSize: 1,
      maxMutationsPerRun: 1, // enforce budget across multiple pages
    });
    expect(secondRun.appliedMutations).toBe(1);
    expect(secondRun.skipped.budget_exhausted).toBe(1);
    expect(secondRun.pendingWork).toBe(true);

    // Third run finishes remaining eligible record
    const thirdRun = await wrapper.sweep({
      scope: SCOPE,
      dryRun: false,
      pageSize: 1,
      maxMutationsPerRun: 5,
    });
    expect(thirdRun.appliedMutations).toBe(1);
    expect(fake.records.has("m-archive-fail")).toBe(false);
    expect(fake.records.has("m-delete-fail-once")).toBe(false);
    expect(fake.records.has("m-pinned-midflight")).toBe(true);
    expect(fake.records.has("m-other-scope")).toBe(true);

    const deleteFailKeys = archivedRequests
      .filter((r) => r.sourceId === "m-delete-fail-once")
      .map((r) => r.key);
    expect(deleteFailKeys).toHaveLength(2);
    expect(deleteFailKeys[0]).toBe(firstDeleteFailArchiveKey);
    expect(deleteFailKeys[1]).toBe(firstDeleteFailArchiveKey);
  });

  it("supports direct delete without calling archiveSink and reconciles already-absent sources", async () => {
    const fake = new FakeMem0QdrantStore();
    fake.seed({
      id: "m-expired",
      memory: "Explicitly expired soft memory",
      payload: {
        user_id: "alice",
        [DECAY_METADATA_KEYS.tenant]: "acme",
        [DECAY_METADATA_KEYS.timeStart]: NOW - DAY,
        [DECAY_METADATA_KEYS.timeEnd]: NOW - DAY,
        [DECAY_METADATA_KEYS.ingestedAt]: NOW - DAY,
        [DECAY_METADATA_KEYS.policy]: DEFAULT_POLICY_IDS.soft,
        [DECAY_METADATA_KEYS.tier]: 0,
        [DECAY_METADATA_KEYS.expiresAt]: NOW - 1000,
      },
    });
    fake.seed({
      id: "m-vanished",
      memory: "Disappears before delete",
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

    fake.hooks.beforeGet = async (id) => {
      if (id === "m-vanished") {
        fake.records.delete("m-vanished");
      }
    };

    const archiveFn = vi.fn();
    const wrapper = createDecayMemory({
      storeId: "store-1",
      memory: fake.asMem0(),
      nowMs: () => NOW,
      qdrantReader: createQdrantReader({
        client: fake.asQdrantClient(),
        collectionName: "memories",
      }),
      scopeSweepPolicy: { action: "delete", expiryAction: "delete" },
      archiveSink: { archive: archiveFn },
    });

    const report = await wrapper.sweep({ scope: SCOPE, dryRun: false });
    expect(archiveFn).not.toHaveBeenCalled();
    expect(fake.records.has("m-expired")).toBe(false);
    expect(fake.records.has("m-vanished")).toBe(false);
    expect(report.errors).toHaveLength(0);
    expect(report.appliedMutations).toBe(1); // m-vanished reconciled as already absent
    expect(OTHER_SCOPE.userId).toBe("bob");
  });
});
