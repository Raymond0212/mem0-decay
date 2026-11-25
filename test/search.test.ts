import { describe, expect, it } from "vitest";
import {
  DECAY_METADATA_KEYS,
  DEFAULT_ALPHA,
  DEFAULT_POLICY_IDS,
  MS_PER_DAY,
  createCurveRegistry,
  createDecayMemory,
  createDecayRouter,
  createPolicyRegistry,
  createQdrantReader,
  decayCurves,
  preserveEventSpanMetadata,
} from "../src/index.js";
import { FakeMem0QdrantStore } from "./helpers/fake-store.js";

const DAY = MS_PER_DAY;
const NOW = Date.UTC(2026, 5, 1, 12, 0, 0); // 2026-06-01T12:00:00Z
const SCOPE = { tenant: "acme", userId: "alice" };

describe("add() and time-aware search()", () => {
  it("stamps managed metadata, preserves infer, and protects the decay_ namespace", async () => {
    const fake = new FakeMem0QdrantStore();
    const wrapper = createDecayMemory({
      storeId: "store-1",
      memory: fake.asMem0(),
      nowMs: () => NOW,
    });

    const eventMs = NOW - 10 * DAY;
    const expiresMs = NOW + 30 * DAY;

    const res = await wrapper.add("Deployed release v2.1", {
      scope: SCOPE,
      eventTime: eventMs,
      policyId: "ephemeral",
      expiresAt: expiresMs,
      infer: false,
      metadata: { category: "ops" },
    });

    expect(res.results).toHaveLength(1);
    expect(fake.addCalls).toHaveLength(1);
    const call = fake.addCalls[0]!;
    expect(call.config.userId).toBe("alice");
    expect(call.config.infer).toBe(false);
    expect(call.config.metadata).toMatchObject({
      category: "ops",
      [DECAY_METADATA_KEYS.timeStart]: eventMs,
      [DECAY_METADATA_KEYS.timeEnd]: eventMs,
      [DECAY_METADATA_KEYS.ingestedAt]: NOW,
      [DECAY_METADATA_KEYS.policy]: DEFAULT_POLICY_IDS.ephemeral,
      [DECAY_METADATA_KEYS.tier]: 0,
      [DECAY_METADATA_KEYS.expiresAt]: expiresMs,
      [DECAY_METADATA_KEYS.tenant]: "acme",
    });

    // Reject caller attempts to overwrite reserved decay_* keys
    await expect(
      wrapper.add("Bad metadata", {
        scope: SCOPE,
        metadata: { decay_tier: 99 },
      }),
    ).rejects.toThrow(RangeError);

    // Ordinary text edits preserve event span unless explicitly corrected
    const existing = call.config.metadata!;
    const preserved = preserveEventSpanMetadata(existing, { category: "ops-edited" });
    expect(preserved[DECAY_METADATA_KEYS.timeStart]).toBe(eventMs);
    expect(preserved[DECAY_METADATA_KEYS.timeEnd]).toBe(eventMs);
    expect(preserved.category).toBe("ops-edited");
  });

  it("ranks 30-day soft record (0.80 -> 0.786 at default 5% alpha) behind fresh record (0.79), supports adjustable alpha, and preserves baseScore", async () => {
    expect(DEFAULT_ALPHA).toBe(0.05);

    const fake = new FakeMem0QdrantStore();
    fake.seed({
      id: "old-soft",
      memory: "Old preference",
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
    fake.seed({
      id: "fresh-soft",
      memory: "Fresh preference",
      baseScore: 0.79,
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

    const wrapper = createDecayMemory({
      storeId: "store-1",
      memory: fake.asMem0(),
      nowMs: () => NOW,
    });

    // Default alpha = 0.05: old-soft has timeWeight = 0.65 -> 0.8 - 0.8 * (1 - 0.65) * 0.05 = 0.786 < 0.79
    const top2 = await wrapper.search("preference", { scope: SCOPE, topK: 2 });
    expect(top2.mode).toBe("recency");
    expect(top2.results.map((r) => r.id)).toEqual(["fresh-soft", "old-soft"]);
    expect(top2.results[0]!.baseScore).toBe(0.79);
    expect(top2.results[0]!.timeWeight).toBe(1);
    expect(top2.results[0]!.adjustedScore).toBeCloseTo(0.79, 12);
    expect(top2.results[1]!.baseScore).toBe(0.8);
    expect(top2.results[1]!.timeWeight).toBeCloseTo(0.65, 12);
    expect(top2.results[1]!.adjustedScore).toBeCloseTo(0.786, 12);
    expect(top2.results[1]!.tier).toBe(0);
    expect(top2.results[1]!.timeSpan).toEqual({
      start: NOW - 30 * DAY,
      end: NOW - 30 * DAY,
    });

    const top1 = await wrapper.search("preference", { scope: SCOPE, topK: 1 });
    expect(top1.results.map((r) => r.id)).toEqual(["fresh-soft"]);

    // Per-query adjustable alpha: alpha = 0 disables decay penalty; alpha = 1 applies full (1 - timeWeight)
    const noPenalty = await wrapper.search("preference", { scope: SCOPE, topK: 2, alpha: 0 });
    expect(noPenalty.results.map((r) => r.id)).toEqual(["old-soft", "fresh-soft"]);
    expect(noPenalty.results[0]!.adjustedScore).toBeCloseTo(0.8, 12);

    const fullPenalty = await wrapper.search("preference", { scope: SCOPE, topK: 2, alpha: 1 });
    expect(fullPenalty.results.find((r) => r.id === "old-soft")!.adjustedScore).toBeCloseTo(0.52, 12);

    // Store-level adjustable alpha
    const customAlphaWrapper = createDecayMemory({
      storeId: "store-1",
      memory: fake.asMem0(),
      nowMs: () => NOW,
      alpha: 0.2,
    });
    const storeCustomRes = await customAlphaWrapper.search("preference", { scope: SCOPE, topK: 2 });
    // 0.8 - 0.8 * 0.35 * 0.2 = 0.744
    expect(storeCustomRes.results.find((r) => r.id === "old-soft")!.adjustedScore).toBeCloseTo(0.744, 12);
  });

  it("applies the same weighted alpha formula for exponential, 5-parameter sigmoid, and custom curves", async () => {
    const registry = createPolicyRegistry([
      {
        id: "sig@v1",
        curve: decayCurves.sigmoid({
          lower: 0.1,
          upper: 0.9,
          midpoint: 30,
          slope: 0.2,
          asymmetry: 2,
        }),
        floor: 0.2,
      },
      {
        id: "linear@v1",
        curve: (ageDays) => Math.max(0, (100 - ageDays) / 100),
        floor: 0.1,
      },
    ]);

    const fake = new FakeMem0QdrantStore();
    fake.seed({
      id: "rec-sig",
      memory: "Sigmoid memory",
      baseScore: 0.8,
      payload: {
        user_id: "alice",
        [DECAY_METADATA_KEYS.tenant]: "acme",
        [DECAY_METADATA_KEYS.timeStart]: NOW - 30 * DAY,
        [DECAY_METADATA_KEYS.timeEnd]: NOW - 30 * DAY,
        [DECAY_METADATA_KEYS.ingestedAt]: NOW - 30 * DAY,
        [DECAY_METADATA_KEYS.policy]: "sig@v1",
        [DECAY_METADATA_KEYS.tier]: 0,
      },
    });
    fake.seed({
      id: "rec-linear",
      memory: "Linear memory",
      baseScore: 0.8,
      payload: {
        user_id: "alice",
        [DECAY_METADATA_KEYS.tenant]: "acme",
        [DECAY_METADATA_KEYS.timeStart]: NOW - 50 * DAY,
        [DECAY_METADATA_KEYS.timeEnd]: NOW - 50 * DAY,
        [DECAY_METADATA_KEYS.ingestedAt]: NOW - 50 * DAY,
        [DECAY_METADATA_KEYS.policy]: "linear@v1",
        [DECAY_METADATA_KEYS.tier]: 0,
      },
    });

    const wrapper = createDecayMemory({
      storeId: "store-1",
      memory: fake.asMem0(),
      nowMs: () => NOW,
      policyResolver: registry,
    });

    const res = await wrapper.search("memory", { scope: SCOPE, topK: 5 });
    const sig = res.results.find((r) => r.id === "rec-sig")!;
    const lin = res.results.find((r) => r.id === "rec-linear")!;
    // 5-param sigmoid at day 30: strength = 0.1 + 0.8 * 0.5^2 = 0.3 -> timeWeight = 0.2 + 0.8 * 0.3 = 0.44
    // adjustedScore = 0.8 - 0.8 * (1 - 0.44) * 0.05 = 0.7776
    expect(sig.timeWeight).toBeCloseTo(0.44, 12);
    expect(sig.adjustedScore).toBeCloseTo(0.7776, 12);
    // Linear at day 50 has strength = 0.5 -> timeWeight = 0.1 + 0.9 * 0.5 = 0.55 -> 0.8 - 0.8 * 0.45 * 0.05 = 0.782
    expect(lin.timeWeight).toBeCloseTo(0.55, 12);
    expect(lin.adjustedScore).toBeCloseTo(0.782, 12);
  });

  it("handles strict range pre-filtering, straddling summary exclusion, soft range, anchor, and backfills", async () => {
    const fake = new FakeMem0QdrantStore();

    // Historical backfill: old event time (2025-03-10), very recent ingestion time (NOW)
    const mar10 = Date.UTC(2025, 2, 10, 10, 0, 0);
    fake.seed({
      id: "backfill-in-range",
      memory: "March 10 meeting notes",
      baseScore: 0.72,
      payload: {
        user_id: "alice",
        [DECAY_METADATA_KEYS.tenant]: "acme",
        [DECAY_METADATA_KEYS.timeStart]: mar10,
        [DECAY_METADATA_KEYS.timeEnd]: mar10,
        [DECAY_METADATA_KEYS.ingestedAt]: NOW,
        [DECAY_METADATA_KEYS.policy]: DEFAULT_POLICY_IDS.soft,
        [DECAY_METADATA_KEYS.tier]: 0,
      },
    });

    // Summary straddling the strict window [2025-03-05, 2025-03-15] -> must be excluded in strict mode
    fake.seed({
      id: "straddling-summary",
      memory: "March summary",
      baseScore: 0.95,
      payload: {
        user_id: "alice",
        [DECAY_METADATA_KEYS.tenant]: "acme",
        [DECAY_METADATA_KEYS.timeStart]: Date.UTC(2025, 2, 1),
        [DECAY_METADATA_KEYS.timeEnd]: Date.UTC(2025, 2, 20),
        [DECAY_METADATA_KEYS.ingestedAt]: NOW,
        [DECAY_METADATA_KEYS.policy]: DEFAULT_POLICY_IDS.tier1Summary,
        [DECAY_METADATA_KEYS.tier]: 1,
      },
    });

    // Pinned record far from the query window (2025-03-29 = 14 days after 2025-03-15 end)
    fake.seed({
      id: "pinned-outside",
      memory: "Pinned architecture rule",
      baseScore: 0.9,
      payload: {
        user_id: "alice",
        [DECAY_METADATA_KEYS.tenant]: "acme",
        [DECAY_METADATA_KEYS.timeStart]: Date.UTC(2025, 2, 30, 0, 0, 0),
        [DECAY_METADATA_KEYS.timeEnd]: Date.UTC(2025, 2, 30, 0, 0, 0),
        [DECAY_METADATA_KEYS.ingestedAt]: Date.UTC(2025, 2, 30, 0, 0, 0),
        [DECAY_METADATA_KEYS.policy]: DEFAULT_POLICY_IDS.pinned,
        [DECAY_METADATA_KEYS.tier]: 0,
      },
    });

    let currentNow = NOW;
    const wrapper = createDecayMemory({
      storeId: "store-1",
      memory: fake.asMem0(),
      nowMs: () => currentNow,
    });

    // 1. Strict range [2025-03-05, 2025-03-15] (date-only includes full end day -> < 2025-03-16T00:00Z)
    const strictRes = await wrapper.search("notes", {
      scope: SCOPE,
      topK: 5,
      range: { from: "2025-03-05", to: "2025-03-15" },
    });
    expect(strictRes.mode).toBe("strict_range");
    expect(strictRes.results.map((r) => r.id)).toEqual(["backfill-in-range"]);
    expect(strictRes.results[0]!.adjustedScore).toBe(0.72);
    expect(strictRes.results[0]!.timeWeight).toBe(1);

    // Verify range filters were pushed down into Mem0 search
    const lastSearchCall = fake.searchCalls.at(-1)!;
    expect(lastSearchCall.config.filters?.[DECAY_METADATA_KEYS.timeStart]).toEqual({
      gte: Date.UTC(2025, 2, 5, 0, 0, 0),
    });
    expect(lastSearchCall.config.filters?.[DECAY_METADATA_KEYS.timeEnd]).toEqual({
      lt: Date.UTC(2025, 2, 16, 0, 0, 0),
    });

    // 2. Soft range ignores today's date AND weights pinned records by query distance
    const soft1 = await wrapper.search("notes", {
      scope: SCOPE,
      topK: 5,
      range: { from: "2025-03-05", to: "2025-03-15", strict: false, halfLifeDays: 14 },
    });
    currentNow = NOW + 365 * DAY; // Advance today's clock by 1 year
    const soft2 = await wrapper.search("notes", {
      scope: SCOPE,
      topK: 5,
      range: { from: "2025-03-05", to: "2025-03-15", strict: false, halfLifeDays: 14 },
    });
    expect(soft2.results.map((r) => [r.id, r.adjustedScore])).toEqual(
      soft1.results.map((r) => [r.id, r.adjustedScore]),
    );
    const pinnedInSoft = soft1.results.find((r) => r.id === "pinned-outside")!;
    // 2025-03-30T00:00Z is exactly 14 days after 2025-03-16T00:00Z -> weight = 0.5 -> 0.9 - 0.9 * 0.5 * 0.05 = 0.8775
    expect(pinnedInSoft.timeWeight).toBeCloseTo(0.5, 12);
    expect(pinnedInSoft.adjustedScore).toBeCloseTo(0.8775, 12);

    // 3. Anchor query also ignores today's date and weights pinned records symmetrically
    const anchor1 = await wrapper.search("notes", {
      scope: SCOPE,
      topK: 5,
      anchor: { around: mar10, halfLifeDays: 10 },
    });
    const inAnchor = anchor1.results.find((r) => r.id === "backfill-in-range")!;
    expect(inAnchor.timeWeight).toBe(1);
    const straddlingInAnchor = anchor1.results.find((r) => r.id === "straddling-summary")!;
    expect(straddlingInAnchor.timeWeight).toBe(1); // span [Mar 1, Mar 20] contains Mar 10
    const pinnedInAnchor = anchor1.results.find((r) => r.id === "pinned-outside")!;
    expect(pinnedInAnchor.timeWeight).toBeLessThan(0.5);

    // 4. Backfill uses old event time (not recent ingestion time) for no-date recency
    currentNow = NOW;
    const recency = await wrapper.search("notes", { scope: SCOPE, topK: 5 });
    const backfillRecency = recency.results.find((r) => r.id === "backfill-in-range")!;
    expect(backfillRecency.timeWeight).toBeCloseTo(0.3, 3); // > 1 year old -> near floor 0.3
  });

  it("handles legacy createdAt fallback, neutral weight for untimestamped records, timezone dates, and validation", async () => {
    const fake = new FakeMem0QdrantStore();
    fake.seed({
      id: "legacy-dated",
      memory: "Legacy with createdAt",
      baseScore: 0.8,
      createdAt: new Date(NOW - 30 * DAY).toISOString(),
      payload: { user_id: "alice" },
    });
    fake.seed({
      id: "legacy-undated",
      memory: "Legacy with no timestamp",
      baseScore: 0.6,
      payload: { user_id: "alice" },
    });

    const wrapper = createDecayMemory({
      storeId: "store-1",
      memory: fake.asMem0(),
      nowMs: () => NOW,
      maxCandidates: 20,
    });

    const res = await wrapper.search("legacy", { scope: SCOPE, topK: 5 });
    const undated = res.results.find((r) => r.id === "legacy-undated")!;
    const dated = res.results.find((r) => r.id === "legacy-dated")!;
    expect(undated.timeWeight).toBe(1);
    expect(undated.adjustedScore).toBe(0.6);
    expect(dated.adjustedScore).toBeCloseTo(0.786, 12);

    // Validation errors
    await expect(
      wrapper.search("x", { scope: SCOPE, topK: 21 }),
    ).rejects.toThrow(RangeError);
    await expect(
      wrapper.search("x", { scope: SCOPE, alpha: -0.01 }),
    ).rejects.toThrow(RangeError);
    await expect(
      wrapper.search("x", { scope: SCOPE, alpha: 1.01 }),
    ).rejects.toThrow(RangeError);
    expect(() =>
      createDecayMemory({
        storeId: "store-bad-alpha",
        memory: fake.asMem0(),
        alpha: Number.NaN,
      }),
    ).toThrow(RangeError);
    await expect(
      wrapper.search("x", {
        scope: SCOPE,
        range: { from: "2026-05-10", to: "2026-05-05" },
      }),
    ).rejects.toThrow(RangeError);
    await expect(
      wrapper.search("x", {
        scope: SCOPE,
        range: { from: "2026-05-01", to: "2026-05-02" },
        anchor: { around: "2026-05-01" },
      }),
    ).rejects.toThrow(RangeError);

    // Timezone-aware date-only range: 2026-01-15 in Asia/Tokyo (UTC+9) starts 2026-01-14T15:00:00Z
    await wrapper.search("tz", {
      scope: SCOPE,
      topK: 5,
      range: { from: "2026-01-15", to: "2026-01-15", timeZone: "Asia/Tokyo" },
    });
    const tzCall = fake.searchCalls.at(-1)!;
    expect(tzCall.config.filters?.[DECAY_METADATA_KEYS.timeStart]).toEqual({
      gte: Date.UTC(2026, 0, 14, 15, 0, 0),
    });
    expect(tzCall.config.filters?.[DECAY_METADATA_KEYS.timeEnd]).toEqual({
      lt: Date.UTC(2026, 0, 15, 15, 0, 0),
    });
  });

  it("reuses a single created curve everywhere: policy, sweep, store defaultQueryCurve, recency override, soft range, anchor, and router", async () => {
    // Create one 5-parameter sigmoid curve once
    const sharedCurve = decayCurves.sigmoid({
      lower: 0.1,
      upper: 0.9,
      midpoint: 20,
      slope: 0.3,
      asymmetry: 2,
    });

    const policyRegistry = createPolicyRegistry([
      {
        id: "shared_sig@v1",
        curve: sharedCurve,
        floor: 0,
        minStrength: 0.35,
        sweepEligible: true,
      },
    ]);

    const fake = new FakeMem0QdrantStore();
    fake.seed({
      id: "m-20d",
      memory: "20-day note",
      baseScore: 0.8,
      payload: {
        user_id: "alice",
        [DECAY_METADATA_KEYS.tenant]: "acme",
        [DECAY_METADATA_KEYS.timeStart]: NOW - 20 * DAY,
        [DECAY_METADATA_KEYS.timeEnd]: NOW - 20 * DAY,
        [DECAY_METADATA_KEYS.ingestedAt]: NOW - 20 * DAY,
        [DECAY_METADATA_KEYS.policy]: "shared_sig@v1",
        [DECAY_METADATA_KEYS.tier]: 0,
      },
    });

    const wrapper = createDecayMemory({
      storeId: "store-universal",
      memory: fake.asMem0(),
      nowMs: () => NOW,
      policyResolver: policyRegistry,
      defaultQueryCurve: sharedCurve,
      qdrantReader: createQdrantReader({
        client: fake.asQdrantClient(),
        collectionName: "memories",
      }),
      scopeSweepPolicy: { action: "delete" },
    });

    const expectedStrengthAt20 = sharedCurve(20); // 0.3

    // 1. Policy recency ranking uses sharedCurve
    const recency = await wrapper.search("note", { scope: SCOPE, topK: 5 });
    expect(recency.results[0]!.timeWeight).toBeCloseTo(expectedStrengthAt20, 12);

    // 2. Per-query curve override in recency mode
    const steeperCurve = decayCurves.exponential(10); // at 20d -> 0.25
    const overridden = await wrapper.search("note", {
      scope: SCOPE,
      topK: 5,
      curve: steeperCurve,
    });
    expect(overridden.results[0]!.timeWeight).toBeCloseTo(0.25, 12);

    // 3. Store defaultQueryCurve used automatically in anchor mode (20d from NOW)
    const anchorDefault = await wrapper.search("note", {
      scope: SCOPE,
      topK: 5,
      anchor: { around: NOW },
    });
    expect(anchorDefault.results[0]!.timeWeight).toBeCloseTo(expectedStrengthAt20, 12);

    // 3b. Exact anchor match (distMs === 0) MUST keep timeWeight = 1 even when sharedCurve(0) < 1
    expect(sharedCurve(0)).toBeLessThan(1);
    const exactAnchor = await wrapper.search("note", {
      scope: SCOPE,
      topK: 5,
      anchor: { around: NOW - 20 * DAY, curve: sharedCurve },
    });
    expect(exactAnchor.results[0]!.timeWeight).toBe(1);

    // 4. Explicit curve on soft_range and anchor
    const softRange = await wrapper.search("note", {
      scope: SCOPE,
      topK: 5,
      range: { from: NOW, to: NOW + DAY, strict: false, curve: steeperCurve },
    });
    expect(softRange.results[0]!.timeWeight).toBeCloseTo(0.25, 12);

    // 5. Router defaultQueryCurve and per-query curve fan-out (including store-level curveRegistry specs)
    const storeCurveRegistry = createCurveRegistry().register("half_at_any_age", () => () => 0.5);
    const storeWithRegistry = createDecayMemory({
      storeId: "store-with-registry",
      memory: fake.asMem0(),
      nowMs: () => NOW,
      policyResolver: policyRegistry,
      curveRegistry: storeCurveRegistry,
    });
    const router = createDecayRouter({
      nowMs: () => NOW,
      defaultQueryCurve: steeperCurve,
    });
    router.registerStore(wrapper, { activeWriter: true });
    const routerAnchor = await router.search("note", {
      scope: SCOPE,
      topK: 5,
      anchor: { around: NOW },
    });
    expect(routerAnchor.results[0]!.timeWeight).toBeCloseTo(0.25, 12);

    const routerForRegistryStore = createDecayRouter({ nowMs: () => NOW });
    routerForRegistryStore.registerStore(storeWithRegistry, { activeWriter: true });
    const routerSpecSearch = await routerForRegistryStore.search("note", {
      scope: SCOPE,
      topK: 5,
      curve: { kind: "half_at_any_age" },
    });
    expect(routerSpecSearch.results[0]!.timeWeight).toBeCloseTo(0.5, 12);

    // 6. Sweep eligibility uses the exact same sharedCurve (0.3 < minStrength 0.35 -> eligible)
    const drySweep = await wrapper.sweep({ scope: SCOPE, dryRun: true });
    expect(drySweep.eligible).toBe(1);
  });
});
