import type {
  Clock,
  CompactionScopeConfig,
  CompactionSourceRef,
  CompactionSummarizer,
  CompactionWorkRecord,
  CompactionWorkStore,
  DecayPolicyDefinition,
  ManagedDecayMetadata,
  Mem0Memory,
  MemoryItem,
  PolicyResolver,
  QdrantMemoryRecord,
  QdrantReader,
  ResolvedScope,
  SweepError,
  WriteCoordinator,
} from "./contracts.js";
import {
  DECAY_METADATA_KEYS,
  computeRevisionFingerprint,
  deriveCompactionKey,
  extractScopeIdentityFromPayload,
  parseManagedMetadata,
  payloadMatchesScope,
} from "./metadata.js";
import { DEFAULT_POLICY_IDS, MS_PER_DAY } from "./policy.js";

export const DEFAULT_COMPACTION_TIME_BUCKET_MS = 7 * MS_PER_DAY;
export const DEFAULT_COMPACTION_SIMILARITY_THRESHOLD = 0.75;
export const DEFAULT_COMPACTION_MIN_GROUP_SIZE = 2;
export const DEFAULT_COMPACTION_MAX_GROUP_SIZE = 8;

export interface CompactionCandidate {
  id: string;
  memory: string;
  scope: ResolvedScope;
  metadata: ManagedDecayMetadata;
  policy: DecayPolicyDefinition;
  revision: string;
  vector?: number[];
  rawPayload: Record<string, unknown>;
}

export interface PlannedCompactionGroup {
  compactionKey: string;
  storeId: string;
  scope: ResolvedScope;
  targetTier: 1;
  sources: CompactionCandidate[];
  timeSpan: { start: number; end: number };
  inheritedExpiresAt?: number;
}

export interface CompactionPlanResult {
  groups: PlannedCompactionGroup[];
  unclustered: CompactionCandidate[];
}

export class InMemoryCompactionWorkStore implements CompactionWorkStore {
  private readonly records = new Map<string, CompactionWorkRecord>();

  async listPending(storeId: string, scope: ResolvedScope): Promise<CompactionWorkRecord[]> {
    const out: CompactionWorkRecord[] = [];
    for (const rec of this.records.values()) {
      if (rec.status === "completed") continue;
      if (rec.storeId !== storeId) continue;
      if (!scopesEqual(rec.scope, scope)) continue;
      out.push(cloneWorkRecord(rec));
    }
    out.sort((a, b) => a.createdAtMs - b.createdAtMs || (a.compactionKey < b.compactionKey ? -1 : 1));
    return out;
  }

  async get(compactionKey: string): Promise<CompactionWorkRecord | null> {
    const rec = this.records.get(compactionKey);
    return rec ? cloneWorkRecord(rec) : null;
  }

  async save(record: CompactionWorkRecord): Promise<void> {
    this.records.set(record.compactionKey, cloneWorkRecord(record));
  }
}

function cloneWorkRecord(rec: CompactionWorkRecord): CompactionWorkRecord {
  return {
    ...rec,
    scope: { ...rec.scope },
    sources: rec.sources.map((s) => ({ ...s })),
    completedSourceIds: [...rec.completedSourceIds],
  };
}

export function scopesEqual(a: ResolvedScope, b: ResolvedScope): boolean {
  return (
    a.tenant === b.tenant &&
    (a.userId ?? undefined) === (b.userId ?? undefined) &&
    (a.agentId ?? undefined) === (b.agentId ?? undefined) &&
    (a.runId ?? undefined) === (b.runId ?? undefined)
  );
}

export function cosineSimilarity(a: number[], b: number[]): number | null {
  if (a.length === 0 || a.length !== b.length) {
    return null;
  }
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    const va = a[i]!;
    const vb = b[i]!;
    if (!Number.isFinite(va) || !Number.isFinite(vb)) {
      return null;
    }
    dot += va * vb;
    normA += va * va;
    normB += vb * vb;
  }
  if (normA === 0 || normB === 0) {
    return null;
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

function sortedSourceRefs(sources: CompactionSourceRef[]): CompactionSourceRef[] {
  return [...sources].sort((a, b) => {
    if (a.id !== b.id) return a.id < b.id ? -1 : 1;
    return a.revision < b.revision ? -1 : a.revision > b.revision ? 1 : 0;
  });
}

function sourceRefsEqual(a: CompactionSourceRef[], b: CompactionSourceRef[]): boolean {
  const sa = sortedSourceRefs(a);
  const sb = sortedSourceRefs(b);
  if (sa.length !== sb.length) return false;
  for (let i = 0; i < sa.length; i++) {
    if (sa[i]!.id !== sb[i]!.id || sa[i]!.revision !== sb[i]!.revision) {
      return false;
    }
  }
  return true;
}

export function verifySummaryLineage(
  summary: { id: string; payload: Record<string, unknown> },
  expected: {
    scope: ResolvedScope;
    compactionKey: string;
    sources: CompactionSourceRef[];
    tenantField?: string;
  },
): boolean {
  if (
    !payloadMatchesScope(summary.payload, expected.scope, {
      requireExactIdentity: true,
      tenantField: expected.tenantField,
    })
  ) {
    return false;
  }
  const parsed = parseManagedMetadata(summary.payload);
  if (parsed.status !== "valid") {
    return false;
  }
  const meta = parsed.metadata;
  if (meta.decay_tier !== 1) return false;
  if (meta.decay_compaction_key !== expected.compactionKey) return false;
  if (!meta.decay_sources) return false;
  return sourceRefsEqual(meta.decay_sources, expected.sources);
}

/**
 * Group eligible Tier 0 records within the same tenant, store, exact Mem0 identity,
 * time bucket, compatible retention policy, and embedding space.
 */
export function planCompactionGroups(
  storeId: string,
  candidates: CompactionCandidate[],
  config: CompactionScopeConfig = {},
): CompactionPlanResult {
  const timeBucketMs = config.timeBucketMs ?? DEFAULT_COMPACTION_TIME_BUCKET_MS;
  const similarityThreshold =
    config.similarityThreshold ?? DEFAULT_COMPACTION_SIMILARITY_THRESHOLD;
  const minGroupSize = Math.max(2, config.minGroupSize ?? DEFAULT_COMPACTION_MIN_GROUP_SIZE);
  const maxGroupSize = Math.max(minGroupSize, config.maxGroupSize ?? DEFAULT_COMPACTION_MAX_GROUP_SIZE);

  if (!Number.isFinite(timeBucketMs) || timeBucketMs <= 0) {
    throw new RangeError("compaction.timeBucketMs must be a finite number > 0");
  }
  if (!Number.isFinite(similarityThreshold) || similarityThreshold < -1 || similarityThreshold > 1) {
    throw new RangeError("compaction.similarityThreshold must be in [-1, 1]");
  }

  const buckets = new Map<string, CompactionCandidate[]>();
  const unclustered: CompactionCandidate[] = [];

  for (const candidate of candidates) {
    if (candidate.metadata.decay_tier !== 0) {
      unclustered.push(candidate);
      continue;
    }
    if (!candidate.vector || candidate.vector.length === 0) {
      unclustered.push(candidate);
      continue;
    }

    const timeBucket = Math.floor(candidate.metadata.decay_time_end / timeBucketMs);
    const compatKey = candidate.policy.retentionCompatibilityKey ?? candidate.policy.id;
    const bucketKey = JSON.stringify([
      candidate.scope.tenant,
      storeId,
      candidate.scope.userId ?? null,
      candidate.scope.agentId ?? null,
      candidate.scope.runId ?? null,
      candidate.metadata.decay_tier,
      timeBucket,
      compatKey,
      candidate.vector.length,
    ]);

    const list = buckets.get(bucketKey);
    if (list) {
      list.push(candidate);
    } else {
      buckets.set(bucketKey, [candidate]);
    }
  }

  const groups: PlannedCompactionGroup[] = [];

  const sortedBucketKeys = Array.from(buckets.keys()).sort();
  for (const key of sortedBucketKeys) {
    const items = buckets.get(key)!;
    // Stable deterministic input order
    items.sort((a, b) => {
      if (a.metadata.decay_time_start !== b.metadata.decay_time_start) {
        return a.metadata.decay_time_start - b.metadata.decay_time_start;
      }
      if (a.metadata.decay_time_end !== b.metadata.decay_time_end) {
        return a.metadata.decay_time_end - b.metadata.decay_time_end;
      }
      return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
    });

    const assigned = new Set<string>();
    for (let i = 0; i < items.length; i++) {
      const seed = items[i]!;
      if (assigned.has(seed.id)) continue;

      const cluster: CompactionCandidate[] = [seed];
      assigned.add(seed.id);

      for (let j = i + 1; j < items.length && cluster.length < maxGroupSize; j++) {
        const candidate = items[j]!;
        if (assigned.has(candidate.id)) continue;

        // Ensure cosine similarity meets threshold against every member in cluster
        let matchesAll = true;
        for (const member of cluster) {
          const sim = cosineSimilarity(member.vector!, candidate.vector!);
          if (sim === null || sim < similarityThreshold) {
            matchesAll = false;
            break;
          }
        }
        if (matchesAll) {
          cluster.push(candidate);
          assigned.add(candidate.id);
        }
      }

      if (cluster.length < minGroupSize) {
        for (const item of cluster) {
          unclustered.push(item);
        }
        continue;
      }

      // Defensive cross-scope check
      for (const item of cluster) {
        if (!scopesEqual(item.scope, seed.scope)) {
          throw new Error("Cross-scope compaction group rejected");
        }
      }

      const sourceRefs = sortedSourceRefs(
        cluster.map((c) => ({ id: c.id, revision: c.revision })),
      );
      const compactionKey = deriveCompactionKey({
        storeId,
        scope: seed.scope,
        targetTier: 1,
        sources: sourceRefs,
      });

      let minStart = Number.POSITIVE_INFINITY;
      let maxEnd = Number.NEGATIVE_INFINITY;
      let earliestExpiry: number | undefined;
      for (const item of cluster) {
        if (item.metadata.decay_time_start < minStart) minStart = item.metadata.decay_time_start;
        if (item.metadata.decay_time_end > maxEnd) maxEnd = item.metadata.decay_time_end;
        if (item.metadata.decay_expires_at !== undefined) {
          earliestExpiry =
            earliestExpiry === undefined
              ? item.metadata.decay_expires_at
              : Math.min(earliestExpiry, item.metadata.decay_expires_at);
        }
      }

      groups.push({
        compactionKey,
        storeId,
        scope: { ...seed.scope },
        targetTier: 1,
        sources: cluster,
        timeSpan: { start: minStart, end: maxEnd },
        ...(earliestExpiry !== undefined ? { inheritedExpiresAt: earliestExpiry } : {}),
      });
    }
  }

  return { groups, unclustered };
}

export interface MemoryGetInspectResult {
  item: MemoryItem;
  payload: Record<string, unknown>;
}

/**
 * Reconstruct a flat payload object from Mem0's `memory.get(id)` response,
 * combining top-level identity fields (`user_id`, `agent_id`, `run_id`) with `metadata`.
 */
export function buildFlatPayloadFromMemoryItem(item: MemoryItem): Record<string, unknown> {
  const rec = item as MemoryItem & {
    user_id?: string;
    agent_id?: string;
    run_id?: string;
  };
  return {
    ...(item.metadata && typeof item.metadata === "object" ? item.metadata : {}),
    ...(rec.user_id !== undefined ? { user_id: rec.user_id } : {}),
    ...(rec.agent_id !== undefined ? { agent_id: rec.agent_id } : {}),
    ...(rec.run_id !== undefined ? { run_id: rec.run_id } : {}),
    data: item.memory,
  };
}

export interface CompactionExecutionContext {
  storeId: string;
  scope: ResolvedScope;
  runNowMs: number;
  nowMs: Clock;
  memory: Mem0Memory;
  qdrantReader: QdrantReader;
  policyResolver: PolicyResolver;
  writeCoordinator: WriteCoordinator;
  summarizer?: CompactionSummarizer;
  workStore?: CompactionWorkStore;
  tenantField?: string;
  compactionConfig?: CompactionScopeConfig;
  getRemainingBudget: () => number;
  consumeBudget: (count: number) => void;
  onRevisionConflict: () => void;
  onError: (err: SweepError) => void;
  onUnresolved: (work: CompactionWorkRecord) => void;
  onExpiredSourceDuringPrep: (sourceId: string) => Promise<void>;
}

export interface CompactionStepResult {
  appliedMutations: number;
  pendingWork: boolean;
}

function isSourceStillEligibleForCompaction(
  item: MemoryItem,
  expectedRef: CompactionSourceRef,
  scope: ResolvedScope,
  policyResolver: PolicyResolver,
  runNowMs: number,
  currentNowMs: number,
  tenantField?: string,
): { ok: true } | { ok: false; expired?: boolean; reason: string } {
  const payload = buildFlatPayloadFromMemoryItem(item);
  if (!payloadMatchesScope(payload, scope, { requireExactIdentity: true, tenantField })) {
    return { ok: false, reason: "scope_mismatch" };
  }
  const parsed = parseManagedMetadata(payload);
  if (parsed.status !== "valid") {
    return { ok: false, reason: "malformed_metadata" };
  }
  const meta = parsed.metadata;
  const policy = policyResolver.resolve(meta.decay_policy);
  if (!policy || policy.exemptFromSweep) {
    return { ok: false, reason: "exempt_or_unknown_policy" };
  }

  const effectiveNowMs = Math.max(runNowMs, currentNowMs);

  // Check if source expired while waiting/preparing (checked before revision mismatch
  // if expiry deadline has been reached so it routes cleanly to expiry policy)
  if (meta.decay_expires_at !== undefined && meta.decay_expires_at <= effectiveNowMs) {
    return { ok: false, expired: true, reason: "expired" };
  }

  const identity = extractScopeIdentityFromPayload(payload);
  const effectiveScope: ResolvedScope = {
    tenant: identity.tenant ?? scope.tenant,
    ...(identity.userId ? { userId: identity.userId } : {}),
    ...(identity.agentId ? { agentId: identity.agentId } : {}),
    ...(identity.runId ? { runId: identity.runId } : {}),
  };
  const currentRevision = computeRevisionFingerprint({
    memory: item.memory,
    scope: effectiveScope,
    metadata: meta,
  });

  if (currentRevision !== expectedRef.revision) {
    return { ok: false, reason: "revision_changed" };
  }

  if (!policy.sweepEligible || meta.decay_tier !== 0) {
    return { ok: false, reason: "ineligible_policy_or_tier" };
  }
  if (runNowMs - meta.decay_ingested_at < policy.ingestionGraceMs) {
    return { ok: false, reason: "grace_period" };
  }
  const ageDays = Math.max(0, (runNowMs - meta.decay_time_end) / MS_PER_DAY);
  const { belowThreshold } = policy.criterion(ageDays);
  if (!belowThreshold) {
    return { ok: false, reason: "above_threshold" };
  }
  return { ok: true };
}

async function retireCompactionSources(
  ctx: CompactionExecutionContext,
  work: CompactionWorkRecord,
): Promise<CompactionStepResult> {
  let appliedMutations = 0;
  let pendingWork = false;
  const completedSet = new Set(work.completedSourceIds);
  let allProcessed = true;

  for (const sourceRef of work.sources) {
    if (completedSet.has(sourceRef.id)) {
      continue;
    }

    if (ctx.getRemainingBudget() < 1) {
      pendingWork = true;
      allProcessed = false;
      break;
    }

    try {
      const outcome = await ctx.writeCoordinator.run(ctx.storeId, async () => {
        const current = await ctx.memory.get(sourceRef.id);
        if (!current) {
          // Already deleted on a previous attempt: reconcile as completed.
          return { action: "reconciled" as const };
        }
        const check = isSourceStillEligibleForCompaction(
          current,
          sourceRef,
          work.scope,
          ctx.policyResolver,
          ctx.runNowMs,
          ctx.nowMs(),
          ctx.tenantField,
        );
        if (!check.ok) {
          if (check.expired) {
            return { action: "expired" as const };
          }
          return { action: "conflict" as const };
        }
        await ctx.memory.delete(sourceRef.id);
        return { action: "deleted" as const };
      });

      if (outcome.action === "deleted") {
        ctx.consumeBudget(1);
        appliedMutations += 1;
        completedSet.add(sourceRef.id);
      } else if (outcome.action === "reconciled") {
        completedSet.add(sourceRef.id);
      } else if (outcome.action === "expired") {
        await ctx.onExpiredSourceDuringPrep(sourceRef.id);
        completedSet.add(sourceRef.id);
      } else {
        // Preserve changed/re-pinned source and report conflict
        ctx.onRevisionConflict();
        completedSet.add(sourceRef.id);
      }

      work.completedSourceIds = Array.from(completedSet);
      work.updatedAtMs = ctx.nowMs();
      if (ctx.workStore) {
        await ctx.workStore.save(work);
      }
    } catch (err) {
      allProcessed = false;
      pendingWork = true;
      ctx.onError({
        stage: "compact",
        sourceId: sourceRef.id,
        compactionKey: work.compactionKey,
        message: err instanceof Error ? err.message : String(err),
      });
    }
  }

  if (allProcessed) {
    work.status = "completed";
    work.updatedAtMs = ctx.nowMs();
    if (ctx.workStore) {
      await ctx.workStore.save(work);
    }
  }

  return { appliedMutations, pendingWork };
}

/**
 * Resume unfinished compaction work records before forming any new groups.
 */
export async function resumePendingCompactions(
  ctx: CompactionExecutionContext,
  dryRun: boolean,
): Promise<{
  appliedMutations: number;
  plannedMutations: number;
  pendingWork: boolean;
  reservedSourceIds: Set<string>;
}> {
  const reservedSourceIds = new Set<string>();
  if (!ctx.workStore) {
    return { appliedMutations: 0, plannedMutations: 0, pendingWork: false, reservedSourceIds };
  }

  const pendingRecords = await ctx.workStore.listPending(ctx.storeId, ctx.scope);
  let appliedMutations = 0;
  let plannedMutations = 0;
  let pendingWork = false;

  for (const work of pendingRecords) {
    for (const s of work.sources) {
      reservedSourceIds.add(s.id);
    }
    const remainingCount = work.sources.filter(
      (s) => !work.completedSourceIds.includes(s.id),
    ).length;

    if (dryRun) {
      if (work.status === "uncertain") {
        ctx.onUnresolved(work);
        pendingWork = true;
      } else {
        plannedMutations += remainingCount;
        if (remainingCount > 0) {
          pendingWork = true;
        }
      }
      continue;
    }

    // Attempt to resolve the summary by compaction key first
    let verifiedSummary: QdrantMemoryRecord | null = null;
    try {
      const found = await ctx.qdrantReader.findSummary(work.scope, work.compactionKey);
      if (
        found &&
        verifySummaryLineage(found, {
          scope: work.scope,
          compactionKey: work.compactionKey,
          sources: work.sources,
          tenantField: ctx.tenantField,
        })
      ) {
        verifiedSummary = found;
      }
    } catch (err) {
      ctx.onError({
        stage: "resume",
        compactionKey: work.compactionKey,
        message: err instanceof Error ? err.message : String(err),
      });
      pendingWork = true;
      ctx.onUnresolved(work);
      continue;
    }

    if (verifiedSummary) {
      work.summaryId = verifiedSummary.id;
      work.status = "confirmed";
      work.updatedAtMs = ctx.nowMs();
      await ctx.workStore.save(work);

      const step = await retireCompactionSources(ctx, work);
      appliedMutations += step.appliedMutations;
      if (step.pendingWork) pendingWork = true;
      continue;
    }

    // No verified summary found: if status is uncertain or pending without confirmed write,
    // preserve sources and surface for reconciliation before retrying an add.
    work.status = "uncertain";
    work.updatedAtMs = ctx.nowMs();
    await ctx.workStore.save(work);
    ctx.onUnresolved(work);
    pendingWork = true;
  }

  return { appliedMutations, plannedMutations, pendingWork, reservedSourceIds };
}

/**
 * Execute one newly planned Tier 0 -> Tier 1 compaction group.
 */
export async function executeCompactionGroup(
  ctx: CompactionExecutionContext,
  group: PlannedCompactionGroup,
): Promise<CompactionStepResult> {
  if (!ctx.summarizer) {
    throw new Error("Compaction requires a configured summarizer");
  }

  // Reserve 1 summary write + 1 deletion per source BEFORE invoking the summarizer
  const requiredBudget = 1 + group.sources.length;
  if (ctx.getRemainingBudget() < requiredBudget) {
    return { appliedMutations: 0, pendingWork: true };
  }

  // Step 1: Call summarizer and validate output
  let summaryText: string;
  try {
    const sumResult = await ctx.summarizer.summarize({
      storeId: group.storeId,
      scope: group.scope,
      sources: group.sources.map((s) => ({
        id: s.id,
        revision: s.revision,
        memory: s.memory,
        timeSpan: {
          start: s.metadata.decay_time_start,
          end: s.metadata.decay_time_end,
        },
        metadata: s.rawPayload,
      })),
    });
    if (!sumResult || typeof sumResult.summary !== "string" || sumResult.summary.trim() === "") {
      throw new Error("Summarizer returned an empty summary");
    }
    summaryText = sumResult.summary.trim();
  } catch (err) {
    ctx.onError({
      stage: "compact",
      compactionKey: group.compactionKey,
      message: err instanceof Error ? err.message : String(err),
    });
    return { appliedMutations: 0, pendingWork: true };
  }

  const targetTierPolicyId =
    ctx.compactionConfig?.targetTierPolicyId ?? DEFAULT_POLICY_IDS.tier1Summary;
  const targetTierPolicy = ctx.policyResolver.resolveOrThrow(targetTierPolicyId);

  const sourceRefs = sortedSourceRefs(
    group.sources.map((s) => ({ id: s.id, revision: s.revision })),
  );

  // Step 2: Under coordinator, recheck all sources, check existing summary, or write + verify summary
  let appliedMutations = 0;
  let expiredDuringPrep: string[] = [];

  const commitResult = await ctx.writeCoordinator.run(ctx.storeId, async () => {
    const expiredIds: string[] = [];
    for (const source of group.sources) {
      const current = await ctx.memory.get(source.id);
      if (!current) {
        return { outcome: "conflict" as const, expiredIds };
      }
      const check = isSourceStillEligibleForCompaction(
        current,
        { id: source.id, revision: source.revision },
        group.scope,
        ctx.policyResolver,
        ctx.runNowMs,
        ctx.nowMs(),
        ctx.tenantField,
      );
      if (!check.ok) {
        if (check.expired) {
          expiredIds.push(source.id);
        } else {
          return { outcome: "conflict" as const, expiredIds };
        }
      }
    }

    if (expiredIds.length > 0) {
      return { outcome: "expired_during_prep" as const, expiredIds };
    }

    const work: CompactionWorkRecord = {
      compactionKey: group.compactionKey,
      storeId: group.storeId,
      scope: { ...group.scope },
      targetTier: 1,
      sources: sourceRefs,
      completedSourceIds: [],
      status: "pending",
      createdAtMs: ctx.nowMs(),
      updatedAtMs: ctx.nowMs(),
    };
    if (ctx.workStore) {
      await ctx.workStore.save(work);
    }

    // Reuse an already-verified summary matching this exact compactionKey
    const existingSummary = await ctx.qdrantReader.findSummary(group.scope, group.compactionKey);
    if (
      existingSummary &&
      verifySummaryLineage(existingSummary, {
        scope: group.scope,
        compactionKey: group.compactionKey,
        sources: sourceRefs,
        tenantField: ctx.tenantField,
      })
    ) {
      work.summaryId = existingSummary.id;
      work.status = "confirmed";
      work.updatedAtMs = ctx.nowMs();
      if (ctx.workStore) {
        await ctx.workStore.save(work);
      }
      return { outcome: "reused" as const, work, expiredIds };
    }

    const summaryMetadata: Record<string, unknown> = {
      [DECAY_METADATA_KEYS.timeStart]: group.timeSpan.start,
      [DECAY_METADATA_KEYS.timeEnd]: group.timeSpan.end,
      [DECAY_METADATA_KEYS.ingestedAt]: ctx.nowMs(),
      [DECAY_METADATA_KEYS.policy]: targetTierPolicy.id,
      [DECAY_METADATA_KEYS.tier]: 1,
      [DECAY_METADATA_KEYS.compactionKey]: group.compactionKey,
      [DECAY_METADATA_KEYS.sources]: sourceRefs,
      [DECAY_METADATA_KEYS.tenant]: group.scope.tenant,
    };
    if (group.inheritedExpiresAt !== undefined) {
      summaryMetadata[DECAY_METADATA_KEYS.expiresAt] = group.inheritedExpiresAt;
    }

    let createdSummaryId: string | undefined;
    try {
      const addRes = await ctx.memory.add(summaryText, {
        ...(group.scope.userId ? { userId: group.scope.userId } : {}),
        ...(group.scope.agentId ? { agentId: group.scope.agentId } : {}),
        ...(group.scope.runId ? { runId: group.scope.runId } : {}),
        infer: false,
        metadata: summaryMetadata,
      });

      const created = addRes?.results?.[0];
      if (!created?.id) {
        throw new Error("Summary add did not return a memory ID");
      }
      createdSummaryId = created.id;

      // Read back and verify persisted lineage before retiring any source
      const readBack = await ctx.memory.get(createdSummaryId);
      if (!readBack) {
        throw new Error(`Summary read-back returned null for ID '${createdSummaryId}'`);
      }
      const verified = verifySummaryLineage(
        { id: readBack.id, payload: buildFlatPayloadFromMemoryItem(readBack) },
        {
          scope: group.scope,
          compactionKey: group.compactionKey,
          sources: sourceRefs,
          tenantField: ctx.tenantField,
        },
      );
      if (!verified) {
        throw new Error(`Summary lineage verification failed for ID '${createdSummaryId}'`);
      }
    } catch (err) {
      work.status = "uncertain";
      work.updatedAtMs = ctx.nowMs();
      work.error = err instanceof Error ? err.message : String(err);
      if (ctx.workStore) {
        await ctx.workStore.save(work);
      }
      return {
        outcome: "uncertain" as const,
        work,
        error: work.error,
        expiredIds,
      };
    }

    work.summaryId = createdSummaryId;
    work.status = "confirmed";
    work.updatedAtMs = ctx.nowMs();
    if (ctx.workStore) {
      await ctx.workStore.save(work);
    }
    return { outcome: "created" as const, work, expiredIds };
  });

  expiredDuringPrep = commitResult.expiredIds;

  if (commitResult.outcome === "expired_during_prep") {
    for (const id of expiredDuringPrep) {
      await ctx.onExpiredSourceDuringPrep(id);
    }
    return { appliedMutations, pendingWork: false };
  }

  if (commitResult.outcome === "conflict") {
    ctx.onRevisionConflict();
    return { appliedMutations, pendingWork: false };
  }

  if (commitResult.outcome === "uncertain") {
    ctx.onUnresolved(commitResult.work);
    ctx.onError({
      stage: "compact",
      compactionKey: group.compactionKey,
      message: commitResult.error ?? "Uncertain summary write outcome",
    });
    return { appliedMutations, pendingWork: true };
  }

  if (commitResult.outcome === "created") {
    ctx.consumeBudget(1);
    appliedMutations += 1;
  }

  const retireStep = await retireCompactionSources(ctx, commitResult.work);
  return {
    appliedMutations: appliedMutations + retireStep.appliedMutations,
    pendingWork: retireStep.pendingWork,
  };
}
