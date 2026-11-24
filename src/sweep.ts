import {
  buildFlatPayloadFromMemoryItem,
  executeCompactionGroup,
  planCompactionGroups,
  resumePendingCompactions,
  type CompactionCandidate,
  type CompactionExecutionContext,
} from "./compaction.js";
import {
  assertValidScope,
  type ArchiveSink,
  type Clock,
  type CompactionSummarizer,
  type CompactionWorkRecord,
  type CompactionWorkStore,
  type DecayPolicyDefinition,
  type ExpiryActionType,
  type ManagedDecayMetadata,
  type Mem0Memory,
  type PlannedMutation,
  type PolicyResolver,
  type QdrantMemoryRecord,
  type QdrantReader,
  type ResolvedScope,
  type ScanCursor,
  type ScopeSweepPolicy,
  type ScopeSweepPolicyResolver,
  type SweepError,
  type SweepOptions,
  type SweepReport,
  type SweepSkipReason,
  type TenantBinding,
  type WriteCoordinator,
} from "./contracts.js";
import {
  DECAY_METADATA_KEYS,
  buildArchiveKey,
  computeRevisionFingerprint,
  extractScopeIdentityFromPayload,
  parseManagedMetadata,
  payloadMatchesScope,
} from "./metadata.js";
import { MS_PER_DAY } from "./policy.js";
import { assertScopeMatchesTenantBinding } from "./search.js";

export const DEFAULT_SWEEP_PAGE_SIZE = 100;
export const DEFAULT_MAX_MUTATIONS_PER_RUN = 50;

export interface SweepEngineConfig {
  storeId: string;
  memory: Mem0Memory;
  nowMs: Clock;
  policyResolver: PolicyResolver;
  writeCoordinator: WriteCoordinator;
  tenantBinding?: TenantBinding;
  qdrantReader?: QdrantReader;
  scopeSweepPolicy?: ScopeSweepPolicy | ScopeSweepPolicyResolver;
  archiveSink?: ArchiveSink;
  summarizer?: CompactionSummarizer;
  compactionWorkStore?: CompactionWorkStore;
}

interface EvaluatedEligibleRecord {
  record: QdrantMemoryRecord;
  scope: ResolvedScope;
  metadata: ManagedDecayMetadata;
  policy: DecayPolicyDefinition;
  revision: string;
  trigger: "decay" | "expired";
  action: "archive" | "delete" | "compact";
}

function createEmptySkipCounts(): Record<SweepSkipReason, number> {
  return {
    unmanaged: 0,
    malformed_metadata: 0,
    unknown_policy: 0,
    exempt_policy: 0,
    ranking_only: 0,
    grace_period: 0,
    above_threshold: 0,
    tier_exempt: 0,
    action_disabled: 0,
    budget_exhausted: 0,
    revision_conflict: 0,
    scope_mismatch: 0,
    unclustered: 0,
  };
}

function evaluateRecordEligibility(input: {
  record: QdrantMemoryRecord;
  scope: ResolvedScope;
  runNowMs: number;
  policyResolver: PolicyResolver;
  scopePolicy: ScopeSweepPolicy;
  tenantField?: string;
}): { eligible: true; value: EvaluatedEligibleRecord } | { eligible: false; reason: SweepSkipReason } {
  const { record, scope, runNowMs, policyResolver, scopePolicy, tenantField } = input;

  if (!payloadMatchesScope(record.payload, scope, { requireExactIdentity: false, tenantField })) {
    return { eligible: false, reason: "scope_mismatch" };
  }

  const parsed = parseManagedMetadata(record.payload);
  if (parsed.status === "unmanaged") {
    return { eligible: false, reason: "unmanaged" };
  }
  if (parsed.status === "malformed") {
    return { eligible: false, reason: "malformed_metadata" };
  }

  const meta = parsed.metadata;
  const policy = policyResolver.resolve(meta.decay_policy);
  if (!policy) {
    return { eligible: false, reason: "unknown_policy" };
  }

  if (policy.exemptFromSweep) {
    return { eligible: false, reason: "exempt_policy" };
  }

  const identity = extractScopeIdentityFromPayload(record.payload);
  const recordScope: ResolvedScope = {
    tenant: identity.tenant ?? scope.tenant,
    ...(identity.userId ? { userId: identity.userId } : {}),
    ...(identity.agentId ? { agentId: identity.agentId } : {}),
    ...(identity.runId ? { runId: identity.runId } : {}),
  };

  if (!recordScope.userId && !recordScope.agentId && !recordScope.runId) {
    return { eligible: false, reason: "scope_mismatch" };
  }

  const revision = computeRevisionFingerprint({
    memory: record.memory,
    scope: recordScope,
    metadata: meta,
  });

  // Explicit expiry evaluated independently of curve strength
  if (meta.decay_expires_at !== undefined && meta.decay_expires_at <= runNowMs) {
    const expiryAction: ExpiryActionType | undefined =
      scopePolicy.expiryAction ??
      (scopePolicy.action === "archive" || scopePolicy.action === "delete"
        ? scopePolicy.action
        : undefined);

    if (!expiryAction) {
      return { eligible: false, reason: "action_disabled" };
    }

    return {
      eligible: true,
      value: {
        record,
        scope: recordScope,
        metadata: meta,
        policy,
        revision,
        trigger: "expired",
        action: expiryAction,
      },
    };
  }

  if (scopePolicy.action === "none") {
    return { eligible: false, reason: "action_disabled" };
  }

  // V1 compacts only Tier 0 -> Tier 1
  if (scopePolicy.action === "compact" && meta.decay_tier !== 0) {
    return { eligible: false, reason: "tier_exempt" };
  }

  if (!policy.sweepEligible) {
    return { eligible: false, reason: "ranking_only" };
  }

  if (runNowMs - meta.decay_ingested_at < policy.ingestionGraceMs) {
    return { eligible: false, reason: "grace_period" };
  }

  const ageDays = Math.max(0, (runNowMs - meta.decay_time_end) / MS_PER_DAY);
  const { belowThreshold } = policy.criterion(ageDays);
  if (!belowThreshold) {
    return { eligible: false, reason: "above_threshold" };
  }

  return {
    eligible: true,
    value: {
      record,
      scope: recordScope,
      metadata: meta,
      policy,
      revision,
      trigger: "decay",
      action: scopePolicy.action,
    },
  };
}

async function applyArchiveOrDelete(input: {
  config: SweepEngineConfig;
  item: EvaluatedEligibleRecord;
  action: "archive" | "delete";
  scopePolicy: ScopeSweepPolicy;
  runNowMs: number;
  tenantField?: string;
}): Promise<
  | { status: "deleted" }
  | { status: "reconciled" }
  | { status: "conflict" }
  | { status: "error"; stage: "archive" | "delete"; message: string }
> {
  const { config, item, action, scopePolicy, runNowMs, tenantField } = input;

  if (action === "archive") {
    if (!config.archiveSink) {
      return {
        status: "error",
        stage: "archive",
        message: "Archive action requires a configured archiveSink",
      };
    }
    const archiveKey = buildArchiveKey(config.storeId, item.record.id, item.revision);
    try {
      const ack = await config.archiveSink.archive({
        key: archiveKey,
        storeId: config.storeId,
        scope: item.scope,
        sourceId: item.record.id,
        revision: item.revision,
        memory: item.record.memory,
        metadata: item.record.payload,
        archivedAtMs: config.nowMs(),
      });
      if (ack && ack.confirmed === false) {
        return {
          status: "error",
          stage: "archive",
          message: `Archive sink did not confirm copy for key '${archiveKey}'`,
        };
      }
    } catch (err) {
      return {
        status: "error",
        stage: "archive",
        message: err instanceof Error ? err.message : String(err),
      };
    }
  }

  // Recheck scope, revision, and eligibility under the write coordinator immediately before delete
  try {
    return await config.writeCoordinator.run(config.storeId, async () => {
      const current = await config.memory.get(item.record.id);
      if (!current) {
        // Source already absent on retry: reconcile as completed
        return { status: "reconciled" as const };
      }

      const currentPayload = buildFlatPayloadFromMemoryItem(current);
      const recheck = evaluateRecordEligibility({
        record: {
          id: current.id,
          memory: current.memory,
          payload: currentPayload,
          createdAt: current.createdAt,
          updatedAt: current.updatedAt,
        },
        scope: item.scope,
        runNowMs,
        policyResolver: config.policyResolver,
        scopePolicy,
        tenantField,
      });

      if (!recheck.eligible || recheck.value.revision !== item.revision) {
        return { status: "conflict" as const };
      }

      await config.memory.delete(item.record.id);
      return { status: "deleted" as const };
    });
  } catch (err) {
    return {
      status: "error",
      stage: "delete",
      message: err instanceof Error ? err.message : String(err),
    };
  }
}

export async function executeSweep(
  config: SweepEngineConfig,
  options: SweepOptions,
): Promise<SweepReport> {
  if (!config.qdrantReader) {
    throw new Error("sweep() requires a configured qdrantReader");
  }

  const scope = assertValidScope(options.scope);
  assertScopeMatchesTenantBinding(scope, config.tenantBinding);

  const dryRun = options.dryRun !== false;
  const pageSize = options.pageSize ?? DEFAULT_SWEEP_PAGE_SIZE;
  const maxScanned = options.maxScanned ?? Number.POSITIVE_INFINITY;
  const maxMutationsPerRun = options.maxMutationsPerRun ?? DEFAULT_MAX_MUTATIONS_PER_RUN;

  if (!Number.isInteger(pageSize) || pageSize <= 0) {
    throw new RangeError(`pageSize must be a positive integer, received: ${String(pageSize)}`);
  }
  if (maxScanned !== Number.POSITIVE_INFINITY && (!Number.isInteger(maxScanned) || maxScanned <= 0)) {
    throw new RangeError(`maxScanned must be a positive integer, received: ${String(maxScanned)}`);
  }
  if (!Number.isInteger(maxMutationsPerRun) || maxMutationsPerRun < 0) {
    throw new RangeError(
      `maxMutationsPerRun must be an integer >= 0, received: ${String(maxMutationsPerRun)}`,
    );
  }

  const runNowMs = config.nowMs();
  const scopePolicy: ScopeSweepPolicy =
    typeof config.scopeSweepPolicy === "function"
      ? await config.scopeSweepPolicy(scope, config.storeId)
      : (config.scopeSweepPolicy ?? { action: "none" });

  const tenantField =
    config.tenantBinding?.mode === "metadata"
      ? (config.tenantBinding.field ?? DECAY_METADATA_KEYS.tenant)
      : DECAY_METADATA_KEYS.tenant;

  const skipped = createEmptySkipCounts();
  const plannedActions: PlannedMutation[] = [];
  const errors: SweepError[] = [];
  const unresolvedCompactions: CompactionWorkRecord[] = [];

  let scanned = 0;
  let eligible = 0;
  let plannedMutations = 0;
  let appliedMutations = 0;
  let remainingBudget = maxMutationsPerRun;
  let pendingWork = false;
  let partial = false;

  const compactionCtx: CompactionExecutionContext = {
    storeId: config.storeId,
    scope,
    runNowMs,
    nowMs: config.nowMs,
    memory: config.memory,
    qdrantReader: config.qdrantReader,
    policyResolver: config.policyResolver,
    writeCoordinator: config.writeCoordinator,
    summarizer: config.summarizer,
    workStore: config.compactionWorkStore,
    tenantField,
    compactionConfig: scopePolicy.compaction,
    getRemainingBudget: () => remainingBudget,
    consumeBudget: (count: number) => {
      remainingBudget = Math.max(0, remainingBudget - count);
    },
    onRevisionConflict: () => {
      skipped.revision_conflict += 1;
    },
    onError: (err) => {
      errors.push(err);
    },
    onUnresolved: (work) => {
      unresolvedCompactions.push(work);
    },
    onExpiredSourceDuringPrep: async (sourceId: string) => {
      const expiryAction = scopePolicy.expiryAction;
      if (!expiryAction || remainingBudget < 1) {
        skipped.action_disabled += 1;
        return;
      }
      const current = await config.memory.get(sourceId);
      if (!current) return;
      const payload = buildFlatPayloadFromMemoryItem(current);
      const effectiveNowMs = Math.max(runNowMs, config.nowMs());
      const recheck = evaluateRecordEligibility({
        record: {
          id: current.id,
          memory: current.memory,
          payload,
          createdAt: current.createdAt,
          updatedAt: current.updatedAt,
        },
        scope,
        runNowMs: effectiveNowMs,
        policyResolver: config.policyResolver,
        scopePolicy,
        tenantField,
      });
      if (!recheck.eligible) return;
      const res = await applyArchiveOrDelete({
        config,
        item: recheck.value,
        action: expiryAction,
        scopePolicy,
        runNowMs: effectiveNowMs,
        tenantField,
      });
      if (res.status === "deleted") {
        appliedMutations += 1;
        remainingBudget = Math.max(0, remainingBudget - 1);
      } else if (res.status === "conflict") {
        skipped.revision_conflict += 1;
      } else if (res.status === "error") {
        errors.push({ stage: res.stage, sourceId, message: res.message });
      }
    },
  };

  // Step 1: Resume pending compaction work before forming new groups
  const resumeResult = await resumePendingCompactions(compactionCtx, dryRun);
  appliedMutations += resumeResult.appliedMutations;
  plannedMutations += resumeResult.plannedMutations;
  if (resumeResult.pendingWork) {
    pendingWork = true;
  }

  // Step 2: Paginated Qdrant cursor scan
  const withVector = scopePolicy.action === "compact";
  let cursor: ScanCursor | null = options.cursor ?? null;
  let hasMore = false;

  const archiveOrDeleteCandidates: EvaluatedEligibleRecord[] = [];
  const compactionCandidates: CompactionCandidate[] = [];

  while (scanned < maxScanned) {
    const remainingToScan = maxScanned - scanned;
    const limit = Math.min(pageSize, remainingToScan);

    let page;
    try {
      page = await config.qdrantReader.scan(scope, {
        cursor,
        limit,
        withVector,
      });
    } catch (err) {
      errors.push({
        stage: "scan",
        message: err instanceof Error ? err.message : String(err),
      });
      partial = true;
      pendingWork = true;
      break;
    }

    for (const record of page.records) {
      scanned += 1;

      if (resumeResult.reservedSourceIds.has(record.id)) {
        continue;
      }

      const evaluation = evaluateRecordEligibility({
        record,
        scope,
        runNowMs,
        policyResolver: config.policyResolver,
        scopePolicy,
        tenantField,
      });

      if (!evaluation.eligible) {
        skipped[evaluation.reason] += 1;
        continue;
      }

      eligible += 1;
      const item = evaluation.value;
      if (item.action === "compact") {
        compactionCandidates.push({
          id: item.record.id,
          memory: item.record.memory,
          scope: item.scope,
          metadata: item.metadata,
          policy: item.policy,
          revision: item.revision,
          vector: item.record.vector,
          rawPayload: item.record.payload,
        });
      } else {
        archiveOrDeleteCandidates.push(item);
      }
    }

    cursor = page.nextCursor;
    if (cursor === null) {
      hasMore = false;
      break;
    }

    if (scanned >= maxScanned) {
      hasMore = true;
      partial = true;
      pendingWork = true;
      break;
    }
  }

  // Step 3: Plan and optionally execute archive / delete actions
  for (const item of archiveOrDeleteCandidates) {
    const action = item.action as "archive" | "delete";
    if (dryRun) {
      if (remainingBudget < 1) {
        skipped.budget_exhausted += 1;
        pendingWork = true;
        continue;
      }
      remainingBudget -= 1;
      plannedMutations += 1;
      plannedActions.push({
        action,
        reason: item.trigger,
        sourceIds: [item.record.id],
        estimatedMutations: 1,
      });
      continue;
    }

    if (remainingBudget < 1) {
      skipped.budget_exhausted += 1;
      pendingWork = true;
      continue;
    }

    plannedMutations += 1;
    plannedActions.push({
      action,
      reason: item.trigger,
      sourceIds: [item.record.id],
      estimatedMutations: 1,
    });

    const outcome = await applyArchiveOrDelete({
      config,
      item,
      action,
      scopePolicy,
      runNowMs,
      tenantField,
    });

    if (outcome.status === "deleted") {
      appliedMutations += 1;
      remainingBudget -= 1;
    } else if (outcome.status === "reconciled") {
      // Already removed on retry; completed without double-charging deletion
    } else if (outcome.status === "conflict") {
      skipped.revision_conflict += 1;
    } else {
      errors.push({
        stage: outcome.stage,
        sourceId: item.record.id,
        message: outcome.message,
      });
      pendingWork = true;
    }
  }

  // Step 4: Plan and optionally execute Tier 0 -> Tier 1 compaction groups
  if (compactionCandidates.length > 0) {
    const plan = planCompactionGroups(
      config.storeId,
      compactionCandidates,
      scopePolicy.compaction,
    );

    skipped.unclustered += plan.unclustered.length;

    for (const group of plan.groups) {
      const estimatedMutations = 1 + group.sources.length;

      if (remainingBudget < estimatedMutations) {
        skipped.budget_exhausted += group.sources.length;
        pendingWork = true;
        continue;
      }

      plannedMutations += estimatedMutations;
      plannedActions.push({
        action: "compact",
        reason: "decay",
        sourceIds: group.sources.map((s) => s.id),
        compactionKey: group.compactionKey,
        estimatedMutations,
      });

      if (dryRun) {
        remainingBudget -= estimatedMutations;
        continue;
      }

      const step = await executeCompactionGroup(compactionCtx, group);
      appliedMutations += step.appliedMutations;
      if (step.pendingWork) {
        pendingWork = true;
      }
    }
  }

  return {
    storeId: config.storeId,
    scope,
    dryRun,
    referenceNowMs: runNowMs,
    scanned,
    eligible,
    skipped,
    plannedMutations,
    appliedMutations,
    plannedActions,
    errors,
    unresolvedCompactions,
    nextCursor: hasMore ? cursor : null,
    hasMore,
    pendingWork: pendingWork || hasMore || unresolvedCompactions.length > 0,
    partial,
  };
}
