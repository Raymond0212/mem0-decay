import {
  computeEffectiveAgeDays,
  readAccessBatchSafe,
  serializeAccessKey,
} from "./access-store.js";
import {
  assertValidScope,
  type AccessKey,
  type AccessSidecarConfig,
  type Clock,
  type DecayPolicyDefinition,
  type DecaySearchItem,
  type DecaySearchOptions,
  type DecaySearchResult,
  type Mem0Memory,
  type MemoryItem,
  type PolicyResolver,
  type ResolvedScope,
  type SearchFilters,
  type TenantBinding,
} from "./contracts.js";
import {
  decayCurves,
  resolveDecayCurve,
  type CurveRegistry,
  type DecayCurve,
  type DecayCurveInput,
} from "./curves.js";
import {
  DECAY_METADATA_KEYS,
  assertNoReservedMetadataKeys,
  extractLegacyCreatedAtMs,
  normalizeAnchorSpan,
  normalizeDateRange,
  parseManagedMetadata,
} from "./metadata.js";
import { DEFAULT_POLICY_IDS, MS_PER_DAY } from "./policy.js";

export const DEFAULT_TOP_K = 10;
export const DEFAULT_OVERFETCH_FACTOR = 4;
export const DEFAULT_MAX_CANDIDATES = 200;
export const DEFAULT_QUERY_HALF_LIFE_DAYS = 14;
export const DEFAULT_ALPHA = 0.05;

export interface SearchEngineConfig {
  storeId: string;
  memory: Mem0Memory;
  nowMs: Clock;
  policyResolver: PolicyResolver;
  defaultPolicyId: string;
  tenantBinding?: TenantBinding;
  defaultTimeZone: string;
  overfetchFactor: number;
  maxCandidates: number;
  defaultQueryHalfLifeDays: number;
  defaultQueryCurve?: DecayCurveInput;
  curveRegistry?: CurveRegistry;
  alpha?: number;
  accessSidecar?: AccessSidecarConfig;
}

export function assertScopeMatchesTenantBinding(
  scope: ResolvedScope,
  tenantBinding?: TenantBinding,
): void {
  if (tenantBinding?.mode === "collection" && scope.tenant !== tenantBinding.tenant) {
    throw new RangeError(
      `Scope tenant '${scope.tenant}' does not match collection tenant binding '${tenantBinding.tenant}'`,
    );
  }
}

export function buildMem0ScopeFilters(
  scope: ResolvedScope,
  tenantBinding?: TenantBinding,
  extraFilters?: SearchFilters,
): SearchFilters {
  if (extraFilters) {
    assertNoReservedMetadataKeys(extraFilters);
  }

  const filters: SearchFilters = {
    ...(extraFilters ?? {}),
  };

  if (scope.userId !== undefined) filters.user_id = scope.userId;
  else delete filters.user_id;

  if (scope.agentId !== undefined) filters.agent_id = scope.agentId;
  else delete filters.agent_id;

  if (scope.runId !== undefined) filters.run_id = scope.runId;
  else delete filters.run_id;

  if (tenantBinding?.mode === "metadata") {
    const field = tenantBinding.field ?? DECAY_METADATA_KEYS.tenant;
    filters[field] = scope.tenant;
  }

  return filters;
}

function extractBaseScore(item: MemoryItem): number {
  if (typeof item.rerankScore === "number" && Number.isFinite(item.rerankScore)) {
    return item.rerankScore;
  }
  if (typeof item.score === "number" && Number.isFinite(item.score)) {
    return item.score;
  }
  return 0;
}

function intervalDistanceMs(
  aStart: number,
  aEnd: number,
  bStart: number,
  bEnd: number,
): number {
  if (aEnd < bStart) {
    return bStart - aEnd;
  }
  if (aStart > bEnd) {
    return aStart - bEnd;
  }
  return 0;
}

export function validateAlpha(alpha: number, label = "alpha"): number {
  if (typeof alpha !== "number" || !Number.isFinite(alpha) || alpha < 0 || alpha > 1) {
    throw new RangeError(`${label} must be a finite number in [0, 1], received: ${String(alpha)}`);
  }
  return alpha;
}

function computeRecencyTimeWeight(input: {
  policy: DecayPolicyDefinition;
  rawAgeDays: number;
  referenceNowMs: number;
  usage: { lastUsedAt: number; accessCount: number } | undefined;
  maxAgeReductionDays?: number;
  curveOverride?: DecayCurve;
}): number {
  const { policy, rawAgeDays, referenceNowMs, usage, maxAgeReductionDays, curveOverride } = input;
  if (policy.fullRecencyWeight) {
    return 1;
  }
  const effectiveAgeDays = computeEffectiveAgeDays({
    rawAgeDays,
    referenceNowMs,
    usage,
    maxAgeReductionDays,
  });
  if (curveOverride) {
    return curveOverride.withFloor(policy.floor).evaluate(effectiveAgeDays);
  }
  return policy.weightedCurve.evaluate(effectiveAgeDays);
}

function compareRankedItems(a: DecaySearchItem, b: DecaySearchItem): number {
  if (b.adjustedScore !== a.adjustedScore) {
    return b.adjustedScore - a.adjustedScore;
  }
  if (b.baseScore !== a.baseScore) {
    return b.baseScore - a.baseScore;
  }
  if (a.id === b.id) {
    return 0;
  }
  return a.id < b.id ? -1 : 1;
}

export async function executeDecaySearch(
  config: SearchEngineConfig,
  query: string,
  options: DecaySearchOptions,
): Promise<DecaySearchResult> {
  if (typeof query !== "string" || query.trim() === "") {
    throw new RangeError("query must be a non-empty string");
  }

  const scope = assertValidScope(options.scope);
  assertScopeMatchesTenantBinding(scope, config.tenantBinding);

  if (options.range && options.anchor) {
    throw new RangeError("Cannot specify both range and anchor in the same search");
  }

  const topK = options.topK ?? DEFAULT_TOP_K;
  if (!Number.isInteger(topK) || topK <= 0) {
    throw new RangeError(`topK must be a positive integer, received: ${String(topK)}`);
  }
  if (topK > config.maxCandidates) {
    throw new RangeError(
      `Requested topK (${topK}) exceeds configured maxCandidates (${config.maxCandidates})`,
    );
  }

  const alpha = validateAlpha(options.alpha ?? config.alpha ?? DEFAULT_ALPHA);

  const referenceNowMs =
    options.referenceNowMs !== undefined ? options.referenceNowMs : config.nowMs();
  if (!Number.isFinite(referenceNowMs)) {
    throw new RangeError("referenceNowMs must be a finite epoch millisecond timestamp");
  }

  const filters = buildMem0ScopeFilters(scope, config.tenantBinding, options.filters);

  const defaultQueryCurve: DecayCurve =
    config.defaultQueryCurve !== undefined
      ? resolveDecayCurve(config.defaultQueryCurve, config.curveRegistry)
      : decayCurves.exponential(config.defaultQueryHalfLifeDays);

  const searchCurveOverride: DecayCurve | undefined =
    options.curve !== undefined
      ? resolveDecayCurve(options.curve, config.curveRegistry)
      : undefined;

  let mode: DecaySearchResult["mode"] = "recency";
  let rangeBounds: { fromMs: number; toMs: number } | undefined;
  let anchorSpan: { start: number; end: number } | undefined;
  let queryCurve: DecayCurve = searchCurveOverride ?? defaultQueryCurve;

  if (options.range) {
    if (options.range.curve !== undefined && options.range.halfLifeDays !== undefined) {
      throw new RangeError("Provide either range.curve or range.halfLifeDays, not both");
    }
    const tz = options.range.timeZone ?? config.defaultTimeZone;
    rangeBounds = normalizeDateRange(options.range.from, options.range.to, tz);
    const strict = options.range.strict !== false;
    mode = strict ? "strict_range" : "soft_range";
    if (!strict) {
      if (options.range.curve !== undefined) {
        queryCurve = resolveDecayCurve(options.range.curve, config.curveRegistry);
      } else if (options.range.halfLifeDays !== undefined) {
        queryCurve = decayCurves.exponential(options.range.halfLifeDays);
      }
    } else {
      if (options.range.curve !== undefined) {
        resolveDecayCurve(options.range.curve, config.curveRegistry);
      }
      filters[DECAY_METADATA_KEYS.timeStart] = { gte: rangeBounds.fromMs };
      filters[DECAY_METADATA_KEYS.timeEnd] = { lt: rangeBounds.toMs };
    }
  } else if (options.anchor) {
    if (options.anchor.curve !== undefined && options.anchor.halfLifeDays !== undefined) {
      throw new RangeError("Provide either anchor.curve or anchor.halfLifeDays, not both");
    }
    const tz = options.anchor.timeZone ?? config.defaultTimeZone;
    anchorSpan = normalizeAnchorSpan(options.anchor.around, tz);
    mode = "anchor";
    if (options.anchor.curve !== undefined) {
      queryCurve = resolveDecayCurve(options.anchor.curve, config.curveRegistry);
    } else if (options.anchor.halfLifeDays !== undefined) {
      queryCurve = decayCurves.exponential(options.anchor.halfLifeDays);
    }
  }

  const fetchLimit =
    mode === "strict_range"
      ? topK
      : Math.min(topK * config.overfetchFactor, config.maxCandidates);

  const mem0Result = await config.memory.search(query, {
    topK: fetchLimit,
    filters,
    ...(options.threshold !== undefined ? { threshold: options.threshold } : {}),
    ...(options.rerank !== undefined ? { rerank: options.rerank } : {}),
    ...(options.showExpired !== undefined ? { showExpired: options.showExpired } : {}),
  });

  const rawCandidates = mem0Result?.results ?? [];

  // Batch-read usage sidecar ONLY in no-date recency mode
  let usageMap = new Map<string, { lastUsedAt: number; accessCount: number }>();
  if (mode === "recency" && config.accessSidecar && rawCandidates.length > 0) {
    const keys: AccessKey[] = rawCandidates.map((item) => ({
      tenant: scope.tenant,
      storeId: config.storeId,
      memoryId: item.id,
    }));
    usageMap = await readAccessBatchSafe(config.accessSidecar, keys);
  }

  const defaultPolicy =
    config.policyResolver.resolve(config.defaultPolicyId) ??
    config.policyResolver.resolveOrThrow(DEFAULT_POLICY_IDS.soft);
  const tenantField =
    config.tenantBinding?.mode === "metadata"
      ? (config.tenantBinding.field ?? DECAY_METADATA_KEYS.tenant)
      : DECAY_METADATA_KEYS.tenant;

  const ranked: DecaySearchItem[] = [];

  for (const item of rawCandidates) {
    const metadata: Record<string, unknown> =
      item.metadata && typeof item.metadata === "object" ? { ...item.metadata } : {};

    // Tenant isolation safeguard for records carrying a tenant field
    const recordTenant = metadata[tenantField];
    if (recordTenant !== undefined && recordTenant !== scope.tenant) {
      continue;
    }

    const parsed = parseManagedMetadata(metadata);
    const managed = parsed.status === "valid" ? parsed.metadata : null;
    const policy = managed ? config.policyResolver.resolve(managed.decay_policy) : undefined;

    // Exact visibility expiry enforcement when configured on the record's policy
    if (
      !options.showExpired &&
      managed?.decay_expires_at !== undefined &&
      managed.decay_expires_at <= referenceNowMs &&
      policy?.enforceExactVisibilityExpiry === true
    ) {
      continue;
    }

    const baseScore = extractBaseScore(item);
    let timeWeight = 1;

    if (mode === "strict_range" && rangeBounds) {
      if (
        !managed ||
        managed.decay_time_start < rangeBounds.fromMs ||
        managed.decay_time_end >= rangeBounds.toMs
      ) {
        continue;
      }
      timeWeight = 1;
    } else if (mode === "soft_range" && rangeBounds) {
      if (!managed) {
        continue;
      }
      const distMs = intervalDistanceMs(
        managed.decay_time_start,
        managed.decay_time_end,
        rangeBounds.fromMs,
        rangeBounds.toMs,
      );
      timeWeight = distMs === 0 ? 1 : queryCurve.evaluate(distMs / MS_PER_DAY);
    } else if (mode === "anchor" && anchorSpan) {
      if (!managed) {
        continue;
      }
      const distMs = intervalDistanceMs(
        managed.decay_time_start,
        managed.decay_time_end,
        anchorSpan.start,
        anchorSpan.end,
      );
      timeWeight = distMs === 0 ? 1 : queryCurve.evaluate(distMs / MS_PER_DAY);
    } else {
      // No-date recency mode
      const usageKey = serializeAccessKey({
        tenant: scope.tenant,
        storeId: config.storeId,
        memoryId: item.id,
      });
      const usage = usageMap.get(usageKey);

      if (managed) {
        timeWeight = policy
          ? computeRecencyTimeWeight({
              policy,
              rawAgeDays: Math.max(0, (referenceNowMs - managed.decay_time_end) / MS_PER_DAY),
              referenceNowMs,
              usage,
              maxAgeReductionDays: config.accessSidecar?.maxAgeReductionDays,
              curveOverride: searchCurveOverride,
            })
          : 1;
      } else {
        const legacyTs = extractLegacyCreatedAtMs(item.createdAt, metadata);
        timeWeight =
          legacyTs === null
            ? 1
            : computeRecencyTimeWeight({
                policy: defaultPolicy,
                rawAgeDays: Math.max(0, (referenceNowMs - legacyTs) / MS_PER_DAY),
                referenceNowMs,
                usage,
                maxAgeReductionDays: config.accessSidecar?.maxAgeReductionDays,
                curveOverride: searchCurveOverride,
              });
      }
    }

    const adjustedScore = baseScore - baseScore * (1 - timeWeight) * alpha;

    ranked.push({
      id: item.id,
      memory: item.memory,
      storeId: config.storeId,
      baseScore,
      adjustedScore,
      timeWeight,
      timeSpan: managed
        ? { start: managed.decay_time_start, end: managed.decay_time_end }
        : null,
      tier: managed ? managed.decay_tier : null,
      policyId: managed ? managed.decay_policy : null,
      ingestedAt: managed ? managed.decay_ingested_at : null,
      expiresAt: managed?.decay_expires_at ?? null,
      ...(item.createdAt !== undefined ? { createdAt: item.createdAt } : {}),
      ...(item.updatedAt !== undefined ? { updatedAt: item.updatedAt } : {}),
      metadata,
    });
  }

  ranked.sort(compareRankedItems);

  return {
    results: ranked.slice(0, topK),
    referenceNowMs,
    candidateCount: rawCandidates.length,
    mode,
  };
}
