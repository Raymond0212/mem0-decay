import type { AddMemoryOptions, Memory, MemoryItem, Message, SearchFilters, SearchMemoryOptions, SearchResult } from "mem0ai/oss";
import type { DecayCriterion, DecayCurve } from "./curves.js";

export type { AddMemoryOptions, MemoryItem, Message, SearchFilters, SearchMemoryOptions, SearchResult };

/**
 * Minimal subset of Mem0 OSS 3.3.1 Memory required by the wrapper.
 */
export type Mem0Memory = Pick<Memory, "add" | "search" | "get" | "delete">;

/**
 * Optional update support when a host exposes ordinary text edits through the wrapper.
 */
export type Mem0UpdatableMemory = Mem0Memory & Partial<Pick<Memory, "update">>;

/**
 * Authorized scope resolved by the Mem0-Plugins router.
 * Includes a tenant and at least one of Mem0's userId, agentId, or runId.
 */
export interface ResolvedScope {
  tenant: string;
  userId?: string;
  agentId?: string;
  runId?: string;
}

export type Clock = () => number;

export type DateInput = number | Date | string;

export type TenantBinding =
  | { mode: "collection"; tenant: string }
  | { mode: "metadata"; field?: string };

export interface WriteCoordinator {
  run<T>(storeId: string, operation: () => Promise<T>): Promise<T>;
}

export interface DecayPolicyDefinition {
  /** Canonical registered policy identifier (e.g. "soft@v1", "ephemeral@v1"). */
  id: string;
  /** Ranking floor in [0, 1]. timeWeight = floor + (1 - floor) * strength. */
  floor: number;
  /** Decay curve returning raw strength in [0, 1]. */
  curve: DecayCurve;
  /** Callable criterion returning raw strength and strict `< minStrength` threshold status. */
  criterion: DecayCriterion;
  /** Raw strength threshold for sweep eligibility. */
  minStrength: number;
  /** If true, always receives weight 1 in no-date recency search. */
  fullRecencyWeight: boolean;
  /** If true, exempt from automated sweep mutations (archive/delete/compact/expiry). */
  exemptFromSweep: boolean;
  /** If true, eligible for curve-based sweep actions after ingestion grace. */
  sweepEligible: boolean;
  /** Minimum elapsed milliseconds after `decay_ingested_at` before curve-based sweep eligibility. */
  ingestionGraceMs: number;
  /** Optional default retention deadline offset from write time in ms. */
  defaultTtlMs?: number;
  /** If true, wrapper hides records past `decay_expires_at` during normal search unless `showExpired: true`. */
  enforceExactVisibilityExpiry?: boolean;
  /** Optional compatibility group tag for compaction grouping. */
  retentionCompatibilityKey?: string;
}

export interface PolicyResolver {
  resolve(policyId: string): DecayPolicyDefinition | undefined;
  resolveOrThrow(policyId: string): DecayPolicyDefinition;
  has(policyId: string): boolean;
  listIds(): string[];
}

export interface CompactionSourceRef {
  id: string;
  revision: string;
}

export interface ManagedDecayMetadata {
  decay_time_start: number;
  decay_time_end: number;
  decay_ingested_at: number;
  decay_policy: string;
  decay_tier: number;
  decay_tenant?: string;
  decay_expires_at?: number;
  decay_compaction_key?: string;
  decay_sources?: CompactionSourceRef[];
}

export interface TimeSpan {
  start: number;
  end: number;
}

export interface DecayAddOptions {
  scope: ResolvedScope;
  eventTime?: DateInput;
  eventSpan?: { start: DateInput; end: DateInput };
  policyId?: string;
  expiresAt?: DateInput;
  metadata?: Record<string, unknown>;
  filters?: SearchFilters;
  infer?: boolean;
  expirationDate?: string | null;
}

export interface RangeQuery {
  from: DateInput;
  to: DateInput;
  /** Defaults to true (strict pre-filtered [from, to) window). */
  strict?: boolean;
  /** Half-life in days for soft range fade outside the window. */
  halfLifeDays?: number;
  /** Timezone for date-only (YYYY-MM-DD) inputs. Defaults to store/UTC. */
  timeZone?: string;
}

export interface AnchorQuery {
  around: DateInput;
  /** Half-life in days for symmetric fade from the anchor. */
  halfLifeDays?: number;
  /** Timezone for date-only (YYYY-MM-DD) inputs. Defaults to store/UTC. */
  timeZone?: string;
}

export interface DecaySearchOptions {
  scope: ResolvedScope;
  topK?: number;
  threshold?: number;
  rerank?: boolean;
  showExpired?: boolean;
  filters?: SearchFilters;
  range?: RangeQuery;
  anchor?: AnchorQuery;
  /** Optional frozen clock override (used by multi-store routers). */
  referenceNowMs?: number;
}

export interface DecaySearchItem {
  id: string;
  memory: string;
  storeId: string;
  baseScore: number;
  adjustedScore: number;
  timeWeight: number;
  timeSpan: TimeSpan | null;
  tier: number | null;
  policyId: string | null;
  ingestedAt: number | null;
  expiresAt: number | null;
  createdAt?: string;
  updatedAt?: string;
  metadata: Record<string, unknown>;
}

export interface DecaySearchResult {
  results: DecaySearchItem[];
  referenceNowMs: number;
  candidateCount: number;
  mode: "recency" | "strict_range" | "soft_range" | "anchor";
}

export type ScanCursor = string | number;

export interface QdrantMemoryRecord {
  id: string;
  memory: string;
  payload: Record<string, unknown>;
  vector?: number[];
  createdAt?: string;
  updatedAt?: string;
}

export interface QdrantScanOptions {
  cursor?: ScanCursor | null;
  limit?: number;
  withVector?: boolean;
}

export interface QdrantScanPage {
  records: QdrantMemoryRecord[];
  nextCursor: ScanCursor | null;
}

export interface QdrantReader {
  scan(scope: ResolvedScope, options?: QdrantScanOptions): Promise<QdrantScanPage>;
  findSummary(scope: ResolvedScope, compactionKey: string): Promise<QdrantMemoryRecord | null>;
}

export type SweepActionType = "none" | "archive" | "delete" | "compact";
export type ExpiryActionType = "archive" | "delete";

export interface CompactionScopeConfig {
  /** Time bucket width in ms for grouping Tier 0 records. Default: 7 days. */
  timeBucketMs?: number;
  /** Minimum cosine similarity in [-1, 1] to cluster records. Default: 0.75. */
  similarityThreshold?: number;
  /** Minimum sources per group (>= 2). Default: 2. */
  minGroupSize?: number;
  /** Maximum sources per group. Default: 8. */
  maxGroupSize?: number;
  /** Policy ID stamped onto Tier 1 summaries. Default: "tier1_summary@v1". */
  targetTierPolicyId?: string;
}

export interface ScopeSweepPolicy {
  action: SweepActionType;
  expiryAction?: ExpiryActionType;
  compaction?: CompactionScopeConfig;
}

export type ScopeSweepPolicyResolver = (
  scope: ResolvedScope,
  storeId: string,
) => ScopeSweepPolicy | Promise<ScopeSweepPolicy>;

export type SweepSkipReason =
  | "unmanaged"
  | "malformed_metadata"
  | "unknown_policy"
  | "exempt_policy"
  | "ranking_only"
  | "grace_period"
  | "above_threshold"
  | "tier_exempt"
  | "action_disabled"
  | "budget_exhausted"
  | "revision_conflict"
  | "scope_mismatch"
  | "unclustered";

export interface ArchiveRequest {
  key: string;
  storeId: string;
  scope: ResolvedScope;
  sourceId: string;
  revision: string;
  memory: string;
  metadata: Record<string, unknown>;
  archivedAtMs: number;
}

export interface ArchiveAck {
  confirmed: boolean;
  key: string;
  location?: string;
}

export interface ArchiveSink {
  archive(request: ArchiveRequest): Promise<ArchiveAck | void>;
}

export interface CompactionSourceInput {
  id: string;
  revision: string;
  memory: string;
  timeSpan: TimeSpan;
  metadata: Record<string, unknown>;
}

export interface CompactionSummarizeRequest {
  storeId: string;
  scope: ResolvedScope;
  sources: CompactionSourceInput[];
}

export interface CompactionSummarizeResult {
  summary: string;
}

export interface CompactionSummarizer {
  summarize(request: CompactionSummarizeRequest): Promise<CompactionSummarizeResult>;
}

export type CompactionWorkStatus = "pending" | "confirmed" | "uncertain" | "completed";

export interface CompactionWorkRecord {
  compactionKey: string;
  storeId: string;
  scope: ResolvedScope;
  targetTier: 1;
  sources: CompactionSourceRef[];
  completedSourceIds: string[];
  summaryId?: string;
  status: CompactionWorkStatus;
  createdAtMs: number;
  updatedAtMs: number;
  error?: string;
}

export interface CompactionWorkStore {
  listPending(storeId: string, scope: ResolvedScope): Promise<CompactionWorkRecord[]>;
  get(compactionKey: string): Promise<CompactionWorkRecord | null>;
  save(record: CompactionWorkRecord): Promise<void>;
}

export interface AccessKey {
  tenant: string;
  storeId: string;
  memoryId: string;
}

export interface AccessRecord {
  lastUsedAt: number;
  accessCount: number;
}

export interface AccessStore {
  getMany(keys: AccessKey[]): Promise<Map<string, AccessRecord>>;
  recordUse(key: AccessKey, eventId: string, usedAtMs: number): Promise<void>;
  close?(): Promise<void>;
}

export interface AccessSidecarConfig {
  store: AccessStore;
  /** Maximum days that confirmed usage can reduce effective ranking age. Default: 14. */
  maxAgeReductionDays?: number;
  /** Timeout in ms before falling back to age-only ranking. Default: 100. */
  readTimeoutMs?: number;
}

export interface SweepOptions {
  scope: ResolvedScope;
  cursor?: ScanCursor | null;
  pageSize?: number;
  maxScanned?: number;
  maxMutationsPerRun?: number;
  /** Defaults to true. */
  dryRun?: boolean;
}

export interface PlannedMutation {
  action: "archive" | "delete" | "compact";
  reason: "decay" | "expired" | "resume_compaction";
  sourceIds: string[];
  compactionKey?: string;
  estimatedMutations: number;
}

export interface SweepError {
  stage: "scan" | "archive" | "delete" | "compact" | "resume";
  sourceId?: string;
  compactionKey?: string;
  message: string;
}

export interface SweepReport {
  storeId: string;
  scope: ResolvedScope;
  dryRun: boolean;
  referenceNowMs: number;
  scanned: number;
  eligible: number;
  skipped: Record<SweepSkipReason, number>;
  plannedMutations: number;
  appliedMutations: number;
  plannedActions: PlannedMutation[];
  errors: SweepError[];
  unresolvedCompactions: CompactionWorkRecord[];
  nextCursor: ScanCursor | null;
  hasMore: boolean;
  pendingWork: boolean;
  partial: boolean;
}

export interface DecayMemoryDependencies {
  storeId: string;
  memory: Mem0UpdatableMemory;
  nowMs?: Clock;
  policyResolver?: PolicyResolver;
  writeCoordinator?: WriteCoordinator;
  defaultPolicyId?: string;
  tenantBinding?: TenantBinding;
  defaultTimeZone?: string;
  overfetchFactor?: number;
  maxCandidates?: number;
  defaultQueryHalfLifeDays?: number;
  qdrantReader?: QdrantReader;
  scopeSweepPolicy?: ScopeSweepPolicy | ScopeSweepPolicyResolver;
  archiveSink?: ArchiveSink;
  summarizer?: CompactionSummarizer;
  compactionWorkStore?: CompactionWorkStore;
  accessSidecar?: AccessSidecarConfig;
}

export interface DecayMemory {
  readonly storeId: string;
  add(messages: string | Message[], options: DecayAddOptions): Promise<SearchResult>;
  search(query: string, options: DecaySearchOptions): Promise<DecaySearchResult>;
  sweep(options: SweepOptions): Promise<SweepReport>;
  recordUse(scope: ResolvedScope, memoryId: string, eventId: string, usedAtMs?: number): Promise<void>;
}

export function assertValidScope(scope: ResolvedScope): ResolvedScope {
  if (!scope || typeof scope !== "object") {
    throw new TypeError("scope is required");
  }
  if (typeof scope.tenant !== "string" || scope.tenant.trim() === "") {
    throw new RangeError("scope.tenant must be a non-empty string");
  }
  const userId = normalizeOptionalId(scope.userId, "scope.userId");
  const agentId = normalizeOptionalId(scope.agentId, "scope.agentId");
  const runId = normalizeOptionalId(scope.runId, "scope.runId");
  if (!userId && !agentId && !runId) {
    throw new RangeError("scope must include at least one of userId, agentId, or runId");
  }
  return {
    tenant: scope.tenant.trim(),
    ...(userId ? { userId } : {}),
    ...(agentId ? { agentId } : {}),
    ...(runId ? { runId } : {}),
  };
}

function normalizeOptionalId(value: string | undefined, name: string): string | undefined {
  if (value === undefined) {
    return undefined;
  }
  if (typeof value !== "string" || value.trim() === "") {
    throw new RangeError(`${name} must be a non-empty string when provided`);
  }
  return value.trim();
}
