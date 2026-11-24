import { createHash } from "node:crypto";
import type {
  CompactionSourceRef,
  DateInput,
  ManagedDecayMetadata,
  ResolvedScope,
  TimeSpan,
} from "./contracts.js";

export const DECAY_METADATA_PREFIX = "decay_";

export const DECAY_METADATA_KEYS = {
  timeStart: "decay_time_start",
  timeEnd: "decay_time_end",
  ingestedAt: "decay_ingested_at",
  policy: "decay_policy",
  tier: "decay_tier",
  expiresAt: "decay_expires_at",
  compactionKey: "decay_compaction_key",
  sources: "decay_sources",
  tenant: "decay_tenant",
} as const;

const DATE_ONLY_REGEX = /^(\d{4})-(\d{2})-(\d{2})$/;

export function assertNoReservedMetadataKeys(
  metadata: Record<string, unknown> | undefined,
): void {
  if (!metadata) {
    return;
  }
  for (const key of Object.keys(metadata)) {
    if (key.startsWith(DECAY_METADATA_PREFIX)) {
      throw new RangeError(
        `Metadata key '${key}' is reserved for mem0-decay (namespace '${DECAY_METADATA_PREFIX}*')`,
      );
    }
  }
}

export function isDateOnlyString(input: DateInput): input is string {
  return typeof input === "string" && DATE_ONLY_REGEX.test(input.trim());
}

/**
 * Convert a local wall-clock calendar date (YYYY-MM-DD 00:00:00.000) in the given IANA timezone
 * into UTC epoch milliseconds without external dependencies.
 */
export function zonedStartOfDayMs(year: number, month: number, day: number, timeZone: string): number {
  // Validate timezone and calendar date rollover
  const probe = new Date(Date.UTC(year, month - 1, day, 0, 0, 0, 0));
  if (
    probe.getUTCFullYear() !== year ||
    probe.getUTCMonth() !== month - 1 ||
    probe.getUTCDate() !== day
  ) {
    throw new RangeError(`Invalid calendar date: ${year}-${month}-${day}`);
  }

  if (timeZone === "UTC" || timeZone === "Etc/UTC") {
    return probe.getTime();
  }

  const formatter = new Intl.DateTimeFormat("en-US", {
    timeZone,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hourCycle: "h23",
  });

  const getLocalAsUtcMs = (utcMs: number): number => {
    const parts = formatter.formatToParts(new Date(utcMs));
    let y = 0;
    let m = 0;
    let d = 0;
    let h = 0;
    let min = 0;
    let sec = 0;
    for (const part of parts) {
      if (part.type === "year") y = Number(part.value);
      else if (part.type === "month") m = Number(part.value);
      else if (part.type === "day") d = Number(part.value);
      else if (part.type === "hour") h = Number(part.value) % 24;
      else if (part.type === "minute") min = Number(part.value);
      else if (part.type === "second") sec = Number(part.value);
    }
    return Date.UTC(y, m - 1, d, h, min, sec, 0);
  };

  const targetLocalMs = Date.UTC(year, month - 1, day, 0, 0, 0, 0);
  let guess = targetLocalMs;
  for (let i = 0; i < 4; i++) {
    const localAtGuess = getLocalAsUtcMs(guess);
    const diff = localAtGuess - targetLocalMs;
    if (diff === 0) {
      break;
    }
    guess -= diff;
  }
  return guess;
}

function parseDateOnlyParts(dateStr: string): { year: number; month: number; day: number } {
  const match = DATE_ONLY_REGEX.exec(dateStr.trim());
  if (!match) {
    throw new RangeError(`Invalid date-only string: '${dateStr}'`);
  }
  return {
    year: Number(match[1]),
    month: Number(match[2]),
    day: Number(match[3]),
  };
}

function addOneCalendarDay(year: number, month: number, day: number): { year: number; month: number; day: number } {
  const next = new Date(Date.UTC(year, month - 1, day + 1, 0, 0, 0, 0));
  return {
    year: next.getUTCFullYear(),
    month: next.getUTCMonth() + 1,
    day: next.getUTCDate(),
  };
}

/**
 * Parse a timestamp input to epoch ms. Date-only strings resolve to start of day in `timeZone`.
 */
export function parseTimestampMs(
  input: DateInput,
  options: { timeZone?: string; endOfDayExclusive?: boolean; label?: string } = {},
): number {
  const label = options.label ?? "timestamp";
  const timeZone = options.timeZone ?? "UTC";

  if (typeof input === "number") {
    if (!Number.isFinite(input)) {
      throw new RangeError(`${label} must be a finite epoch millisecond number`);
    }
    return Math.trunc(input);
  }

  if (input instanceof Date) {
    const ms = input.getTime();
    if (!Number.isFinite(ms)) {
      throw new RangeError(`${label} is an invalid Date`);
    }
    return ms;
  }

  if (typeof input === "string") {
    const trimmed = input.trim();
    if (trimmed === "") {
      throw new RangeError(`${label} cannot be empty`);
    }
    if (DATE_ONLY_REGEX.test(trimmed)) {
      const { year, month, day } = parseDateOnlyParts(trimmed);
      if (options.endOfDayExclusive) {
        const next = addOneCalendarDay(year, month, day);
        return zonedStartOfDayMs(next.year, next.month, next.day, timeZone);
      }
      return zonedStartOfDayMs(year, month, day, timeZone);
    }
    const parsed = Date.parse(trimmed);
    if (!Number.isFinite(parsed)) {
      throw new RangeError(`${label} is not a valid date string: '${input}'`);
    }
    return parsed;
  }

  throw new TypeError(`${label} must be a number, Date, or string`);
}

export function normalizeDateRange(
  from: DateInput,
  to: DateInput,
  timeZone = "UTC",
): { fromMs: number; toMs: number } {
  const fromMs = parseTimestampMs(from, { timeZone, endOfDayExclusive: false, label: "from" });
  const toMs = parseTimestampMs(to, {
    timeZone,
    endOfDayExclusive: isDateOnlyString(to),
    label: "to",
  });
  if (fromMs >= toMs) {
    throw new RangeError(`Invalid date range: 'from' (${fromMs}) must be strictly before 'to' (${toMs})`);
  }
  return { fromMs, toMs };
}

export function normalizeAnchorSpan(around: DateInput, timeZone = "UTC"): TimeSpan {
  if (isDateOnlyString(around)) {
    const start = parseTimestampMs(around, { timeZone, endOfDayExclusive: false, label: "around" });
    const end = parseTimestampMs(around, { timeZone, endOfDayExclusive: true, label: "around" });
    return { start, end };
  }
  const ms = parseTimestampMs(around, { timeZone, label: "around" });
  return { start: ms, end: ms };
}

export type ParsedManagedMetadataResult =
  | { status: "unmanaged" }
  | { status: "malformed"; reason: string }
  | { status: "valid"; metadata: ManagedDecayMetadata };

function isFiniteIntOrNumber(v: unknown): v is number {
  return typeof v === "number" && Number.isFinite(v);
}

function parseCompactionSources(raw: unknown): CompactionSourceRef[] | null {
  if (!Array.isArray(raw) || raw.length === 0) {
    return null;
  }
  const result: CompactionSourceRef[] = [];
  for (const item of raw) {
    if (!item || typeof item !== "object") {
      return null;
    }
    const rec = item as Record<string, unknown>;
    if (typeof rec.id !== "string" || rec.id.trim() === "") {
      return null;
    }
    if (typeof rec.revision !== "string" || rec.revision.trim() === "") {
      return null;
    }
    result.push({ id: rec.id, revision: rec.revision });
  }
  return result;
}

export function parseManagedMetadata(
  payload: Record<string, unknown> | undefined,
): ParsedManagedMetadataResult {
  if (!payload || typeof payload !== "object") {
    return { status: "unmanaged" };
  }

  const hasAnyDecayKey = Object.keys(payload).some((k) => k.startsWith(DECAY_METADATA_PREFIX));
  if (!hasAnyDecayKey) {
    return { status: "unmanaged" };
  }

  const timeStart = payload[DECAY_METADATA_KEYS.timeStart];
  const timeEnd = payload[DECAY_METADATA_KEYS.timeEnd];
  const ingestedAt = payload[DECAY_METADATA_KEYS.ingestedAt];
  const policy = payload[DECAY_METADATA_KEYS.policy];
  const tier = payload[DECAY_METADATA_KEYS.tier];

  if (!isFiniteIntOrNumber(timeStart)) {
    return { status: "malformed", reason: "decay_time_start must be a finite number" };
  }
  if (!isFiniteIntOrNumber(timeEnd)) {
    return { status: "malformed", reason: "decay_time_end must be a finite number" };
  }
  if (timeStart > timeEnd) {
    return { status: "malformed", reason: "decay_time_start must be <= decay_time_end" };
  }
  if (!isFiniteIntOrNumber(ingestedAt)) {
    return { status: "malformed", reason: "decay_ingested_at must be a finite number" };
  }
  if (typeof policy !== "string" || policy.trim() === "") {
    return { status: "malformed", reason: "decay_policy must be a non-empty string" };
  }
  if (typeof tier !== "number" || !Number.isInteger(tier) || tier < 0) {
    return { status: "malformed", reason: "decay_tier must be an integer >= 0" };
  }

  const expiresAt = payload[DECAY_METADATA_KEYS.expiresAt];
  if (expiresAt !== undefined && expiresAt !== null && !isFiniteIntOrNumber(expiresAt)) {
    return { status: "malformed", reason: "decay_expires_at must be a finite number when present" };
  }

  const tenant = payload[DECAY_METADATA_KEYS.tenant];
  if (tenant !== undefined && (typeof tenant !== "string" || tenant.trim() === "")) {
    return { status: "malformed", reason: "decay_tenant must be a non-empty string when present" };
  }

  const compactionKey = payload[DECAY_METADATA_KEYS.compactionKey];
  if (compactionKey !== undefined && (typeof compactionKey !== "string" || compactionKey.trim() === "")) {
    return { status: "malformed", reason: "decay_compaction_key must be a non-empty string when present" };
  }

  const rawSources = payload[DECAY_METADATA_KEYS.sources];
  let sources: CompactionSourceRef[] | undefined;
  if (rawSources !== undefined) {
    const parsedSources = parseCompactionSources(rawSources);
    if (!parsedSources) {
      return { status: "malformed", reason: "decay_sources must be a non-empty array of { id, revision }" };
    }
    sources = parsedSources;
  }

  const metadata: ManagedDecayMetadata = {
    decay_time_start: timeStart,
    decay_time_end: timeEnd,
    decay_ingested_at: ingestedAt,
    decay_policy: policy.trim(),
    decay_tier: tier,
    ...(typeof tenant === "string" ? { decay_tenant: tenant.trim() } : {}),
    ...(isFiniteIntOrNumber(expiresAt) ? { decay_expires_at: expiresAt } : {}),
    ...(typeof compactionKey === "string" ? { decay_compaction_key: compactionKey.trim() } : {}),
    ...(sources ? { decay_sources: sources } : {}),
  };

  return { status: "valid", metadata };
}

export function extractLegacyCreatedAtMs(
  createdAt: unknown,
  metadata?: Record<string, unknown>,
): number | null {
  const candidate = createdAt ?? metadata?.createdAt ?? metadata?.created_at;
  if (candidate === undefined || candidate === null) {
    return null;
  }
  if (typeof candidate === "number" && Number.isFinite(candidate)) {
    return candidate;
  }
  if (typeof candidate === "string" && candidate.trim() !== "") {
    const parsed = Date.parse(candidate);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return null;
}

export function extractScopeIdentityFromPayload(
  payload: Record<string, unknown> | undefined,
): { tenant?: string; userId?: string; agentId?: string; runId?: string } {
  if (!payload) {
    return {};
  }
  const tenant =
    typeof payload[DECAY_METADATA_KEYS.tenant] === "string"
      ? (payload[DECAY_METADATA_KEYS.tenant] as string)
      : undefined;
  const userId = typeof payload.user_id === "string" ? payload.user_id : undefined;
  const agentId = typeof payload.agent_id === "string" ? payload.agent_id : undefined;
  const runId = typeof payload.run_id === "string" ? payload.run_id : undefined;
  return {
    ...(tenant ? { tenant } : {}),
    ...(userId ? { userId } : {}),
    ...(agentId ? { agentId } : {}),
    ...(runId ? { runId } : {}),
  };
}

export function payloadMatchesScope(
  payload: Record<string, unknown> | undefined,
  scope: ResolvedScope,
  options: { requireExactIdentity?: boolean; tenantField?: string } = {},
): boolean {
  if (!payload) {
    return false;
  }
  const tenantField = options.tenantField ?? DECAY_METADATA_KEYS.tenant;
  const payloadTenant = payload[tenantField];
  if (payloadTenant !== undefined && payloadTenant !== scope.tenant) {
    return false;
  }

  const payloadUserId = typeof payload.user_id === "string" ? payload.user_id : undefined;
  const payloadAgentId = typeof payload.agent_id === "string" ? payload.agent_id : undefined;
  const payloadRunId = typeof payload.run_id === "string" ? payload.run_id : undefined;

  if (options.requireExactIdentity) {
    return (
      payloadUserId === scope.userId &&
      payloadAgentId === scope.agentId &&
      payloadRunId === scope.runId
    );
  }

  if (scope.userId !== undefined && payloadUserId !== scope.userId) return false;
  if (scope.agentId !== undefined && payloadAgentId !== scope.agentId) return false;
  if (scope.runId !== undefined && payloadRunId !== scope.runId) return false;
  return true;
}

/**
 * Compute a deterministic revision fingerprint over text, scope, and eligibility metadata.
 */
export function computeRevisionFingerprint(input: {
  memory: string;
  scope: ResolvedScope;
  metadata: ManagedDecayMetadata;
}): string {
  const canonical = JSON.stringify({
    memory: input.memory,
    tenant: input.scope.tenant,
    userId: input.scope.userId ?? null,
    agentId: input.scope.agentId ?? null,
    runId: input.scope.runId ?? null,
    decay_time_start: input.metadata.decay_time_start,
    decay_time_end: input.metadata.decay_time_end,
    decay_ingested_at: input.metadata.decay_ingested_at,
    decay_policy: input.metadata.decay_policy,
    decay_tier: input.metadata.decay_tier,
    decay_expires_at: input.metadata.decay_expires_at ?? null,
  });
  return createHash("sha256").update(canonical).digest("hex");
}

/**
 * Derive a stable archive key containing store, source ID, and revision.
 */
export function buildArchiveKey(storeId: string, sourceId: string, revision: string): string {
  return `${storeId}:${sourceId}:${revision}`;
}

/**
 * Derive a deterministic compaction key from store, scope, target tier, and sorted source IDs + revisions.
 */
export function deriveCompactionKey(input: {
  storeId: string;
  scope: ResolvedScope;
  targetTier: number;
  sources: CompactionSourceRef[];
}): string {
  const sortedSources = [...input.sources].sort((a, b) => {
    if (a.id !== b.id) return a.id < b.id ? -1 : 1;
    return a.revision < b.revision ? -1 : a.revision > b.revision ? 1 : 0;
  });

  const canonical = JSON.stringify({
    storeId: input.storeId,
    tenant: input.scope.tenant,
    userId: input.scope.userId ?? null,
    agentId: input.scope.agentId ?? null,
    runId: input.scope.runId ?? null,
    targetTier: input.targetTier,
    sources: sortedSources,
  });

  const digest = createHash("sha256").update(canonical).digest("hex");
  return `cmp_t${input.targetTier}_${digest}`;
}

/**
 * Helper for ordinary text edits: preserves managed event span and ingestion timestamps
 * unless the caller explicitly provides a corrected event span.
 */
export function preserveEventSpanMetadata(
  existingMetadata: Record<string, unknown> | undefined,
  updates: Record<string, unknown> = {},
  options: { allowEventSpanCorrection?: { start: number; end: number } } = {},
): Record<string, unknown> {
  assertNoReservedMetadataKeys(updates);
  const preserved: Record<string, unknown> = {
    ...(existingMetadata ?? {}),
    ...updates,
  };
  if (options.allowEventSpanCorrection) {
    const { start, end } = options.allowEventSpanCorrection;
    if (!Number.isFinite(start) || !Number.isFinite(end) || start > end) {
      throw new RangeError("Corrected event span must have finite start <= end");
    }
    preserved[DECAY_METADATA_KEYS.timeStart] = start;
    preserved[DECAY_METADATA_KEYS.timeEnd] = end;
  }
  return preserved;
}
