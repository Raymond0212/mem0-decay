import type { AccessKey, AccessRecord, AccessSidecarConfig, AccessStore } from "./contracts.js";
import { MS_PER_DAY } from "./policy.js";

export const DEFAULT_MAX_AGE_REDUCTION_DAYS = 14;
export const DEFAULT_ACCESS_READ_TIMEOUT_MS = 100;

export function serializeAccessKey(key: AccessKey): string {
  return JSON.stringify([key.tenant, key.storeId, key.memoryId]);
}

/**
 * In-memory reference implementation of the optional AccessStore port.
 * Scopes keys by tenant, storeId, and memoryId; deduplicates retries by eventId;
 * and keeps `lastUsedAt` monotonic (older events never move `lastUsedAt` backward).
 */
export class InMemoryAccessStore implements AccessStore {
  private readonly records = new Map<string, AccessRecord>();
  private readonly seenEvents = new Set<string>();
  private closed = false;

  async getMany(keys: AccessKey[]): Promise<Map<string, AccessRecord>> {
    this.assertOpen();
    const out = new Map<string, AccessRecord>();
    for (const key of keys) {
      const serialized = serializeAccessKey(key);
      const record = this.records.get(serialized);
      if (record) {
        out.set(serialized, { ...record });
      }
    }
    return out;
  }

  async recordUse(key: AccessKey, eventId: string, usedAtMs: number): Promise<void> {
    this.assertOpen();
    if (typeof eventId !== "string" || eventId.trim() === "") {
      throw new RangeError("eventId must be a non-empty string");
    }
    if (typeof usedAtMs !== "number" || !Number.isFinite(usedAtMs)) {
      throw new RangeError("usedAtMs must be a finite epoch millisecond timestamp");
    }

    const serializedKey = serializeAccessKey(key);
    const dedupeKey = `${serializedKey}::${eventId.trim()}`;
    if (this.seenEvents.has(dedupeKey)) {
      return;
    }
    this.seenEvents.add(dedupeKey);

    const existing = this.records.get(serializedKey);
    if (!existing) {
      this.records.set(serializedKey, {
        lastUsedAt: Math.trunc(usedAtMs),
        accessCount: 1,
      });
      return;
    }

    existing.accessCount += 1;
    if (usedAtMs > existing.lastUsedAt) {
      existing.lastUsedAt = Math.trunc(usedAtMs);
    }
  }

  async close(): Promise<void> {
    this.closed = true;
  }

  private assertOpen(): void {
    if (this.closed) {
      throw new Error("AccessStore is closed");
    }
  }
}

/**
 * Compute effective ranking age (in days) given raw event age and an optional confirmed usage record.
 * Reduces effective age by at most `maxAgeReductionDays` based on `lastUsedAt`, never below 0,
 * and never increases age beyond `rawAgeDays`.
 */
export function computeEffectiveAgeDays(input: {
  rawAgeDays: number;
  referenceNowMs: number;
  usage?: AccessRecord;
  maxAgeReductionDays?: number;
}): number {
  const { rawAgeDays, referenceNowMs, usage } = input;
  const maxReduction = input.maxAgeReductionDays ?? DEFAULT_MAX_AGE_REDUCTION_DAYS;
  if (!usage || !Number.isFinite(usage.lastUsedAt) || maxReduction <= 0 || rawAgeDays <= 0) {
    return Math.max(0, rawAgeDays);
  }

  const sinceLastUseDays = Math.max(0, (referenceNowMs - usage.lastUsedAt) / MS_PER_DAY);
  if (sinceLastUseDays >= rawAgeDays) {
    return rawAgeDays;
  }

  const reduction = Math.min(maxReduction, rawAgeDays - sinceLastUseDays);
  return Math.max(0, rawAgeDays - reduction);
}

/**
 * Batch-read usage records with a timeout guard.
 * Returns an empty Map on timeout or sidecar failure so retrieval falls back to age-only ranking.
 */
export async function readAccessBatchSafe(
  sidecar: AccessSidecarConfig | undefined,
  keys: AccessKey[],
): Promise<Map<string, AccessRecord>> {
  if (!sidecar || keys.length === 0) {
    return new Map();
  }

  const timeoutMs = sidecar.readTimeoutMs ?? DEFAULT_ACCESS_READ_TIMEOUT_MS;
  let timer: ReturnType<typeof setTimeout> | undefined;

  try {
    const readPromise = sidecar.store.getMany(keys);
    const timeoutPromise = new Promise<never>((_, reject) => {
      timer = setTimeout(() => reject(new Error("Access sidecar read timeout")), timeoutMs);
    });
    const result = await Promise.race([readPromise, timeoutPromise]);
    return result instanceof Map ? result : new Map();
  } catch {
    return new Map();
  } finally {
    if (timer !== undefined) {
      clearTimeout(timer);
    }
  }
}
