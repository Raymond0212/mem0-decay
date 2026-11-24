import {
  assertValidScope,
  type DecayAddOptions,
  type DecayMemory,
  type DecayMemoryDependencies,
  type DecaySearchOptions,
  type DecaySearchResult,
  type Message,
  type ResolvedScope,
  type SearchResult,
  type SweepOptions,
  type SweepReport,
} from "./contracts.js";
import {
  DECAY_METADATA_KEYS,
  assertNoReservedMetadataKeys,
  parseTimestampMs,
} from "./metadata.js";
import { DEFAULT_POLICY_IDS, defaultPolicyRegistry } from "./policy.js";
import { createInProcessWriteCoordinator } from "./router.js";
import {
  DEFAULT_MAX_CANDIDATES,
  DEFAULT_OVERFETCH_FACTOR,
  DEFAULT_QUERY_HALF_LIFE_DAYS,
  assertScopeMatchesTenantBinding,
  executeDecaySearch,
} from "./search.js";
import { executeSweep } from "./sweep.js";

export function createDecayMemory(dependencies: DecayMemoryDependencies): DecayMemory {
  if (!dependencies || typeof dependencies !== "object") {
    throw new TypeError("dependencies object is required");
  }
  if (typeof dependencies.storeId !== "string" || dependencies.storeId.trim() === "") {
    throw new RangeError("storeId must be a non-empty string");
  }
  if (!dependencies.memory || typeof dependencies.memory !== "object") {
    throw new TypeError("memory instance is required");
  }

  const storeId = dependencies.storeId.trim();
  const memory = dependencies.memory;
  const nowMs = dependencies.nowMs ?? (() => Date.now());
  const policyResolver = dependencies.policyResolver ?? defaultPolicyRegistry;
  const writeCoordinator =
    dependencies.writeCoordinator ?? createInProcessWriteCoordinator();
  const defaultPolicyId = dependencies.defaultPolicyId ?? DEFAULT_POLICY_IDS.soft;
  const defaultTimeZone = dependencies.defaultTimeZone ?? "UTC";
  const overfetchFactor = dependencies.overfetchFactor ?? DEFAULT_OVERFETCH_FACTOR;
  const maxCandidates = dependencies.maxCandidates ?? DEFAULT_MAX_CANDIDATES;
  const defaultQueryHalfLifeDays =
    dependencies.defaultQueryHalfLifeDays ?? DEFAULT_QUERY_HALF_LIFE_DAYS;

  if (!Number.isInteger(overfetchFactor) || overfetchFactor <= 0) {
    throw new RangeError(`overfetchFactor must be a positive integer, received: ${String(overfetchFactor)}`);
  }
  if (!Number.isInteger(maxCandidates) || maxCandidates <= 0) {
    throw new RangeError(`maxCandidates must be a positive integer, received: ${String(maxCandidates)}`);
  }
  if (!Number.isFinite(defaultQueryHalfLifeDays) || defaultQueryHalfLifeDays <= 0) {
    throw new RangeError(
      `defaultQueryHalfLifeDays must be a finite number > 0, received: ${String(defaultQueryHalfLifeDays)}`,
    );
  }

  return {
    storeId,

    async add(messages: string | Message[], options: DecayAddOptions): Promise<SearchResult> {
      const scope = assertValidScope(options.scope);
      assertScopeMatchesTenantBinding(scope, dependencies.tenantBinding);
      assertNoReservedMetadataKeys(options.metadata);

      if (options.eventTime !== undefined && options.eventSpan !== undefined) {
        throw new RangeError("Provide either eventTime or eventSpan, not both");
      }

      const policy = policyResolver.resolveOrThrow(options.policyId ?? defaultPolicyId);
      const writeNowMs = nowMs();
      if (!Number.isFinite(writeNowMs)) {
        throw new RangeError("Clock returned non-finite timestamp");
      }

      let timeStart: number;
      let timeEnd: number;
      if (options.eventSpan) {
        timeStart = parseTimestampMs(options.eventSpan.start, {
          timeZone: defaultTimeZone,
          label: "eventSpan.start",
        });
        timeEnd = parseTimestampMs(options.eventSpan.end, {
          timeZone: defaultTimeZone,
          label: "eventSpan.end",
        });
        if (timeStart > timeEnd) {
          throw new RangeError(
            `eventSpan.start (${timeStart}) must be <= eventSpan.end (${timeEnd})`,
          );
        }
      } else if (options.eventTime !== undefined) {
        const eventMs = parseTimestampMs(options.eventTime, {
          timeZone: defaultTimeZone,
          label: "eventTime",
        });
        timeStart = eventMs;
        timeEnd = eventMs;
      } else {
        timeStart = writeNowMs;
        timeEnd = writeNowMs;
      }

      let expiresAtMs: number | undefined;
      if (options.expiresAt !== undefined) {
        expiresAtMs = parseTimestampMs(options.expiresAt, {
          timeZone: defaultTimeZone,
          label: "expiresAt",
        });
      } else if (policy.defaultTtlMs !== undefined) {
        expiresAtMs = writeNowMs + policy.defaultTtlMs;
      }

      const metadata: Record<string, unknown> = {
        ...(options.metadata ?? {}),
        [DECAY_METADATA_KEYS.timeStart]: timeStart,
        [DECAY_METADATA_KEYS.timeEnd]: timeEnd,
        [DECAY_METADATA_KEYS.ingestedAt]: writeNowMs,
        [DECAY_METADATA_KEYS.policy]: policy.id,
        [DECAY_METADATA_KEYS.tier]: 0,
        [DECAY_METADATA_KEYS.tenant]: scope.tenant,
      };

      if (expiresAtMs !== undefined) {
        metadata[DECAY_METADATA_KEYS.expiresAt] = expiresAtMs;
      }

      if (
        dependencies.tenantBinding?.mode === "metadata" &&
        dependencies.tenantBinding.field &&
        dependencies.tenantBinding.field !== DECAY_METADATA_KEYS.tenant
      ) {
        metadata[dependencies.tenantBinding.field] = scope.tenant;
      }

      return writeCoordinator.run(storeId, async () => {
        return memory.add(messages, {
          ...(scope.userId !== undefined ? { userId: scope.userId } : {}),
          ...(scope.agentId !== undefined ? { agentId: scope.agentId } : {}),
          ...(scope.runId !== undefined ? { runId: scope.runId } : {}),
          metadata,
          ...(options.filters !== undefined ? { filters: options.filters } : {}),
          ...(options.infer !== undefined ? { infer: options.infer } : {}),
          ...(options.expirationDate !== undefined
            ? { expirationDate: options.expirationDate }
            : {}),
        });
      });
    },

    async search(query: string, options: DecaySearchOptions): Promise<DecaySearchResult> {
      return executeDecaySearch(
        {
          storeId,
          memory,
          nowMs,
          policyResolver,
          defaultPolicyId,
          tenantBinding: dependencies.tenantBinding,
          defaultTimeZone,
          overfetchFactor,
          maxCandidates,
          defaultQueryHalfLifeDays,
          accessSidecar: dependencies.accessSidecar,
        },
        query,
        options,
      );
    },

    async sweep(options: SweepOptions): Promise<SweepReport> {
      return executeSweep(
        {
          storeId,
          memory,
          nowMs,
          policyResolver,
          writeCoordinator,
          tenantBinding: dependencies.tenantBinding,
          qdrantReader: dependencies.qdrantReader,
          scopeSweepPolicy: dependencies.scopeSweepPolicy,
          archiveSink: dependencies.archiveSink,
          summarizer: dependencies.summarizer,
          compactionWorkStore: dependencies.compactionWorkStore,
        },
        options,
      );
    },

    async recordUse(
      scope: ResolvedScope,
      memoryId: string,
      eventId: string,
      usedAtMs?: number,
    ): Promise<void> {
      const validScope = assertValidScope(scope);
      assertScopeMatchesTenantBinding(validScope, dependencies.tenantBinding);
      if (typeof memoryId !== "string" || memoryId.trim() === "") {
        throw new RangeError("memoryId must be a non-empty string");
      }
      if (!dependencies.accessSidecar) {
        return;
      }
      const ts = usedAtMs ?? nowMs();
      await dependencies.accessSidecar.store.recordUse(
        {
          tenant: validScope.tenant,
          storeId,
          memoryId: memoryId.trim(),
        },
        eventId,
        ts,
      );
    },
  };
}
