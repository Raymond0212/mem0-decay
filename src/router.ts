import {
  assertValidScope,
  type AccessStore,
  type Clock,
  type DecayAddOptions,
  type DecayMemory,
  type DecaySearchItem,
  type DecaySearchOptions,
  type DecaySearchResult,
  type Message,
  type ResolvedScope,
  type SearchResult,
  type SweepOptions,
  type SweepReport,
  type WriteCoordinator,
} from "./contracts.js";
import {
  resolveDecayCurve,
  type CurveRegistry,
  type DecayCurveInput,
} from "./curves.js";
import { DEFAULT_TOP_K, validateAlpha } from "./search.js";

export interface InProcessWriteCoordinatorOptions {
  /**
   * When true (default), enforces a strict single-writer boundary across all stores
   * coordinated by this router. When false, serializes writes per storeId.
   */
  serializeAllStores?: boolean;
}

/**
 * FIFO async write coordinator enforcing the single-writer boundary for agent writes,
 * ordinary edits, and mutating sweeps.
 */
export function createInProcessWriteCoordinator(
  options: InProcessWriteCoordinatorOptions = {},
): WriteCoordinator {
  const serializeAll = options.serializeAllStores !== false;
  const tails = new Map<string, Promise<unknown>>();

  return {
    async run<T>(storeId: string, operation: () => Promise<T>): Promise<T> {
      const lockKey = serializeAll ? "__global__" : storeId;
      const previous = tails.get(lockKey) ?? Promise.resolve();

      let release!: () => void;
      const gate = new Promise<void>((resolve) => {
        release = resolve;
      });
      const nextTail = previous.then(() => gate);
      tails.set(lockKey, nextTail);

      await previous;
      try {
        return await operation();
      } finally {
        release();
        if (tails.get(lockKey) === nextTail) {
          tails.delete(lockKey);
        }
      }
    },
  };
}

export type RouterOperation = "add" | "search" | "sweep" | "recordUse";

export interface RouterAuthorizationContext {
  principal?: string;
  scope: ResolvedScope;
  operation: RouterOperation;
  storeId: string;
}

export type ScopeAuthorizer = (
  ctx: RouterAuthorizationContext,
) => boolean | Promise<boolean>;

export interface DecayRouterConfig {
  nowMs?: Clock;
  writeCoordinator?: WriteCoordinator;
  authorize?: ScopeAuthorizer;
  accessStore?: AccessStore;
  /** Optional router-wide default `alpha` in [0, 1] when per-query `alpha` is not specified. */
  alpha?: number;
  /** Optional router-wide default decay curve for soft-range and anchor distance queries. */
  defaultQueryCurve?: DecayCurveInput;
  /** Optional custom `CurveRegistry` for resolving `{ kind }` curve specs passed to router search. */
  curveRegistry?: CurveRegistry;
}

export interface RouterSearchOptions extends DecaySearchOptions {
  principal?: string;
  /** Optional explicit subset of stores to fan out to. Defaults to all authorized stores. */
  storeIds?: string[];
}

export interface RouterAddOptions extends DecayAddOptions {
  principal?: string;
}

export interface RouterSweepJob {
  storeId: string;
  principal?: string;
  options: SweepOptions;
}

export interface DecayRouter {
  readonly writeCoordinator: WriteCoordinator;
  registerStore(store: DecayMemory, options?: { activeWriter?: boolean }): void;
  setActiveWriteStore(storeId: string): void;
  getActiveWriteStoreId(): string | null;
  listStoreIds(): string[];
  add(messages: string | Message[], options: RouterAddOptions): Promise<SearchResult & { storeId: string }>;
  search(query: string, options: RouterSearchOptions): Promise<DecaySearchResult>;
  sweep(job: RouterSweepJob): Promise<SweepReport>;
  recordUse(
    storeId: string,
    scope: ResolvedScope,
    memoryId: string,
    eventId: string,
    options?: { principal?: string; usedAtMs?: number },
  ): Promise<void>;
  shutdown(): Promise<void>;
}

function compareMergedItems(a: DecaySearchItem, b: DecaySearchItem): number {
  if (b.adjustedScore !== a.adjustedScore) {
    return b.adjustedScore - a.adjustedScore;
  }
  if (b.baseScore !== a.baseScore) {
    return b.baseScore - a.baseScore;
  }
  if (a.storeId !== b.storeId) {
    return a.storeId < b.storeId ? -1 : 1;
  }
  if (a.id === b.id) {
    return 0;
  }
  return a.id < b.id ? -1 : 1;
}

/**
 * Mem0-Plugins router composing per-store DecayMemory wrappers with:
 * - scope authorization,
 * - frozen-clock multi-store search fan-out and adjusted-score merge,
 * - one active write destination for agent writes,
 * - fixed-target sweep dispatch through the shared write coordinator.
 */
export function createDecayRouter(config: DecayRouterConfig = {}): DecayRouter {
  const nowMs = config.nowMs ?? (() => Date.now());
  const writeCoordinator =
    config.writeCoordinator ?? createInProcessWriteCoordinator({ serializeAllStores: true });
  const authorize: ScopeAuthorizer = config.authorize ?? (() => true);
  const routerAlpha =
    config.alpha !== undefined ? validateAlpha(config.alpha, "config.alpha") : undefined;
  const routerDefaultQueryCurve =
    config.defaultQueryCurve !== undefined
      ? resolveDecayCurve(config.defaultQueryCurve, config.curveRegistry)
      : undefined;

  const stores = new Map<string, DecayMemory>();
  let activeWriteStoreId: string | null = null;

  async function assertAuthorized(ctx: RouterAuthorizationContext): Promise<void> {
    const allowed = await authorize(ctx);
    if (!allowed) {
      throw new Error(
        `Unauthorized ${ctx.operation} for tenant '${ctx.scope.tenant}' on store '${ctx.storeId}'`,
      );
    }
  }

  return {
    writeCoordinator,

    registerStore(store: DecayMemory, options?: { activeWriter?: boolean }): void {
      if (stores.has(store.storeId)) {
        throw new RangeError(`Store '${store.storeId}' is already registered`);
      }
      stores.set(store.storeId, store);
      if (options?.activeWriter || activeWriteStoreId === null) {
        activeWriteStoreId = store.storeId;
      }
    },

    setActiveWriteStore(storeId: string): void {
      if (!stores.has(storeId)) {
        throw new RangeError(`Cannot select unregistered active write store '${storeId}'`);
      }
      activeWriteStoreId = storeId;
    },

    getActiveWriteStoreId(): string | null {
      return activeWriteStoreId;
    },

    listStoreIds(): string[] {
      return Array.from(stores.keys());
    },

    async add(
      messages: string | Message[],
      options: RouterAddOptions,
    ): Promise<SearchResult & { storeId: string }> {
      if (!activeWriteStoreId) {
        throw new Error("No active write store registered on router");
      }
      const targetStore = stores.get(activeWriteStoreId)!;
      const scope = assertValidScope(options.scope);
      await assertAuthorized({
        principal: options.principal,
        scope,
        operation: "add",
        storeId: targetStore.storeId,
      });

      const res = await targetStore.add(messages, {
        ...options,
        scope,
      });
      return { ...res, storeId: targetStore.storeId };
    },

    async search(query: string, options: RouterSearchOptions): Promise<DecaySearchResult> {
      const scope = assertValidScope(options.scope);
      const effectiveAlpha =
        options.alpha !== undefined ? validateAlpha(options.alpha) : routerAlpha;
      const resolveQueryCurve = (input: DecayCurveInput): DecayCurveInput =>
        config.curveRegistry ? resolveDecayCurve(input, config.curveRegistry) : input;

      const resolvedTopCurve =
        options.curve !== undefined ? resolveQueryCurve(options.curve) : undefined;
      const resolvedRange =
        options.range !== undefined
          ? {
              ...options.range,
              ...(options.range.curve !== undefined
                ? { curve: resolveQueryCurve(options.range.curve) }
                : options.range.strict === false &&
                    options.range.halfLifeDays === undefined &&
                    resolvedTopCurve === undefined &&
                    routerDefaultQueryCurve !== undefined
                  ? { curve: routerDefaultQueryCurve }
                  : {}),
            }
          : undefined;
      const resolvedAnchor =
        options.anchor !== undefined
          ? {
              ...options.anchor,
              ...(options.anchor.curve !== undefined
                ? { curve: resolveQueryCurve(options.anchor.curve) }
                : options.anchor.halfLifeDays === undefined &&
                    resolvedTopCurve === undefined &&
                    routerDefaultQueryCurve !== undefined
                  ? { curve: routerDefaultQueryCurve }
                  : {}),
            }
          : undefined;
      const referenceNowMs = options.referenceNowMs ?? nowMs();
      const topK = options.topK ?? DEFAULT_TOP_K;

      const candidateStoreIds = options.storeIds ?? Array.from(stores.keys());
      const authorizedStores: DecayMemory[] = [];

      for (const storeId of candidateStoreIds) {
        const store = stores.get(storeId);
        if (!store) {
          throw new RangeError(`Unknown store '${storeId}' requested in search`);
        }
        const allowed = await authorize({
          principal: options.principal,
          scope,
          operation: "search",
          storeId,
        });
        if (allowed) {
          authorizedStores.push(store);
        }
      }

      if (authorizedStores.length === 0) {
        return {
          results: [],
          referenceNowMs,
          candidateCount: 0,
          mode: options.range
            ? options.range.strict === false
              ? "soft_range"
              : "strict_range"
            : options.anchor
              ? "anchor"
              : "recency",
        };
      }

      const perStoreResults = await Promise.all(
        authorizedStores.map((store) =>
          store.search(query, {
            ...options,
            scope,
            topK,
            referenceNowMs,
            ...(effectiveAlpha !== undefined ? { alpha: effectiveAlpha } : {}),
            ...(resolvedTopCurve !== undefined ? { curve: resolvedTopCurve } : {}),
            ...(resolvedRange !== undefined ? { range: resolvedRange } : {}),
            ...(resolvedAnchor !== undefined ? { anchor: resolvedAnchor } : {}),
          }),
        ),
      );

      const merged: DecaySearchItem[] = [];
      let totalCandidates = 0;
      for (const res of perStoreResults) {
        totalCandidates += res.candidateCount;
        merged.push(...res.results);
      }

      merged.sort(compareMergedItems);

      return {
        results: merged.slice(0, topK),
        referenceNowMs,
        candidateCount: totalCandidates,
        mode: perStoreResults[0]!.mode,
      };
    },

    async sweep(job: RouterSweepJob): Promise<SweepReport> {
      const store = stores.get(job.storeId);
      if (!store) {
        throw new RangeError(`Unknown sweep target store '${job.storeId}'`);
      }
      const scope = assertValidScope(job.options.scope);
      await assertAuthorized({
        principal: job.principal,
        scope,
        operation: "sweep",
        storeId: store.storeId,
      });

      // The sweep target store stays fixed for the entire run, independent of activeWriteStoreId.
      return store.sweep({
        ...job.options,
        scope,
      });
    },

    async recordUse(
      storeId: string,
      scope: ResolvedScope,
      memoryId: string,
      eventId: string,
      options?: { principal?: string; usedAtMs?: number },
    ): Promise<void> {
      const store = stores.get(storeId);
      if (!store) {
        throw new RangeError(`Unknown store '${storeId}' for recordUse`);
      }
      const validScope = assertValidScope(scope);
      await assertAuthorized({
        principal: options?.principal,
        scope: validScope,
        operation: "recordUse",
        storeId: store.storeId,
      });
      await store.recordUse(validScope, memoryId, eventId, options?.usedAtMs);
    },

    async shutdown(): Promise<void> {
      if (config.accessStore?.close) {
        await config.accessStore.close();
      }
    },
  };
}
