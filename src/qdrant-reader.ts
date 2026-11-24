import {
  assertValidScope,
  type QdrantMemoryRecord,
  type QdrantReader,
  type QdrantScanOptions,
  type QdrantScanPage,
  type ResolvedScope,
  type ScanCursor,
  type TenantBinding,
} from "./contracts.js";
import { DECAY_METADATA_KEYS, payloadMatchesScope } from "./metadata.js";
import { assertScopeMatchesTenantBinding } from "./search.js";

export interface QdrantScrollPoint {
  id: string | number;
  payload?: Record<string, unknown> | null;
  vector?: number[] | number[][] | Record<string, unknown> | null;
}

export interface QdrantScrollResponse {
  points: QdrantScrollPoint[];
  next_page_offset?: string | number | Record<string, unknown> | null;
}

export interface QdrantFilterCondition {
  key: string;
  match?: { value: string | number | boolean };
}

export interface QdrantClientLike {
  scroll(
    collectionName: string,
    params: {
      filter?: { must: QdrantFilterCondition[] };
      limit?: number;
      offset?: string | number | null;
      with_payload?: boolean;
      with_vector?: boolean;
    },
  ): Promise<QdrantScrollResponse>;
  createPayloadIndex?(
    collectionName: string,
    params: {
      field_name: string;
      field_schema: "integer" | "keyword" | "float" | "bool";
      wait?: boolean;
    },
  ): Promise<unknown>;
}

export interface QdrantReaderConfig {
  client: QdrantClientLike;
  collectionName: string;
  tenantBinding?: TenantBinding;
  defaultPageSize?: number;
}

function buildScopeMustConditions(
  scope: ResolvedScope,
  tenantBinding?: TenantBinding,
): QdrantFilterCondition[] {
  const must: QdrantFilterCondition[] = [];

  if (scope.userId !== undefined) {
    must.push({ key: "user_id", match: { value: scope.userId } });
  }
  if (scope.agentId !== undefined) {
    must.push({ key: "agent_id", match: { value: scope.agentId } });
  }
  if (scope.runId !== undefined) {
    must.push({ key: "run_id", match: { value: scope.runId } });
  }
  if (tenantBinding?.mode === "metadata") {
    must.push({
      key: tenantBinding.field ?? DECAY_METADATA_KEYS.tenant,
      match: { value: scope.tenant },
    });
  }
  return must;
}

function extractDenseVector(raw: QdrantScrollPoint["vector"]): number[] | undefined {
  if (!raw) {
    return undefined;
  }
  if (Array.isArray(raw) && raw.length > 0 && typeof raw[0] === "number") {
    return raw as number[];
  }
  return undefined;
}

function normalizeCursor(offset: QdrantScrollResponse["next_page_offset"]): ScanCursor | null {
  if (typeof offset === "string" || typeof offset === "number") {
    return offset;
  }
  return null;
}

function mapPointToRecord(point: QdrantScrollPoint, withVector: boolean): QdrantMemoryRecord {
  const payload: Record<string, unknown> =
    point.payload && typeof point.payload === "object" ? { ...point.payload } : {};
  const memory = typeof payload.data === "string" ? payload.data : "";
  const createdAt = typeof payload.createdAt === "string" ? payload.createdAt : undefined;
  const updatedAt = typeof payload.updatedAt === "string" ? payload.updatedAt : undefined;
  const vector = withVector ? extractDenseVector(point.vector) : undefined;

  return {
    id: String(point.id),
    memory,
    payload,
    ...(vector ? { vector } : {}),
    ...(createdAt !== undefined ? { createdAt } : {}),
    ...(updatedAt !== undefined ? { updatedAt } : {}),
  };
}

export function createQdrantReader(config: QdrantReaderConfig): QdrantReader {
  if (!config.collectionName || config.collectionName.trim() === "") {
    throw new RangeError("collectionName must be a non-empty string");
  }
  const collectionName = config.collectionName.trim();
  const defaultPageSize = config.defaultPageSize ?? 100;
  const tenantField =
    config.tenantBinding?.mode === "metadata"
      ? (config.tenantBinding.field ?? DECAY_METADATA_KEYS.tenant)
      : DECAY_METADATA_KEYS.tenant;

  return {
    async scan(scope: ResolvedScope, options: QdrantScanOptions = {}): Promise<QdrantScanPage> {
      const validScope = assertValidScope(scope);
      assertScopeMatchesTenantBinding(validScope, config.tenantBinding);

      const limit = options.limit ?? defaultPageSize;
      if (!Number.isInteger(limit) || limit <= 0) {
        throw new RangeError(`scan limit must be a positive integer, received: ${String(limit)}`);
      }
      const withVector = Boolean(options.withVector);
      const must = buildScopeMustConditions(validScope, config.tenantBinding);

      const response = await config.client.scroll(collectionName, {
        filter: { must },
        limit,
        offset: options.cursor ?? undefined,
        with_payload: true,
        with_vector: withVector,
      });

      const records = (response.points ?? []).map((p) => mapPointToRecord(p, withVector));
      return {
        records,
        nextCursor: normalizeCursor(response.next_page_offset),
      };
    },

    async findSummary(
      scope: ResolvedScope,
      compactionKey: string,
    ): Promise<QdrantMemoryRecord | null> {
      const validScope = assertValidScope(scope);
      assertScopeMatchesTenantBinding(validScope, config.tenantBinding);

      if (typeof compactionKey !== "string" || compactionKey.trim() === "") {
        throw new RangeError("compactionKey must be a non-empty string");
      }

      const must = [
        ...buildScopeMustConditions(validScope, config.tenantBinding),
        {
          key: DECAY_METADATA_KEYS.compactionKey,
          match: { value: compactionKey.trim() },
        },
      ];

      const response = await config.client.scroll(collectionName, {
        filter: { must },
        limit: 5,
        with_payload: true,
        with_vector: false,
      });

      for (const point of response.points ?? []) {
        const record = mapPointToRecord(point, false);
        if (payloadMatchesScope(record.payload, validScope, { requireExactIdentity: true, tenantField })) {
          return record;
        }
      }
      return null;
    },
  };
}

/**
 * Provision the numeric time indexes and keyword compaction/scope indexes required on Qdrant.
 */
export async function ensureQdrantDecayIndexes(
  client: QdrantClientLike,
  collectionName: string,
  options: { tenantField?: string } = {},
): Promise<void> {
  if (typeof client.createPayloadIndex !== "function") {
    throw new Error("Qdrant client does not expose createPayloadIndex");
  }

  const integerFields = [
    DECAY_METADATA_KEYS.timeStart,
    DECAY_METADATA_KEYS.timeEnd,
    DECAY_METADATA_KEYS.ingestedAt,
    DECAY_METADATA_KEYS.expiresAt,
    DECAY_METADATA_KEYS.tier,
  ];

  const keywordFields = [
    DECAY_METADATA_KEYS.compactionKey,
    DECAY_METADATA_KEYS.policy,
    options.tenantField ?? DECAY_METADATA_KEYS.tenant,
    "user_id",
    "agent_id",
    "run_id",
  ];

  for (const field_name of integerFields) {
    await client.createPayloadIndex(collectionName, {
      field_name,
      field_schema: "integer",
      wait: true,
    });
  }

  for (const field_name of keywordFields) {
    await client.createPayloadIndex(collectionName, {
      field_name,
      field_schema: "keyword",
      wait: true,
    });
  }
}
