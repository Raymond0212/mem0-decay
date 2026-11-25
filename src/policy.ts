import type { DecayPolicyDefinition, PolicyResolver } from "./contracts.js";
import {
  DEFAULT_HALF_LIFE_DAYS,
  DEFAULT_MIN_STRENGTH,
  decayCurves,
  resolveDecayCurve,
  type CurveRegistry,
  type DecayCurve,
  type DecayCurveInput,
} from "./curves.js";

export const MS_PER_HOUR = 60 * 60 * 1000;
export const MS_PER_DAY = 24 * MS_PER_HOUR;

export const DEFAULT_POLICY_IDS = {
  soft: "soft@v1",
  ephemeral: "ephemeral@v1",
  pinned: "pinned@v1",
  evergreen: "evergreen@v1",
  tier1Summary: "tier1_summary@v1",
} as const;

export interface PolicyRegistrationInput {
  id: string;
  aliases?: string[];
  floor?: number;
  halfLifeDays?: number;
  curve?: DecayCurveInput;
  minStrength?: number;
  fullRecencyWeight?: boolean;
  exemptFromSweep?: boolean;
  sweepEligible?: boolean;
  ingestionGraceMs?: number;
  defaultTtlMs?: number;
  enforceExactVisibilityExpiry?: boolean;
  retentionCompatibilityKey?: string;
}

export function definePolicy(
  input: PolicyRegistrationInput,
  options: { curveRegistry?: CurveRegistry } = {},
): DecayPolicyDefinition {
  if (typeof input.id !== "string" || input.id.trim() === "") {
    throw new RangeError("Policy id must be a non-empty string");
  }
  const id = input.id.trim();
  if (input.curve !== undefined && input.halfLifeDays !== undefined) {
    throw new RangeError(`Policy '${id}' must provide either curve or halfLifeDays, not both`);
  }

  const floor = input.floor ?? 0;
  if (!Number.isFinite(floor) || floor < 0 || floor > 1) {
    throw new RangeError(`Policy '${id}' floor must be in [0, 1], received: ${String(floor)}`);
  }

  const ingestionGraceMs = input.ingestionGraceMs ?? 0;
  if (!Number.isFinite(ingestionGraceMs) || ingestionGraceMs < 0) {
    throw new RangeError(
      `Policy '${id}' ingestionGraceMs must be a finite number >= 0, received: ${String(ingestionGraceMs)}`,
    );
  }

  if (input.defaultTtlMs !== undefined && (!Number.isFinite(input.defaultTtlMs) || input.defaultTtlMs <= 0)) {
    throw new RangeError(
      `Policy '${id}' defaultTtlMs must be a finite number > 0, received: ${String(input.defaultTtlMs)}`,
    );
  }

  const minStrength = input.minStrength ?? DEFAULT_MIN_STRENGTH;
  const fullRecencyWeight = input.fullRecencyWeight ?? false;
  const exemptFromSweep = input.exemptFromSweep ?? false;
  const sweepEligible = exemptFromSweep ? false : (input.sweepEligible ?? false);

  const curve: DecayCurve =
    input.curve !== undefined
      ? resolveDecayCurve(input.curve, options.curveRegistry)
      : fullRecencyWeight
        ? decayCurves.constant(1)
        : decayCurves.exponential({ halfLifeDays: input.halfLifeDays ?? DEFAULT_HALF_LIFE_DAYS });

  const weightedCurve: DecayCurve = fullRecencyWeight
    ? decayCurves.constant(1)
    : curve.withFloor(floor);

  const criterion = curve.toCriterion(minStrength);

  return {
    id,
    floor,
    curve,
    weightedCurve,
    criterion,
    minStrength,
    fullRecencyWeight,
    exemptFromSweep,
    sweepEligible,
    ingestionGraceMs,
    ...(input.defaultTtlMs !== undefined ? { defaultTtlMs: input.defaultTtlMs } : {}),
    ...(input.enforceExactVisibilityExpiry !== undefined
      ? { enforceExactVisibilityExpiry: input.enforceExactVisibilityExpiry }
      : {}),
    ...(input.retentionCompatibilityKey !== undefined
      ? { retentionCompatibilityKey: input.retentionCompatibilityKey }
      : {}),
  };
}

export interface PolicyRegistry extends PolicyResolver {
  register(input: PolicyRegistrationInput): DecayPolicyDefinition;
  registerAlias(alias: string, targetPolicyId: string): void;
}

export function createPolicyRegistry(
  initialPolicies: PolicyRegistrationInput[] = [],
  options: { includeDefaults?: boolean; curveRegistry?: CurveRegistry } = {},
): PolicyRegistry {
  const byId = new Map<string, DecayPolicyDefinition>();
  const aliases = new Map<string, string>();

  const register = (input: PolicyRegistrationInput): DecayPolicyDefinition => {
    const def = definePolicy(input, { curveRegistry: options.curveRegistry });
    byId.set(def.id, def);
    if (input.aliases) {
      for (const alias of input.aliases) {
        if (typeof alias !== "string" || alias.trim() === "") {
          throw new RangeError("Policy alias must be a non-empty string");
        }
        aliases.set(alias.trim(), def.id);
      }
    }
    return def;
  };

  const registerAlias = (alias: string, targetPolicyId: string): void => {
    const cleanAlias = alias.trim();
    const cleanTarget = targetPolicyId.trim();
    if (!cleanAlias || !byId.has(cleanTarget)) {
      throw new RangeError(`Cannot alias '${alias}' to unregistered policy '${targetPolicyId}'`);
    }
    aliases.set(cleanAlias, cleanTarget);
  };

  if (options.includeDefaults !== false) {
    register({
      id: DEFAULT_POLICY_IDS.soft,
      aliases: ["soft"],
      halfLifeDays: 30,
      floor: 0.3,
      minStrength: DEFAULT_MIN_STRENGTH,
      fullRecencyWeight: false,
      exemptFromSweep: false,
      sweepEligible: false,
      ingestionGraceMs: 0,
      retentionCompatibilityKey: "soft",
    });

    register({
      id: DEFAULT_POLICY_IDS.ephemeral,
      aliases: ["ephemeral"],
      halfLifeDays: 7,
      floor: 0,
      minStrength: DEFAULT_MIN_STRENGTH,
      fullRecencyWeight: false,
      exemptFromSweep: false,
      sweepEligible: true,
      ingestionGraceMs: 24 * MS_PER_HOUR,
      retentionCompatibilityKey: "ephemeral",
    });

    register({
      id: DEFAULT_POLICY_IDS.pinned,
      aliases: ["pinned"],
      floor: 1,
      minStrength: 0,
      fullRecencyWeight: true,
      exemptFromSweep: true,
      sweepEligible: false,
      ingestionGraceMs: 0,
      retentionCompatibilityKey: "pinned",
    });

    register({
      id: DEFAULT_POLICY_IDS.evergreen,
      aliases: ["evergreen"],
      floor: 1,
      minStrength: 0,
      fullRecencyWeight: true,
      exemptFromSweep: true,
      sweepEligible: false,
      ingestionGraceMs: 0,
      retentionCompatibilityKey: "evergreen",
    });

    register({
      id: DEFAULT_POLICY_IDS.tier1Summary,
      aliases: ["tier1_summary"],
      halfLifeDays: 90,
      floor: 0.3,
      minStrength: DEFAULT_MIN_STRENGTH,
      fullRecencyWeight: false,
      exemptFromSweep: false,
      sweepEligible: false,
      ingestionGraceMs: 24 * MS_PER_HOUR,
      retentionCompatibilityKey: "tier1_summary",
    });
  }

  for (const policy of initialPolicies) {
    register(policy);
  }

  const resolve = (policyId: string): DecayPolicyDefinition | undefined => {
    if (typeof policyId !== "string" || policyId.trim() === "") {
      return undefined;
    }
    const trimmed = policyId.trim();
    const direct = byId.get(trimmed);
    if (direct) {
      return direct;
    }
    const target = aliases.get(trimmed);
    return target ? byId.get(target) : undefined;
  };

  return {
    register,
    registerAlias,
    resolve,
    resolveOrThrow(policyId: string): DecayPolicyDefinition {
      const found = resolve(policyId);
      if (!found) {
        throw new RangeError(`Unknown decay policy: '${policyId}'`);
      }
      return found;
    },
    has(policyId: string): boolean {
      return resolve(policyId) !== undefined;
    },
    listIds(): string[] {
      return Array.from(byId.keys());
    },
  };
}

export const defaultPolicyRegistry: PolicyRegistry = createPolicyRegistry();
