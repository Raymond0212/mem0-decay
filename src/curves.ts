/** Pure function signature returning decay strength in [0, 1] for `ageDays >= 0`. */
export type DecayCurveFn = (ageDays: number) => number;

/**
 * Strategy pattern contract for any decay curve implementation.
 * Custom curve classes or objects only need to implement `kind` and `evaluate(ageDays)`.
 */
export interface DecayCurveStrategy<TParams extends object = object> {
  readonly kind: string;
  readonly params?: Readonly<TParams>;
  evaluate(ageDays: number): number;
}

/**
 * Declarative, JSON-serializable curve specification resolved via a `CurveRegistry`.
 */
export interface DecayCurveSpec {
  kind: string;
  [key: string]: unknown;
}

/**
 * First-class callable decay curve combining:
 * - direct invocation `curve(ageDays)`,
 * - Strategy contract `curve.evaluate(ageDays)` and inspectable `kind` / `params`,
 * - Decorator composition `curve.withFloor(floor)`,
 * - Criterion adapter `curve.toCriterion(minStrength)`.
 */
export interface DecayCurve<TParams extends object = object>
  extends DecayCurveStrategy<TParams> {
  (ageDays: number): number;
  readonly params: Readonly<TParams>;
  /** Decorator: bounds strength into `[floor, 1]` via `floor + (1 - floor) * strength`. */
  withFloor(floor: number): DecayCurve<FlooredCurveParams>;
  /** Adapter: binds this curve to a `minStrength` threshold to form a `DecayCriterion`. */
  toCriterion(minStrength?: number): DecayCriterion;
}

/**
 * Polymorphic curve input accepted everywhere across policies, search queries,
 * store/router defaults, and criteria.
 */
export type DecayCurveInput =
  | DecayCurve
  | DecayCurveStrategy
  | DecayCurveFn
  | DecayCriterion
  | DecayCurveSpec;

export interface ExponentialCurveOptions {
  halfLifeDays?: number;
}

export interface ExponentialCurveParams {
  halfLifeDays: number;
}

export interface SigmoidCurveOptions {
  /** Lower asymptote in [0, 1] approached as ageDays -> +infinity. Defaults to 0. */
  lower?: number;
  /** Upper asymptote in [0, 1] approached as ageDays -> -infinity. Defaults to 1. */
  upper?: number;
  /** Shift / inflection age in days (finite number). Defaults to 30 (`DEFAULT_SIGMOID_MIDPOINT_DAYS`). */
  midpoint?: number;
  /** Decay steepness > 0 (per day). Defaults to 0.2 (`DEFAULT_SIGMOID_SLOPE`). */
  slope?: number;
  /** Asymmetry exponent > 0 (`1` yields symmetric 4-parameter logistic). Defaults to 1. */
  asymmetry?: number;
}

export interface SigmoidCurveParams {
  lower: number;
  upper: number;
  midpoint: number;
  slope: number;
  asymmetry: number;
}

export interface ConstantCurveOptions {
  strength?: number;
}

export interface ConstantCurveParams {
  strength: number;
}

export interface FlooredCurveParams {
  baseKind: string;
  floor: number;
}

export interface DecayCriterionOptions {
  curve?: DecayCurveInput;
  halfLifeDays?: number;
  minStrength?: number;
  curveRegistry?: CurveRegistry;
}

export interface DecayEvaluation {
  strength: number;
  belowThreshold: boolean;
}

export interface DecayCriterion {
  (ageDays: number): DecayEvaluation;
  readonly curve: DecayCurve;
  readonly minStrength: number;
  evaluate(ageDays: number): DecayEvaluation;
}

export type CurveFactory<TOptions = Record<string, unknown>> = (
  options?: TOptions,
) => DecayCurve | DecayCurveStrategy | DecayCurveFn;

export interface CurveRegistry {
  register<TOptions = Record<string, unknown>>(
    kind: string,
    factory: CurveFactory<TOptions>,
  ): this;
  create(spec: DecayCurveSpec): DecayCurve;
  resolve(input: DecayCurveInput): DecayCurve;
  has(kind: string): boolean;
  listKinds(): string[];
}

export const DEFAULT_HALF_LIFE_DAYS = 30;
export const DEFAULT_MIN_STRENGTH = 0.05;
export const DEFAULT_SIGMOID_LOWER = 0;
export const DEFAULT_SIGMOID_UPPER = 1;
export const DEFAULT_SIGMOID_MIDPOINT_DAYS = DEFAULT_HALF_LIFE_DAYS;
export const DEFAULT_SIGMOID_SLOPE = 0.2;
export const DEFAULT_SIGMOID_ASYMMETRY = 1;

const DECAY_CURVE_BRAND = Symbol.for("mem0-decay.curve");
const DECAY_CRITERION_BRAND = Symbol.for("mem0-decay.criterion");

export function assertValidAgeDays(ageDays: number): void {
  if (typeof ageDays !== "number" || !Number.isFinite(ageDays) || ageDays < 0) {
    throw new RangeError(`ageDays must be a finite number >= 0, received: ${String(ageDays)}`);
  }
}

export function assertValidStrength(strength: number, source = "curve"): number {
  if (typeof strength !== "number" || !Number.isFinite(strength) || strength < 0 || strength > 1) {
    throw new RangeError(
      `${source} must return a finite strength in [0, 1], received: ${String(strength)}`,
    );
  }
  return strength;
}

function assertFinite(value: number, name: string): void {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new RangeError(`${name} must be a finite number, received: ${String(value)}`);
  }
}

function assertPositiveFinite(value: number, name: string): void {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    throw new RangeError(`${name} must be a finite number > 0, received: ${String(value)}`);
  }
}

function assertUnitInterval(value: number, name: string): void {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0 || value > 1) {
    throw new RangeError(`${name} must be a finite number in [0, 1], received: ${String(value)}`);
  }
}

export function isDecayCurve(value: unknown): value is DecayCurve {
  return (
    typeof value === "function" &&
    Boolean((value as unknown as Record<symbol, unknown>)[DECAY_CURVE_BRAND])
  );
}

export function isDecayCriterion(value: unknown): value is DecayCriterion {
  return (
    typeof value === "function" &&
    Boolean((value as unknown as Record<symbol, unknown>)[DECAY_CRITERION_BRAND])
  );
}

/**
 * Adapter turning any `DecayCurveStrategy` into a first-class callable `DecayCurve`
 * with Decorator (`withFloor`) and Criterion (`toCriterion`) methods attached.
 */
export function createCallableCurve<TParams extends object>(
  strategy: DecayCurveStrategy<TParams>,
  options: { validateOutput?: boolean } = {},
): DecayCurve<TParams> {
  if (!strategy || typeof strategy !== "object" || typeof strategy.evaluate !== "function") {
    throw new TypeError("strategy must be an object implementing evaluate(ageDays)");
  }
  if (typeof strategy.kind !== "string" || strategy.kind.trim() === "") {
    throw new RangeError("strategy.kind must be a non-empty string");
  }

  const kind = strategy.kind.trim();
  const frozenParams = Object.freeze({ ...(strategy.params ?? ({} as TParams)) }) as Readonly<TParams>;
  const shouldValidateOutput = options.validateOutput !== false;

  const evaluate = (ageDays: number): number => {
    assertValidAgeDays(ageDays);
    const raw = strategy.evaluate(ageDays);
    return shouldValidateOutput ? assertValidStrength(raw, "decay curve") : raw;
  };

  const fn = ((ageDays: number): number => evaluate(ageDays)) as DecayCurve<TParams>;

  Object.defineProperties(fn, {
    [DECAY_CURVE_BRAND]: { value: true, enumerable: false },
    kind: { value: kind, enumerable: true, writable: false },
    params: { value: frozenParams, enumerable: true, writable: false },
    evaluate: { value: evaluate, enumerable: true, writable: false },
    withFloor: {
      value: (floor: number): DecayCurve<FlooredCurveParams> => createFlooredCurve(fn, floor),
      enumerable: false,
      writable: false,
    },
    toCriterion: {
      value: (minStrength?: number): DecayCriterion =>
        createDecayCriterion({ curve: fn, minStrength }),
      enumerable: false,
      writable: false,
    },
  });

  return fn;
}

/** Decorator implementation bounding a base curve's strength into `[floor, 1]`. */
function createFlooredCurve(base: DecayCurve, floor: number): DecayCurve<FlooredCurveParams> {
  assertUnitInterval(floor, "floor");
  const params: FlooredCurveParams = { baseKind: base.kind, floor };
  return createCallableCurve(
    {
      kind: "floored",
      params,
      evaluate(ageDays: number): number {
        const strength = base.evaluate(ageDays);
        if (floor === 0) {
          return strength;
        }
        if (floor === 1) {
          return 1;
        }
        return Math.min(1, Math.max(0, floor + (1 - floor) * strength));
      },
    },
    { validateOutput: false },
  );
}

/**
 * Primary Factory facade for constructing reusable `DecayCurve` instances.
 */
export const decayCurves = {
  /** Exponential half-life decay: `0.5 ^ (ageDays / halfLifeDays)`. */
  exponential(options?: number | ExponentialCurveOptions): DecayCurve<ExponentialCurveParams> {
    if (
      options !== undefined &&
      typeof options !== "number" &&
      (typeof options !== "object" || options === null)
    ) {
      throw new TypeError("exponential options must be a number or options object");
    }
    const halfLifeDays =
      typeof options === "number"
        ? options
        : (options?.halfLifeDays ?? DEFAULT_HALF_LIFE_DAYS);
    assertPositiveFinite(halfLifeDays, "halfLifeDays");

    return createCallableCurve(
      {
        kind: "exponential",
        params: { halfLifeDays },
        evaluate(ageDays: number): number {
          if (ageDays === 0) {
            return 1;
          }
          return Math.pow(0.5, ageDays / halfLifeDays);
        },
      },
      { validateOutput: false },
    );
  },

  /**
   * 5-parameter generalized logistic (Richards) sigmoid:
   * `lower + (upper - lower) / (1 + exp(slope * (ageDays - midpoint))) ^ asymmetry`.
   */
  sigmoid(options?: SigmoidCurveOptions): DecayCurve<SigmoidCurveParams> {
    if (options !== undefined && (typeof options !== "object" || options === null)) {
      throw new TypeError("sigmoid options must be an object when provided");
    }

    const lower = options?.lower ?? DEFAULT_SIGMOID_LOWER;
    const upper = options?.upper ?? DEFAULT_SIGMOID_UPPER;
    const midpoint = options?.midpoint ?? DEFAULT_SIGMOID_MIDPOINT_DAYS;
    const slope = options?.slope ?? DEFAULT_SIGMOID_SLOPE;
    const asymmetry = options?.asymmetry ?? DEFAULT_SIGMOID_ASYMMETRY;

    assertUnitInterval(lower, "lower");
    assertUnitInterval(upper, "upper");
    if (lower > upper) {
      throw new RangeError(`lower (${lower}) must be <= upper (${upper})`);
    }
    assertFinite(midpoint, "midpoint");
    assertPositiveFinite(slope, "slope");
    assertPositiveFinite(asymmetry, "asymmetry");

    const span = upper - lower;

    return createCallableCurve(
      {
        kind: "sigmoid",
        params: { lower, upper, midpoint, slope, asymmetry },
        evaluate(ageDays: number): number {
          if (span === 0) {
            return lower;
          }
          const z = slope * (ageDays - midpoint);
          const logDenom = z > 40 ? z : Math.log1p(Math.exp(z));
          const factor = z === 0 ? Math.pow(0.5, asymmetry) : Math.exp(-asymmetry * logDenom);
          const strength = lower + span * factor;
          return Math.min(1, Math.max(0, strength));
        },
      },
      { validateOutput: false },
    );
  },

  /** Constant (age-invariant) strength in `[0, 1]` (e.g. `1` for pinned/evergreen). */
  constant(options?: number | ConstantCurveOptions): DecayCurve<ConstantCurveParams> {
    if (
      options !== undefined &&
      typeof options !== "number" &&
      (typeof options !== "object" || options === null)
    ) {
      throw new TypeError("constant options must be a number or options object");
    }
    const strength = typeof options === "number" ? options : (options?.strength ?? 1);
    assertUnitInterval(strength, "strength");

    return createCallableCurve(
      {
        kind: "constant",
        params: { strength },
        evaluate(): number {
          return strength;
        },
      },
      { validateOutput: false },
    );
  },

  /** Adapter wrapping any user-supplied `(ageDays) => number` function with runtime validation. */
  custom(
    fn: DecayCurveFn,
    options: { kind?: string; params?: Record<string, unknown> } = {},
  ): DecayCurve<Record<string, unknown>> {
    if (typeof fn !== "function") {
      throw new TypeError("custom curve must be a function");
    }
    const kind = options.kind?.trim() || "custom";
    return createCallableCurve(
      {
        kind,
        params: options.params ?? {},
        evaluate: fn,
      },
      { validateOutput: true },
    );
  },

  /** Universal adapter normalizing any `DecayCurveInput` into a reusable `DecayCurve`. */
  from(input: DecayCurveInput, registry: CurveRegistry = defaultCurveRegistry): DecayCurve {
    return resolveDecayCurve(input, registry);
  },
} as const;

/**
 * Factory + Open Registry pattern for registering custom curve kinds and resolving
 * any `DecayCurveInput` (instance, strategy object, function, criterion, or `{ kind }` spec)
 * into a unified callable `DecayCurve`.
 */
export function createCurveRegistry(options: { includeDefaults?: boolean } = {}): CurveRegistry {
  const factories = new Map<string, CurveFactory<unknown>>();

  const registry: CurveRegistry = {
    register<TOptions = Record<string, unknown>>(
      kind: string,
      factory: CurveFactory<TOptions>,
    ): CurveRegistry {
      if (typeof kind !== "string" || kind.trim() === "") {
        throw new RangeError("Curve kind must be a non-empty string");
      }
      if (typeof factory !== "function") {
        throw new TypeError(`Factory for curve kind '${kind}' must be a function`);
      }
      factories.set(kind.trim(), factory as CurveFactory<unknown>);
      return registry;
    },

    create(spec: DecayCurveSpec): DecayCurve {
      if (!spec || typeof spec !== "object" || Array.isArray(spec)) {
        throw new TypeError("Curve spec must be an object with a non-empty 'kind' string");
      }
      if (typeof spec.kind !== "string" || spec.kind.trim() === "") {
        throw new RangeError("Curve spec.kind must be a non-empty string");
      }
      const kind = spec.kind.trim();
      const factory = factories.get(kind);
      if (!factory) {
        throw new RangeError(`Unknown decay curve kind: '${kind}'`);
      }
      const { kind: _ignored, ...rest } = spec;
      const built = factory(rest);
      if (isDecayCurve(built)) {
        return built;
      }
      if (typeof built === "function") {
        return decayCurves.custom(built, { kind, params: rest });
      }
      if (built && typeof built === "object" && typeof built.evaluate === "function") {
        return createCallableCurve(built);
      }
      throw new TypeError(
        `Curve factory '${kind}' must return a DecayCurve, DecayCurveStrategy, or function`,
      );
    },

    resolve(input: DecayCurveInput): DecayCurve {
      if (isDecayCurve(input)) {
        return input;
      }
      if (isDecayCriterion(input)) {
        return input.curve;
      }
      if (typeof input === "function") {
        return decayCurves.custom(input);
      }
      if (input && typeof input === "object" && !Array.isArray(input)) {
        if (typeof (input as DecayCurveStrategy).evaluate === "function") {
          const strategy = input as DecayCurveStrategy;
          const kind =
            typeof strategy.kind === "string" && strategy.kind.trim() !== ""
              ? strategy.kind.trim()
              : "custom";
          return createCallableCurve({
            kind,
            params: strategy.params ?? {},
            evaluate: (ageDays: number) => strategy.evaluate(ageDays),
          });
        }
        if (typeof (input as DecayCurveSpec).kind === "string") {
          return registry.create(input as DecayCurveSpec);
        }
      }
      throw new TypeError(
        "Invalid decay curve: expected a DecayCurve, DecayCurveStrategy, (ageDays) => number function, or { kind } spec",
      );
    },

    has(kind: string): boolean {
      if (typeof kind !== "string" || kind.trim() === "") {
        return false;
      }
      return factories.has(kind.trim());
    },

    listKinds(): string[] {
      return Array.from(factories.keys());
    },
  };

  if (options.includeDefaults !== false) {
    registry.register<ExponentialCurveOptions>("exponential", (opts) =>
      decayCurves.exponential(opts),
    );
    registry.register<SigmoidCurveOptions>("sigmoid", (opts) => decayCurves.sigmoid(opts));
    registry.register<ConstantCurveOptions>("constant", (opts) => decayCurves.constant(opts));
  }

  return registry;
}

export const defaultCurveRegistry: CurveRegistry = createCurveRegistry();

/**
 * Normalizes any `DecayCurveInput` into a reusable callable `DecayCurve`.
 * Idempotent when given an existing `DecayCurve`.
 */
export function resolveDecayCurve(
  input: DecayCurveInput,
  registry: CurveRegistry = defaultCurveRegistry,
): DecayCurve {
  return registry.resolve(input);
}

export function createDecayCriterion(options?: DecayCriterionOptions): DecayCriterion {
  if (options !== undefined && (typeof options !== "object" || options === null)) {
    throw new TypeError("options must be an object when provided");
  }

  if (options?.curve !== undefined && options?.halfLifeDays !== undefined) {
    throw new RangeError("Provide either curve or halfLifeDays, not both");
  }

  const minStrength = options?.minStrength ?? DEFAULT_MIN_STRENGTH;
  assertUnitInterval(minStrength, "minStrength");

  const curve: DecayCurve =
    options?.curve !== undefined
      ? resolveDecayCurve(options.curve, options.curveRegistry)
      : decayCurves.exponential({
          halfLifeDays: options?.halfLifeDays ?? DEFAULT_HALF_LIFE_DAYS,
        });

  const evaluate = (ageDays: number): DecayEvaluation => {
    const strength = curve.evaluate(ageDays);
    return {
      strength,
      belowThreshold: strength < minStrength,
    };
  };

  const criterion = ((ageDays: number): DecayEvaluation => evaluate(ageDays)) as DecayCriterion;

  Object.defineProperties(criterion, {
    [DECAY_CRITERION_BRAND]: { value: true, enumerable: false },
    curve: { value: curve, enumerable: true, writable: false },
    minStrength: { value: minStrength, enumerable: true, writable: false },
    evaluate: { value: evaluate, enumerable: true, writable: false },
  });

  return criterion;
}

export const evaluateDecay: DecayCriterion = createDecayCriterion();
