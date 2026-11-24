export type DecayCurve = (ageDays: number) => number;

export interface ExponentialCurveOptions {
  halfLifeDays?: number;
}

export interface SigmoidCurveOptions {
  activeDays: number;
  fadeDays: number;
}

export interface DecayCriterionOptions {
  curve?: DecayCurve;
  halfLifeDays?: number;
  minStrength?: number;
}

export interface DecayEvaluation {
  strength: number;
  belowThreshold: boolean;
}

export type DecayCriterion = (ageDays: number) => DecayEvaluation;

export const DEFAULT_HALF_LIFE_DAYS = 30;
export const DEFAULT_MIN_STRENGTH = 0.05;

const LN_9 = Math.log(9);

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

function assertPositiveFinite(value: number, name: string): void {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    throw new RangeError(`${name} must be a finite number > 0, received: ${String(value)}`);
  }
}

function assertNonNegativeFinite(value: number, name: string): void {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    throw new RangeError(`${name} must be a finite number >= 0, received: ${String(value)}`);
  }
}

function assertUnitInterval(value: number, name: string): void {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0 || value > 1) {
    throw new RangeError(`${name} must be a finite number in [0, 1], received: ${String(value)}`);
  }
}

export const decayCurves = {
  exponential(options?: number | ExponentialCurveOptions): DecayCurve {
    const halfLifeDays =
      typeof options === "number"
        ? options
        : (options?.halfLifeDays ?? DEFAULT_HALF_LIFE_DAYS);
    assertPositiveFinite(halfLifeDays, "halfLifeDays");

    return (ageDays: number): number => {
      assertValidAgeDays(ageDays);
      if (ageDays === 0) {
        return 1;
      }
      return Math.pow(0.5, ageDays / halfLifeDays);
    };
  },

  sigmoid(options: SigmoidCurveOptions): DecayCurve {
    if (!options || typeof options !== "object") {
      throw new TypeError("sigmoid options must be an object with activeDays and fadeDays");
    }
    const { activeDays, fadeDays } = options;
    assertNonNegativeFinite(activeDays, "activeDays");
    assertPositiveFinite(fadeDays, "fadeDays");

    const midpoint = activeDays + fadeDays / 2;
    const slope = (2 * LN_9) / fadeDays;

    return (ageDays: number): number => {
      assertValidAgeDays(ageDays);
      const exponent = slope * (ageDays - midpoint);
      if (exponent > 700) {
        return 0;
      }
      if (exponent < -700) {
        return 1;
      }
      return 1 / (1 + Math.exp(exponent));
    };
  },
} as const;

export function createDecayCriterion(options?: DecayCriterionOptions): DecayCriterion {
  if (options !== undefined && (typeof options !== "object" || options === null)) {
    throw new TypeError("options must be an object when provided");
  }

  if (options?.curve !== undefined && options?.halfLifeDays !== undefined) {
    throw new RangeError("Provide either curve or halfLifeDays, not both");
  }

  const minStrength = options?.minStrength ?? DEFAULT_MIN_STRENGTH;
  assertUnitInterval(minStrength, "minStrength");

  let curve: DecayCurve;
  if (options?.curve !== undefined) {
    if (typeof options.curve !== "function") {
      throw new TypeError("curve must be a function");
    }
    curve = options.curve;
  } else {
    curve = decayCurves.exponential({
      halfLifeDays: options?.halfLifeDays ?? DEFAULT_HALF_LIFE_DAYS,
    });
  }

  return (ageDays: number): DecayEvaluation => {
    assertValidAgeDays(ageDays);
    const rawStrength = curve(ageDays);
    const strength = assertValidStrength(rawStrength, "decay curve");
    return {
      strength,
      belowThreshold: strength < minStrength,
    };
  };
}

export const evaluateDecay: DecayCriterion = createDecayCriterion();
