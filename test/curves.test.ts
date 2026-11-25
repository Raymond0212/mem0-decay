import { describe, expect, it } from "vitest";
import {
  DEFAULT_HALF_LIFE_DAYS,
  DEFAULT_MIN_STRENGTH,
  DEFAULT_SIGMOID_ASYMMETRY,
  DEFAULT_SIGMOID_LOWER,
  DEFAULT_SIGMOID_MIDPOINT_DAYS,
  DEFAULT_SIGMOID_SLOPE,
  DEFAULT_SIGMOID_UPPER,
  createCurveRegistry,
  createDecayCriterion,
  decayCurves,
  evaluateDecay,
  isDecayCriterion,
  isDecayCurve,
  resolveDecayCurve,
  type DecayCurveStrategy,
} from "../src/curves.js";

describe("curves", () => {
  it("evaluates default exponential decay at 30 days as 0.5", () => {
    expect(DEFAULT_HALF_LIFE_DAYS).toBe(30);
    expect(DEFAULT_MIN_STRENGTH).toBe(0.05);

    const atZero = evaluateDecay(0);
    expect(atZero).toEqual({ strength: 1, belowThreshold: false });

    const atThirty = evaluateDecay(30);
    expect(atThirty.strength).toBeCloseTo(0.5, 12);
    expect(atThirty.belowThreshold).toBe(false);

    const farFuture = evaluateDecay(300);
    expect(farFuture.strength).toBeLessThan(0.05);
    expect(farFuture.belowThreshold).toBe(true);
  });

  it("uses strict < comparison at the threshold boundary", () => {
    const criterion = createDecayCriterion({ halfLifeDays: 30, minStrength: 0.5 });
    const atBoundary = criterion(30);
    expect(atBoundary.strength).toBe(0.5);
    expect(atBoundary.belowThreshold).toBe(false);

    const justPast = criterion(30.0001);
    expect(justPast.strength).toBeLessThan(0.5);
    expect(justPast.belowThreshold).toBe(true);
  });

  it("evaluates 5-parameter generalized logistic (Richards) sigmoid accurately across all parameters and extremes", () => {
    expect(DEFAULT_SIGMOID_LOWER).toBe(0);
    expect(DEFAULT_SIGMOID_UPPER).toBe(1);
    expect(DEFAULT_SIGMOID_MIDPOINT_DAYS).toBe(30);
    expect(DEFAULT_SIGMOID_SLOPE).toBe(0.2);
    expect(DEFAULT_SIGMOID_ASYMMETRY).toBe(1);

    // Defaults: lower=0, upper=1, midpoint=30, slope=0.2, asymmetry=1
    const defaultSig = decayCurves.sigmoid();
    expect(defaultSig(30)).toBe(0.5);
    expect(defaultSig(0)).toBeGreaterThan(0.99);
    expect(defaultSig(1000)).toBeCloseTo(0, 12);

    // Symmetric anchors (asymmetry=1, lower=0, upper=1) within 1e-12
    const symmetric = decayCurves.sigmoid({ midpoint: 30, slope: Math.log(9) / 7 });
    expect(Math.abs(symmetric(23) - 0.9)).toBeLessThan(1e-12);
    expect(Math.abs(symmetric(30) - 0.5)).toBeLessThan(1e-12);
    expect(Math.abs(symmetric(37) - 0.1)).toBeLessThan(1e-12);

    // Full 5-parameter Richards curve: lower=0.1, upper=0.9, midpoint=30, slope=0.25, asymmetry=2
    const fiveParam = decayCurves.sigmoid({
      lower: 0.1,
      upper: 0.9,
      midpoint: 30,
      slope: 0.25,
      asymmetry: 2,
    });
    // At midpoint: 0.1 + (0.9 - 0.1) / (1 + 1)^2 = 0.1 + 0.8 * 0.25 = 0.3
    expect(fiveParam(30)).toBeCloseTo(0.3, 12);
    expect(fiveParam(0)).toBeGreaterThan(0.89);
    expect(fiveParam(0)).toBeLessThanOrEqual(0.9);
    expect(fiveParam(1000)).toBeCloseTo(0.1, 12);
    expect(fiveParam(10)).toBeGreaterThan(fiveParam(30));
    expect(fiveParam(30)).toBeGreaterThan(fiveParam(50));

    // Degenerate flat band lower === upper
    const flat = decayCurves.sigmoid({ lower: 0.4, upper: 0.4 });
    expect(flat(0)).toBe(0.4);
    expect(flat(100)).toBe(0.4);

    // Log-space stability when slope * (ageDays - midpoint) > 700 with small asymmetry
    const extremeZ = decayCurves.sigmoid({
      lower: 0,
      upper: 1,
      midpoint: 0,
      slope: 1000,
      asymmetry: 0.001,
    });
    // z = 10000 -> (1 + exp(10000))^(-0.001) ≈ exp(-10)
    expect(extremeZ(10)).toBeCloseTo(Math.exp(-10), 12);
  });

  it("supports custom curves and validates custom outputs", () => {
    const linear = createDecayCriterion({
      curve: (ageDays) => Math.max(0, (10 - ageDays) / 10),
      minStrength: 0.2,
    });
    expect(linear(5)).toEqual({ strength: 0.5, belowThreshold: false });
    expect(linear(8)).toEqual({ strength: 0.2, belowThreshold: false });
    expect(linear(9)).toEqual({ strength: 0.1, belowThreshold: true });

    const outOfRangeHigh = createDecayCriterion({ curve: () => 1.01 });
    expect(() => outOfRangeHigh(1)).toThrow(RangeError);

    const outOfRangeLow = createDecayCriterion({ curve: () => -0.01 });
    expect(() => outOfRangeLow(1)).toThrow(RangeError);

    const nonFinite = createDecayCriterion({ curve: () => Number.NaN });
    expect(() => nonFinite(1)).toThrow(RangeError);
  });

  it("rejects invalid ages and invalid configuration", () => {
    expect(() => evaluateDecay(-1)).toThrow(RangeError);
    expect(() => evaluateDecay(Number.NaN)).toThrow(RangeError);
    expect(() => evaluateDecay(Number.POSITIVE_INFINITY)).toThrow(RangeError);

    expect(() => decayCurves.exponential({ halfLifeDays: 0 })).toThrow(RangeError);
    expect(() => decayCurves.exponential({ halfLifeDays: -5 })).toThrow(RangeError);
    expect(() => decayCurves.exponential({ halfLifeDays: Number.NaN })).toThrow(RangeError);

    expect(() => decayCurves.sigmoid({ lower: -0.1 })).toThrow(RangeError);
    expect(() => decayCurves.sigmoid({ upper: 1.1 })).toThrow(RangeError);
    expect(() => decayCurves.sigmoid({ lower: 0.8, upper: 0.2 })).toThrow(RangeError);
    expect(() => decayCurves.sigmoid({ midpoint: Number.NaN })).toThrow(RangeError);
    expect(() => decayCurves.sigmoid({ midpoint: Number.POSITIVE_INFINITY })).toThrow(RangeError);
    expect(() => decayCurves.sigmoid({ slope: 0 })).toThrow(RangeError);
    expect(() => decayCurves.sigmoid({ slope: -0.2 })).toThrow(RangeError);
    expect(() => decayCurves.sigmoid({ asymmetry: 0 })).toThrow(RangeError);
    expect(() => decayCurves.sigmoid({ asymmetry: -1 })).toThrow(RangeError);

    expect(() => createDecayCriterion({ minStrength: -0.1 })).toThrow(RangeError);
    expect(() => createDecayCriterion({ minStrength: 1.1 })).toThrow(RangeError);
    expect(() =>
      createDecayCriterion({
        curve: decayCurves.exponential(10),
        halfLifeDays: 20,
      }),
    ).toThrow(RangeError);
  });

  it("implements Strategy, Factory, Decorator (withFloor), Adapter (resolveDecayCurve / toCriterion), and CurveRegistry patterns", () => {
    // 1. Factory creates callable Strategy objects with inspectable kind & frozen params
    const sigCurve = decayCurves.sigmoid({
      lower: 0.1,
      upper: 0.9,
      midpoint: 20,
      slope: 0.3,
      asymmetry: 2,
    });
    expect(isDecayCurve(sigCurve)).toBe(true);
    expect(sigCurve.kind).toBe("sigmoid");
    expect(sigCurve.params).toEqual({
      lower: 0.1,
      upper: 0.9,
      midpoint: 20,
      slope: 0.3,
      asymmetry: 2,
    });
    expect(sigCurve(20)).toBeCloseTo(sigCurve.evaluate(20), 12);
    expect(sigCurve.evaluate(20)).toBeCloseTo(0.3, 12);
    expect(resolveDecayCurve(sigCurve)).toBe(sigCurve); // idempotent

    // 2. Decorator: withFloor + Adapter: toCriterion
    const floored = sigCurve.withFloor(0.5);
    expect(floored.kind).toBe("floored");
    expect(floored.params).toEqual({ baseKind: "sigmoid", floor: 0.5 });
    expect(floored(20)).toBeCloseTo(0.5 + 0.5 * 0.3, 12);

    const crit = sigCurve.toCriterion(0.35);
    expect(isDecayCriterion(crit)).toBe(true);
    expect(crit.curve).toBe(sigCurve);
    expect(crit(20).strength).toBeCloseTo(0.3, 12);
    expect(crit(20).belowThreshold).toBe(true);
    expect(resolveDecayCurve(crit)).toBe(sigCurve);

    // 3. User-defined OOP Strategy class + Custom CurveRegistry
    class WeibullDecayStrategy implements DecayCurveStrategy<{ scaleDays: number; shape: number }> {
      readonly kind = "weibull";
      readonly params: Readonly<{ scaleDays: number; shape: number }>;
      constructor(private readonly scaleDays: number, private readonly shape: number) {
        this.params = Object.freeze({ scaleDays, shape });
      }
      evaluate(ageDays: number): number {
        return Math.exp(-Math.pow(ageDays / this.scaleDays, this.shape));
      }
    }

    const adaptedStrategy = resolveDecayCurve(new WeibullDecayStrategy(20, 2));
    expect(isDecayCurve(adaptedStrategy)).toBe(true);
    expect(adaptedStrategy.kind).toBe("weibull");
    expect(adaptedStrategy(20)).toBeCloseTo(Math.exp(-1), 12);

    const registry = createCurveRegistry();
    expect(registry.listKinds()).toEqual(["exponential", "sigmoid", "constant"]);
    registry.register<{ scaleDays?: number; shape?: number }>(
      "weibull",
      (opts) => new WeibullDecayStrategy(opts?.scaleDays ?? 30, opts?.shape ?? 1.5),
    );
    expect(registry.has("weibull")).toBe(true);

    const fromSpec = registry.create({ kind: "weibull", scaleDays: 20, shape: 2 });
    expect(fromSpec.kind).toBe("weibull");
    expect(fromSpec(20)).toBeCloseTo(Math.exp(-1), 12);

    expect(decayCurves.exponential(10).evaluate(10)).toBeCloseTo(0.5, 12);
    expect(decayCurves.constant(0.75).evaluate(999)).toBe(0.75);
    expect(() => registry.create({ kind: "unknown" })).toThrow(RangeError);
  });
});
