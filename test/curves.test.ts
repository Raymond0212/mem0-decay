import { describe, expect, it } from "vitest";
import {
  DEFAULT_HALF_LIFE_DAYS,
  DEFAULT_MIN_STRENGTH,
  createDecayCriterion,
  decayCurves,
  evaluateDecay,
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

  it("matches reversed sigmoid anchors within 1e-12", () => {
    const curve = decayCurves.sigmoid({ activeDays: 23, fadeDays: 14 });
    expect(Math.abs(curve(23) - 0.9)).toBeLessThan(1e-12);
    expect(Math.abs(curve(30) - 0.5)).toBeLessThan(1e-12);
    expect(Math.abs(curve(37) - 0.1)).toBeLessThan(1e-12);
    expect(curve(0)).toBeGreaterThan(0.9);
    expect(curve(1000)).toBeGreaterThanOrEqual(0);
    expect(curve(1000)).toBeLessThanOrEqual(1);
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

    expect(() => decayCurves.sigmoid({ activeDays: -1, fadeDays: 10 })).toThrow(RangeError);
    expect(() => decayCurves.sigmoid({ activeDays: 10, fadeDays: 0 })).toThrow(RangeError);
    expect(() => decayCurves.sigmoid({ activeDays: 10, fadeDays: -2 })).toThrow(RangeError);

    expect(() => createDecayCriterion({ minStrength: -0.1 })).toThrow(RangeError);
    expect(() => createDecayCriterion({ minStrength: 1.1 })).toThrow(RangeError);
    expect(() =>
      createDecayCriterion({
        curve: decayCurves.exponential(10),
        halfLifeDays: 20,
      }),
    ).toThrow(RangeError);
  });
});
