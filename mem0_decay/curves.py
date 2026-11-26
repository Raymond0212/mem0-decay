"""Composable decay curves, ranking floors, and raw-strength criteria."""

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from .contracts import finite, nonempty, unit_interval

DEFAULT_HALF_LIFE_DAYS = 30
DEFAULT_MIN_STRENGTH = 0.05


def positive(value, name):
    if not finite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


@dataclass(frozen=True, slots=True)
class DecayCurve:
    kind: str
    evaluator: Callable[[float], float] = field(repr=False)
    params: Mapping = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "kind", nonempty(self.kind, "curve kind"))
        if not callable(self.evaluator):
            raise TypeError("curve evaluator must be callable")
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))

    def evaluate(self, age_days):
        if not finite(age_days) or age_days < 0:
            raise ValueError("age_days must be finite and nonnegative")
        return unit_interval(self.evaluator(age_days), "curve strength")

    __call__ = evaluate

    def with_floor(self, floor):
        unit_interval(floor, "floor")
        return DecayCurve(
            f"floored:{self.kind}",
            lambda age: floor + (1 - floor) * self(age),
            {"floor": floor, "base": self},
        )

    def to_criterion(self, min_strength=DEFAULT_MIN_STRENGTH):
        return DecayCriterion(self, min_strength)


@dataclass(frozen=True, slots=True)
class DecayEvaluation:
    strength: float
    below_threshold: bool


@dataclass(frozen=True, slots=True)
class DecayCriterion:
    curve: DecayCurve
    min_strength: float = DEFAULT_MIN_STRENGTH

    def __post_init__(self):
        unit_interval(self.min_strength, "min_strength")

    def evaluate(self, age_days):
        strength = self.curve(age_days)
        return DecayEvaluation(strength, strength < self.min_strength)

    __call__ = evaluate


def exponential(half_life_days=DEFAULT_HALF_LIFE_DAYS):
    positive(half_life_days, "half_life_days")
    return DecayCurve(
        "exponential",
        lambda age: 0.5 ** (age / half_life_days),
        {"half_life_days": half_life_days},
    )


def sigmoid(*, lower=0, upper=1, midpoint=30, slope=0.2, asymmetry=1):
    unit_interval(lower, "lower")
    unit_interval(upper, "upper")
    if lower > upper or not finite(midpoint):
        raise ValueError("sigmoid requires lower <= upper and a finite midpoint")
    positive(slope, "slope")
    positive(asymmetry, "asymmetry")

    def strength(age):
        z = slope * (age - midpoint)
        log_denom = z if z > 40 else math.log1p(math.exp(z))
        return lower + (upper - lower) * math.exp(-asymmetry * log_denom)

    return DecayCurve(
        "sigmoid",
        strength,
        dict(
            lower=lower,
            upper=upper,
            midpoint=midpoint,
            slope=slope,
            asymmetry=asymmetry,
        ),
    )


def constant(strength=1):
    unit_interval(strength, "strength")
    return DecayCurve("constant", lambda _: strength, {"strength": strength})


def custom(evaluator, *, kind="custom", params=None):
    return DecayCurve(kind, evaluator, {} if params is None else params)


class CurveRegistry:
    def __init__(self, *, include_defaults=True):
        self._factories = {}
        if include_defaults:
            self.register("exponential", lambda opts: exponential(**opts))
            self.register("sigmoid", lambda opts: sigmoid(**opts))
            self.register("constant", lambda opts: constant(**opts))

    def register(self, kind, factory):
        if not callable(factory):
            raise TypeError("curve factory must be callable")
        self._factories[nonempty(kind, "curve kind")] = factory
        return self

    def create(self, spec):
        params = dict(spec)
        kind = nonempty(params.pop("kind", None), "curve kind")
        if kind not in self._factories:
            raise ValueError(f"Unknown decay curve kind: {kind!r}")
        curve = self._factories[kind](params)
        if isinstance(curve, DecayCurve):
            return curve
        evaluator = getattr(curve, "evaluate", curve)
        return custom(
            evaluator,
            kind=getattr(curve, "kind", kind),
            params=getattr(curve, "params", params),
        )

    def resolve(self, value):
        if isinstance(value, DecayCurve):
            return value
        if isinstance(value, DecayCriterion):
            return value.curve
        if isinstance(value, Mapping):
            return self.create(value)
        if callable(value):
            return custom(value)
        if callable(getattr(value, "evaluate", None)):
            return custom(
                value.evaluate,
                kind=getattr(value, "kind", "custom"),
                params=getattr(value, "params", {}),
            )
        raise TypeError("expected a decay curve, callable, strategy, or curve spec")

    def has(self, kind):
        return isinstance(kind, str) and kind.strip() in self._factories

    def list_kinds(self):
        return list(self._factories)


default_curve_registry = CurveRegistry()


def resolve_decay_curve(value, registry=None):
    return (registry if registry is not None else default_curve_registry).resolve(value)


def create_decay_criterion(
    *,
    curve=None,
    half_life_days=None,
    min_strength=DEFAULT_MIN_STRENGTH,
    curve_registry=None,
):
    if curve is not None and half_life_days is not None:
        raise ValueError("provide either curve or half_life_days")
    resolved = (
        resolve_decay_curve(curve, curve_registry)
        if curve is not None
        else exponential(
            DEFAULT_HALF_LIFE_DAYS if half_life_days is None else half_life_days
        )
    )
    return resolved.to_criterion(min_strength)


evaluate_decay = create_decay_criterion()
