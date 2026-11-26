"""Revisioned lifecycle policy registry."""

from dataclasses import dataclass

from .contracts import MS_PER_DAY, finite, nonempty, unit_interval
from .curves import (
    DEFAULT_HALF_LIFE_DAYS,
    DEFAULT_MIN_STRENGTH,
    DecayCriterion,
    DecayCurve,
    constant,
    exponential,
    positive,
    resolve_decay_curve,
)

DEFAULT_POLICY_IDS = {
    name: f"{name}@v1"
    for name in ("soft", "ephemeral", "pinned", "evergreen", "tier1_summary")
}


@dataclass(frozen=True, slots=True)
class DecayPolicy:
    id: str
    floor: float
    curve: DecayCurve
    weighted_curve: DecayCurve
    criterion: DecayCriterion
    full_recency_weight: bool
    exempt_from_sweep: bool
    sweep_eligible: bool
    ingestion_grace_ms: float
    default_ttl_ms: float | None
    enforce_exact_visibility_expiry: bool
    retention_compatibility_key: str | None

    @property
    def min_strength(self):
        return self.criterion.min_strength


def define_policy(
    id,
    *,
    curve=None,
    curve_registry=None,
    half_life_days=None,
    floor=0,
    min_strength=DEFAULT_MIN_STRENGTH,
    full_recency_weight=False,
    exempt_from_sweep=False,
    sweep_eligible=False,
    ingestion_grace_ms=0,
    default_ttl_ms=None,
    enforce_exact_visibility_expiry=False,
    retention_compatibility_key=None,
):
    id = nonempty(id, "policy id")
    unit_interval(floor, "floor")
    if curve is not None and half_life_days is not None:
        raise ValueError("provide either curve or half_life_days")
    if not finite(ingestion_grace_ms) or ingestion_grace_ms < 0:
        raise ValueError("ingestion_grace_ms must be finite and nonnegative")
    if default_ttl_ms is not None:
        positive(default_ttl_ms, "default_ttl_ms")
    if curve is not None:
        raw = resolve_decay_curve(curve, curve_registry)
    elif full_recency_weight:
        raw = constant()
    else:
        raw = exponential(
            DEFAULT_HALF_LIFE_DAYS if half_life_days is None else half_life_days
        )
    weighted = constant() if full_recency_weight else raw.with_floor(floor)
    return DecayPolicy(
        id,
        floor,
        raw,
        weighted,
        raw.to_criterion(min_strength),
        full_recency_weight,
        exempt_from_sweep,
        sweep_eligible and not exempt_from_sweep,
        ingestion_grace_ms,
        default_ttl_ms,
        enforce_exact_visibility_expiry,
        retention_compatibility_key,
    )


class PolicyRegistry:
    def __init__(
        self, initial_policies=(), *, include_defaults=True, curve_registry=None
    ):
        self._policies = {}
        self._aliases = {}
        self._curves = curve_registry
        if include_defaults:
            defaults = {
                "soft": {"half_life_days": 30, "floor": 0.3},
                "ephemeral": {
                    "half_life_days": 7,
                    "sweep_eligible": True,
                    "ingestion_grace_ms": MS_PER_DAY,
                },
                "pinned": {
                    "floor": 1,
                    "min_strength": 0,
                    "full_recency_weight": True,
                    "exempt_from_sweep": True,
                },
                "evergreen": {
                    "floor": 1,
                    "min_strength": 0,
                    "full_recency_weight": True,
                    "exempt_from_sweep": True,
                },
                "tier1_summary": {
                    "half_life_days": 90,
                    "floor": 0.3,
                    "ingestion_grace_ms": MS_PER_DAY,
                },
            }
            for alias, options in defaults.items():
                self.register(
                    DEFAULT_POLICY_IDS[alias],
                    aliases=(alias,),
                    retention_compatibility_key=alias,
                    **options,
                )
        for options in initial_policies:
            self.register(**options)

    def register(self, id, *, aliases=(), **options):
        policy = define_policy(id, curve_registry=self._curves, **options)
        aliases = [nonempty(alias, "policy alias") for alias in aliases]
        self._policies[policy.id] = policy
        for alias in aliases:
            self._aliases[alias] = policy.id
        return policy

    def register_alias(self, alias, target_policy_id):
        alias = nonempty(alias, "policy alias")
        target = nonempty(target_policy_id, "target policy id")
        if target not in self._policies:
            raise ValueError(f"Unknown decay policy: {target!r}")
        self._aliases[alias] = target

    def resolve(self, policy_id):
        if not isinstance(policy_id, str):
            return None
        policy_id = policy_id.strip()
        return self._policies.get(policy_id) or self._policies.get(
            self._aliases.get(policy_id)
        )

    def resolve_or_raise(self, policy_id):
        policy = self.resolve(policy_id)
        if policy is None:
            raise ValueError(f"Unknown decay policy: {policy_id!r}")
        return policy

    def has(self, policy_id):
        return self.resolve(policy_id) is not None

    def list_ids(self):
        return list(self._policies)


default_policy_registry = PolicyRegistry()
