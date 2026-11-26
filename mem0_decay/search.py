"""Mem0 candidate retrieval and time-aware ranking."""

from collections.abc import Mapping

from .access_store import (
    AccessKey,
    compute_effective_age_days,
    read_access_batch_safe,
    serialize_access_key,
)
from .contracts import (
    IDENTITY_FIELDS,
    MS_PER_DAY,
    DecaySearchItem,
    DecaySearchResult,
    TimeSpan,
    call,
    finite,
    nonempty,
    positive_int,
    unit_interval,
)
from .curves import exponential, resolve_decay_curve
from .metadata import (
    EXPIRES_AT,
    INGESTED_AT,
    POLICY,
    TENANT,
    TIER,
    TIME_END,
    TIME_START,
    assert_no_reserved_metadata_keys,
    js_order,
    legacy_created_at_ms,
    normalize_anchor_span,
    normalize_date_range,
    parse_managed_metadata,
)
from .policy import DEFAULT_POLICY_IDS

DEFAULT_TOP_K = 10
DEFAULT_OVERFETCH_FACTOR = 4
DEFAULT_MAX_CANDIDATES = 200
DEFAULT_QUERY_HALF_LIFE_DAYS = 14
DEFAULT_ALPHA = 0.05


def build_mem0_scope_filters(scope, tenant_binding=None, filters=None):
    assert_no_reserved_metadata_keys(filters)
    tenant_field = (
        tenant_binding.field
        if tenant_binding and tenant_binding.mode == "metadata"
        else TENANT
    )
    _validate_nested_filters(filters or {}, {*IDENTITY_FIELDS, tenant_field})
    scoped = {
        key: value
        for key, value in (filters or {}).items()
        if key not in IDENTITY_FIELDS
    }
    scoped.update(scope.identity())
    if tenant_binding and tenant_binding.mode == "metadata":
        scoped[tenant_binding.field] = scope.tenant
    return scoped


def _validate_nested_filters(filters, scope_fields):
    """Protect scope constraints when the SDK flattens caller-provided boolean filters."""
    for operator in ("AND", "OR", "NOT", "$or", "$not"):
        if operator not in filters:
            continue
        children = filters[operator]
        if not isinstance(children, list) or any(
            not isinstance(child, Mapping) for child in children
        ):
            raise ValueError(f"{operator} must be a list of filter mappings")
        for child in children:
            assert_no_reserved_metadata_keys(child)
            if scope_fields.intersection(child):
                raise ValueError(
                    "Scope fields cannot be set inside boolean search filters"
                )
            _validate_nested_filters(child, scope_fields)


def rank_key(item):
    return (
        -item.adjusted_score,
        -item.base_score,
        js_order(item.store_id),
        js_order(item.id),
    )


def _score(item):
    for key in ("rerank_score", "rerankScore", "score"):
        candidate = item.get(key)
        if finite(candidate):
            return candidate
    return 0


def query_mode(range_query, anchor):
    if range_query is not None and anchor is not None:
        raise ValueError("cannot specify range and anchor in one search")
    if range_query is not None:
        return "strict_range" if range_query.strict else "soft_range"
    return "anchor" if anchor is not None else "recency"


def _query_curve(options, default, registry):
    if options.curve is not None:
        if options.half_life_days is not None:
            raise ValueError("provide either query curve or half_life_days")
        return resolve_decay_curve(options.curve, registry)
    if options.half_life_days is not None:
        return exponential(options.half_life_days)
    return default


async def execute_decay_search(
    store,
    query,
    *,
    scope,
    top_k=DEFAULT_TOP_K,
    alpha=None,
    curve=None,
    threshold=None,
    rerank=None,
    show_expired=None,
    filters=None,
    range=None,
    anchor=None,
    reference_now_ms=None,
):
    nonempty(query, "query")
    positive_int(top_k, "top_k")
    if top_k > store.max_candidates:
        raise ValueError("top_k exceeds max_candidates")
    mode = query_mode(range, anchor)
    now = store.clock() if reference_now_ms is None else reference_now_ms
    if not finite(now):
        raise ValueError("reference_now_ms must be finite")
    alpha = store.alpha if alpha is None else unit_interval(alpha, "alpha")
    scoped = build_mem0_scope_filters(scope, store.tenant_binding, filters)
    override = (
        resolve_decay_curve(curve, store.curve_registry) if curve is not None else None
    )
    temporal_curve = override or store.default_query_curve
    bounds = None
    if range is not None:
        temporal_curve = _query_curve(range, temporal_curve, store.curve_registry)
        bounds = normalize_date_range(
            range.start, range.end, range.time_zone or store.default_time_zone
        )
        if mode == "strict_range":
            scoped[TIME_START] = {"gte": bounds.start}
            scoped[TIME_END] = {"lt": bounds.end}
    elif anchor is not None:
        temporal_curve = _query_curve(anchor, temporal_curve, store.curve_registry)
        bounds = normalize_anchor_span(
            anchor.around, anchor.time_zone or store.default_time_zone
        )

    limit = (
        top_k
        if mode == "strict_range"
        else min(top_k * store.overfetch_factor, store.max_candidates)
    )
    opts = {
        key: value
        for key, value in dict(
            threshold=threshold, rerank=rerank, show_expired=show_expired
        ).items()
        if value is not None
    }
    response = await call(
        store.memory.search, query, top_k=limit, filters=scoped, **opts
    )
    candidates = (response or {}).get("results") or []
    recorded_use = {}
    if mode == "recency":
        recorded_use = await read_access_batch_safe(
            store.access_sidecar,
            [
                AccessKey(scope.tenant, store.store_id, str(item["id"]))
                for item in candidates
            ],
        )

    default_policy = store.policy_resolver.resolve(
        store.default_policy_id
    ) or store.policy_resolver.resolve_or_raise(DEFAULT_POLICY_IDS["soft"])
    results = []
    for item in candidates:
        item_id = str(item["id"])
        metadata = dict(item.get("metadata") or {})
        if any(
            field in metadata and metadata[field] != scope.tenant
            for field in (TENANT, store.tenant_field)
        ):
            continue
        if store.metadata_binding and metadata.get(store.tenant_field) != scope.tenant:
            continue
        _, managed = parse_managed_metadata(metadata)
        policy = (
            store.policy_resolver.resolve(managed[POLICY])
            if managed
            else default_policy
        )
        span = TimeSpan(managed[TIME_START], managed[TIME_END]) if managed else None
        expiry = managed.get(EXPIRES_AT) if managed else None
        if (
            not show_expired
            and expiry is not None
            and expiry <= now
            and policy
            and policy.enforce_exact_visibility_expiry
        ):
            continue

        created = item.get("created_at", item.get("createdAt"))
        weight = 1
        if bounds is not None:
            if span is None:
                continue
            if mode == "strict_range":
                if span.start < bounds.start or span.end >= bounds.end:
                    continue
            else:
                distance = max(0, bounds.start - span.end, span.start - bounds.end)
                if distance:
                    weight = temporal_curve(distance / MS_PER_DAY)
        elif policy and not policy.full_recency_weight:
            event_end = span.end if span else legacy_created_at_ms(created, metadata)
            if event_end is not None:
                age = max(0, (now - event_end) / MS_PER_DAY)
                if store.access_sidecar is not None:
                    key = serialize_access_key(
                        AccessKey(scope.tenant, store.store_id, item_id)
                    )
                    age = compute_effective_age_days(
                        age,
                        now,
                        recorded_use.get(key),
                        store.access_sidecar.max_age_reduction_days,
                    )
                weight = (
                    policy.floor + (1 - policy.floor) * override(age)
                    if override is not None
                    else policy.weighted_curve(age)
                )

        score = _score(item)
        results.append(
            DecaySearchItem(
                item_id,
                item["memory"],
                store.store_id,
                score,
                score - score * (1 - weight) * alpha,
                weight,
                span,
                managed[TIER] if managed else None,
                managed[POLICY] if managed else None,
                managed[INGESTED_AT] if managed else None,
                expiry,
                metadata,
                created,
                item.get("updated_at", item.get("updatedAt")),
            )
        )

    results.sort(key=rank_key)
    return DecaySearchResult(results[:top_k], now, len(candidates), mode)
