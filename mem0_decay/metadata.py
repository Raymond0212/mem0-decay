"""Date normalization and the persisted metadata format shared with TypeScript."""

import hashlib
import json
import re
from collections.abc import Mapping
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

from .contracts import IDENTITY_FIELDS, ResolvedScope, TimeSpan, finite

DECAY_METADATA_PREFIX = "decay_"
TIME_START = "decay_time_start"
TIME_END = "decay_time_end"
INGESTED_AT = "decay_ingested_at"
POLICY = "decay_policy"
TIER = "decay_tier"
TENANT = "decay_tenant"
EXPIRES_AT = "decay_expires_at"
COMPACTION_KEY = "decay_compaction_key"
SOURCES = "decay_sources"
_DATE_ONLY = re.compile(r"\d{4}-\d{2}-\d{2}")


def assert_no_reserved_metadata_keys(metadata):
    for key in metadata or {}:
        if key.startswith(DECAY_METADATA_PREFIX):
            raise ValueError(f"Metadata key {key!r} is reserved for mem0-decay")


def is_date_only(value):
    return (isinstance(value, date) and not isinstance(value, datetime)) or (
        isinstance(value, str) and _DATE_ONLY.fullmatch(value.strip()) is not None
    )


def parse_timestamp_ms(value, *, time_zone="UTC", end_of_day_exclusive=False):
    if finite(value):
        return int(value)
    if is_date_only(value):
        day = date.fromisoformat(value.strip()) if isinstance(value, str) else value
        if end_of_day_exclusive:
            day += timedelta(days=1)
        value = datetime.combine(day, time.min, tzinfo=ZoneInfo(time_zone))
    elif isinstance(value, str):
        value = datetime.fromisoformat(value.strip())
    if not isinstance(value, datetime):
        raise ValueError(
            "timestamp must be an epoch millisecond number, date, or ISO datetime"
        )
    if value.tzinfo is None:
        value = value.replace(tzinfo=ZoneInfo(time_zone))
    return int(value.timestamp() * 1000)


def normalize_date_range(start, end, time_zone="UTC"):
    start_ms = parse_timestamp_ms(start, time_zone=time_zone)
    end_ms = parse_timestamp_ms(
        end, time_zone=time_zone, end_of_day_exclusive=is_date_only(end)
    )
    if start_ms >= end_ms:
        raise ValueError("range start must precede range end")
    return TimeSpan(start_ms, end_ms)


def normalize_anchor_span(around, time_zone="UTC"):
    start = parse_timestamp_ms(around, time_zone=time_zone)
    end = (
        parse_timestamp_ms(around, time_zone=time_zone, end_of_day_exclusive=True)
        if is_date_only(around)
        else start
    )
    return TimeSpan(start, end)


def parse_managed_metadata(payload):
    if not payload or not any(key.startswith(DECAY_METADATA_PREFIX) for key in payload):
        return "unmanaged", None
    begin, end, ingested = (
        payload.get(key) for key in (TIME_START, TIME_END, INGESTED_AT)
    )
    policy, tier = payload.get(POLICY), payload.get(TIER)
    if not all(map(finite, (begin, end, ingested))) or begin > end:
        return "malformed", None
    if (
        not isinstance(policy, str)
        or not policy.strip()
        or not finite(tier)
        or tier != int(tier)
        or tier < 0
    ):
        return "malformed", None

    result = {
        TIME_START: begin,
        TIME_END: end,
        INGESTED_AT: ingested,
        POLICY: policy.strip(),
        TIER: int(tier),
    }
    for key in (TENANT, COMPACTION_KEY):
        if key in payload:
            value = payload[key]
            if not isinstance(value, str) or not value.strip():
                return "malformed", None
            result[key] = value.strip()
    expires = payload.get(EXPIRES_AT)
    if expires is not None:
        if not finite(expires):
            return "malformed", None
        result[EXPIRES_AT] = expires
    if SOURCES in payload:
        sources = payload[SOURCES]
        if not isinstance(sources, list) or not sources:
            return "malformed", None
        if not all(
            isinstance(source, Mapping)
            and all(
                isinstance(source.get(key), str) and source[key].strip()
                for key in ("id", "revision")
            )
            for source in sources
        ):
            return "malformed", None
        result[SOURCES] = [
            {"id": source["id"], "revision": source["revision"]} for source in sources
        ]
    return "valid", result


def legacy_created_at_ms(created_at, metadata):
    for value in (created_at, metadata.get("createdAt"), metadata.get("created_at")):
        if value is None:
            continue
        try:
            return parse_timestamp_ms(value)
        except (ValueError, OverflowError):
            return None
    return None


def payload_matches_scope(
    payload, scope, *, exact=False, tenant_field=TENANT, require_tenant=False
):
    if payload is None or (
        (require_tenant or tenant_field in payload)
        and payload.get(tenant_field) != scope.tenant
    ):
        return False
    if TENANT in payload and payload[TENANT] != scope.tenant:
        return False
    identity = {
        key: value
        for key in IDENTITY_FIELDS
        if isinstance(value := payload.get(key), str) and value
    }
    expected = scope.identity()
    return (
        identity == expected
        if exact
        else all(identity.get(key) == value for key, value in expected.items())
    )


def scope_from_payload(payload, tenant):
    return ResolvedScope(
        payload.get(TENANT, tenant),
        **{
            key: payload[key]
            for key in IDENTITY_FIELDS
            if isinstance(payload.get(key), str) and payload[key]
        },
    )


def item_payload(item):
    metadata = dict(item.get("metadata") or {})
    metadata.update({key: item[key] for key in IDENTITY_FIELDS if key in item})
    return metadata


def stable_json(value):
    """Match JSON.stringify for the string, numeric, list, and object data used in stored hashes."""
    if isinstance(value, dict):
        return (
            "{"
            + ",".join(
                f"{stable_json(key)}:{stable_json(item)}" for key, item in value.items()
            )
            + "}"
        )
    if isinstance(value, (tuple, list)):
        return "[" + ",".join(map(stable_json, value)) + "]"
    if type(value) is float:
        if value == 0:
            return "0"
        text = repr(value)
        if 1e-6 <= abs(value) < 1e21:
            fixed = format(Decimal(text), "f")
            return fixed.rstrip("0").rstrip(".") if "." in fixed else fixed
        return re.sub(r"e([+-])0(\d+)$", r"e\1\2", text)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _scope_key(scope):
    return dict(
        tenant=scope.tenant,
        userId=scope.user_id,
        agentId=scope.agent_id,
        runId=scope.run_id,
    )


def _digest(value):
    serialized = stable_json(value)
    # JSON.stringify escapes a lone surrogate while leaving ordinary Unicode intact.
    if any(0xD800 <= ord(char) <= 0xDFFF for char in serialized):
        serialized = "".join(
            f"\\u{ord(char):04x}" if 0xD800 <= ord(char) <= 0xDFFF else char
            for char in serialized
        )
    return hashlib.sha256(serialized.encode()).hexdigest()


def compute_revision_fingerprint(memory, scope, metadata):
    canonical = {"memory": memory, **_scope_key(scope)}
    canonical.update(
        {
            key: metadata.get(key)
            for key in (TIME_START, TIME_END, INGESTED_AT, POLICY, TIER, EXPIRES_AT)
        }
    )
    return _digest(canonical)


def build_archive_key(store_id, source_id, revision):
    return f"{store_id}:{source_id}:{revision}"


def js_order(text):
    return text.encode("utf-16-be", errors="surrogatepass")


def sorted_source_refs(sources):
    return [
        {"id": source["id"], "revision": source["revision"]}
        for source in sorted(
            sources,
            key=lambda source: (js_order(source["id"]), js_order(source["revision"])),
        )
    ]


def derive_compaction_key(store_id, scope, target_tier, sources):
    canonical = dict(
        storeId=store_id,
        **_scope_key(scope),
        targetTier=target_tier,
        sources=sorted_source_refs(sources),
    )
    return f"cmp_t{target_tier}_{_digest(canonical)}"


def preserve_event_span_metadata(
    existing_metadata=None, updates=None, *, allow_event_span_correction=None
):
    assert_no_reserved_metadata_keys(updates)
    result = {**(existing_metadata or {}), **(updates or {})}
    if allow_event_span_correction is not None:
        start, end = allow_event_span_correction.start, allow_event_span_correction.end
        if not finite(start) or not finite(end) or start > end:
            raise ValueError("corrected event span requires finite start <= end")
        result.update({TIME_START: start, TIME_END: end})
    return result
