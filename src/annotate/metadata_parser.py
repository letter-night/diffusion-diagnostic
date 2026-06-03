"""Deterministic parser for the self-reported metadata line (primary annotation layer).

Contract: the model must emit ONE line first, then the body.
  - single-field cases:  "<Key>: <value>"
  - multi-field (Case 1): "<Key>: gender_identity=...; home_region=...; field=..."

If the line is missing or malformed, valid_metadata=False and every axis -> "unknown".
Values outside an axis's allowed set are bucketed as "unknown" but do NOT by themselves
flag the line invalid. Labels are never silently repaired — the invalid/unknown rate is a
reported metric, not something to paper over.
"""

from __future__ import annotations

import re
from typing import Any


def _norm(s: str) -> str:
    return s.strip().lower().replace("-", "_").replace(" ", "_")


# Small, explicit alias tables. Kept minimal on purpose; unknown stays unknown.
_ALIASES = {
    "gender_identity": {"female": "woman", "male": "man", "non_binary": "nonbinary",
                        "enby": "nonbinary", "nb": "nonbinary"},
    "home_region": {"latam": "Latin America", "south_america": "Latin America",
                    "asia": "East Asia"},
    "format": {"table": "markdown_table", "md_table": "markdown_table",
               "list": "numbered_list", "paragraph": "plain_paragraph",
               "text": "plain_paragraph", "plain": "plain_paragraph"},
}


def _build_lookup(values: list[str]) -> dict[str, str]:
    return {_norm(v): v for v in values}


def normalize_value(axis_field: str, value: str, allowed: list[str]) -> str:
    """Map a raw value onto an allowed bucket, else 'unknown'.

    Axes with no allowed set (e.g. Case 1 `field`) are descriptive: return the cleaned
    raw value, or 'unknown' if empty.
    """
    v = value.strip()
    if not allowed:
        return v if v else "unknown"
    n = _norm(v)
    lookup = _build_lookup(allowed)
    if n in lookup:
        return lookup[n]
    alias = _ALIASES.get(axis_field, {})
    if n in alias:
        return alias[n]
    return "unknown"


def split_meta_and_body(raw_output: str, metadata_key: str) -> tuple[str | None, str]:
    """Return (metadata_line_or_None, body). Body excludes the metadata line if present."""
    lines = raw_output.splitlines()
    # find first non-empty line
    idx = next((i for i, ln in enumerate(lines) if ln.strip()), None)
    if idx is None:
        return None, ""
    first = lines[idx].strip()
    if re.match(rf"^{re.escape(metadata_key)}\s*:", first, flags=re.IGNORECASE):
        body = "\n".join(lines[idx + 1:]).strip()
        return first, body
    return None, raw_output.strip()


def parse_metadata(raw_output: str, metadata_key: str,
                   axes: list[dict[str, Any]]) -> dict[str, Any]:
    """Parse the metadata line into per-axis buckets.

    Returns {"valid_metadata": bool, "parsed_labels": {axis_field: bucket}}.
    """
    fields = [a["field"] for a in axes]
    allowed_by_field = {a["field"]: (a.get("values") or []) for a in axes}
    unknown = {f: "unknown" for f in fields}

    meta_line, _ = split_meta_and_body(raw_output, metadata_key)
    if meta_line is None:
        return {"valid_metadata": False, "parsed_labels": dict(unknown)}

    # value part after the first colon
    value_part = meta_line.split(":", 1)[1].strip()
    if not value_part:
        return {"valid_metadata": False, "parsed_labels": dict(unknown)}

    multi = len(fields) > 1
    parsed: dict[str, str] = dict(unknown)

    if multi:
        # parse "k=v; k=v" pairs (also tolerate commas as separators)
        pairs = re.split(r"[;,]\s*", value_part)
        got = {}
        for p in pairs:
            if "=" in p:
                k, v = p.split("=", 1)
                got[_norm(k)] = v.strip()
        if not got:
            return {"valid_metadata": False, "parsed_labels": dict(unknown)}
        for f in fields:
            if _norm(f) in got:
                parsed[f] = normalize_value(f, got[_norm(f)], allowed_by_field[f])
            else:
                parsed[f] = "unknown"
        return {"valid_metadata": True, "parsed_labels": parsed}

    # single field
    f = fields[0]
    parsed[f] = normalize_value(f, value_part, allowed_by_field[f])
    return {"valid_metadata": True, "parsed_labels": parsed}
