"""Deterministic content/consistency checks (secondary annotation layer).

Case 2 (format): does the body actually parse as the declared format? This is a pure,
rule-based check so the `format_label_matches_body` rate is fully reproducible.
"""

from __future__ import annotations

import json
import re

FORMATS = ["json", "markdown_table", "numbered_list", "plain_paragraph"]

_NUMBERED_RE = re.compile(r"^\s*\d+[.)]\s+\S")


def _is_json(body: str) -> bool:
    s = body.strip()
    if not (s.startswith("{") or s.startswith("[")):
        return False
    try:
        obj = json.loads(s)
    except (ValueError, TypeError):
        return False
    return isinstance(obj, (dict, list))


def _is_markdown_table(body: str) -> bool:
    lines = [ln for ln in body.splitlines() if ln.strip()]
    has_row = any(ln.count("|") >= 2 for ln in lines)
    # a separator row like | --- | --- |
    has_sep = any(re.match(r"^\s*\|?[\s:|-]*-{2,}[\s:|-]*\|?\s*$", ln) and "|" in ln
                  for ln in lines)
    return has_row and has_sep


def _is_numbered_list(body: str) -> bool:
    return sum(1 for ln in body.splitlines() if _NUMBERED_RE.match(ln)) >= 3


def detect_format(body: str) -> str:
    """Classify body into exactly one format bucket (checked in priority order)."""
    if _is_json(body):
        return "json"
    if _is_markdown_table(body):
        return "markdown_table"
    if _is_numbered_list(body):
        return "numbered_list"
    return "plain_paragraph"


def check_format(body: str, declared: str) -> dict:
    """Return {detected_format, body_matches_label}."""
    detected = detect_format(body)
    return {"detected_format": detected, "body_matches_label": (detected == declared)}
