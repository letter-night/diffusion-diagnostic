"""Aggregate annotated completions into per-case, per-axis distribution summaries.

For each axis: empirical distribution vs. target (side by side), TV / L1 / L2, entropy,
normalized entropy, KL vs uniform and vs target, invalid-metadata rate, and bootstrap CIs
(over prompt IDs) for TV and for each bucket proportion. Plus corpus diversity metrics
over the free-text bodies, and (Case 2) the body-matches-label rate.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from . import metrics


def _group_by_prompt(records: list[dict], field: str) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for r in records:
        groups[r["prompt_id"]].append(r["parsed_labels"].get(field, "unknown"))
    return dict(groups)


def aggregate_axis(records: list[dict], axis: dict, n_boot: int, ci: float,
                   seed: int) -> dict[str, Any]:
    field = axis["field"]
    target = axis.get("target", {}) or {}
    values = list(axis.get("values") or [])
    scored = bool(target)  # axes without a target are descriptive only

    labels = [r["parsed_labels"].get(field, "unknown") for r in records]
    # support = declared values + 'unknown' (+ any surprise observed buckets)
    support = list(dict.fromkeys([*values, *labels, "unknown"]))
    emp = metrics.empirical_distribution(labels, support=support)
    counts = {k: sum(1 for l in labels if l == k) for k in support}

    n_total = len(labels)
    n_unknown = counts.get("unknown", 0)
    out: dict[str, Any] = {
        "axis": axis["name"],
        "field": field,
        "n": n_total,
        "scored": scored,
        "support": support,
        "counts": counts,
        "empirical": emp,
        "target": target,
        "unknown_rate": (n_unknown / n_total) if n_total else 0.0,
        "normalized_entropy": metrics.normalized_entropy(
            {k: emp[k] for k in support if k != "unknown"} or emp),
        "kl_vs_uniform": metrics.kl_divergence(
            emp, metrics.uniform_over([k for k in support if k != "unknown"] or support)),
    }

    if scored:
        # Score only over the target's support (exclude 'unknown' from the comparison
        # distribution but report unknown_rate separately).
        keys = list(target.keys())
        emp_scored = metrics.empirical_distribution(
            [l for l in labels if l in keys], support=keys)
        out["empirical_scored"] = emp_scored
        out["tv"] = metrics.tv_distance(emp_scored, target)
        out["l1"] = metrics.l1_moment_error(emp_scored, target)
        out["l2"] = metrics.l2_moment_error(emp_scored, target)
        out["kl_vs_target"] = metrics.kl_divergence(emp_scored, target)

        groups = _group_by_prompt(records, field)
        # TV CI: recompute empirical over target keys for each bootstrap resample
        def _tv_stat(items, keys=keys, target=target):
            sub = [i for i in items if i in keys]
            e = metrics.empirical_distribution(sub, support=keys)
            return metrics.tv_distance(e, target)

        out["tv_ci"] = metrics.bootstrap_ci(groups, _tv_stat, n_boot=n_boot, ci=ci, seed=seed)
        out["proportion_ci"] = metrics.proportion_cis(
            groups, keys, n_boot=n_boot, ci=ci, seed=seed)

    return out


def aggregate_case(records: list[dict], case_cfg: dict, variant: str,
                   n_boot: int = 1000, ci: float = 0.95, seed: int = 0) -> dict[str, Any]:
    case_id = case_cfg["case_id"]
    recs = [r for r in records
            if r.get("case_id") == case_id and r.get("prompt_variant") == variant]

    bodies = [r.get("body", "") for r in recs]
    n = len(recs)
    invalid = sum(1 for r in recs if not r.get("valid_metadata", False))

    summary: dict[str, Any] = {
        "case_id": case_id,
        "variant": variant,
        "model_name": recs[0]["model_name"] if recs else None,
        "n_completions": n,
        "n_prompts": len({r["prompt_id"] for r in recs}),
        "invalid_metadata_rate": (invalid / n) if n else 0.0,
        "diversity": {
            "distinct_2": metrics.distinct_n(bodies, 2),
            "distinct_3": metrics.distinct_n(bodies, 3),
            "repetition_rate": metrics.repetition_rate(bodies),
            "type_token_ratio": metrics.type_token_ratio(bodies),
        },
        "axes": [],
    }

    if case_cfg.get("content_check") == "format":
        checked = [r for r in recs if "body_matches_label" in r]
        summary["format_label_matches_body_rate"] = (
            sum(1 for r in checked if r["body_matches_label"]) / len(checked)
            if checked else 0.0)

    for axis in case_cfg["axes"]:
        summary["axes"].append(aggregate_axis(recs, axis, n_boot, ci, seed))

    # Case 1: joint gender×region counts for the heatmap (JSON-serializable list).
    if case_id == "C1_BIO_DEMOGRAPHIC":
        joint: dict[tuple, int] = defaultdict(int)
        for r in recs:
            g = r["parsed_labels"].get("gender_identity", "unknown")
            reg = r["parsed_labels"].get("home_region", "unknown")
            joint[(g, reg)] += 1
        summary["joint_gender_region"] = [[g, reg, c] for (g, reg), c in joint.items()]

    return summary
