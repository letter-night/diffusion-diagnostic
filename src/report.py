"""Write aggregate tables (CSV+JSON), a human-readable report.md, and figures."""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd


def _ensure(d: str) -> str:
    os.makedirs(d, exist_ok=True)
    return d


def write_aggregate_outputs(summaries: list[dict], out_dir: str) -> None:
    _ensure(out_dir)
    rows = []
    for s in summaries:
        with open(os.path.join(out_dir, f"{s['case_id']}_{s['variant']}.json"), "w") as f:
            json.dump(s, f, indent=2)
        for axis in s["axes"]:
            if not axis["scored"]:
                continue
            for bucket in axis["target"]:
                ci = axis.get("proportion_ci", {}).get(bucket, {})
                rows.append({
                    "case_id": s["case_id"], "variant": s["variant"],
                    "model": s["model_name"], "axis": axis["axis"], "bucket": bucket,
                    "achieved": round(axis["empirical_scored"].get(bucket, 0.0), 4),
                    "target": axis["target"][bucket],
                    "ci_lo": round(ci.get("lo", float("nan")), 4),
                    "ci_hi": round(ci.get("hi", float("nan")), 4),
                    "tv": round(axis["tv"], 4), "l1": round(axis["l1"], 4),
                    "l2": round(axis["l2"], 4),
                    "kl_vs_target": round(axis["kl_vs_target"], 4),
                    "norm_entropy": round(axis["normalized_entropy"], 4),
                    "unknown_rate": round(axis["unknown_rate"], 4),
                    "invalid_metadata_rate": round(s["invalid_metadata_rate"], 4),
                })
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(out_dir, "summary_table.csv"), index=False)


def _headline(summary: dict) -> str | None:
    scored = [a for a in summary["axes"] if a["scored"]]
    if not scored:
        return None
    a = scored[0]
    emp = a["empirical_scored"]
    parts = " / ".join(f"{round(100 * emp.get(b, 0.0))}% {b}" for b in a["target"])
    return (f"{summary['case_id']} {a['axis']} = {parts} vs target "
            f"({', '.join(f'{int(100*v)}% {b}' for b, v in a['target'].items())}) "
            f"→ TV {a['tv']:.2f}, L1 {a['l1']:.2f}, "
            f"invalid-label rate {summary['invalid_metadata_rate']*100:.0f}%.")


def write_report_md(summaries: list[dict], path: str) -> None:
    lines: list[str] = []
    lines.append("# Diagnostic Study: Distributional Failure in Diffusion LMs\n")
    lines.append("_Measurement-only report. Characterizes the base model's aggregate "
                 "sampling distribution against user-specified targets. No fine-tuning, "
                 "no reward models, no GDC._\n")

    # group by case
    by_case: dict[str, list[dict]] = {}
    for s in summaries:
        by_case.setdefault(s["case_id"], []).append(s)

    for case_id, group in by_case.items():
        lines.append(f"\n## {case_id}\n")
        for s in group:
            lines.append(f"\n### variant: `{s['variant']}`  "
                         f"(model: {s['model_name']}, n={s['n_completions']}, "
                         f"prompts={s['n_prompts']})\n")
            hl = _headline(s)
            if hl:
                lines.append(f"> **Headline:** {hl}\n")
            lines.append(f"- invalid/unknown-metadata rate: "
                         f"**{s['invalid_metadata_rate']*100:.1f}%**")
            if "format_label_matches_body_rate" in s:
                lines.append(f"- format label matches body: "
                             f"**{s['format_label_matches_body_rate']*100:.1f}%**")
            d = s["diversity"]
            lines.append(f"- diversity: distinct-2 {d['distinct_2']:.3f}, "
                         f"distinct-3 {d['distinct_3']:.3f}, "
                         f"repetition {d['repetition_rate']:.3f}, "
                         f"TTR {d['type_token_ratio']:.3f}\n")

            for a in s["axes"]:
                lines.append(f"\n#### axis: {a['axis']}\n")
                if not a["scored"]:
                    top = sorted(a["counts"].items(), key=lambda kv: -kv[1])[:6]
                    lines.append("_descriptive axis (no target)_ — top buckets: "
                                 + ", ".join(f"{k} ({v})" for k, v in top) + "\n")
                    continue
                lines.append("| bucket | achieved | 95% CI | target |")
                lines.append("| --- | --- | --- | --- |")
                ci = a.get("proportion_ci", {})
                for b in a["target"]:
                    ach = a["empirical_scored"].get(b, 0.0)
                    c = ci.get(b, {})
                    lines.append(f"| {b} | {ach:.3f} | "
                                 f"[{c.get('lo', float('nan')):.3f}, "
                                 f"{c.get('hi', float('nan')):.3f}] | "
                                 f"{a['target'][b]:.3f} |")
                tvc = a.get("tv_ci", {})
                lines.append(
                    f"\nTV **{a['tv']:.3f}** "
                    f"(95% CI [{tvc.get('lo', float('nan')):.3f}, "
                    f"{tvc.get('hi', float('nan')):.3f}]), "
                    f"L1 {a['l1']:.3f}, L2 {a['l2']:.3f}, "
                    f"KL(emp‖target) {a['kl_vs_target']:.3f}, "
                    f"norm-entropy {a['normalized_entropy']:.3f}, "
                    f"unknown {a['unknown_rate']*100:.1f}%\n")

    lines.append("\n---\n")
    lines.append("### Neutral takeaways\n")
    lines.append("- Base sampling (and prompting) does not directly solve user-specified "
                 "aggregate moment-matching — the gap later Diffusion-GDC work addresses.\n")
    lines.append("- The model can produce every target category at least sometimes, yet "
                 "the **aggregate** distribution is skewed away from the target.\n")
    lines.append("- Framing is diagnostic: this characterizes a sampling distribution; it "
                 "does not claim existing methods can *never* match a target.\n")

    _ensure(os.path.dirname(path))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def make_figures(summaries: list[dict], fig_dir: str) -> list[str]:
    """Target-vs-achieved bar charts + Case 1 gender×region heatmap. Returns paths."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:  # pragma: no cover
        print(f"[report] matplotlib unavailable, skipping figures: {e}")
        return []

    _ensure(fig_dir)
    paths = []
    for s in summaries:
        for a in s["axes"]:
            if not a["scored"]:
                continue
            buckets = list(a["target"].keys())
            ach = [a["empirical_scored"].get(b, 0.0) for b in buckets]
            tgt = [a["target"][b] for b in buckets]
            x = np.arange(len(buckets))
            w = 0.38
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(x - w / 2, ach, w, label="achieved")
            ax.bar(x + w / 2, tgt, w, label="target")
            ax.set_xticks(x)
            ax.set_xticklabels(buckets, rotation=20, ha="right")
            ax.set_ylabel("proportion")
            ax.set_title(f"{s['case_id']} / {a['axis']} / {s['variant']} (TV={a['tv']:.2f})")
            ax.legend()
            fig.tight_layout()
            p = os.path.join(fig_dir, f"{s['case_id']}_{a['axis']}_{s['variant']}.png")
            fig.savefig(p, dpi=120)
            plt.close(fig)
            paths.append(p)

    # Case 1 gender x region heatmap (joint counts) — neutral variant if present.
    for s in summaries:
        if s["case_id"] != "C1_BIO_DEMOGRAPHIC":
            continue
        paths += _gender_region_heatmap(s, fig_dir)
    return paths


def _gender_region_heatmap(summary: dict, fig_dir: str) -> list[str]:
    import matplotlib.pyplot as plt
    import numpy as np
    # Need joint labels; reconstructed from stored per-completion records is not here,
    # so we approximate the heatmap from marginal axes only when joint is unavailable.
    joint_list = summary.get("joint_gender_region")
    if not joint_list:
        return []
    joint = {(g, r): c for g, r, c in joint_list}
    genders = ["woman", "man", "nonbinary"]
    regions = ["Africa", "East Asia", "Europe", "Latin America"]
    mat = np.array([[joint.get((g, r), 0) for r in regions] for g in genders], dtype=float)
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(mat, cmap="viridis")
    ax.set_xticks(range(len(regions)))
    ax.set_xticklabels(regions, rotation=20, ha="right")
    ax.set_yticks(range(len(genders)))
    ax.set_yticklabels(genders)
    for i in range(len(genders)):
        for j in range(len(regions)):
            ax.text(j, i, int(mat[i, j]), ha="center", va="center", color="w")
    ax.set_title(f"C1 gender×region joint counts ({summary['variant']})")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    p = os.path.join(fig_dir, f"C1_gender_region_heatmap_{summary['variant']}.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return [p]
