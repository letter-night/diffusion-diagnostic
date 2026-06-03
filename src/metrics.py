"""Distributional + diversity metrics for the diagnostic study.

All distribution metrics operate on dicts {label: value} and align on a shared key set
(the union of both sides) so missing buckets count as zero. Bootstrap CIs resample
*prompt IDs* (not individual completions), because completions sharing a prompt are
correlated and treating them as independent would understate the CI.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Callable, Iterable, Sequence

import numpy as np

EPS = 1e-12


# --------------------------------------------------------------------------- #
# Distribution helpers
# --------------------------------------------------------------------------- #
def counts_to_proportions(counts: dict[str, float]) -> dict[str, float]:
    total = sum(counts.values())
    if total <= 0:
        return {k: 0.0 for k in counts}
    return {k: v / total for k, v in counts.items()}


def empirical_distribution(labels: Iterable[str],
                           support: Sequence[str] | None = None) -> dict[str, float]:
    """Counts -> proportions over `support` (or observed labels if support is None)."""
    c = Counter(labels)
    keys = list(support) if support is not None else list(c.keys())
    counts = {k: float(c.get(k, 0)) for k in keys}
    return counts_to_proportions(counts)


def _aligned(p: dict[str, float], q: dict[str, float]) -> tuple[list[float], list[float]]:
    keys = list(dict.fromkeys([*p.keys(), *q.keys()]))
    return [p.get(k, 0.0) for k in keys], [q.get(k, 0.0) for k in keys]


def tv_distance(p: dict[str, float], q: dict[str, float]) -> float:
    """Total variation distance = 1/2 * sum_j |p_j - q_j|. In [0, 1]."""
    pv, qv = _aligned(p, q)
    return 0.5 * sum(abs(a - b) for a, b in zip(pv, qv))


def l1_moment_error(p: dict[str, float], q: dict[str, float]) -> float:
    pv, qv = _aligned(p, q)
    return sum(abs(a - b) for a, b in zip(pv, qv))


def l2_moment_error(p: dict[str, float], q: dict[str, float]) -> float:
    pv, qv = _aligned(p, q)
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(pv, qv)))


def shannon_entropy(p: dict[str, float], base: float = 2.0) -> float:
    s = sum(p.values())
    if s <= 0:
        return 0.0
    out = 0.0
    for v in p.values():
        if v > 0:
            pr = v / s
            out -= pr * math.log(pr, base)
    return out


def normalized_entropy(p: dict[str, float], base: float = 2.0) -> float:
    """Entropy / log(k); 1.0 = uniform, 0.0 = point mass. k = number of buckets."""
    k = len(p)
    if k <= 1:
        return 0.0
    return shannon_entropy(p, base) / math.log(k, base)


def kl_divergence(p: dict[str, float], q: dict[str, float], base: float = 2.0) -> float:
    """KL(p || q) with q smoothed by EPS to avoid div-by-zero. Nats if base=e."""
    pv, qv = _aligned(p, q)
    sp, sq = sum(pv) or 1.0, sum(qv) or 1.0
    out = 0.0
    for a, b in zip(pv, qv):
        pa = a / sp
        qb = max(b / sq, EPS)
        if pa > 0:
            out += pa * math.log(pa / qb, base)
    return out


def uniform_over(keys: Sequence[str]) -> dict[str, float]:
    k = len(keys)
    return {key: (1.0 / k if k else 0.0) for key in keys}


# --------------------------------------------------------------------------- #
# Diversity / repetition (over free-text bodies)
# --------------------------------------------------------------------------- #
def _tokens(text: str) -> list[str]:
    return text.split()


def _ngrams(tokens: Sequence[str], n: int) -> list[tuple]:
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)] if len(tokens) >= n else []


def distinct_n(texts: Iterable[str], n: int) -> float:
    """Corpus-level Distinct-n: unique n-grams / total n-grams across all texts."""
    total, uniq = 0, set()
    for t in texts:
        grams = _ngrams(_tokens(t), n)
        total += len(grams)
        uniq.update(grams)
    return (len(uniq) / total) if total else 0.0


def type_token_ratio(texts: Iterable[str]) -> float:
    total, types = 0, set()
    for t in texts:
        toks = _tokens(t)
        total += len(toks)
        types.update(toks)
    return (len(types) / total) if total else 0.0


def repetition_rate(texts: Iterable[str]) -> float:
    """Mean per-text fraction of unigram tokens that are repeats (1 - unique/total)."""
    rates = []
    for t in texts:
        toks = _tokens(t)
        if toks:
            rates.append(1.0 - len(set(toks)) / len(toks))
    return float(np.mean(rates)) if rates else 0.0


# --------------------------------------------------------------------------- #
# Bootstrap CIs over prompt IDs
# --------------------------------------------------------------------------- #
def bootstrap_ci(groups: dict[str, list],
                 stat_fn: Callable[[list], float],
                 n_boot: int = 1000,
                 ci: float = 0.95,
                 seed: int = 0) -> dict[str, float]:
    """Bootstrap a scalar statistic by resampling prompt-ID groups with replacement.

    groups: {prompt_id: [items...]}; stat_fn maps a flat item list -> scalar.
    Returns {point, lo, hi}. Resampling whole groups preserves within-prompt correlation.
    """
    keys = list(groups.keys())
    all_items = [it for k in keys for it in groups[k]]
    point = float(stat_fn(all_items)) if all_items else float("nan")
    if not keys or n_boot <= 0:
        return {"point": point, "lo": point, "hi": point}

    rng = np.random.default_rng(seed)
    n = len(keys)
    stats = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        pick = rng.integers(0, n, size=n)
        items = [it for i in pick for it in groups[keys[i]]]
        stats[b] = stat_fn(items) if items else float("nan")
    alpha = (1.0 - ci) / 2.0
    lo = float(np.nanpercentile(stats, 100 * alpha))
    hi = float(np.nanpercentile(stats, 100 * (1 - alpha)))
    return {"point": point, "lo": lo, "hi": hi}


def proportion_cis(groups: dict[str, list],
                   support: Sequence[str],
                   n_boot: int = 1000,
                   ci: float = 0.95,
                   seed: int = 0) -> dict[str, dict[str, float]]:
    """Bootstrap CI for each bucket's proportion. Returns {bucket: {point, lo, hi}}."""
    out = {}
    for j, bucket in enumerate(support):
        out[bucket] = bootstrap_ci(
            groups,
            lambda items, b=bucket: (items.count(b) / len(items)) if items else 0.0,
            n_boot=n_boot, ci=ci, seed=seed + j,
        )
    return out
