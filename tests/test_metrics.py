import math

import pytest

from src import metrics


def test_tv_distance_known():
    p = {"a": 0.5, "b": 0.5}
    q = {"a": 0.25, "b": 0.75}
    assert metrics.tv_distance(p, q) == pytest.approx(0.25)


def test_tv_identical_is_zero_and_disjoint_is_one():
    p = {"a": 1.0}
    assert metrics.tv_distance(p, p) == pytest.approx(0.0)
    assert metrics.tv_distance({"a": 1.0}, {"b": 1.0}) == pytest.approx(1.0)


def test_l1_l2_moment_error():
    p = {"a": 0.5, "b": 0.5}
    q = {"a": 0.0, "b": 1.0}
    assert metrics.l1_moment_error(p, q) == pytest.approx(1.0)
    assert metrics.l2_moment_error(p, q) == pytest.approx(math.sqrt(0.5))


def test_entropy_uniform_is_max_normalized_one():
    p = {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}
    assert metrics.shannon_entropy(p, base=2) == pytest.approx(2.0)
    assert metrics.normalized_entropy(p) == pytest.approx(1.0)


def test_entropy_point_mass_is_zero():
    p = {"a": 1.0, "b": 0.0, "c": 0.0}
    assert metrics.shannon_entropy(p) == pytest.approx(0.0)
    assert metrics.normalized_entropy(p) == pytest.approx(0.0)


def test_kl_self_is_zero_and_nonnegative():
    p = {"a": 0.3, "b": 0.7}
    assert metrics.kl_divergence(p, p) == pytest.approx(0.0, abs=1e-9)
    q = {"a": 0.5, "b": 0.5}
    assert metrics.kl_divergence(p, q) >= 0.0


def test_empirical_distribution_with_support():
    # Normalization is over the support's counts: labels outside `support`
    # (here "unknown") are excluded from the denominator. So a = 2/3, b = 1/3.
    labels = ["a", "a", "b", "unknown"]
    emp = metrics.empirical_distribution(labels, support=["a", "b", "c"])
    assert emp["a"] == pytest.approx(2 / 3)
    assert emp["b"] == pytest.approx(1 / 3)
    assert emp["c"] == pytest.approx(0.0)
    assert sum(emp.values()) == pytest.approx(1.0)


def test_distinct_n_and_repetition():
    texts = ["the cat sat", "the cat sat"]
    # 4 unique bigrams? "the cat","cat sat" repeated -> 2 unique / 4 total = 0.5
    assert metrics.distinct_n(texts, 2) == pytest.approx(0.5)
    assert metrics.repetition_rate(["a a b"]) == pytest.approx(1 - 2 / 3)
    assert metrics.type_token_ratio(["a a b"]) == pytest.approx(2 / 3)


def test_bootstrap_ci_brackets_point_and_is_deterministic():
    groups = {f"p{i}": ["man"] * 8 + ["woman"] * 2 for i in range(12)}
    stat = lambda items: items.count("man") / len(items)
    out1 = metrics.bootstrap_ci(groups, stat, n_boot=200, seed=0)
    out2 = metrics.bootstrap_ci(groups, stat, n_boot=200, seed=0)
    assert out1 == out2  # deterministic given seed
    assert out1["point"] == pytest.approx(0.8)
    assert out1["lo"] <= out1["point"] <= out1["hi"]


def test_proportion_cis_cover_all_buckets():
    groups = {f"p{i}": ["a", "b"] for i in range(5)}
    cis = metrics.proportion_cis(groups, ["a", "b", "c"], n_boot=100, seed=1)
    assert set(cis) == {"a", "b", "c"}
    assert cis["a"]["point"] == pytest.approx(0.5)
    assert cis["c"]["point"] == pytest.approx(0.0)
