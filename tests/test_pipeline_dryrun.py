"""End-to-end dry-run pipeline test: sample -> annotate -> report on the stub."""

import json
import os

import pytest

from src import run


@pytest.fixture
def isolated_results(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "RESULTS", str(tmp_path))
    return str(tmp_path)


def _run(argv):
    run.main(argv)


def test_full_pipeline_case2(isolated_results):
    res = isolated_results
    _run(["sample", "--model", "dry_run", "--case", "case2",
          "--n-prompts", "8", "--n-per-prompt", "16", "--variant", "neutral"])
    raw = os.path.join(res, "raw", "C2_FORMAT_neutral_dry-run-stub.jsonl")
    assert os.path.exists(raw)
    recs = [json.loads(l) for l in open(raw)]
    assert len(recs) == 8 * 16
    # sampling hyperparameters logged with every record
    assert all("sampling" in r and "steps" in r["sampling"] for r in recs)

    _run(["annotate", "--model", "dry_run", "--case", "case2", "--variant", "neutral"])
    ann = os.path.join(res, "annotations", "C2_FORMAT_neutral_dry-run-stub.jsonl")
    arecs = [json.loads(l) for l in open(ann)]
    assert all("parsed_labels" in r and "valid_metadata" in r for r in arecs)
    assert all("body_matches_label" in r for r in arecs)  # content check ran

    _run(["report", "--model", "dry_run", "--case", "case2", "--variant", "neutral",
          "--n-boot", "100"])
    summ = os.path.join(res, "aggregate", "C2_FORMAT_neutral.json")
    assert os.path.exists(summ)
    assert os.path.exists(os.path.join(res, "report.md"))
    s = json.load(open(summ))
    axis = s["axes"][0]
    assert 0.0 <= axis["tv"] <= 1.0
    assert set(axis["target"]) == {"json", "markdown_table", "numbered_list", "plain_paragraph"}
    # bootstrap CI brackets the point estimate
    assert axis["tv_ci"]["lo"] <= axis["tv_ci"]["point"] <= axis["tv_ci"]["hi"]


def test_dry_run_produces_skew_and_some_invalid(isolated_results):
    res = isolated_results
    _run(["sample", "--model", "dry_run", "--case", "case1",
          "--n-prompts", "12", "--n-per-prompt", "24", "--variant", "neutral"])
    _run(["annotate", "--model", "dry_run", "--case", "case1", "--variant", "neutral"])
    _run(["report", "--model", "dry_run", "--case", "case1", "--variant", "neutral",
          "--n-boot", "100"])
    s = json.load(open(os.path.join(res, "aggregate", "C1_BIO_DEMOGRAPHIC_neutral.json")))
    gender = next(a for a in s["axes"] if a["field"] == "gender_identity")
    # The stub deliberately collapses onto a modal bucket -> TV should be clearly nonzero.
    assert gender["tv"] > 0.1
    # Some malformed metadata should appear (rate configured ~4%).
    assert s["invalid_metadata_rate"] > 0.0
    # joint heatmap data present for Case 1
    assert s.get("joint_gender_region")


def test_resume_skips_existing(isolated_results):
    res = isolated_results
    argv = ["sample", "--model", "dry_run", "--case", "case4",
            "--n-prompts", "4", "--n-per-prompt", "8", "--variant", "neutral"]
    _run(argv)
    raw = os.path.join(res, "raw", "C4_STYLE_neutral_dry-run-stub.jsonl")
    n1 = sum(1 for _ in open(raw))
    _run(argv)  # second run should add nothing
    n2 = sum(1 for _ in open(raw))
    assert n1 == n2 == 4 * 8
