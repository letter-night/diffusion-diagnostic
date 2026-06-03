"""Sampler interface + dry-run stub.

The pipeline talks to models only through `DiffusionSampler.generate`, which returns
a raw completion string. The dry-run stub fabricates *skewed* completions (including a
configurable rate of malformed metadata lines) so the whole annotate -> metrics ->
aggregate -> report pipeline is testable without a GPU. Crucially, the stub encodes its
chosen labels only into the output text — exactly like a real model — so the annotator
must re-parse them and is never handed ground truth.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class SamplingParams:
    steps: int = 128
    gen_length: int = 256
    block_length: int = 32
    temperature: float = 0.0
    remasking: str = "low_confidence"

    @classmethod
    def from_dict(cls, d: dict | None) -> "SamplingParams":
        d = d or {}
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GenContext:
    """Per-call context. Real samplers ignore everything except `prompt`."""
    case_id: str = ""
    prompt_id: str = ""
    variant: str = "neutral"
    sample_idx: int = 0
    metadata_key: str = ""
    # axes: list of {"field": str, "values": [..], "target": {..}}
    axes: list[dict[str, Any]] = field(default_factory=list)


class DiffusionSampler:
    """Abstract sampler. Subclasses implement `generate`."""

    backend = "abstract"

    def __init__(self, model_cfg: dict):
        self.model_cfg = model_cfg
        self.model_name = model_cfg.get("model_name", self.backend)

    def generate(self, prompt: str, *, seed: int, params: SamplingParams,
                 ctx: GenContext) -> str:
        raise NotImplementedError


def _hash_unit(*parts: Any) -> float:
    """Deterministic float in [0,1) from arbitrary parts (no global RNG state)."""
    h = hashlib.sha256("||".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:13], 16) / float(16 ** 13)


def _hash_int(n: int, *parts: Any) -> int:
    return int(_hash_unit(*parts) * n) % n if n > 0 else 0


# Modal bucket each axis collapses onto in the synthetic data (illustrative).
_MODAL = {
    "gender_identity": "man",
    "home_region": "Europe",
    "format": "json",
    "topic": "efficiency",
    "style": "definition_first",
}

_FIELDS_BY_REGION = {
    "Africa": "glaciology", "East Asia": "marine ecology",
    "Europe": "particle physics", "Latin America": "agronomy",
}


def _skewed_choice(values: list[str], target: dict, skew: float, *salt) -> str:
    """Pick a value from a distribution collapsed toward a modal bucket.

    modal prob = skew + (1-skew)/k ; others share the rest evenly.
    This deliberately diverges from `target` so the diagnostic has something to measure.
    """
    k = len(values)
    if k == 0:
        return "unknown"
    modal = _MODAL.get(_axis_name_for(values), values[_hash_int(k, "modal", *salt)])
    if modal not in values:
        modal = values[0]
    base = (1.0 - skew) / k
    probs = [base + (skew if v == modal else 0.0) for v in values]
    s = sum(probs)
    probs = [p / s for p in probs]
    r = _hash_unit("choice", *salt)
    acc = 0.0
    for v, p in zip(values, probs):
        acc += p
        if r <= acc:
            return v
    return values[-1]


def _axis_name_for(values: list[str]) -> str:
    """Best-effort axis key from its value set (used only for modal lookup)."""
    sets = {
        "gender_identity": {"woman", "man", "nonbinary"},
        "home_region": {"Africa", "East Asia", "Europe", "Latin America"},
        "format": {"json", "markdown_table", "numbered_list", "plain_paragraph"},
        "topic": {"evaluation", "safety", "efficiency", "multilinguality"},
        "style": {"definition_first", "analogy_centered", "socratic_tutor", "formal_academic"},
    }
    vv = set(values)
    for name, sv in sets.items():
        if vv == sv:
            return name
    return values[0] if values else ""


def _render_body(case_id: str, labels: dict, salt) -> str:
    """Produce body text consistent with chosen labels (so content checks mostly match)."""
    if case_id == "C2_FORMAT":
        fmt = labels.get("format", "plain_paragraph")
        # ~12% of the time, emit a body that does NOT match the declared format,
        # so body_matches_label rate is a non-trivial reported metric.
        if _hash_unit("mismatch", *salt) < 0.12:
            fmt = {"json": "plain_paragraph", "plain_paragraph": "json",
                   "markdown_table": "numbered_list",
                   "numbered_list": "markdown_table"}.get(fmt, "plain_paragraph")
        if fmt == "json":
            return json.dumps({"answer": "concise response", "items": [1, 2, 3]})
        if fmt == "markdown_table":
            return "| key | value |\n| --- | --- |\n| a | 1 |\n| b | 2 |"
        if fmt == "numbered_list":
            return "1. First point.\n2. Second point.\n3. Third point."
        return "This is a concise plain-paragraph answer to the task."
    if case_id == "C4_STYLE":
        return f"[{labels.get('style','?')}] A 100-150 word explanation tailored to the audience."
    if case_id == "C3_TOPIC":
        return f"Idea: a concrete project on {labels.get('topic','?')} with a clear evaluation plan."
    if case_id == "C1_BIO_DEMOGRAPHIC":
        return ("A fictional figure whose life unfolded across study and practice, "
                "remembered for quiet, lasting contributions.")
    return "Synthetic body."


class DryRunSampler(DiffusionSampler):
    backend = "dry_run"

    def __init__(self, model_cfg: dict):
        super().__init__(model_cfg)
        cfg = model_cfg.get("dry_run", {})
        self.invalid_rate = float(cfg.get("invalid_metadata_rate", 0.04))
        self.skew = float(cfg.get("skew", 0.7))

    def generate(self, prompt: str, *, seed: int, params: SamplingParams,
                 ctx: GenContext) -> str:
        salt = (ctx.case_id, ctx.prompt_id, ctx.variant, ctx.sample_idx, seed)
        # Explicit/weak prompting nudges the model a little, but does NOT fix the skew.
        skew = self.skew * {"neutral": 1.0, "weak": 0.85, "explicit": 0.7}.get(ctx.variant, 1.0)

        labels: dict[str, str] = {}
        for axis in ctx.axes:
            fld = axis["field"]
            vals = axis.get("values") or []
            if not vals:
                continue
            labels[fld] = _skewed_choice(vals, axis.get("target", {}), skew, fld, *salt)

        # Case 1 free-text `field` axis, correlated with region for flavor.
        if ctx.case_id == "C1_BIO_DEMOGRAPHIC":
            labels["field"] = _FIELDS_BY_REGION.get(labels.get("home_region", ""), "history")

        meta_line = self._format_meta(ctx, labels)
        body = _render_body(ctx.case_id, labels, salt)

        # Inject malformed/missing metadata at the configured rate.
        if _hash_unit("invalid", *salt) < self.invalid_rate:
            mode = _hash_int(3, "invalidmode", *salt)
            if mode == 0:
                return body                                   # missing line entirely
            if mode == 1:
                return f"{ctx.metadata_key} = {list(labels.values())[0] if labels else 'x'}\n\n{body}"
            return f"(no structured metadata)\n\n{body}"      # unparseable

        return f"{meta_line}\n\n{body}"

    @staticmethod
    def _format_meta(ctx: GenContext, labels: dict) -> str:
        if ctx.case_id == "C1_BIO_DEMOGRAPHIC":
            inner = "; ".join(
                f"{k}={labels.get(k,'unknown')}"
                for k in ("gender_identity", "home_region", "field"))
            return f"{ctx.metadata_key}: {inner}"
        # single-field cases: one axis
        val = next(iter(labels.values()), "unknown")
        return f"{ctx.metadata_key}: {val}"


def get_sampler(model_cfg: dict) -> DiffusionSampler:
    backend = model_cfg.get("backend", "dry_run")
    if backend == "dry_run":
        return DryRunSampler(model_cfg)
    if backend == "llada":
        from .llada import LLaDASampler
        return LLaDASampler(model_cfg)
    if backend == "mdlm":
        from .mdlm import MDLMSampler
        return MDLMSampler(model_cfg)
    raise ValueError(f"Unknown sampler backend: {backend!r}")
