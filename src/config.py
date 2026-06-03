"""Config loading + prompt rendering."""

from __future__ import annotations

import json
import os
from typing import Any

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_cases(path: str | None = None) -> dict[str, dict]:
    path = path or os.path.join(ROOT, "config", "cases.yaml")
    raw = _load_yaml(path)
    cases = {c["case_id"]: c for c in raw["cases"]}
    # attach variants table to each case for convenience
    for c in cases.values():
        c["_variants"] = raw["variants"]
    return cases


def load_models(path: str | None = None) -> dict[str, Any]:
    path = path or os.path.join(ROOT, "config", "models.yaml")
    return _load_yaml(path)


def model_config(model_key: str, models: dict | None = None) -> dict:
    models = models or load_models()
    if model_key not in models["models"]:
        raise KeyError(f"Unknown model {model_key!r}. Known: {list(models['models'])}")
    cfg = dict(models["models"][model_key])
    # merge default sampling under model-specific sampling
    sampling = dict(models.get("defaults", {}).get("sampling", {}))
    sampling.update(cfg.get("sampling", {}))
    cfg["sampling"] = sampling
    return cfg


def load_prompts(case_cfg: dict) -> list[dict]:
    path = os.path.join(ROOT, case_cfg["prompts_file"])
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def render_prompt(case_cfg: dict, variant: str, fields: dict) -> str:
    variants = case_cfg["_variants"]
    if variant not in variants:
        raise KeyError(f"Unknown variant {variant!r}. Known: {list(variants)}")
    nudge = variants[variant].get("nudge", "") or ""
    if "{explicit_target}" in nudge:
        nudge = nudge.format(explicit_target=case_cfg.get("explicit_target_text", "").strip())
    meta = case_cfg["metadata_instruction"].strip()
    template = case_cfg["template"].strip()
    return template.format(meta=meta, nudge=nudge, **fields).strip()
