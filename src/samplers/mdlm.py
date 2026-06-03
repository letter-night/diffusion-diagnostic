"""MDLM sampler wrapper (secondary, smaller, base-trained -> less coherent).

Stub wiring: MDLM's reference sampling lives in the kuleshov-group/mdlm repo and is not a
drop-in HF generate(). This wrapper documents the integration point and fails loudly until
the repo's sampler is wired in, so dry-run remains the path for pipeline testing.
"""

from __future__ import annotations

from .base import DiffusionSampler, SamplingParams, GenContext


class MDLMSampler(DiffusionSampler):
    backend = "mdlm"

    def __init__(self, model_cfg: dict):
        super().__init__(model_cfg)
        import torch  # noqa: F401
        self.torch = torch
        # TODO: load MDLM checkpoint + tokenizer from model_cfg["hf_repo"] using the
        # kuleshov-group/mdlm reference code (clone the repo, import its sampler).

    def generate(self, prompt: str, *, seed: int, params: SamplingParams,
                 ctx: GenContext) -> str:
        raise NotImplementedError(
            "MDLM sampling is not wired up. Integrate kuleshov-group/mdlm's reference "
            "sampler here, or use --model dry_run / --model llada.")
