"""LLaDA-8B-Instruct sampler wrapper (iterative denoising, not autoregressive).

Heavy imports (torch/transformers) are deferred to __init__ so that importing this module
never breaks a dry-run-only environment. Run on a >=24GB bf16 GPU with transformers==4.38.2.

The denoising loop here follows LLaDA's reference generate() shape (block-wise low-confidence
remasking). Confirm a single prompt produces coherent, parseable output before scaling
(spec step 7). If you have the official `generate` from the LLaDA repo on PYTHONPATH, prefer
wiring that in at `_denoise` instead of this minimal reimplementation.
"""

from __future__ import annotations

from .base import DiffusionSampler, SamplingParams, GenContext


class LLaDASampler(DiffusionSampler):
    backend = "llada"

    def __init__(self, model_cfg: dict):
        super().__init__(model_cfg)
        import torch  # noqa: F401  (fail loudly here if torch missing)
        import transformers
        from transformers import AutoModel, AutoTokenizer

        want = model_cfg.get("transformers_version")
        if want and transformers.__version__ != want:
            raise RuntimeError(
                f"LLaDA expects transformers=={want}, found {transformers.__version__}. "
                "Install requirements-gpu.txt with the pinned version.")

        self.torch = torch
        repo = model_cfg["hf_repo"]
        dtype = getattr(torch, model_cfg.get("dtype", "bfloat16"))
        trust = model_cfg.get("trust_remote_code", True)
        self.tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=trust)
        self.model = AutoModel.from_pretrained(
            repo, torch_dtype=dtype, trust_remote_code=trust,
        ).to("cuda").eval()
        self.mask_id = getattr(self.model.config, "mask_token_id", 126336)

    def _build_input(self, prompt: str):
        # LLaDA-Instruct expects the chat template.
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)
        return self.tokenizer(text, return_tensors="pt").input_ids.to("cuda")

    def generate(self, prompt: str, *, seed: int, params: SamplingParams,
                 ctx: GenContext) -> str:
        torch = self.torch
        torch.manual_seed(seed)
        input_ids = self._build_input(prompt)
        out = self._denoise(input_ids, params)
        gen = out[0, input_ids.shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()

    def _denoise(self, input_ids, params: SamplingParams):
        """Block-wise iterative denoising with low-confidence remasking."""
        torch = self.torch
        with torch.no_grad():
            B, L0 = input_ids.shape
            gen_len = params.gen_length
            block = max(1, params.block_length)
            x = torch.cat([
                input_ids,
                torch.full((B, gen_len), self.mask_id, dtype=torch.long, device=input_ids.device),
            ], dim=1)
            n_blocks = (gen_len + block - 1) // block
            steps_per_block = max(1, params.steps // n_blocks)

            for b in range(n_blocks):
                lo = L0 + b * block
                hi = min(L0 + (b + 1) * block, L0 + gen_len)
                for _ in range(steps_per_block):
                    mask = (x == self.mask_id)
                    mask[:, :lo] = False
                    mask[:, hi:] = False
                    if not mask.any():
                        break
                    logits = self.model(x).logits
                    if params.temperature and params.temperature > 0:
                        probs = torch.softmax(logits / params.temperature, dim=-1)
                        pred = torch.multinomial(
                            probs.view(-1, probs.shape[-1]), 1).view(x.shape)
                        conf = probs.max(dim=-1).values
                    else:
                        conf, pred = torch.softmax(logits, dim=-1).max(dim=-1)
                    # Reveal the highest-confidence masked positions; keep the rest masked.
                    conf = torch.where(mask, conf, torch.full_like(conf, -1.0))
                    k = max(1, int(mask.sum().item() / max(1, steps_per_block)))
                    for row in range(B):
                        idx = torch.topk(conf[row], k=min(k, int(mask[row].sum().item()))).indices
                        x[row, idx] = pred[row, idx]
            return x
