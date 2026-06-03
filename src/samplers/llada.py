"""LLaDA-8B-Instruct sampler wrapper (iterative denoising, not autoregressive).

The denoising loop is a faithful port of the official reference generate() from
https://github.com/ML-GSAI/LLaDA (block-wise, low-confidence remasking, Gumbel-noise
sampling). Heavy imports (torch/transformers) are deferred to __init__ so importing this
module never breaks a dry-run-only environment. Run on a >=24GB bf16 GPU with
transformers==4.38.2. Confirm a single prompt yields coherent, parseable output before
scaling (spec step 7).
"""

from __future__ import annotations

from .base import DiffusionSampler, SamplingParams, GenContext

MASK_ID = 126336  # LLaDA mask token id


def _add_gumbel_noise(logits, temperature, torch):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def _num_transfer_tokens(mask_index, steps, torch):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    out = torch.zeros(mask_num.size(0), steps, device=mask_index.device,
                      dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        out[i, : remainder[i]] += 1
    return out


class LLaDASampler(DiffusionSampler):
    backend = "llada"

    def __init__(self, model_cfg: dict):
        super().__init__(model_cfg)
        import torch
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
        self.mask_id = getattr(self.model.config, "mask_token_id", MASK_ID)

    def _build_input(self, prompt: str):
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

    def _denoise(self, prompt, params: SamplingParams):
        """Faithful port of the official LLaDA generate() (batch supported, cfg_scale=0)."""
        import numpy as np
        import torch.nn.functional as F
        torch = self.torch

        gen_length = params.gen_length
        block_length = params.block_length
        steps = params.steps
        temperature = params.temperature
        remasking = params.remasking
        mask_id = self.mask_id

        # The official loop requires these to divide evenly; round to safe values.
        if gen_length % block_length != 0:
            block_length = gen_length  # single block fallback
        num_blocks = gen_length // block_length
        if steps % num_blocks != 0:
            steps = (steps // num_blocks) * num_blocks or num_blocks
        steps_per_block = steps // num_blocks

        with torch.no_grad():
            x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id,
                           dtype=torch.long, device=self.model.device)
            x[:, : prompt.shape[1]] = prompt.clone()
            P = prompt.shape[1]

            for nb in range(num_blocks):
                blk = x[:, P + nb * block_length: P + (nb + 1) * block_length] == mask_id
                ntt = _num_transfer_tokens(blk, steps_per_block, torch)
                for i in range(steps_per_block):
                    mask_index = (x == mask_id)
                    logits = self.model(x).logits
                    logits_noisy = _add_gumbel_noise(logits, temperature, torch)
                    x0 = torch.argmax(logits_noisy, dim=-1)

                    if remasking == "low_confidence":
                        p = F.softmax(logits.to(torch.float64), dim=-1)
                        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                    elif remasking == "random":
                        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                    else:
                        raise NotImplementedError(remasking)

                    # never reveal beyond the current block
                    x0_p[:, P + (nb + 1) * block_length:] = -np.inf
                    x0 = torch.where(mask_index, x0, x)
                    confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, -np.inf))

                    transfer = torch.zeros_like(x0, dtype=torch.bool)
                    for j in range(confidence.shape[0]):
                        k = int(ntt[j, i].item())
                        if k > 0:
                            _, sel = torch.topk(confidence[j], k=k)
                            transfer[j, sel] = True
                    x[transfer] = x0[transfer]
            return x
