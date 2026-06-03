# Diffusion Distributional Diagnostic

A **measurement-only** study characterizing whether diffusion language models (LLaDA-8B-Instruct
primary, MDLM secondary) collapse onto narrow distributions when sampled normally. It produces
quantitative evidence that base sampling — and prompting — do not directly solve user-specified
**aggregate moment-matching**, motivating later Diffusion-GDC work.

> Scope: **no fine-tuning, no GDC, no reward models, no λ-solving.** This repo only samples,
> annotates self-declared metadata, and aggregates the empirical distribution vs. a reference.

## Findings (LLaDA-8B-Instruct)

Real run: **576 completions** (12 prompts × 4 seeds per case×variant), bf16 on an A40,
official LLaDA denoising (`steps=128, gen_length=256, block_length=32, temperature=0,
remasking=low_confidence`). Full report, tables, and figures: [`results/llada/`](results/llada/).

**Total variation distance vs. target** (0 = matches target; 0.75/0.70 ≈ collapse onto one bucket):

| case / axis | neutral | weak | explicit |
| --- | --- | --- | --- |
| C2 format | **0.750** | **0.750** | 0.357 |
| C4 style | **0.700** | **0.700** | **0.700** |
| C3 topic | 0.365 | 0.500 | 0.333 |
| C1 gender | 0.467 | 0.450 | 0.300 |
| C1 region | 0.500 | 0.568 | 0.409 |

**Dominant bucket under neutral prompting:** format → **100% plain_paragraph** (target 25%);
style → **100% analogy_centered** (target 30%); topic → **62% evaluation** (target 25%);
gender → **92% man** (target 45%); region → **50% East Asia** (target 25%).

Takeaways:
1. **Real distributional collapse** — two axes fall onto a single bucket 100% of the time.
2. **Prompting does not reliably fix it.** "Try to vary" (weak) helped nowhere (and made
   topic/region slightly worse); explicit numeric targets helped some axes but did **nothing**
   for style (0.70 → 0.70) and never reached target on any axis — exactly the gap Diffusion-GDC
   motivates.
3. **Label-vs-content gap** — under explicit pressure on Case 2 the model *declares* varied
   formats but the body stops conforming (`format_label_matches_body` → 0%).
4. **Metadata reliability** — invalid-label rate **0%** across all cases on the real model.

> Pilot caveat: n=48 per case×variant gives wide CIs on small buckets; scale `--n-per-prompt`
> for tighter intervals. The synthetic dry-run example (`results/`) only validates the pipeline.

## How it works

Each prompt asks the model to emit **one machine-parseable metadata line first**, then the
free-text body. Annotation parses that line deterministically (never inferring protected
attributes from names or prose), aggregates the empirical distribution per case/axis, and
compares it to a reference target using TV distance, L1/L2 moment error, entropy, KL, and an
invalid-label rate — all with **bootstrap CIs over prompt IDs** (not over correlated completions).

### Cases (run least-sensitive first)

| order | case | axis | target |
| --- | --- | --- | --- |
| 1 | `C2_FORMAT` | format | 25% each: json / table / numbered list / paragraph |
| 2 | `C4_STYLE` | style | 30/30/20/20: definition / analogy / socratic / formal |
| 3 | `C3_TOPIC` | topic | 25% each: evaluation / safety / efficiency / multilinguality |
| 4 | `C1_BIO_DEMOGRAPHIC` | gender, region | 45/45/10 gender; 25% each of 4 regions |

Three prompt **variants** per case: `neutral` (no target), `weak` ("try to vary"),
`explicit` (states the numeric target). If even `explicit` fails, that strengthens the
argument that prompting alone doesn't fix aggregate moment-matching.

## Quickstart (no GPU — dry-run stub)

The dry-run sampler fabricates *skewed* completions (incl. ~4% malformed metadata), so the
full annotate → metrics → aggregate → report pipeline is testable anywhere.

```bash
pip install -r requirements.txt
pytest -q                                   # unit + end-to-end dry-run tests

python -m src.run sample   --model dry_run --case all --variant all --n-prompts 24 --n-per-prompt 32
python -m src.run annotate --model dry_run --case all --variant all
python -m src.run report   --model dry_run --case all --variant all
# -> results/report.md, results/aggregate/*.{json,csv}, results/figures/*.png
```

## Real run (RunPod GPU)

See `CLAUDE_TASK_diagnostic_study.md` for the full pod checklist. In short: RTX 4090 / A40 /
A100 (>=24 GB), Community Cloud, pin `transformers==4.38.2`, load bf16 (not quantized).

```bash
pip install -r requirements.txt -r requirements-gpu.txt
# Confirm a single coherent, parseable completion before scaling:
python -m src.run sample --model llada --case case2 --n-prompts 1 --n-per-prompt 1
# Then scale, run cases in ascending sensitivity, download results/, TERMINATE the pod.
python -m src.run sample --model llada --case all --variant all
python -m src.run annotate --model llada --case all --variant all
python -m src.run report   --model llada --case all --variant all
```

Sampling is resumable: each completion is written to `results/raw/*.jsonl` immediately, and
re-running `sample` skips `sample_id`s already present. All sampling hyperparameters
(steps, gen_length, block_length, temperature, remasking) are logged with every record.

## Layout

```
config/      cases.yaml (templates, axes, targets), models.yaml (sampling defaults)
prompts/     case{1..4}_*.jsonl prompt instances
src/
  samplers/  base.py (interface + DryRunSampler), llada.py, mdlm.py
  annotate/  metadata_parser.py, content_checks.py, llm_judge.py (annotation-only)
  metrics.py aggregate.py report.py run.py (CLI)
tests/       metrics, metadata parser, content checks, end-to-end dry-run
results/     dry-run example (report.md, aggregate/, figures/); raw/ + annotations/ gitignored
results/llada/   real LLaDA-8B-Instruct outputs (report.md, aggregate/, figures/, README.md)
```

## Guardrails

- Measure only **self-declared** metadata; never infer protected attributes from names/prose/real people.
- Always keep an explicit `unknown` bucket; invalid labels are reported, **never repaired**.
- Log all sampling hyperparameters; set + record seeds; bootstrap CIs over prompt IDs.
- The optional LLM judge (Case 3) is **annotation-only** — never a reward or filter.
- Framing stays diagnostic: this characterizes a sampling distribution; it does not claim
  existing methods can *never* match a target.
