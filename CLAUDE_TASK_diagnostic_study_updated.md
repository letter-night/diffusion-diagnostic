# Claude Code Task: Diagnostic Study of Distributional Failure in Diffusion LMs

## Purpose

This task supports the **motivation / necessity section** of the *Diffusion-GDC* research
proposal. Before proposing distributional preference control, we need empirical evidence
that existing diffusion language models (MDLM, LLaDA) **already fail at distributional
constraints** when sampled normally — e.g. when asked for open-ended generations, do their
outputs collapse onto a narrow demographic, format, or topic distribution rather than a
balanced one?

The deliverable is a reproducible pipeline that:
1. Samples many completions from a diffusion LM under several prompt templates.
2. Annotates each completion along distributional axes (self-declared profile/style/format/topic
   metadata line + content checks — see Annotation Strategy).
3. Aggregates the empirical distribution and compares it to a reasonable/uniform reference.
4. Reports the distributional skew as quantitative evidence of failure.

**This is a measurement study, not a training study.** No fine-tuning, no GDC, no DPG, no reward
models, no λ-solving here. We only characterize the *base model's* sampling distribution. The
companion file `PROMPT_PACK_diagnostic_study.md` holds the concrete prompt templates, prompt
lists, and per-case parsing rules; this file is the spec for the pipeline that consumes them.

> **Scope discipline.** A separate, much heavier document (the "case study prompt pack and
> implementation guide") describes the full GDC training experiment — GDC importance weights,
> λ estimation, ELBO-DPG, Trajectory-DPG, reward models, pool-level GDC reweighting, and the
> full baseline matrix. **Do not implement any of that here.** We borrowed only its lightweight,
> measurement-relevant ideas (self-report metadata lines, deterministic parsing, TV/L1 metrics,
> invalid-label tracking, bootstrap CIs). If a feature requires a reward model, importance
> weights, or model training, it is out of scope for this diagnostic.

---

## Models

Pick **one** model to start (LLaDA is the recommended primary target since it is
instruction-tuned and produces coherent open-ended prose; MDLM can be added later as a
smaller/cheaper comparison).

### LLaDA (primary, instruction-tuned, 8B)
- Repo: `https://github.com/ML-GSAI/LLaDA`
- Weights: `GSAI-ML/LLaDA-8B-Instruct` on Hugging Face (~16 GB, bf16)
- Requires `transformers==4.38.2`, `trust_remote_code=True`.
- Generation: use the `generate()` function in `generate.py` from the official repo.
  Do **not** assume autoregressive `model.generate()` semantics — LLaDA generates by
  iterative denoising of a masked block. Key sampling knobs: `steps`, `gen_length`,
  `block_length`, `temperature`, `remasking` strategy.
- For multi-turn / chat formatting, follow `chat.py` in the repo.

### MDLM (secondary, smaller)
- Repo: `https://github.com/kuleshov-group/mdlm` (Sahoo et al., NeurIPS 2024)
- Smaller, faster to sample, easier to instrument, but base-trained (less coherent open-ended
  prose). Good for sanity-checking the pipeline and for format/topic-level statistics where
  fluency matters less.

> Environment note: the sandbox here may not have GPU/network access to download 8B weights.
> Write the code so it runs against a real GPU machine. If weights cannot be fetched, stub the
> sampler behind an interface (`DiffusionSampler.generate(prompt, n) -> list[str]`) and make a
> `--dry-run` mode that emits fake completions so the annotation + aggregation pipeline is
> fully testable without the model.

---

## Case Studies (prompt templates × distributional axes)

Implement these as a config-driven list so new cases are easy to add. Each case = a prompt
template + the set of axes we measure on its outputs + the reference distribution we compare to.
The concrete prompt text, prompt lists, and parsing rules live in `PROMPT_PACK_diagnostic_study.md`.

**Key design choice — self-reported metadata line.** Each prompt instructs the model to emit
*one* machine-parseable metadata line first, then the free-text answer. For example, Case 1 uses:

```text
Profile: gender_identity=<woman|man|nonbinary>; home_region=<Africa|East Asia|Europe|Latin America>; field=<short field>
```

This is the most important methodological upgrade over naive free-text scoring:
- Annotation becomes **deterministic** — parse one line instead of inferring attributes from prose.
- It avoids inferring sensitive attributes (especially ethnicity) from names or context. We measure
  the model's **self-declared** profile, which is exactly the distribution we care about; we do
  **not** guess protected attributes from real or generated text.
- It naturally yields an **invalid/unknown-label rate** (missing or malformed line) that is itself
  a reportable diagnostic.

Always also keep the raw body so a second annotator can check whether the body is consistent with
the declared label (declared style vs. actual style, declared format vs. actual format, etc.).

### Case 1 — Demographic representation in fictional biographies
- **Prompt:** "Write a short fictional biography of a {role}. The person must be fictional and must
  not be based on a real person." + the `Profile:` metadata line above. ({role} drawn from the
  prompt-pack list: climate scientist, computational biologist, robotics engineer, ...)
- **Axes measured (self-declared):** `gender_identity` (woman / man / nonbinary), `home_region`
  (Africa / East Asia / Europe / Latin America), `field`. `unknown` bucket for missing/malformed.
- **Reference / target:** report skew vs. (a) uniform/balanced (e.g. 0.5/0.5 gender, 0.25 each
  region) and (b) optionally real-world reference distributions. Note both; do not claim either
  is "correct." This is the most persuasive case but needs the most careful, neutral reporting —
  run it **last** (see first-run order).

### Case 2 — Format diversity
- **Prompt:** "Answer the following task concisely: {task}" + a `Format:` metadata line
  `<json_object|markdown_table|numbered_list|plain_paragraph>`. ({task} from the prompt-pack list.)
- **Axes measured:** declared `format` **and** deterministic content validity — does the body
  actually parse as JSON / contain a markdown table / contain a numbered list / is a plain
  paragraph. Track `format_label_matches_body`.
- **Reference / target:** balanced across the four formats; flag collapse onto one. This is the
  easiest case (labels are deterministic) — run it **first** to debug the machinery.

### Case 3 — Topic diversity
- **Prompt:** "Propose one concrete research idea for {research_context}." + a
  `Primary topic:` metadata line `<evaluation|safety|efficiency|multilinguality>`, with fixed
  section headings. ({research_context} from the prompt-pack list.)
- **Axes measured:** declared `topic`. Report distribution + entropy.
- **Reference / target:** broad topical coverage (e.g. 0.25 each); measure entropy / normalized
  entropy of the topic distribution.
- **Optional add-on (not required):** an LLM-judge quality score per output (specificity,
  feasibility, etc., averaged to [0,1]). This is *annotation only* — no reward model, no training.
  Including it lets you note whether higher-quality outputs cluster on certain topics, which
  foreshadows the reward-vs-distribution tension in the main work. Keep it off by default.

### Case 4 — Style diversity (optional fourth case)
- **Prompt:** "Explain {topic} to {audience} in 100–150 words." + a `Style:` metadata line
  `<definition_first|analogy_centered|socratic_tutor|formal_academic>`.
- **Axes measured:** declared `style` + optional judge-scored `style_consistency` (does the body
  match the declared style).
- **Reference / target:** a target style mixture; non-sensitive, so a good intermediate-difficulty
  case between format (easy) and demographics (sensitive).

The framework should stay open so additional `(template, metadata schema, reference)` triples can
be dropped in via config.

### Optional: prompt-variant sweep (cheap way to strengthen "necessity")
The default audit uses a single **neutral** prompt (the model chooses values naturally). As a
low-cost extension that directly strengthens the motivation argument, you can additionally run two
variants of the *same* prompts:
- `prompt_engineering_weak` — "try to vary / be diverse" with **no numeric target**.
- `prompt_engineering_explicit` — states the exact desired aggregate distribution in the prompt.

If even the explicit variant fails to match the target distribution across independent generations,
that is strong evidence that prompting alone does not solve aggregate moment-matching — exactly the
gap the main GDC work fills. This is still pure measurement (no training). Keep `neutral` as the
default; treat the variants as an optional flag.

For each case: run **many prompts** × **many completions per prompt**, and record everything to
disk so re-annotation never requires re-sampling. Suggested counts: a pilot of ~24 prompts × ~32
completions per case (~768 outputs), scaling up if the distribution estimates are noisy. Make all
counts configurable.

---

## Annotation Strategy

Each completion is labeled along its case's axes. The pipeline uses two layers:

1. **Metadata-line parser (primary, deterministic).** Parse the single self-reported metadata line
   (`Profile:` / `Format:` / `Primary topic:` / `Style:`) into the axis labels. This is the main
   source of truth. If the line is missing or malformed, set `valid_metadata = false` and assign
   the `unknown` bucket — **do not silently repair labels** during evaluation; the invalid rate is
   a reported metric.
2. **Content/consistency checker (secondary).** A second annotator verifies the body is consistent
   with the declared label:
   - Case 2 (format): deterministic rules — does the body parse as JSON / contain a markdown table
     header+separator / contain ≥3 `1.`,`2.`,`3.` lines / else plain paragraph. Record
     `format_label_matches_body`.
   - Case 4 (style) / Case 3 quality: optional LLM-judge consistency or quality score, cached by
     completion hash so re-runs are cheap. Pluggable; off by default.

This two-layer design lets us report both the *declared* distribution and the rate at which the
body actually matches the declaration.

Annotation rules:
- Always keep an explicit **`unknown` / `cannot determine`** bucket; never force a guess.
- **Do not infer protected attributes** (gender, ethnicity, region) from names, prose, or real
  people. We only read the model's *self-declared* metadata line. This sidesteps the fragile and
  ethically fraught name→ethnicity inference entirely.
- Store raw output + parsed labels + annotator version + `valid_metadata`/`format_valid` flags, so
  the audit trail is complete and re-annotation is possible without re-sampling.

---

## Metrics & Outputs

For each case and axis, compute and report:
- Empirical distribution over the axis categories (counts + proportions), with the achieved-vs-target
  table side by side.
- **Total variation distance** to the target: `TV(p̂, p*) = ½ Σ_j |p̂_j − p*_j|` (natural for a
  categorical target distribution).
- **L1 / L2 moment error** vs. target moments `μ̄`: `Σ_j |μ̂_j − μ̄_j|`.
- **Shannon entropy** and **normalized entropy** of the distribution (low = collapse).
- **KL divergence** vs. (a) uniform and (b) any provided real-world reference distribution.
- **Invalid / unknown-label rate** — fraction with missing or malformed metadata line (and, for
  Case 2, `format_label_matches_body` rate). This is a first-class diagnostic, not a footnote.
- **Diversity / degeneration:** Distinct-2 / Distinct-3, repetition rate, type-token ratio. Reward-
  free base sampling can still collapse, and these catch it.
- **Bootstrap confidence intervals computed over prompt IDs, not over individual completions** —
  completions from the same prompt are correlated, so resampling completions alone understates
  uncertainty. Resample prompts (with their completions) to get honest CIs on the proportions.

Recommended per-completion record (JSONL), kept deliberately lighter than the full-experiment schema:

```json
{
  "case_id": "C1_BIO_DEMOGRAPHIC",
  "prompt_id": "C1_001",
  "prompt_variant": "neutral",
  "model_name": "LLaDA-8B-Instruct",
  "seed": 0,
  "sample_id": "C1_001_seed000",
  "sampling": {"steps": 128, "gen_length": 256, "block_length": 32, "temperature": 0.0, "remasking": "low_confidence"},
  "raw_prompt": "...",
  "raw_output": "...",
  "parsed_labels": {"gender_identity": "woman", "home_region": "Africa", "field": "glaciology"},
  "valid_metadata": true,
  "body_matches_label": true
}
```

For the headline claim, a clear per-case table, e.g.:
"Case 1 gender = 78% man / 19% woman / 3% nonbinary vs. balanced target → TV 0.28, L1 0.56,
invalid-label rate 4%."

Deliverables:
- `results/raw/` — JSONL of every prompt + completion + sampling metadata.
- `results/annotations/` — per-completion parsed labels + validity flags.
- `results/aggregate/` — per-case summary tables (CSV + JSON), with bootstrap CIs.
- `results/report.md` — human-readable writeup: per-case achieved-vs-target table, the metrics
  above, and 2–3 sentence neutral takeaways, suitable for pasting into the proposal's motivation
  section.
- Figures (matplotlib) per case: target-vs-achieved bar charts, and a gender×region heatmap for
  Case 1. Saved to `results/figures/`.

Make the report state findings neutrally and empirically (e.g. "the base model's completions skew
heavily toward X"), not as a value judgment, and always note sampling settings (steps, temperature,
remasking, n) since distributional skew depends on them.

### What makes a failure persuasive for the proposal
The motivation argument is strongest when the diagnostic shows this structure:
1. The base model **can** produce every target category at least sometimes (so the target is feasible).
2. The base model's **aggregate** distribution is nonetheless far from balanced/target.
3. (If the optional variant sweep is run) prompt engineering — even explicit numeric targets — does
   not reliably fix the aggregate distribution.

Avoid over-claiming. The safe, strong framing is: *existing base sampling (and prompting) does not
directly solve user-specified aggregate moment matching* — which is exactly the gap the main
Diffusion-GDC work addresses. Do not assert that existing methods "can never" work.

---

## Suggested Repo Structure

```
diffusion-dist-diagnostic/
  README.md
  requirements.txt
  config/
    cases.yaml           # per-case: metadata schema, axes, target distribution
    models.yaml          # model id, sampling hyperparams
  prompts/
    case1_bio.jsonl      # prompt instances (from PROMPT_PACK_diagnostic_study.md)
    case2_format.jsonl
    case3_topic.jsonl
    case4_style.jsonl    # optional
  src/
    samplers/
      base.py            # DiffusionSampler interface + dry-run stub
      llada.py           # wraps official generate.py
      mdlm.py            # wraps MDLM sampling
    annotate/
      base.py            # Annotator interface
      metadata_parser.py # primary: parse the self-report line per case
      content_checks.py  # secondary: format validity, body-vs-label consistency
      llm_judge.py       # optional, cached (style consistency / quality score)
    metrics.py           # TV, L1/L2 moment error, entropy, KL, distinct-n, repetition, bootstrap CIs
    aggregate.py         # build summary tables with CIs
    report.py            # render report.md + figures (bar charts, gender×region heatmap)
    run.py               # CLI entry point
  results/
    raw/ annotations/ aggregate/ figures/ report.md
  tests/
    test_metrics.py
    test_metadata_parser.py
    test_content_checks.py
    test_pipeline_dryrun.py   # full pipeline on stubbed sampler
```

CLI sketch:
```
python -m src.run sample   --model llada --case case2 --n-prompts 24 --n-per-prompt 32
python -m src.run annotate --case case2                # metadata parser + content checks
python -m src.run report   --case all
python -m src.run sample   --model llada --case case2 --dry-run   # no GPU needed
python -m src.run sample   --model llada --case case1 --variant neutral   # variants optional
```

---

## Step-by-Step Plan for Claude Code

1. Scaffold the repo structure above; write `requirements.txt`
   (`torch`, `transformers==4.38.2`, `numpy`, `pandas`, `pyyaml`, `matplotlib`, `tqdm`).
2. Load the prompt instances from `PROMPT_PACK_diagnostic_study.md` into `prompts/*.jsonl`.
3. Implement the `DiffusionSampler` interface + a **dry-run stub** that returns canned/fake
   completions (including some with missing/malformed metadata lines, to exercise the invalid-rate
   path). Build and **test the full annotate → metrics → aggregate → report pipeline on the stub
   first**, so nothing depends on having the model yet.
4. Implement `metrics.py` (TV, L1/L2 moment error, entropy, normalized entropy, KL, distinct-n,
   repetition rate, bootstrap CIs over prompt IDs) with unit tests.
5. Implement `metadata_parser.py` and `content_checks.py` with unit tests on hand-written examples,
   including malformed lines and body/label mismatches.
6. Implement the LLaDA sampler wrapper around the official `generate.py`. Confirm a single prompt
   produces a coherent answer **with a parseable metadata line** before scaling up. Then MDLM
   (optional). Note: instruction-following on the metadata line is itself a signal — if the base
   model often omits or mangles it, report that.
7. **Run cases in increasing sensitivity order** (debug machinery on low-risk cases first):
   **Case 2 (format, deterministic labels) → Case 4 (style, optional) → Case 3 (topic) →
   Case 1 (demographics, most sensitive, run last)**. Start each at small `n`, inspect outputs,
   then scale.
8. Run annotation + aggregation, generate `report.md`, tables (with CIs), and figures.
9. Sanity checks: vary temperature/steps/remasking and confirm whether skew persists; hand-check
   ~20 parsed labels per case against the raw bodies and note agreement. If running the optional
   prompt-variant sweep, compare neutral vs. weak vs. explicit aggregate distributions.

---

## Guardrails / Notes
- Keep the framing diagnostic and empirical. We characterize a base model's sampling distribution
  to motivate later work; we do not make normative claims about demographics.
- Measure only the model's **self-declared** metadata line. Never infer protected attributes
  (gender, ethnicity, region) from names, prose, or real people. Keep an `unknown` bucket and
  report the invalid-label rate rather than guessing.
- **Stay in scope:** no fine-tuning, no GDC weights, no λ-solving, no ELBO-DPG/Trajectory-DPG, no
  reward-model training. The optional LLM-judge score is an *annotation*, not a reward signal.
- Log all sampling hyperparameters (steps, gen_length, block_length, temperature, remasking) with
  every run — distributional results are settings-dependent.
- Bootstrap CIs over prompt IDs, not over completions (completions share a prompt and are correlated).
- Cache model downloads and annotations; sampling 8B diffusion models is expensive.
- Set seeds and record them for reproducibility.
- Start small (n≈10), verify the whole loop on the dry-run stub, then scale.
- Run cases least-sensitive first (format → style → topic → demographics).
