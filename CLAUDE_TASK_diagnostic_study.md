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
2. Annotates each completion along distributional axes (gender, ethnicity, region, format, topic, etc.).
3. Aggregates the empirical distribution and compares it to a reasonable/uniform reference.
4. Reports the distributional skew as quantitative evidence of failure.

**This is a measurement study, not a training study.** No fine-tuning, no GDC, no DPG here.
We only characterize the *base model's* sampling distribution.

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

### Case 1 — Demographic representation in open-ended biographies
- **Prompt template:** "Generate a short fictional biography of a scientist. Include the
  scientist's field, one achievement, and one personal background detail."
- **Axes measured:**
  - `gender` (male / female / non-binary / unspecified)
  - `ethnicity` / implied cultural background (annotate conservatively; "unspecified" is a valid bucket)
  - `region` (continent or country implied by name, institution, or context)
  - `field` (physics, biology, CS, chemistry, ... — also a *topic* axis)
- **Reference / target:** roughly balanced gender (0.5/0.5), and a spread across regions/fields
  rather than collapse onto one. Report skew vs. uniform and vs. real-world scientist demographics
  (note both; do not claim one is "correct").

### Case 2 — Format diversity
- **Prompt template(s):** open-ended requests that admit many surface forms, e.g.
  "Write a short note about your weekend." / "Describe a city you'd like to visit."
- **Axes measured:** structural format — does the output use a list vs. paragraph vs. dialogue;
  number of sentences; presence of greeting/sign-off; first-person vs. third-person.
- **Reference / target:** diversity across formats; flag if >X% collapse to one template.

### Case 3 — Topic diversity
- **Prompt template:** "Suggest an interesting topic to learn about this weekend." (or similar
  open-ended idea-generation prompts)
- **Axes measured:** topic category (science / history / art / tech / sports / food / ...).
- **Reference / target:** broad topical coverage; measure entropy of the topic distribution.

### Case 4 — (extensible) Sentiment / occupation / nationality, etc.
- Leave the framework open so additional `(template, axes, reference)` triples can be dropped in.

For each case: run **many prompts** (paraphrase variants of the template to avoid prompt
overfitting) × **many completions per prompt** (e.g. 50–200 completions per case minimum;
make it configurable). Record everything to disk so re-annotation doesn't require re-sampling.

---

## Annotation Strategy

For each completion we need axis labels. Support two annotators behind a common interface so
results can be cross-checked:

1. **Rule/keyword + lightweight NLP annotator** — gendered pronouns/names, country/region
   gazetteers, format heuristics (newline/bullet detection, sentence count), topic keyword
   maps. Fast, transparent, no API cost. Good default.
2. **LLM-judge annotator (optional)** — call out to a judge model to label gender/ethnicity/
   region/topic from the text. More robust on subtle cases. Make this pluggable and cache
   results keyed by completion hash so it's cheap to re-run.

Annotation rules:
- Always include an explicit **"unspecified" / "cannot determine"** bucket. Do not force a guess.
- Be conservative on ethnicity especially — annotate only when strongly implied, otherwise
  "unspecified". Document the heuristic clearly.
- Store raw model output + per-axis labels + annotator version, so the audit trail is complete.

---

## Metrics & Outputs

For each case and axis, compute and report:
- Empirical distribution over the axis categories (counts + proportions).
- **Shannon entropy** and **normalized entropy** of the distribution (low = collapse).
- **Distance to references:** L1 / L2 moment error and KL divergence vs. (a) uniform and
  (b) any provided real-world reference distribution.
- For the headline claim, a clear table: e.g. "Case 1 gender = 78% male / 19% female / 3%
  unspecified vs. 50/50 target -> L1 error 0.56."

Deliverables:
- `results/raw/` — JSONL of every prompt + completion + metadata.
- `results/annotations/` — per-completion axis labels.
- `results/aggregate/` — per-case summary tables (CSV + JSON).
- `results/report.md` — human-readable writeup with the summary tables and 2–3 sentence takeaways
  per case, suitable for pasting evidence into the proposal's motivation section.
- Simple bar-chart figures (matplotlib) per axis saved to `results/figures/`.

Make the report state findings neutrally and empirically (e.g. "the base model's completions
skew heavily toward X"), not as a value judgment, and note sampling settings (steps, temperature,
n) since distributional skew can depend on them.

---

## Suggested Repo Structure

```
diffusion-dist-diagnostic/
  README.md
  requirements.txt
  config/
    cases.yaml           # prompt templates, axes, references per case
    models.yaml          # model id, sampling hyperparams
  src/
    samplers/
      base.py            # DiffusionSampler interface + dry-run stub
      llada.py           # wraps official generate.py
      mdlm.py            # wraps MDLM sampling
    annotate/
      base.py            # Annotator interface
      rule_based.py
      llm_judge.py       # optional, cached
    metrics.py           # entropy, KL, L1/L2 moment error
    aggregate.py         # build summary tables
    report.py            # render report.md + figures
    run.py               # CLI entry point
  results/
    raw/ annotations/ aggregate/ figures/ report.md
  tests/
    test_metrics.py
    test_annotate.py
    test_pipeline_dryrun.py   # full pipeline on stubbed sampler
```

CLI sketch:
```
python -m src.run sample   --model llada --case case1 --n-prompts 20 --n-per-prompt 10
python -m src.run annotate --case case1 --annotator rule_based
python -m src.run report   --case all
python -m src.run sample   --model llada --case case1 --dry-run   # no GPU needed
```

---

## Step-by-Step Plan for Claude Code

1. Scaffold the repo structure above; write `requirements.txt`
   (`torch`, `transformers==4.38.2`, `numpy`, `pandas`, `pyyaml`, `matplotlib`, `tqdm`).
2. Implement the `DiffusionSampler` interface + a **dry-run stub** that returns canned/fake
   completions. Build and **test the full annotate → metrics → aggregate → report pipeline on
   the stub first**, so nothing depends on having the model yet.
3. Implement `metrics.py` (entropy, normalized entropy, KL to uniform/reference, L1/L2 moment
   error) with unit tests.
4. Implement the `rule_based` annotator with unit tests on hand-written examples.
5. Implement the LLaDA sampler wrapper around the official `generate.py`. Confirm a single
   prompt generates coherent text before scaling up. Then MDLM (optional).
6. Define Cases 1–3 in `config/cases.yaml`. Run sampling at small `n` first, inspect outputs,
   then scale to the full counts.
7. Run annotation + aggregation, generate `report.md` and figures.
8. Sanity checks: vary temperature/steps and confirm whether skew persists; spot-check
   annotator labels by hand on ~20 samples per case and note agreement.

---

## Guardrails / Notes
- Keep the framing diagnostic and empirical. We are characterizing a base model's sampling
  distribution to motivate later work, not making normative claims about demographics.
- Always carry an "unspecified" bucket; never force ethnicity/gender guesses on thin evidence.
- Log all sampling hyperparameters with every run — distributional results are settings-dependent.
- Cache model downloads and annotations; sampling 8B diffusion models is expensive.
- Set seeds and record them for reproducibility.
- Start small (n≈10), verify the whole loop, then scale.
