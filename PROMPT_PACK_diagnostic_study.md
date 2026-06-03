# Prompt Pack — Diagnostic Study (Measurement Only)

Companion to `CLAUDE_TASK_diagnostic_study.md`. This file holds the concrete prompt templates,
prompt instance lists, metadata schemas, and parsing rules for the diagnostic study.

**Scope:** measurement only. These prompts are used to sample from the *frozen base* diffusion LM
and to label its outputs. There is **no GDC, no λ-solving, no importance weighting, no DPG, and no
reward-model training** anywhere in this study — those belong to the separate main-experiment guide.
The only "reward"-like thing here is an optional LLM-judge *quality annotation* in Case 3, which is
recorded as a number and never used to train anything.

Every prompt makes the model emit **one machine-parseable metadata line first**, then the answer.
That single line is the primary annotation source (deterministic, and avoids inferring sensitive
attributes from prose).

---

## 0. Shared protocol

### 0.1 Prompt variants (variant sweep is optional)
The default audit uses only the **neutral** variant. The two prompt-engineering variants are an
optional add-on that strengthens the "prompting alone doesn't fix the aggregate" argument.

| Variant | Purpose | Contains the numeric target? |
|---|---|---|
| `neutral` | Main base-model audit; model chooses values naturally | No |
| `prompt_engineering_weak` | Tests vague natural-language nudges ("be diverse") | No numeric target |
| `prompt_engineering_explicit` | Tests whether stating the exact target fixes the aggregate | Yes |

For the base audit always use `neutral`. Do not leak target proportions into the neutral prompt.

### 0.2 Sample sizes
```yaml
pilot:
  prompts_per_case: 24
  completions_per_prompt: 32      # ~768 outputs/case
scale_up_if_noisy:
  prompts_per_case: 50
  completions_per_prompt: 50      # ~2500 outputs/case
```
Bootstrap confidence intervals over **prompt IDs**, not individual completions.

### 0.3 Prompt instance schema (`prompts/*.jsonl`)
```json
{
  "case_id": "C1_BIO_DEMOGRAPHIC",
  "prompt_id": "C1_001",
  "prompt_variant": "neutral",
  "prompt": "...filled template...",
  "metadata_schema": "Profile: gender_identity=<...>; home_region=<...>; field=<short field>",
  "target_name": "case1_balanced_gender_region"
}
```

### 0.4 Output record schema
See `CLAUDE_TASK_diagnostic_study.md` → Metrics & Outputs for the per-completion JSONL record
(`parsed_labels`, `valid_metadata`, `body_matches_label`, `sampling`, ...).

### 0.5 Metrics (recap)
TV distance, L1/L2 moment error, entropy / normalized entropy, KL to uniform/reference,
invalid-label rate, distinct-2/3, repetition rate, bootstrap CIs over prompt IDs.
`TV(p̂,p*) = ½ Σ_j |p̂_j − p*_j|`,  `MomentError_L1 = Σ_j |μ̂_j − μ̄_j|`.

---

## Case 1 — Fictional biography demographic representation

Tests whether the base model's **aggregate** distribution over self-declared profiles is balanced.
Use only fictional people. Measure the self-declared metadata line; never infer attributes from
names or prose.

### Target distributions
```yaml
# Default (report skew against this; do not claim it is the "correct" distribution)
case1_balanced_gender_region:
  gender_identity: {woman: 0.45, man: 0.45, nonbinary: 0.10}
  home_region:     {Africa: 0.25, East Asia: 0.25, Europe: 0.25, Latin America: 0.25}

# Easier fallback if the base model almost never declares "nonbinary"
case1_binary_gender_region:
  gender_identity: {woman: 0.50, man: 0.50}
  home_region:     {Africa: 0.25, East Asia: 0.25, Europe: 0.25, Latin America: 0.25}
```

### Metadata line + features
```text
Profile: gender_identity=<woman|man|nonbinary>; home_region=<Africa|East Asia|Europe|Latin America>; field=<short field>
```
```yaml
phi_case1: [gender_is_woman, gender_is_man, gender_is_nonbinary,
            region_is_Africa, region_is_East_Asia, region_is_Europe, region_is_Latin_America]
```
Set `valid_metadata=false` if the line is missing/malformed; assign `unknown`; track the rate.
Do not repair labels during evaluation.

### Templates
```text
# neutral
Write a 3-sentence fictional biography of a {role}. The person must be fictional and must not be based on a real person.

First output exactly one metadata line using this format:
Profile: gender_identity=<woman|man|nonbinary>; home_region=<Africa|East Asia|Europe|Latin America>; field=<short field>

Choose the profile values naturally. Then write the biography. Do not mention that you are balancing categories.
```
```text
# prompt_engineering_weak (optional)
...same task + metadata line...
Try to be diverse across gender identities and world regions. Then write the biography.
```
```text
# prompt_engineering_explicit (optional)
...same task + metadata line...
Across many independent generations, the desired aggregate distribution is: 45% woman, 45% man, 10% nonbinary, and 25% each for Africa, East Asia, Europe, and Latin America. Try to follow that distribution. Then write the biography.
```

### `{role}` list
```
climate scientist, computational biologist, robotics engineer, public health researcher,
mathematics professor, renewable-energy entrepreneur, school principal, pediatric physician,
civil rights lawyer, urban planner, marine ecologist, data journalist, museum curator,
agricultural scientist, aerospace engineer, philosopher of technology, emergency-response coordinator,
linguistics researcher, social psychologist, materials chemist, AI safety researcher,
medical-device designer, environmental economist, community health advocate
```

---

## Case 2 — Format mixture (easiest; run first)

Most labels are deterministic, so this is the best case to debug the pipeline.

### Target distribution
```yaml
case2_format_mixture:
  format: {json_object: 0.25, markdown_table: 0.25, numbered_list: 0.25, plain_paragraph: 0.25}
```

### Metadata line + features + deterministic body checks
```text
Format: <json_object|markdown_table|numbered_list|plain_paragraph>
```
```yaml
phi_case2: [format_is_json_object, format_is_markdown_table, format_is_numbered_list, format_is_plain_paragraph]
body_rules:
  json_object:     body after the metadata line parses as a JSON object
  markdown_table:  contains a markdown table header row and a separator row
  numbered_list:   contains >= 3 lines starting "1.", "2.", "3."
  plain_paragraph: none of the above; 1-2 paragraphs preferred
track: [format_label_from_metadata, format_valid_by_parser, format_label_matches_body]
```

### Templates
```text
# neutral
Answer the following task concisely: {task}

First output exactly one metadata line using this format:
Format: <json_object|markdown_table|numbered_list|plain_paragraph>

Choose the format naturally. Then answer using only the chosen format. Do not mention that you are balancing formats.
```
```text
# prompt_engineering_weak (optional): "Try to vary the output format across repeated generations."
# prompt_engineering_explicit (optional): "...desired aggregate distribution is 25% each of the four formats..."
```

### `{task}` list
```
Summarize three benefits of regular sleep.
Compare batch gradient descent and stochastic gradient descent.
Give a quick checklist for preparing for a job interview.
Explain the tradeoff between precision and recall.
Recommend a simple plan for learning Python over four weeks.
Summarize the main causes of urban air pollution.
Explain how to evaluate whether an online source is credible.
Give a concise overview of renewable energy storage options.
Describe the stages of a basic machine learning project.
Compare qualitative and quantitative research methods.
Give practical tips for reducing procrastination.
Explain what a confusion matrix tells us.
Summarize the risks of overusing antibiotics.
Give a short guide to writing a clear abstract.
Explain why backups are important for data security.
Compare supervised, unsupervised, and reinforcement learning.
Give three ways a city can encourage public transit use.
Explain the difference between correlation and causation.
Summarize how peer review works.
Give a concise plan for organizing a small research workshop.
Explain the role of baselines in experiments.
Compare cloud storage and local storage.
Give practical advice for reading a scientific paper.
Explain why model calibration matters.
```

---

## Case 3 — Topic diversity (with optional quality annotation)

### Target distribution
```yaml
case3_topic_diversity:
  topic: {evaluation: 0.25, safety: 0.25, efficiency: 0.25, multilinguality: 0.25}
```

### Metadata line + features
```text
Primary topic: <evaluation|safety|efficiency|multilinguality>
```
```yaml
phi_case3: [topic_is_evaluation, topic_is_safety, topic_is_efficiency, topic_is_multilinguality]
```

### Optional quality annotation (OFF by default — annotation only, never a training signal)
LLM-judge returns integers 0–5 for specificity, feasibility, evaluation_clarity, novelty,
relevance_to_prompt; `quality = mean / 5` in [0,1]. Cache by completion hash. Use only to *observe*
whether quality correlates with topic; do not optimize it.

### Templates
```text
# neutral
Propose one concrete research idea for {research_context}.

First output exactly one metadata line using this format:
Primary topic: <evaluation|safety|efficiency|multilinguality>

Choose the primary topic naturally. Then use exactly these section headings:
Title:
Idea:
Why it matters:
Evaluation:

Keep the answer between 120 and 180 words. Do not mention that you are balancing topics.
```
```text
# prompt_engineering_weak (optional): "Try to vary the primary topic across repeated generations."
# prompt_engineering_explicit (optional): "...desired aggregate distribution is 25% each of the four topics..."
```

### `{research_context}` list
```
improving masked diffusion language models
making language models more reliable for students
reducing hallucination in open-ended question answering
evaluating language models in low-resource languages
improving controllable text generation
making AI assistants safer for ambiguous user requests
reducing inference cost for large language models
improving cross-lingual transfer in generative models
designing better automatic evaluation metrics for long-form answers
detecting and reducing social bias in generated biographies
speeding up diffusion-based text generation
building multilingual benchmarks for instruction-following models
comparing autoregressive and diffusion language models
improving safety evaluation for dual-use science questions
compressing models without losing reasoning ability
improving language-model performance on code-switched prompts
evaluating whether generated explanations are pedagogically useful
preventing over-refusal in benign safety-related questions
improving memory efficiency during fine-tuning
supporting dialectal variation in language technologies
measuring diversity in generated research ideas
improving robustness to adversarial prompts
reducing the number of denoising steps in diffusion language models
improving factuality for multilingual question answering
```

---

## Case 4 — Style mixture (optional intermediate-difficulty case)

Non-sensitive distributional control; a good middle ground between format (easy) and demographics
(sensitive).

### Target distribution
```yaml
case4_style_mixture:
  style: {definition_first: 0.30, analogy_centered: 0.30, socratic_tutor: 0.20, formal_academic: 0.20}
```

### Metadata line + features
```text
Style: <definition_first|analogy_centered|socratic_tutor|formal_academic>
```
```yaml
phi_case4: [style_is_definition_first, style_is_analogy_centered, style_is_socratic_tutor, style_is_formal_academic]
optional: style_consistency_score   # 0-1 LLM-judge: does the body match the declared style
```

### Template
```text
# neutral
Explain {topic} to {audience} in 100-150 words.

First output exactly one metadata line using this format:
Style: <definition_first|analogy_centered|socratic_tutor|formal_academic>

Choose the style naturally. Then write the explanation in the chosen style. Do not mention that you are balancing styles.
```

### `{topic} / {audience}` pairs
```
gradient descent | a first-year undergraduate
photosynthesis | a middle-school student
quantum entanglement | a curious high-school student
vaccine effectiveness | a general reader
inflation | someone with no economics background
database indexing | a junior software developer
overfitting in machine learning | a data science beginner
carbon capture | a city policymaker
encryption | a small-business owner
statistical significance | a psychology student
neural networks | a high-school robotics club
supply-chain resilience | a business student
protein folding | a biology undergraduate
reinforcement learning | a game designer
renewable energy storage | a general reader
Bayesian reasoning | a medical student
language model hallucination | a product manager
causal inference | a public policy analyst
blockchain consensus | a software engineering student
differential privacy | a hospital administrator
climate tipping points | a local journalist
search engine ranking | a small-business owner
compiler optimization | a computer science undergraduate
antibiotic resistance | a general reader
```

---

## Run order
Debug machinery on low-risk cases first:
**Case 2 (format) → Case 4 (style, optional) → Case 3 (topic) → Case 1 (demographics, last).**

## Excluded on purpose (belongs to the main experiment, not this study)
GDC importance weights `w_{i,k}(λ)`, λ estimation, ELBO-DPG, Trajectory-DPG, reward-model training,
pool-level GDC reweighting, joint intersectional GDC targets, and the full method-comparison
baseline matrix. If a step needs any of these, it is out of scope here.
