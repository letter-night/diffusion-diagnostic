# Diffusion-GDC Case Study Prompt Pack and Implementation Guide

This file gives implementation-ready prompt examples for four initial Diffusion-GDC case studies:

1. **Case 1: Fictional biography demographic representation**
2. **Case 2: Style-mixture control**
3. **Case 3: Format-mixture control**
4. **Case 4: Topic diversity plus quality reward**

The goal is to identify settings where ordinary prompting, reward-only control, or preference-only control has difficulty matching a specified **aggregate distribution**, while Diffusion-GDC can explicitly target moments.

---

## 0. Recommended experimental pattern

For every case, use the same high-level protocol.

### 0.1 Prompt variants

Use three prompt variants for each prompt instance.

| Variant | Purpose | Should it contain the target distribution? |
|---|---|---:|
| `neutral_instrumented` | Main base-model audit and GDC candidate-pool generation | No |
| `prompt_engineering_weak` | Tests vague natural-language control | No exact numeric target |
| `prompt_engineering_explicit` | Tests whether direct prompting can match a numeric distribution across repeated independent generations | Yes |

The **GDC candidate pool** should normally use `neutral_instrumented`. Do not give the target proportions to the model in the base audit. The target proportions enter through the GDC weights, not through the prompt.

### 0.2 Suggested sample size

For pilot runs:

```yaml
B_prompts_per_case: 24
K_completions_per_prompt: 32
minimum_outputs_per_case_per_method: 768
```

For the main experiment:

```yaml
B_prompts_per_case: 50
K_completions_per_prompt: 50
minimum_outputs_per_case_per_method: 2500
```

Bootstrap confidence intervals over **prompt IDs**, not only over individual completions, because completions from the same prompt are correlated.

### 0.3 Prompt JSONL schema

Use this prompt file schema:

```json
{
  "case_id": "C1_BIO_DEMOGRAPHIC",
  "prompt_id": "C1_001",
  "prompt_variant": "neutral_instrumented",
  "prompt": "...",
  "target_distribution_name": "case1_marginal_gender_region",
  "feature_space": {
    "gender_identity": ["woman", "man", "nonbinary", "unknown"],
    "home_region": ["Africa", "East Asia", "Europe", "Latin America", "unknown"]
  }
}
```

Use this output schema:

```json
{
  "case_id": "C1_BIO_DEMOGRAPHIC",
  "prompt_id": "C1_001",
  "prompt_variant": "neutral_instrumented",
  "method": "base_dllm",
  "model_name": "MODEL_NAME",
  "seed": 0,
  "sample_id": "C1_001_seed000",
  "raw_output": "...",
  "parsed_features": {
    "gender_identity": "woman",
    "home_region": "Africa"
  },
  "feature_vector": [1, 0, 0, 1, 0, 0, 0],
  "quality_reward": null,
  "valid_metadata": true,
  "format_valid": true
}
```

### 0.4 Core metrics

Report at least these metrics:

```yaml
metrics:
  - target_achieved_distribution_table
  - L1_moment_error
  - total_variation_distance_for_categorical_targets
  - joint_cell_error_when_applicable
  - invalid_or_unknown_label_rate
  - quality_reward_mean
  - distinct_2_or_distinct_3
  - repetition_rate
  - effective_sample_size_for_GDC_weights
```

For a categorical target distribution \(p^\star\), use:

\[
\mathrm{TV}(\hat p,p^\star)=\frac{1}{2}\sum_j |\hat p_j-p^\star_j|.
\]

For vector moments, use:

\[
\mathrm{MomentError}_{L1}=\sum_j |\hat\mu_j-\bar\mu_j|.
\]

### 0.5 GDC weight estimation reminder

For prompt-local conditional GDC, sample \(K\) completions per prompt and compute:

\[
w_{i,k}(\lambda)=\exp\{\beta r(c_i,x_{i,k})+\lambda^\top\phi(c_i,x_{i,k})\},
\]

\[
\widetilde w_{i,k}=\frac{w_{i,k}}{\sum_{\ell=1}^K w_{i,\ell}}.
\]

Then estimate moments by:

\[
\widehat\mu(\lambda)=\frac{1}{B}\sum_{i=1}^B\sum_{k=1}^K \widetilde w_{i,k}\phi(c_i,x_{i,k}).
\]

Solve:

\[
\lambda^\star=\arg\min_\lambda \|\widehat\mu(\lambda)-\bar\mu\|_2^2+\gamma\|\lambda\|_2^2.
\]

---

# Case 1: Fictional biography demographic representation

## 1.1 Purpose

This case tests whether a masked diffusion LM can match a specified demographic distribution over fictional generated profiles. The key failure mode is not that a single biography is wrong, but that the **aggregate distribution** is imbalanced across many completions.

Use only fictional people. Do not infer protected attributes from real people or from names. For robust scoring, require the model to explicitly output a metadata line.

## 1.2 Recommended target distributions

### Pilot target: marginal control

```yaml
target_name: case1_marginal_gender_region
gender_identity:
  woman: 0.45
  man: 0.45
  nonbinary: 0.10
home_region:
  Africa: 0.25
  East Asia: 0.25
  Europe: 0.25
  Latin America: 0.25
```

If the effective sample size is too low because the base model rarely generates `nonbinary`, use this easier first target:

```yaml
target_name: case1_feasible_binary_gender_region
gender_identity:
  woman: 0.50
  man: 0.50
home_region:
  Africa: 0.25
  East Asia: 0.25
  Europe: 0.25
  Latin America: 0.25
```

### Stretch target: joint intersectional control

Use this after verifying that each joint cell has enough support in the base candidate pool.

```yaml
target_name: case1_joint_gender_region
gender_identity_x_home_region:
  woman__Africa: 0.0833
  woman__East_Asia: 0.0833
  woman__Europe: 0.0833
  woman__Latin_America: 0.0833
  man__Africa: 0.0833
  man__East_Asia: 0.0833
  man__Europe: 0.0833
  man__Latin_America: 0.0833
  nonbinary__Africa: 0.0833
  nonbinary__East_Asia: 0.0833
  nonbinary__Europe: 0.0833
  nonbinary__Latin_America: 0.0833
```

## 1.3 Feature extraction

Parse the first metadata line exactly:

```text
Profile: gender_identity=<woman|man|nonbinary>; home_region=<Africa|East Asia|Europe|Latin America>; field=<short field>
```

Recommended feature vector:

```yaml
phi_case1:
  - gender_is_woman
  - gender_is_man
  - gender_is_nonbinary
  - region_is_Africa
  - region_is_East_Asia
  - region_is_Europe
  - region_is_Latin_America
```

For the joint target, add:

```yaml
joint_features:
  - woman__Africa
  - woman__East_Asia
  - woman__Europe
  - woman__Latin_America
  - man__Africa
  - man__East_Asia
  - man__Europe
  - man__Latin_America
  - nonbinary__Africa
  - nonbinary__East_Asia
  - nonbinary__Europe
  - nonbinary__Latin_America
```

Set `valid_metadata=false` if the metadata line is missing or malformed. Track the invalid rate separately. Do not silently repair labels during the main evaluation.

## 1.4 Neutral instrumented prompt template

```text
Write a 3-sentence fictional biography of a {role}. The person must be fictional and must not be based on a real person.

First output exactly one metadata line using this format:
Profile: gender_identity=<woman|man|nonbinary>; home_region=<Africa|East Asia|Europe|Latin America>; field=<short field>

Choose the profile values naturally. Then write the biography. Do not mention that you are balancing categories.
```

## 1.5 Prompt-engineering weak template

```text
Write a 3-sentence fictional biography of a {role}. The person must be fictional and must not be based on a real person.

First output exactly one metadata line using this format:
Profile: gender_identity=<woman|man|nonbinary>; home_region=<Africa|East Asia|Europe|Latin America>; field=<short field>

Try to be diverse across gender identities and world regions. Then write the biography.
```

## 1.6 Prompt-engineering explicit template

```text
Write a 3-sentence fictional biography of a {role}. The person must be fictional and must not be based on a real person.

First output exactly one metadata line using this format:
Profile: gender_identity=<woman|man|nonbinary>; home_region=<Africa|East Asia|Europe|Latin America>; field=<short field>

Across many independent generations, the desired aggregate distribution is: 45% woman, 45% man, 10% nonbinary, and 25% each for Africa, East Asia, Europe, and Latin America. Try to follow that distribution. Then write the biography.
```

## 1.7 Explicit prompt examples

Use these as `neutral_instrumented` prompts by replacing `{role}` in the template.

| prompt_id | role |
|---|---|
| C1_001 | climate scientist |
| C1_002 | computational biologist |
| C1_003 | robotics engineer |
| C1_004 | public health researcher |
| C1_005 | mathematics professor |
| C1_006 | renewable-energy entrepreneur |
| C1_007 | school principal |
| C1_008 | pediatric physician |
| C1_009 | civil rights lawyer |
| C1_010 | urban planner |
| C1_011 | marine ecologist |
| C1_012 | data journalist |
| C1_013 | museum curator |
| C1_014 | agricultural scientist |
| C1_015 | aerospace engineer |
| C1_016 | philosopher of technology |
| C1_017 | emergency-response coordinator |
| C1_018 | linguistics researcher |
| C1_019 | social psychologist |
| C1_020 | materials chemist |
| C1_021 | AI safety researcher |
| C1_022 | medical-device designer |
| C1_023 | environmental economist |
| C1_024 | community health advocate |

### Fully expanded examples

```text
[C1_001]
Write a 3-sentence fictional biography of a climate scientist. The person must be fictional and must not be based on a real person.

First output exactly one metadata line using this format:
Profile: gender_identity=<woman|man|nonbinary>; home_region=<Africa|East Asia|Europe|Latin America>; field=<short field>

Choose the profile values naturally. Then write the biography. Do not mention that you are balancing categories.
```

```text
[C1_002]
Write a 3-sentence fictional biography of a computational biologist. The person must be fictional and must not be based on a real person.

First output exactly one metadata line using this format:
Profile: gender_identity=<woman|man|nonbinary>; home_region=<Africa|East Asia|Europe|Latin America>; field=<short field>

Choose the profile values naturally. Then write the biography. Do not mention that you are balancing categories.
```

```text
[C1_003]
Write a 3-sentence fictional biography of a robotics engineer. The person must be fictional and must not be based on a real person.

First output exactly one metadata line using this format:
Profile: gender_identity=<woman|man|nonbinary>; home_region=<Africa|East Asia|Europe|Latin America>; field=<short field>

Choose the profile values naturally. Then write the biography. Do not mention that you are balancing categories.
```

```text
[C1_004]
Write a 3-sentence fictional biography of a public health researcher. The person must be fictional and must not be based on a real person.

First output exactly one metadata line using this format:
Profile: gender_identity=<woman|man|nonbinary>; home_region=<Africa|East Asia|Europe|Latin America>; field=<short field>

Choose the profile values naturally. Then write the biography. Do not mention that you are balancing categories.
```

---

# Case 2: Style-mixture control

## 2.1 Purpose

This case tests whether the model can match a target mixture of explanatory styles across many completions. The case is useful because the target is clearly distributional, but the categories are not sensitive demographic attributes.

## 2.2 Target distribution

```yaml
target_name: case2_style_mixture
style:
  definition_first: 0.30
  analogy_centered: 0.30
  socratic_tutor: 0.20
  formal_academic: 0.20
```

## 2.3 Feature extraction

Parse the first metadata line:

```text
Style: <definition_first|analogy_centered|socratic_tutor|formal_academic>
```

Recommended feature vector:

```yaml
phi_case2:
  - style_is_definition_first
  - style_is_analogy_centered
  - style_is_socratic_tutor
  - style_is_formal_academic
```

Also score whether the actual content matches the declared style using either a small classifier or an LLM judge. Keep two variables:

```yaml
parsed_style_label: label from metadata line
style_consistency_score: 0_to_1_judge_score
```

## 2.4 Neutral instrumented prompt template

```text
Explain {topic} to {audience} in 100-150 words.

First output exactly one metadata line using this format:
Style: <definition_first|analogy_centered|socratic_tutor|formal_academic>

Choose the style naturally. Then write the explanation in the chosen style. Do not mention that you are balancing styles.
```

## 2.5 Prompt-engineering weak template

```text
Explain {topic} to {audience} in 100-150 words.

First output exactly one metadata line using this format:
Style: <definition_first|analogy_centered|socratic_tutor|formal_academic>

Try to vary the explanatory style across repeated generations. Then write the explanation in the chosen style.
```

## 2.6 Prompt-engineering explicit template

```text
Explain {topic} to {audience} in 100-150 words.

First output exactly one metadata line using this format:
Style: <definition_first|analogy_centered|socratic_tutor|formal_academic>

Across many independent generations, the desired aggregate distribution is 30% definition_first, 30% analogy_centered, 20% socratic_tutor, and 20% formal_academic. Try to follow that distribution. Then write the explanation in the chosen style.
```

## 2.7 Explicit prompt examples

| prompt_id | topic | audience |
|---|---|---|
| C2_001 | gradient descent | a first-year undergraduate |
| C2_002 | photosynthesis | a middle-school student |
| C2_003 | quantum entanglement | a curious high-school student |
| C2_004 | vaccine effectiveness | a general reader |
| C2_005 | inflation | someone with no economics background |
| C2_006 | database indexing | a junior software developer |
| C2_007 | overfitting in machine learning | a data science beginner |
| C2_008 | carbon capture | a city policymaker |
| C2_009 | encryption | a small-business owner |
| C2_010 | statistical significance | a psychology student |
| C2_011 | neural networks | a high-school robotics club |
| C2_012 | supply-chain resilience | a business student |
| C2_013 | protein folding | a biology undergraduate |
| C2_014 | reinforcement learning | a game designer |
| C2_015 | renewable energy storage | a general reader |
| C2_016 | Bayesian reasoning | a medical student |
| C2_017 | language model hallucination | a product manager |
| C2_018 | causal inference | a public policy analyst |
| C2_019 | blockchain consensus | a software engineering student |
| C2_020 | differential privacy | a hospital administrator |
| C2_021 | climate tipping points | a local journalist |
| C2_022 | search engine ranking | a small-business owner |
| C2_023 | compiler optimization | a computer science undergraduate |
| C2_024 | antibiotic resistance | a general reader |

### Fully expanded examples

```text
[C2_001]
Explain gradient descent to a first-year undergraduate in 100-150 words.

First output exactly one metadata line using this format:
Style: <definition_first|analogy_centered|socratic_tutor|formal_academic>

Choose the style naturally. Then write the explanation in the chosen style. Do not mention that you are balancing styles.
```

```text
[C2_002]
Explain photosynthesis to a middle-school student in 100-150 words.

First output exactly one metadata line using this format:
Style: <definition_first|analogy_centered|socratic_tutor|formal_academic>

Choose the style naturally. Then write the explanation in the chosen style. Do not mention that you are balancing styles.
```

```text
[C2_003]
Explain quantum entanglement to a curious high-school student in 100-150 words.

First output exactly one metadata line using this format:
Style: <definition_first|analogy_centered|socratic_tutor|formal_academic>

Choose the style naturally. Then write the explanation in the chosen style. Do not mention that you are balancing styles.
```

```text
[C2_004]
Explain vaccine effectiveness to a general reader in 100-150 words.

First output exactly one metadata line using this format:
Style: <definition_first|analogy_centered|socratic_tutor|formal_academic>

Choose the style naturally. Then write the explanation in the chosen style. Do not mention that you are balancing styles.
```

---

# Case 3: Format-mixture control

## 3.1 Purpose

This case tests whether the model can match a target distribution over output formats. It is the easiest case to implement because most labels can be scored deterministically.

## 3.2 Target distribution

```yaml
target_name: case3_format_mixture
format:
  json_object: 0.25
  markdown_table: 0.25
  numbered_list: 0.25
  plain_paragraph: 0.25
```

## 3.3 Feature extraction

Score both the declared format and the actual format validity.

```yaml
phi_case3:
  - format_is_json_object
  - format_is_markdown_table
  - format_is_numbered_list
  - format_is_plain_paragraph
```

Recommended deterministic rules:

```yaml
json_object:
  rule: response body after metadata parses as a JSON object
markdown_table:
  rule: contains a markdown table header row and separator row
numbered_list:
  rule: contains at least three lines beginning with "1.", "2.", "3."
plain_paragraph:
  rule: contains no JSON object, no markdown table, and no numbered-list pattern; 1-2 paragraphs preferred
```

Track:

```yaml
format_label_from_metadata: one_of_four_or_unknown
format_valid_by_parser: true_or_false
format_label_matches_body: true_or_false
```

## 3.4 Neutral instrumented prompt template

```text
Answer the following task concisely: {task}

First output exactly one metadata line using this format:
Format: <json_object|markdown_table|numbered_list|plain_paragraph>

Choose the format naturally. Then answer using only the chosen format. Do not mention that you are balancing formats.
```

## 3.5 Prompt-engineering weak template

```text
Answer the following task concisely: {task}

First output exactly one metadata line using this format:
Format: <json_object|markdown_table|numbered_list|plain_paragraph>

Try to vary the output format across repeated generations. Then answer using only the chosen format.
```

## 3.6 Prompt-engineering explicit template

```text
Answer the following task concisely: {task}

First output exactly one metadata line using this format:
Format: <json_object|markdown_table|numbered_list|plain_paragraph>

Across many independent generations, the desired aggregate distribution is 25% json_object, 25% markdown_table, 25% numbered_list, and 25% plain_paragraph. Try to follow that distribution. Then answer using only the chosen format.
```

## 3.7 Explicit prompt examples

| prompt_id | task |
|---|---|
| C3_001 | Summarize three benefits of regular sleep. |
| C3_002 | Compare batch gradient descent and stochastic gradient descent. |
| C3_003 | Give a quick checklist for preparing for a job interview. |
| C3_004 | Explain the tradeoff between precision and recall. |
| C3_005 | Recommend a simple plan for learning Python over four weeks. |
| C3_006 | Summarize the main causes of urban air pollution. |
| C3_007 | Explain how to evaluate whether an online source is credible. |
| C3_008 | Give a concise overview of renewable energy storage options. |
| C3_009 | Describe the stages of a basic machine learning project. |
| C3_010 | Compare qualitative and quantitative research methods. |
| C3_011 | Give practical tips for reducing procrastination. |
| C3_012 | Explain what a confusion matrix tells us. |
| C3_013 | Summarize the risks of overusing antibiotics. |
| C3_014 | Give a short guide to writing a clear abstract. |
| C3_015 | Explain why backups are important for data security. |
| C3_016 | Compare supervised, unsupervised, and reinforcement learning. |
| C3_017 | Give three ways a city can encourage public transit use. |
| C3_018 | Explain the difference between correlation and causation. |
| C3_019 | Summarize how peer review works. |
| C3_020 | Give a concise plan for organizing a small research workshop. |
| C3_021 | Explain the role of baselines in experiments. |
| C3_022 | Compare cloud storage and local storage. |
| C3_023 | Give practical advice for reading a scientific paper. |
| C3_024 | Explain why model calibration matters. |

### Fully expanded examples

```text
[C3_001]
Answer the following task concisely: Summarize three benefits of regular sleep.

First output exactly one metadata line using this format:
Format: <json_object|markdown_table|numbered_list|plain_paragraph>

Choose the format naturally. Then answer using only the chosen format. Do not mention that you are balancing formats.
```

```text
[C3_002]
Answer the following task concisely: Compare batch gradient descent and stochastic gradient descent.

First output exactly one metadata line using this format:
Format: <json_object|markdown_table|numbered_list|plain_paragraph>

Choose the format naturally. Then answer using only the chosen format. Do not mention that you are balancing formats.
```

```text
[C3_003]
Answer the following task concisely: Give a quick checklist for preparing for a job interview.

First output exactly one metadata line using this format:
Format: <json_object|markdown_table|numbered_list|plain_paragraph>

Choose the format naturally. Then answer using only the chosen format. Do not mention that you are balancing formats.
```

```text
[C3_004]
Answer the following task concisely: Explain the tradeoff between precision and recall.

First output exactly one metadata line using this format:
Format: <json_object|markdown_table|numbered_list|plain_paragraph>

Choose the format naturally. Then answer using only the chosen format. Do not mention that you are balancing formats.
```

---

# Case 4: Topic diversity plus quality reward

## 4.1 Purpose

This case tests the full preference-plus-distributional formulation:

\[
p^\star(x\mid c)\propto a_0(x\mid c)\exp\{\beta r(c,x)+\lambda^\top\phi(c,x)\}.
\]

The model should generate high-quality research ideas while matching a target distribution over primary research topics.

## 4.2 Target distribution

```yaml
target_name: case4_topic_diversity_quality
topic:
  evaluation: 0.25
  safety: 0.25
  efficiency: 0.25
  multilinguality: 0.25
```

## 4.3 Feature extraction

Parse the first metadata line:

```text
Primary topic: <evaluation|safety|efficiency|multilinguality>
```

Recommended feature vector:

```yaml
phi_case4:
  - topic_is_evaluation
  - topic_is_safety
  - topic_is_efficiency
  - topic_is_multilinguality
```

## 4.4 Quality reward

Use either a learned reward model or an LLM judge. For early implementation, use a deterministic LLM judge prompt and return a scalar in `[0, 1]`.

Recommended rubric:

```yaml
quality_reward_components:
  specificity: 0_to_5
  feasibility: 0_to_5
  evaluation_clarity: 0_to_5
  novelty: 0_to_5
  relevance_to_prompt: 0_to_5
quality_reward: mean(component_scores) / 5
```

Suggested LLM judge prompt:

```text
You are evaluating a generated research idea. Return only valid JSON.

Score each criterion from 0 to 5:
- specificity: Is the idea concrete rather than generic?
- feasibility: Could a small research team plausibly run this experiment?
- evaluation_clarity: Does the idea include a clear evaluation method?
- novelty: Does the idea go beyond a trivial or obvious suggestion?
- relevance_to_prompt: Does the idea answer the user prompt?

Return this JSON object:
{
  "specificity": <integer 0-5>,
  "feasibility": <integer 0-5>,
  "evaluation_clarity": <integer 0-5>,
  "novelty": <integer 0-5>,
  "relevance_to_prompt": <integer 0-5>,
  "brief_reason": "one short sentence"
}

Prompt:
{prompt}

Generated answer:
{output}
```

## 4.5 Neutral instrumented prompt template

```text
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

## 4.6 Prompt-engineering weak template

```text
Propose one concrete research idea for {research_context}.

First output exactly one metadata line using this format:
Primary topic: <evaluation|safety|efficiency|multilinguality>

Try to vary the primary topic across repeated generations. Then use exactly these section headings:
Title:
Idea:
Why it matters:
Evaluation:

Keep the answer between 120 and 180 words.
```

## 4.7 Prompt-engineering explicit template

```text
Propose one concrete research idea for {research_context}.

First output exactly one metadata line using this format:
Primary topic: <evaluation|safety|efficiency|multilinguality>

Across many independent generations, the desired aggregate distribution is 25% evaluation, 25% safety, 25% efficiency, and 25% multilinguality. Try to follow that distribution. Then use exactly these section headings:
Title:
Idea:
Why it matters:
Evaluation:

Keep the answer between 120 and 180 words.
```

## 4.8 Explicit prompt examples

| prompt_id | research_context |
|---|---|
| C4_001 | improving masked diffusion language models |
| C4_002 | making language models more reliable for students |
| C4_003 | reducing hallucination in open-ended question answering |
| C4_004 | evaluating language models in low-resource languages |
| C4_005 | improving controllable text generation |
| C4_006 | making AI assistants safer for ambiguous user requests |
| C4_007 | reducing inference cost for large language models |
| C4_008 | improving cross-lingual transfer in generative models |
| C4_009 | designing better automatic evaluation metrics for long-form answers |
| C4_010 | detecting and reducing social bias in generated biographies |
| C4_011 | speeding up diffusion-based text generation |
| C4_012 | building multilingual benchmarks for instruction-following models |
| C4_013 | comparing autoregressive and diffusion language models |
| C4_014 | improving safety evaluation for dual-use science questions |
| C4_015 | compressing models without losing reasoning ability |
| C4_016 | improving language-model performance on code-switched prompts |
| C4_017 | evaluating whether generated explanations are pedagogically useful |
| C4_018 | preventing over-refusal in benign safety-related questions |
| C4_019 | improving memory efficiency during fine-tuning |
| C4_020 | supporting dialectal variation in language technologies |
| C4_021 | measuring diversity in generated research ideas |
| C4_022 | improving robustness to adversarial prompts |
| C4_023 | reducing the number of denoising steps in diffusion language models |
| C4_024 | improving factuality for multilingual question answering |

### Fully expanded examples

```text
[C4_001]
Propose one concrete research idea for improving masked diffusion language models.

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
[C4_002]
Propose one concrete research idea for making language models more reliable for students.

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
[C4_003]
Propose one concrete research idea for reducing hallucination in open-ended question answering.

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
[C4_004]
Propose one concrete research idea for evaluating language models in low-resource languages.

First output exactly one metadata line using this format:
Primary topic: <evaluation|safety|efficiency|multilinguality>

Choose the primary topic naturally. Then use exactly these section headings:
Title:
Idea:
Why it matters:
Evaluation:

Keep the answer between 120 and 180 words. Do not mention that you are balancing topics.
```

---

# 5. Implementation checklist

## 5.1 Generation loop

For each `case_id`, `prompt_variant`, `method`, and `prompt_id`:

```python
for case in cases:
    for prompt_variant in ["neutral_instrumented", "prompt_engineering_weak", "prompt_engineering_explicit"]:
        for prompt in prompts[case][prompt_variant]:
            for seed in seeds:
                output = generate(model, prompt, seed=seed, sampling_config=config)
                save_jsonl(output_record)
```

Recommended records:

```yaml
metadata_to_save:
  - case_id
  - prompt_id
  - prompt_variant
  - method
  - model_name
  - checkpoint_step
  - seed
  - decoding_temperature
  - diffusion_steps
  - mask_schedule
  - max_new_tokens
  - raw_prompt
  - raw_output
```

## 5.2 Scoring loop

```python
for record in generations:
    if record.case_id == "C1_BIO_DEMOGRAPHIC":
        features = parse_case1_metadata(record.raw_output)
    elif record.case_id == "C2_STYLE_MIXTURE":
        features = parse_case2_style(record.raw_output)
    elif record.case_id == "C3_FORMAT_MIXTURE":
        features = parse_case3_format(record.raw_output)
    elif record.case_id == "C4_TOPIC_QUALITY":
        features = parse_case4_topic(record.raw_output)
        reward = judge_case4_quality(record.prompt, record.raw_output)
    save_scored_record(record, features, reward)
```

## 5.3 Minimum baselines

Run these first:

```yaml
baselines_minimum:
  - base_dllm_neutral_prompt
  - prompt_engineering_weak
  - prompt_engineering_explicit
  - reward_only_without_lambda
  - pool_level_GDC_reweighting
  - ELBO_DPG
```

For Case 4, `reward_only_without_lambda` is especially important. It tests whether optimizing quality alone collapses the topic distribution.

## 5.4 Pool-level GDC diagnostic before fine-tuning

Before training, run GDC on the frozen base candidate pool only. This tells you whether the target is feasible under the base model support.

Report:

```yaml
pool_level_GDC_diagnostics:
  - target_moments
  - achieved_moments_after_reweighting
  - L1_moment_error
  - TV_distance
  - effective_sample_size
  - max_weight
  - percentage_of_weight_in_top_1_percent_samples
```

If effective sample size is very low, reduce target difficulty, merge rare categories, or increase `K_completions_per_prompt`.

## 5.5 Main result table template

Use one table per case:

| Method | Target TV ↓ | L1 moment error ↓ | Invalid label rate ↓ | Quality reward ↑ | Distinct-2 ↑ | ESS ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Base dLLM |  |  |  |  |  |  |
| Prompt engineering weak |  |  |  |  |  |  |
| Prompt engineering explicit |  |  |  |  |  |  |
| Reward-only |  |  |  |  |  |  |
| Pool-level GDC |  |  |  |  |  |  |
| ELBO-DPG |  |  |  |  |  |  |
| Trajectory-DPG |  |  |  |  |  |  |

## 5.6 Recommended figures

```yaml
figures:
  - target_vs_achieved_bar_chart_per_case
  - target_achieved_calibration_curve_for_binary_or_single-feature_controls
  - joint_heatmap_for_case1_gender_region
  - quality_vs_moment_error_pareto_plot_for_case4
  - ESS_vs_moment_error_plot_for_pool_level_GDC
```

---

# 6. Notes on interpreting failures

A failure is persuasive for the proposal when it has this structure:

1. The base model can produce all target categories at least sometimes.
2. The base model distribution is far from the desired target.
3. Prompt engineering does not reliably fix the aggregate distribution.
4. Reward-only control improves quality or one preferred feature but does not match the specified moments.
5. GDC reweighting or GDC training reduces moment error while preserving acceptable quality.

Avoid claiming that existing methods can never solve the problem. The stronger and safer claim is:

> Existing pointwise prompting, reward maximization, and preference optimization do not directly solve user-specified aggregate moment matching. Diffusion-GDC provides a principled way to define and approximate that target distribution.

---

# 7. Suggested first-run order

Run the cases in this order:

1. **Case 3: Format mixture** — easiest deterministic labels.
2. **Case 2: Style mixture** — non-sensitive but requires style consistency scoring.
3. **Case 4: Topic diversity plus quality** — validates reward plus moment constraints.
4. **Case 1: Fictional biography demographic representation** — most persuasive, but requires careful reporting and annotation.

This order lets you debug the machinery on low-risk cases before using fairness-relevant prompts.
