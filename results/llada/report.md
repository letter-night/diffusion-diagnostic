# Diagnostic Study: Distributional Failure in Diffusion LMs

_Measurement-only report. Characterizes the base model's aggregate sampling distribution against user-specified targets. No fine-tuning, no reward models, no GDC._


## C2_FORMAT


### variant: `neutral`  (model: LLaDA-8B-Instruct, n=48, prompts=12)

> **Headline:** C2_FORMAT format = 0% json / 0% markdown_table / 0% numbered_list / 100% plain_paragraph vs target (25% json, 25% markdown_table, 25% numbered_list, 25% plain_paragraph) → TV 0.75, L1 1.50, invalid-label rate 0%.

- invalid/unknown-metadata rate: **0.0%**
- format label matches body: **66.7%**
- diversity: distinct-2 0.231, distinct-3 0.245, repetition 0.180, TTR 0.155


#### axis: format

| bucket | achieved | 95% CI | target |
| --- | --- | --- | --- |
| json | 0.000 | [0.000, 0.000] | 0.250 |
| markdown_table | 0.000 | [0.000, 0.000] | 0.250 |
| numbered_list | 0.000 | [0.000, 0.000] | 0.250 |
| plain_paragraph | 1.000 | [0.750, 1.000] | 0.250 |

TV **0.750** (95% CI [0.750, 0.750]), L1 1.500, L2 0.866, KL(emp‖target) 2.000, norm-entropy 0.000, unknown 8.3%


### variant: `weak`  (model: LLaDA-8B-Instruct, n=48, prompts=12)

> **Headline:** C2_FORMAT format = 0% json / 0% markdown_table / 0% numbered_list / 100% plain_paragraph vs target (25% json, 25% markdown_table, 25% numbered_list, 25% plain_paragraph) → TV 0.75, L1 1.50, invalid-label rate 0%.

- invalid/unknown-metadata rate: **0.0%**
- format label matches body: **66.7%**
- diversity: distinct-2 0.231, distinct-3 0.243, repetition 0.185, TTR 0.158


#### axis: format

| bucket | achieved | 95% CI | target |
| --- | --- | --- | --- |
| json | 0.000 | [0.000, 0.000] | 0.250 |
| markdown_table | 0.000 | [0.000, 0.000] | 0.250 |
| numbered_list | 0.000 | [0.000, 0.000] | 0.250 |
| plain_paragraph | 1.000 | [1.000, 1.000] | 0.250 |

TV **0.750** (95% CI [0.750, 0.750]), L1 1.500, L2 0.866, KL(emp‖target) 2.000, norm-entropy 0.000, unknown 0.0%


### variant: `explicit`  (model: LLaDA-8B-Instruct, n=48, prompts=12)

> **Headline:** C2_FORMAT format = 57% json / 0% markdown_table / 14% numbered_list / 29% plain_paragraph vs target (25% json, 25% markdown_table, 25% numbered_list, 25% plain_paragraph) → TV 0.36, L1 0.71, invalid-label rate 0%.

- invalid/unknown-metadata rate: **0.0%**
- format label matches body: **0.0%**
- diversity: distinct-2 0.171, distinct-3 0.201, repetition 0.494, TTR 0.091


#### axis: format

| bucket | achieved | 95% CI | target |
| --- | --- | --- | --- |
| json | 0.571 | [0.083, 0.583] | 0.250 |
| markdown_table | 0.000 | [0.000, 0.000] | 0.250 |
| numbered_list | 0.143 | [0.000, 0.250] | 0.250 |
| plain_paragraph | 0.286 | [0.000, 0.417] | 0.250 |

TV **0.357** (95% CI [0.250, 0.750]), L1 0.714, L2 0.423, KL(emp‖target) 0.621, norm-entropy 0.689, unknown 41.7%


## C4_STYLE


### variant: `neutral`  (model: LLaDA-8B-Instruct, n=48, prompts=12)

> **Headline:** C4_STYLE style = 0% definition_first / 100% analogy_centered / 0% socratic_tutor / 0% formal_academic vs target (30% definition_first, 30% analogy_centered, 20% socratic_tutor, 20% formal_academic) → TV 0.70, L1 1.40, invalid-label rate 0%.

- invalid/unknown-metadata rate: **0.0%**
- diversity: distinct-2 0.167, distinct-3 0.184, repetition 0.331, TTR 0.095


#### axis: style

| bucket | achieved | 95% CI | target |
| --- | --- | --- | --- |
| definition_first | 0.000 | [0.000, 0.000] | 0.300 |
| analogy_centered | 1.000 | [1.000, 1.000] | 0.300 |
| socratic_tutor | 0.000 | [0.000, 0.000] | 0.200 |
| formal_academic | 0.000 | [0.000, 0.000] | 0.200 |

TV **0.700** (95% CI [0.700, 0.700]), L1 1.400, L2 0.812, KL(emp‖target) 1.737, norm-entropy 0.000, unknown 0.0%


### variant: `weak`  (model: LLaDA-8B-Instruct, n=48, prompts=12)

> **Headline:** C4_STYLE style = 0% definition_first / 100% analogy_centered / 0% socratic_tutor / 0% formal_academic vs target (30% definition_first, 30% analogy_centered, 20% socratic_tutor, 20% formal_academic) → TV 0.70, L1 1.40, invalid-label rate 0%.

- invalid/unknown-metadata rate: **0.0%**
- diversity: distinct-2 0.201, distinct-3 0.221, repetition 0.316, TTR 0.114


#### axis: style

| bucket | achieved | 95% CI | target |
| --- | --- | --- | --- |
| definition_first | 0.000 | [0.000, 0.000] | 0.300 |
| analogy_centered | 1.000 | [1.000, 1.000] | 0.300 |
| socratic_tutor | 0.000 | [0.000, 0.000] | 0.200 |
| formal_academic | 0.000 | [0.000, 0.000] | 0.200 |

TV **0.700** (95% CI [0.700, 0.700]), L1 1.400, L2 0.812, KL(emp‖target) 1.737, norm-entropy 0.000, unknown 0.0%


### variant: `explicit`  (model: LLaDA-8B-Instruct, n=48, prompts=12)

> **Headline:** C4_STYLE style = 0% definition_first / 100% analogy_centered / 0% socratic_tutor / 0% formal_academic vs target (30% definition_first, 30% analogy_centered, 20% socratic_tutor, 20% formal_academic) → TV 0.70, L1 1.40, invalid-label rate 0%.

- invalid/unknown-metadata rate: **0.0%**
- diversity: distinct-2 0.156, distinct-3 0.176, repetition 0.403, TTR 0.083


#### axis: style

| bucket | achieved | 95% CI | target |
| --- | --- | --- | --- |
| definition_first | 0.000 | [0.000, 0.000] | 0.300 |
| analogy_centered | 1.000 | [1.000, 1.000] | 0.300 |
| socratic_tutor | 0.000 | [0.000, 0.000] | 0.200 |
| formal_academic | 0.000 | [0.000, 0.000] | 0.200 |

TV **0.700** (95% CI [0.700, 0.700]), L1 1.400, L2 0.812, KL(emp‖target) 1.737, norm-entropy 0.000, unknown 0.0%


## C3_TOPIC


### variant: `neutral`  (model: LLaDA-8B-Instruct, n=52, prompts=12)

> **Headline:** C3_TOPIC topic = 62% evaluation / 15% safety / 23% efficiency / 0% multilinguality vs target (25% evaluation, 25% safety, 25% efficiency, 25% multilinguality) → TV 0.37, L1 0.73, invalid-label rate 0%.

- invalid/unknown-metadata rate: **0.0%**
- diversity: distinct-2 0.177, distinct-3 0.199, repetition 0.056, TTR 0.116


#### axis: topic

| bucket | achieved | 95% CI | target |
| --- | --- | --- | --- |
| evaluation | 0.615 | [0.333, 0.857] | 0.250 |
| safety | 0.154 | [0.000, 0.417] | 0.250 |
| efficiency | 0.231 | [0.000, 0.467] | 0.250 |
| multilinguality | 0.000 | [0.000, 0.000] | 0.250 |

TV **0.365** (95% CI [0.250, 0.607]), L1 0.731, L2 0.453, KL(emp‖target) 0.665, norm-entropy 0.667, unknown 0.0%


### variant: `weak`  (model: LLaDA-8B-Instruct, n=48, prompts=12)

> **Headline:** C3_TOPIC topic = 75% evaluation / 8% safety / 17% efficiency / 0% multilinguality vs target (25% evaluation, 25% safety, 25% efficiency, 25% multilinguality) → TV 0.50, L1 1.00, invalid-label rate 0%.

- invalid/unknown-metadata rate: **0.0%**
- diversity: distinct-2 0.183, distinct-3 0.204, repetition 0.046, TTR 0.111


#### axis: topic

| bucket | achieved | 95% CI | target |
| --- | --- | --- | --- |
| evaluation | 0.750 | [0.500, 0.917] | 0.250 |
| safety | 0.083 | [0.000, 0.250] | 0.250 |
| efficiency | 0.167 | [0.000, 0.417] | 0.250 |
| multilinguality | 0.000 | [0.000, 0.000] | 0.250 |

TV **0.500** (95% CI [0.333, 0.667]), L1 1.000, L2 0.589, KL(emp‖target) 0.959, norm-entropy 0.520, unknown 0.0%


### variant: `explicit`  (model: LLaDA-8B-Instruct, n=48, prompts=12)

> **Headline:** C3_TOPIC topic = 33% evaluation / 8% safety / 50% efficiency / 8% multilinguality vs target (25% evaluation, 25% safety, 25% efficiency, 25% multilinguality) → TV 0.33, L1 0.67, invalid-label rate 0%.

- invalid/unknown-metadata rate: **0.0%**
- diversity: distinct-2 0.196, distinct-3 0.215, repetition 0.086, TTR 0.122


#### axis: topic

| bucket | achieved | 95% CI | target |
| --- | --- | --- | --- |
| evaluation | 0.333 | [0.083, 0.583] | 0.250 |
| safety | 0.083 | [0.000, 0.250] | 0.250 |
| efficiency | 0.500 | [0.250, 0.750] | 0.250 |
| multilinguality | 0.083 | [0.000, 0.250] | 0.250 |

TV **0.333** (95% CI [0.167, 0.500]), L1 0.667, L2 0.354, KL(emp‖target) 0.374, norm-entropy 0.813, unknown 0.0%


## C1_BIO_DEMOGRAPHIC


### variant: `neutral`  (model: LLaDA-8B-Instruct, n=48, prompts=12)

> **Headline:** C1_BIO_DEMOGRAPHIC gender_identity = 0% woman / 92% man / 8% nonbinary vs target (45% woman, 45% man, 10% nonbinary) → TV 0.47, L1 0.93, invalid-label rate 0%.

- invalid/unknown-metadata rate: **0.0%**
- diversity: distinct-2 0.197, distinct-3 0.231, repetition 0.339, TTR 0.096


#### axis: gender_identity

| bucket | achieved | 95% CI | target |
| --- | --- | --- | --- |
| woman | 0.000 | [0.000, 0.000] | 0.450 |
| man | 0.917 | [0.750, 1.000] | 0.450 |
| nonbinary | 0.083 | [0.000, 0.250] | 0.100 |

TV **0.467** (95% CI [0.450, 0.550]), L1 0.933, L2 0.649, KL(emp‖target) 0.919, norm-entropy 0.261, unknown 0.0%


#### axis: home_region

| bucket | achieved | 95% CI | target |
| --- | --- | --- | --- |
| Africa | 0.000 | [0.000, 0.000] | 0.250 |
| East Asia | 0.500 | [0.250, 0.750] | 0.250 |
| Europe | 0.000 | [0.000, 0.000] | 0.250 |
| Latin America | 0.500 | [0.250, 0.750] | 0.250 |

TV **0.500** (95% CI [0.500, 0.583]), L1 1.000, L2 0.500, KL(emp‖target) 1.000, norm-entropy 0.500, unknown 0.0%


#### axis: field

_descriptive axis (no target)_ — top buckets: free text (12), astrophysics (4), philosophy (4), software science (4), Forensic Medicine (4), chef (4)


### variant: `weak`  (model: LLaDA-8B-Instruct, n=48, prompts=12)

> **Headline:** C1_BIO_DEMOGRAPHIC gender_identity = 0% woman / 75% man / 25% nonbinary vs target (45% woman, 45% man, 10% nonbinary) → TV 0.45, L1 0.90, invalid-label rate 0%.

- invalid/unknown-metadata rate: **0.0%**
- diversity: distinct-2 0.192, distinct-3 0.228, repetition 0.347, TTR 0.095


#### axis: gender_identity

| bucket | achieved | 95% CI | target |
| --- | --- | --- | --- |
| woman | 0.000 | [0.000, 0.000] | 0.450 |
| man | 0.750 | [0.500, 0.919] | 0.450 |
| nonbinary | 0.250 | [0.000, 0.500] | 0.100 |

TV **0.450** (95% CI [0.450, 0.550]), L1 0.900, L2 0.561, KL(emp‖target) 0.883, norm-entropy 0.512, unknown 0.0%


#### axis: home_region

| bucket | achieved | 95% CI | target |
| --- | --- | --- | --- |
| Africa | 0.000 | [0.000, 0.000] | 0.250 |
| East Asia | 0.818 | [0.500, 1.000] | 0.250 |
| Europe | 0.000 | [0.000, 0.000] | 0.250 |
| Latin America | 0.182 | [0.000, 0.417] | 0.250 |

TV **0.568** (95% CI [0.500, 0.750]), L1 1.136, L2 0.673, KL(emp‖target) 1.316, norm-entropy 0.342, unknown 8.3%


#### axis: field

_descriptive axis (no target)_ — top buckets: free text (12), astrophysics (4), philosophy (4), software science (4), integrative medicine (4), Techotics (4)


### variant: `explicit`  (model: LLaDA-8B-Instruct, n=48, prompts=12)

> **Headline:** C1_BIO_DEMOGRAPHIC gender_identity = 25% woman / 75% man / 0% nonbinary vs target (45% woman, 45% man, 10% nonbinary) → TV 0.30, L1 0.60, invalid-label rate 0%.

- invalid/unknown-metadata rate: **0.0%**
- diversity: distinct-2 0.196, distinct-3 0.232, repetition 0.319, TTR 0.095


#### axis: gender_identity

| bucket | achieved | 95% CI | target |
| --- | --- | --- | --- |
| woman | 0.250 | [0.000, 0.500] | 0.450 |
| man | 0.750 | [0.500, 1.000] | 0.450 |
| nonbinary | 0.000 | [0.000, 0.000] | 0.100 |

TV **0.300** (95% CI [0.100, 0.550]), L1 0.600, L2 0.374, KL(emp‖target) 0.341, norm-entropy 0.512, unknown 0.0%


#### axis: home_region

| bucket | achieved | 95% CI | target |
| --- | --- | --- | --- |
| Africa | 0.091 | [0.000, 0.250] | 0.250 |
| East Asia | 0.545 | [0.250, 0.750] | 0.250 |
| Europe | 0.000 | [0.000, 0.000] | 0.250 |
| Latin America | 0.364 | [0.083, 0.583] | 0.250 |

TV **0.409** (95% CI [0.250, 0.568]), L1 0.818, L2 0.434, KL(emp‖target) 0.678, norm-entropy 0.661, unknown 8.3%


#### axis: field

_descriptive axis (no target)_ — top buckets: Biochemistry (4), Mathematics (4), software engineer (4), novelist (4), internaliatrics (4), chef (4)


---

### Neutral takeaways

- Base sampling (and prompting) does not directly solve user-specified aggregate moment-matching — the gap later Diffusion-GDC work addresses.

- The model can produce every target category at least sometimes, yet the **aggregate** distribution is skewed away from the target.

- Framing is diagnostic: this characterizes a sampling distribution; it does not claim existing methods can *never* match a target.

