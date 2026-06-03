# Diagnostic Study: Distributional Failure in Diffusion LMs

_Measurement-only report. Characterizes the base model's aggregate sampling distribution against user-specified targets. No fine-tuning, no reward models, no GDC._


## C2_FORMAT


### variant: `neutral`  (model: dry-run-stub, n=768, prompts=24)

> **Headline:** C2_FORMAT format = 77% json / 9% markdown_table / 7% numbered_list / 7% plain_paragraph vs target (25% json, 25% markdown_table, 25% numbered_list, 25% plain_paragraph) → TV 0.52, L1 1.05, invalid-label rate 4%.

- invalid/unknown-metadata rate: **4.0%**
- format label matches body: **82.8%**
- diversity: distinct-2 0.008, distinct-3 0.011, repetition 0.063, TTR 0.006


#### axis: format

| bucket | achieved | 95% CI | target |
| --- | --- | --- | --- |
| json | 0.773 | [0.708, 0.775] | 0.250 |
| markdown_table | 0.085 | [0.059, 0.104] | 0.250 |
| numbered_list | 0.068 | [0.044, 0.089] | 0.250 |
| plain_paragraph | 0.073 | [0.052, 0.089] | 0.250 |

TV **0.523** (95% CI [0.488, 0.557]), L1 1.047, L2 0.605, KL(emp‖target) 0.870, norm-entropy 0.565, unknown 4.0%


### variant: `weak`  (model: dry-run-stub, n=768, prompts=24)

> **Headline:** C2_FORMAT format = 71% json / 10% markdown_table / 10% numbered_list / 10% plain_paragraph vs target (25% json, 25% markdown_table, 25% numbered_list, 25% plain_paragraph) → TV 0.46, L1 0.92, invalid-label rate 3%.

- invalid/unknown-metadata rate: **3.3%**
- format label matches body: **85.2%**
- diversity: distinct-2 0.008, distinct-3 0.010, repetition 0.080, TTR 0.006


#### axis: format

| bucket | achieved | 95% CI | target |
| --- | --- | --- | --- |
| json | 0.708 | [0.651, 0.716] | 0.250 |
| markdown_table | 0.096 | [0.070, 0.116] | 0.250 |
| numbered_list | 0.100 | [0.068, 0.125] | 0.250 |
| plain_paragraph | 0.097 | [0.074, 0.112] | 0.250 |

TV **0.458** (95% CI [0.424, 0.489]), L1 0.916, L2 0.529, KL(emp‖target) 0.666, norm-entropy 0.667, unknown 3.3%


### variant: `explicit`  (model: dry-run-stub, n=768, prompts=24)

> **Headline:** C2_FORMAT format = 64% json / 13% markdown_table / 12% numbered_list / 12% plain_paragraph vs target (25% json, 25% markdown_table, 25% numbered_list, 25% plain_paragraph) → TV 0.39, L1 0.78, invalid-label rate 4%.

- invalid/unknown-metadata rate: **4.0%**
- format label matches body: **85.7%**
- diversity: distinct-2 0.008, distinct-3 0.010, repetition 0.102, TTR 0.005


#### axis: format

| bucket | achieved | 95% CI | target |
| --- | --- | --- | --- |
| json | 0.639 | [0.582, 0.643] | 0.250 |
| markdown_table | 0.130 | [0.103, 0.148] | 0.250 |
| numbered_list | 0.115 | [0.094, 0.129] | 0.250 |
| plain_paragraph | 0.115 | [0.089, 0.135] | 0.250 |

TV **0.389** (95% CI [0.355, 0.423]), L1 0.778, L2 0.449, KL(emp‖target) 0.485, norm-entropy 0.757, unknown 4.0%


## C4_STYLE


### variant: `neutral`  (model: dry-run-stub, n=512, prompts=16)

> **Headline:** C4_STYLE style = 78% definition_first / 6% analogy_centered / 6% socratic_tutor / 10% formal_academic vs target (30% definition_first, 30% analogy_centered, 20% socratic_tutor, 20% formal_academic) → TV 0.48, L1 0.96, invalid-label rate 5%.

- invalid/unknown-metadata rate: **5.3%**
- diversity: distinct-2 0.005, distinct-3 0.006, repetition 0.000, TTR 0.004


#### axis: style

| bucket | achieved | 95% CI | target |
| --- | --- | --- | --- |
| definition_first | 0.779 | [0.697, 0.779] | 0.300 |
| analogy_centered | 0.062 | [0.043, 0.074] | 0.300 |
| socratic_tutor | 0.060 | [0.039, 0.076] | 0.200 |
| formal_academic | 0.099 | [0.068, 0.125] | 0.200 |

TV **0.479** (95% CI [0.446, 0.517]), L1 0.959, L2 0.562, KL(emp‖target) 0.728, norm-entropy 0.551, unknown 5.3%


### variant: `weak`  (model: dry-run-stub, n=512, prompts=16)

> **Headline:** C4_STYLE style = 73% definition_first / 8% analogy_centered / 11% socratic_tutor / 8% formal_academic vs target (30% definition_first, 30% analogy_centered, 20% socratic_tutor, 20% formal_academic) → TV 0.43, L1 0.85, invalid-label rate 3%.

- invalid/unknown-metadata rate: **3.3%**
- diversity: distinct-2 0.005, distinct-3 0.007, repetition 0.000, TTR 0.004


#### axis: style

| bucket | achieved | 95% CI | target |
| --- | --- | --- | --- |
| definition_first | 0.725 | [0.666, 0.738] | 0.300 |
| analogy_centered | 0.083 | [0.055, 0.105] | 0.300 |
| socratic_tutor | 0.111 | [0.080, 0.135] | 0.200 |
| formal_academic | 0.081 | [0.055, 0.102] | 0.200 |

TV **0.425** (95% CI [0.390, 0.464]), L1 0.851, L2 0.500, KL(emp‖target) 0.570, norm-entropy 0.640, unknown 3.3%


### variant: `explicit`  (model: dry-run-stub, n=512, prompts=16)

> **Headline:** C4_STYLE style = 58% definition_first / 13% analogy_centered / 15% socratic_tutor / 14% formal_academic vs target (30% definition_first, 30% analogy_centered, 20% socratic_tutor, 20% formal_academic) → TV 0.28, L1 0.56, invalid-label rate 6%.

- invalid/unknown-metadata rate: **5.9%**
- diversity: distinct-2 0.006, distinct-3 0.008, repetition 0.000, TTR 0.004


#### axis: style

| bucket | achieved | 95% CI | target |
| --- | --- | --- | --- |
| definition_first | 0.581 | [0.504, 0.588] | 0.300 |
| analogy_centered | 0.133 | [0.092, 0.158] | 0.300 |
| socratic_tutor | 0.149 | [0.111, 0.170] | 0.200 |
| formal_academic | 0.137 | [0.107, 0.150] | 0.200 |

TV **0.281** (95% CI [0.242, 0.319]), L1 0.562, L2 0.337, KL(emp‖target) 0.260, norm-entropy 0.822, unknown 5.9%


## C3_TOPIC


### variant: `neutral`  (model: dry-run-stub, n=512, prompts=16)

> **Headline:** C3_TOPIC topic = 11% evaluation / 8% safety / 77% efficiency / 5% multilinguality vs target (25% evaluation, 25% safety, 25% efficiency, 25% multilinguality) → TV 0.52, L1 1.03, invalid-label rate 4%.

- invalid/unknown-metadata rate: **4.3%**
- diversity: distinct-2 0.005, distinct-3 0.006, repetition 0.101, TTR 0.003


#### axis: topic

| bucket | achieved | 95% CI | target |
| --- | --- | --- | --- |
| evaluation | 0.106 | [0.074, 0.135] | 0.250 |
| safety | 0.076 | [0.053, 0.092] | 0.250 |
| efficiency | 0.767 | [0.699, 0.771] | 0.250 |
| multilinguality | 0.051 | [0.029, 0.070] | 0.250 |

TV **0.517** (95% CI [0.485, 0.549]), L1 1.035, L2 0.599, KL(emp‖target) 0.863, norm-entropy 0.569, unknown 4.3%


### variant: `weak`  (model: dry-run-stub, n=512, prompts=16)

> **Headline:** C3_TOPIC topic = 8% evaluation / 11% safety / 71% efficiency / 11% multilinguality vs target (25% evaluation, 25% safety, 25% efficiency, 25% multilinguality) → TV 0.46, L1 0.91, invalid-label rate 4%.

- invalid/unknown-metadata rate: **3.9%**
- diversity: distinct-2 0.005, distinct-3 0.006, repetition 0.098, TTR 0.003


#### axis: topic

| bucket | achieved | 95% CI | target |
| --- | --- | --- | --- |
| evaluation | 0.081 | [0.061, 0.096] | 0.250 |
| safety | 0.106 | [0.084, 0.119] | 0.250 |
| efficiency | 0.707 | [0.639, 0.723] | 0.250 |
| multilinguality | 0.106 | [0.076, 0.133] | 0.250 |

TV **0.457** (95% CI [0.417, 0.496]), L1 0.915, L2 0.528, KL(emp‖target) 0.667, norm-entropy 0.667, unknown 3.9%


### variant: `explicit`  (model: dry-run-stub, n=512, prompts=16)

> **Headline:** C3_TOPIC topic = 15% evaluation / 12% safety / 59% efficiency / 13% multilinguality vs target (25% evaluation, 25% safety, 25% efficiency, 25% multilinguality) → TV 0.34, L1 0.69, invalid-label rate 4%.

- invalid/unknown-metadata rate: **4.1%**
- diversity: distinct-2 0.005, distinct-3 0.007, repetition 0.105, TTR 0.003


#### axis: topic

| bucket | achieved | 95% CI | target |
| --- | --- | --- | --- |
| evaluation | 0.155 | [0.121, 0.176] | 0.250 |
| safety | 0.122 | [0.094, 0.139] | 0.250 |
| efficiency | 0.595 | [0.527, 0.615] | 0.250 |
| multilinguality | 0.128 | [0.096, 0.150] | 0.250 |

TV **0.345** (95% CI [0.300, 0.388]), L1 0.689, L2 0.399, KL(emp‖target) 0.387, norm-entropy 0.807, unknown 4.1%


## C1_BIO_DEMOGRAPHIC


### variant: `neutral`  (model: dry-run-stub, n=768, prompts=24)

> **Headline:** C1_BIO_DEMOGRAPHIC gender_identity = 10% woman / 80% man / 10% nonbinary vs target (45% woman, 45% man, 10% nonbinary) → TV 0.35, L1 0.70, invalid-label rate 4%.

- invalid/unknown-metadata rate: **3.9%**
- diversity: distinct-2 0.002, distinct-3 0.002, repetition 0.000, TTR 0.002


#### axis: gender_identity

| bucket | achieved | 95% CI | target |
| --- | --- | --- | --- |
| woman | 0.099 | [0.069, 0.122] | 0.450 |
| man | 0.799 | [0.740, 0.801] | 0.450 |
| nonbinary | 0.102 | [0.074, 0.121] | 0.100 |

TV **0.351** (95% CI [0.328, 0.384]), L1 0.702, L2 0.495, KL(emp‖target) 0.449, norm-entropy 0.583, unknown 3.9%


#### axis: home_region

| bucket | achieved | 95% CI | target |
| --- | --- | --- | --- |
| Africa | 0.075 | [0.055, 0.090] | 0.250 |
| East Asia | 0.069 | [0.051, 0.083] | 0.250 |
| Europe | 0.767 | [0.710, 0.766] | 0.250 |
| Latin America | 0.089 | [0.065, 0.107] | 0.250 |

TV **0.517** (95% CI [0.488, 0.547]), L1 1.034, L2 0.597, KL(emp‖target) 0.849, norm-entropy 0.575, unknown 3.9%


#### axis: field

_descriptive axis (no target)_ — top buckets: particle physics (566), agronomy (66), glaciology (55), marine ecology (51), unknown (30)


### variant: `weak`  (model: dry-run-stub, n=768, prompts=24)

> **Headline:** C1_BIO_DEMOGRAPHIC gender_identity = 13% woman / 73% man / 13% nonbinary vs target (45% woman, 45% man, 10% nonbinary) → TV 0.32, L1 0.63, invalid-label rate 4%.

- invalid/unknown-metadata rate: **3.8%**
- diversity: distinct-2 0.002, distinct-3 0.002, repetition 0.000, TTR 0.002


#### axis: gender_identity

| bucket | achieved | 95% CI | target |
| --- | --- | --- | --- |
| woman | 0.134 | [0.104, 0.156] | 0.450 |
| man | 0.733 | [0.672, 0.737] | 0.450 |
| nonbinary | 0.133 | [0.103, 0.155] | 0.100 |

TV **0.316** (95% CI [0.288, 0.342]), L1 0.632, L2 0.426, KL(emp‖target) 0.337, norm-entropy 0.696, unknown 3.8%


#### axis: home_region

| bucket | achieved | 95% CI | target |
| --- | --- | --- | --- |
| Africa | 0.104 | [0.074, 0.129] | 0.250 |
| East Asia | 0.115 | [0.094, 0.130] | 0.250 |
| Europe | 0.697 | [0.639, 0.702] | 0.250 |
| Latin America | 0.084 | [0.064, 0.098] | 0.250 |

TV **0.447** (95% CI [0.413, 0.482]), L1 0.894, L2 0.517, KL(emp‖target) 0.638, norm-entropy 0.681, unknown 3.8%


#### axis: field

_descriptive axis (no target)_ — top buckets: particle physics (515), marine ecology (85), glaciology (77), agronomy (62), unknown (29)


### variant: `explicit`  (model: dry-run-stub, n=768, prompts=24)

> **Headline:** C1_BIO_DEMOGRAPHIC gender_identity = 16% woman / 65% man / 19% nonbinary vs target (45% woman, 45% man, 10% nonbinary) → TV 0.29, L1 0.57, invalid-label rate 3%.

- invalid/unknown-metadata rate: **3.1%**
- diversity: distinct-2 0.002, distinct-3 0.002, repetition 0.000, TTR 0.002


#### axis: gender_identity

| bucket | achieved | 95% CI | target |
| --- | --- | --- | --- |
| woman | 0.163 | [0.142, 0.175] | 0.450 |
| man | 0.648 | [0.592, 0.663] | 0.450 |
| nonbinary | 0.190 | [0.163, 0.206] | 0.100 |

TV **0.287** (95% CI [0.269, 0.304]), L1 0.575, L2 0.360, KL(emp‖target) 0.277, norm-entropy 0.812, unknown 3.1%


#### axis: home_region

| bucket | achieved | 95% CI | target |
| --- | --- | --- | --- |
| Africa | 0.136 | [0.103, 0.159] | 0.250 |
| East Asia | 0.113 | [0.085, 0.135] | 0.250 |
| Europe | 0.616 | [0.560, 0.634] | 0.250 |
| Latin America | 0.136 | [0.104, 0.158] | 0.250 |

TV **0.366** (95% CI [0.326, 0.407]), L1 0.731, L2 0.423, KL(emp‖target) 0.432, norm-entropy 0.784, unknown 3.1%


#### axis: field

_descriptive axis (no target)_ — top buckets: particle physics (458), agronomy (101), glaciology (101), marine ecology (84), unknown (24)


---

### Neutral takeaways

- Base sampling (and prompting) does not directly solve user-specified aggregate moment-matching — the gap later Diffusion-GDC work addresses.

- The model can produce every target category at least sometimes, yet the **aggregate** distribution is skewed away from the target.

- Framing is diagnostic: this characterizes a sampling distribution; it does not claim existing methods can *never* match a target.

