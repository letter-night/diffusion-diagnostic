# Real results — LLaDA-8B-Instruct

Actual diagnostic outputs from **`GSAI-ML/LLaDA-8B-Instruct`** (bf16, A40 48 GB), unlike the
synthetic dry-run example at `results/`. Pilot size: **12 prompts × 4 seeds = 48 completions
per case×variant** (576 total), sampling `steps=128, gen_length=256, block_length=32,
temperature=0, remasking=low_confidence`. Generated with the official LLaDA denoising loop.

Regenerate:

```bash
pip install -r requirements.txt -r requirements-gpu.txt   # torch built for your CUDA driver
python -m src.run sample   --model llada --case all --variant all --n-prompts 12 --n-per-prompt 4
python -m src.run annotate --model llada --case all --variant all --judge
python -m src.run report   --model llada --case all --variant all
```

## Headline: distributional collapse, and prompting does not fix it

Total variation distance vs. target (0 = matches target; 0.75/0.70 ≈ collapse onto one bucket):

| case / axis | neutral | weak | explicit |
| --- | --- | --- | --- |
| C2 format | **0.750** | **0.750** | 0.357 |
| C4 style | **0.700** | **0.700** | **0.700** |
| C3 topic | 0.365 | 0.500 | 0.333 |
| C1 gender | 0.467 | 0.450 | 0.300 |
| C1 region | 0.500 | 0.568 | 0.409 |

Dominant bucket under neutral prompting:

- format → **100% plain_paragraph** (target 25%)
- style → **100% analogy_centered** (target 30%)
- topic → **62% evaluation** (target 25%)
- gender → **92% man** (target 45%)
- region → **50% East Asia** (target 25%)

### Takeaways (diagnostic framing)

1. **Real collapse**, not simulated: two axes fall onto a single bucket 100% of the time.
2. **Prompting does not reliably solve aggregate moment-matching.** "Try to vary" (weak)
   helped nowhere and made topic/region slightly worse; explicit numeric targets helped some
   axes but did **nothing** for style (0.70 → 0.70) and never reached target on any axis.
   This is exactly the gap later Diffusion-GDC work motivates.
3. **Label-vs-content gap:** under explicit pressure on Case 2, the model *declares* varied
   formats but the body stops conforming (`format_label_matches_body` → 0%).
4. **Metadata reliability:** invalid-label rate **0%** across all cases — the self-reported
   metadata line is robust (the ~4% in the dry-run example is synthetic).

Framing stays diagnostic: this characterizes a sampling distribution at these settings; it
does not claim existing methods can *never* match a target. Bootstrap CIs (over prompt IDs)
and per-bucket detail are in `report.md` and `aggregate/`.

> Pilot caveat: n=48 per case×variant gives wide CIs on small buckets; scale `--n-per-prompt`
> for tighter intervals. The raw completions + per-item annotations are not committed (bulky);
> regenerate with the commands above.
