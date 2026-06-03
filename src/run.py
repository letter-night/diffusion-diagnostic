"""CLI entry point: sample -> annotate -> report.

Examples:
  python -m src.run sample   --model dry_run --case case2 --n-prompts 24 --n-per-prompt 32
  python -m src.run sample   --model llada   --case case1 --variant neutral
  python -m src.run sample   --model llada   --case all --dry-run        # force stub
  python -m src.run annotate --case case2 --model dry_run
  python -m src.run report   --case all --model dry_run

Raw completions are written incrementally (resumable): re-running `sample` skips
sample_ids already present in the output file.
"""

from __future__ import annotations

import argparse
import json
import os

from . import aggregate, report
from .config import (ROOT, load_cases, model_config, load_models, load_prompts,
                     render_prompt)
from .samplers.base import GenContext, SamplingParams, get_sampler

RESULTS = os.path.join(ROOT, "results")
CASE_ALIASES = {
    "case1": "C1_BIO_DEMOGRAPHIC", "case2": "C2_FORMAT",
    "case3": "C3_TOPIC", "case4": "C4_STYLE",
}


def _resolve_cases(arg: str, cases: dict) -> list[str]:
    if arg == "all":
        # run order: least sensitive first
        return [cid for cid, _ in sorted(
            cases.items(), key=lambda kv: kv[1].get("run_order", 99))]
    cid = CASE_ALIASES.get(arg, arg)
    if cid not in cases:
        raise SystemExit(f"Unknown case {arg!r}. Known: {list(CASE_ALIASES)} or {list(cases)}")
    return [cid]


def _raw_path(case_id, variant, model_name):
    safe = model_name.replace("/", "_")
    return os.path.join(RESULTS, "raw", f"{case_id}_{variant}_{safe}.jsonl")


def _ann_path(case_id, variant, model_name):
    safe = model_name.replace("/", "_")
    return os.path.join(RESULTS, "annotations", f"{case_id}_{variant}_{safe}.jsonl")


def _read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


# --------------------------------------------------------------------------- #
def cmd_sample(args):
    cases = load_cases()
    models = load_models()
    mcfg = model_config("dry_run" if args.dry_run else args.model, models)
    sampler = get_sampler(mcfg)
    params = SamplingParams.from_dict(mcfg.get("sampling"))
    variants = ["neutral", "weak", "explicit"] if args.variant == "all" else [args.variant]

    for case_id in _resolve_cases(args.case, cases):
        ccfg = cases[case_id]
        prompts = load_prompts(ccfg)
        if args.n_prompts:
            prompts = prompts[: args.n_prompts]
        axes = [{"field": a["field"], "values": a.get("values") or [],
                 "target": a.get("target", {})} for a in ccfg["axes"]]
        for variant in variants:
            path = _raw_path(case_id, variant, sampler.model_name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            done = {r["sample_id"] for r in _read_jsonl(path)}
            written = 0
            with open(path, "a") as out:
                for p in prompts:
                    raw_prompt = render_prompt(ccfg, variant, p["fields"])
                    for s in range(args.n_per_prompt):
                        seed = args.seed + s
                        sid = f"{p['prompt_id']}_{variant}_seed{seed:03d}"
                        if sid in done:
                            continue
                        ctx = GenContext(
                            case_id=case_id, prompt_id=p["prompt_id"], variant=variant,
                            sample_idx=s, metadata_key=ccfg["metadata_key"], axes=axes)
                        raw_output = sampler.generate(
                            raw_prompt, seed=seed, params=params, ctx=ctx)
                        rec = {
                            "case_id": case_id, "prompt_id": p["prompt_id"],
                            "prompt_variant": variant, "model_name": sampler.model_name,
                            "seed": seed, "sample_id": sid,
                            "sampling": params.to_dict(),
                            "raw_prompt": raw_prompt, "raw_output": raw_output,
                        }
                        out.write(json.dumps(rec) + "\n")
                        out.flush()
                        written += 1
            print(f"[sample] {case_id}/{variant}: +{written} "
                  f"(total {len(done)+written}) -> {os.path.relpath(path, ROOT)}")


# --------------------------------------------------------------------------- #
def cmd_annotate(args):
    from .annotate.metadata_parser import parse_metadata, split_meta_and_body
    from .annotate.content_checks import check_format
    cases = load_cases()
    model_name = model_config("dry_run" if args.dry_run else args.model)["model_name"]
    variants = ["neutral", "weak", "explicit"] if args.variant == "all" else [args.variant]

    judge = None
    if args.judge:
        from .annotate.llm_judge import LLMJudge
        judge = LLMJudge(backend=args.judge_backend,
                         cache_path=os.path.join(RESULTS, "annotations", "judge_cache.json"))

    for case_id in _resolve_cases(args.case, cases):
        ccfg = cases[case_id]
        for variant in variants:
            raw = _read_jsonl(_raw_path(case_id, variant, model_name))
            if not raw:
                continue
            out_path = _ann_path(case_id, variant, model_name)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            n = 0
            with open(out_path, "w") as out:
                for r in raw:
                    parsed = parse_metadata(
                        r["raw_output"], ccfg["metadata_key"], ccfg["axes"])
                    _, body = split_meta_and_body(r["raw_output"], ccfg["metadata_key"])
                    rec = dict(r)
                    rec["body"] = body
                    rec["valid_metadata"] = parsed["valid_metadata"]
                    rec["parsed_labels"] = parsed["parsed_labels"]
                    if ccfg.get("content_check") == "format":
                        declared = parsed["parsed_labels"].get("format", "unknown")
                        rec.update(check_format(body, declared))
                    if judge and ccfg.get("llm_judge"):
                        rec["judge"] = judge.score(body, {"case_id": case_id})
                    out.write(json.dumps(rec) + "\n")
                    n += 1
            print(f"[annotate] {case_id}/{variant}: {n} -> "
                  f"{os.path.relpath(out_path, ROOT)}")
    if judge:
        judge.flush()


# --------------------------------------------------------------------------- #
def cmd_report(args):
    cases = load_cases()
    model_name = model_config("dry_run" if args.dry_run else args.model)["model_name"]
    variants = ["neutral", "weak", "explicit"] if args.variant == "all" else [args.variant]

    summaries = []
    for case_id in _resolve_cases(args.case, cases):
        ccfg = cases[case_id]
        for variant in variants:
            recs = _read_jsonl(_ann_path(case_id, variant, model_name))
            if not recs:
                continue
            summaries.append(aggregate.aggregate_case(
                recs, ccfg, variant, n_boot=args.n_boot, ci=args.ci, seed=args.seed))

    if not summaries:
        raise SystemExit("No annotated records found. Run sample + annotate first.")

    agg_dir = os.path.join(RESULTS, "aggregate")
    report.write_aggregate_outputs(summaries, agg_dir)
    report.write_report_md(summaries, os.path.join(RESULTS, "report.md"))
    figs = report.make_figures(summaries, os.path.join(RESULTS, "figures"))
    print(f"[report] {len(summaries)} case×variant summaries -> "
          f"{os.path.relpath(agg_dir, ROOT)}, report.md, {len(figs)} figures")


# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(prog="src.run", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", default="dry_run")
    common.add_argument("--case", default="all")
    common.add_argument("--variant", default="neutral",
                        choices=["neutral", "weak", "explicit", "all"])
    common.add_argument("--dry-run", action="store_true",
                        help="Force the dry-run stub regardless of --model.")

    s = sub.add_parser("sample", parents=[common], help="Generate completions.")
    s.add_argument("--n-prompts", type=int, default=0, help="0 = all prompts in file.")
    s.add_argument("--n-per-prompt", type=int, default=32)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_sample)

    a = sub.add_parser("annotate", parents=[common], help="Parse + content-check.")
    a.add_argument("--judge", action="store_true", help="Run optional LLM judge (Case 3).")
    a.add_argument("--judge-backend", default="offline", choices=["offline", "api"])
    a.set_defaults(func=cmd_annotate)

    r = sub.add_parser("report", parents=[common], help="Aggregate + report + figures.")
    r.add_argument("--n-boot", type=int, default=1000)
    r.add_argument("--ci", type=float, default=0.95)
    r.add_argument("--seed", type=int, default=0)
    r.set_defaults(func=cmd_report)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
