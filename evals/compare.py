"""Single-agent vs multi-agent report pipeline, side by side, from two committed runs (no network).

    python -m evals.compare --single evals/runs/<single>.jsonl --multi evals/runs/<multi>.jsonl [--out PATH]

Writes evals/results/<multi run date>_single_vs_multi.md (or --out). Adds the multi run's verifier audit
(evals.verifier_audit) and evals/verifier_sensitivity.json when they exist, and appends
<out>.notes.md (hand-written: what we learned, the decision) like run notes. The decision rule from the
comparison brief is evaluated mechanically: multi becomes the default only if it is at least as good as
single on every deterministic check AND has no more template fallbacks.
"""
import argparse
import json
import math
import sys
from datetime import date, datetime
from pathlib import Path

from agent.multi_agent import REQUIRED_TOOLS
from evals import multi_metrics
from evals.checks import CHECKS, pass_rates
from evals.dataset import dataset_hash, load_cases
from evals.run import (
    REPO, RESULTS_DIR, _rejection_kind, audit_path, check_dataset_hash, load_run, relative, run_pipeline,
)
from evals.schema import CRITERIA
from services.tracing import estimate_cost

SENSITIVITY_PATH = REPO / "evals" / "verifier_sensitivity.json"


def percentile(xs, p):
    """Nearest-rank percentile (p in 0-100) of a non-empty list."""
    s = sorted(xs)
    return s[max(0, math.ceil(p / 100 * len(s)) - 1)]


def median(xs):
    s = sorted(xs)
    n = len(s)
    return (s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2) if s else None


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def tokens_by_model(record):
    """{model: (input, output)} for a report; records from before by_model are all the agent model."""
    usage = record.get("agent_usage") or {}
    if usage.get("by_model"):
        return {m: (u["input_tokens"], u["output_tokens"]) for m, u in usage["by_model"].items()}
    return {record["agent_model"]: (usage.get("input_tokens", 0), usage.get("output_tokens", 0))}


def report_cost(record):
    """Nominal USD for the report's LLM calls (config/pricing.json at the record's date), and billed USD."""
    day = datetime.fromisoformat(record["recorded_at"]).date()
    nominal = billed = 0.0
    for model, (i, o) in tokens_by_model(record).items():
        c = estimate_cost(model, i, o, on=day)
        if c["nominal_cost_usd"] is None:
            return None, None
        nominal += c["nominal_cost_usd"]
        billed += c["billed_cost_usd"]
    return nominal, billed


def guard_rejections(record):
    """Drafts the guard rejected, and API-level model errors (no draft), as evals.run's summary counts them."""
    kinds = [_rejection_kind(x) for x in record.get("rejected") or []]
    return kinds.count("guard"), kinds.count("model_error")


def side(records):
    """Everything the table needs for one run (records: report records of one run)."""
    rates = pass_rates(records)
    judged = [r for r in records if r.get("judge")]
    fallbacks = [r for r in records if r["report_source"] != "llm"]
    rej = [guard_rejections(r) for r in records]
    latency = [r["agent_usage"]["report_latency_ms"] / 1000 for r in records]
    models = {}
    for r in records:
        for m, (i, o) in tokens_by_model(r).items():
            models.setdefault(m, []).append(i + o)
    costs = [report_cost(r) for r in records]
    times = sorted(datetime.fromisoformat(r["recorded_at"]) for r in records)
    judge_tokens = [sum(v or 0 for v in (r["judge"].get("usage") or {}).values()) for r in judged]
    return {
        "n": len(records), "rates": rates, "fallbacks": fallbacks,
        "judge": {c: [r["judge"]["scores"][c]["score"] for r in judged] for c in CRITERIA},
        "judge_model": judged[0]["judge"]["model"] if judged else None,
        "guard_drafts": sum(d for d, _ in rej), "guard_cases": sum(1 for d, _ in rej if d),
        "model_errors": sum(e for _, e in rej),
        "revisions": multi_metrics.revision_metrics(records),
        "tool_skips": [r["case_id"] for r in records if not set(REQUIRED_TOOLS) <= set(r.get("tool_calls") or [])],
        "reasks": sum(r.get("evidence_reasks") or 0 for r in records),
        "calls": mean([r["agent_usage"]["llm_calls"] for r in records]),
        "tokens": {m: (mean(v) * len(v) / len(records), len(v)) for m, v in models.items()},
        "latency": latency,
        "nominal": None if any(c[0] is None for c in costs) else [c[0] for c in costs],
        "billed": None if any(c[1] is None for c in costs) else [c[1] for c in costs],
        "judge_tokens": mean(judge_tokens),
        "span": (times[-1] - times[0]).total_seconds() if times else None,
        "pipeline": run_pipeline({r["case_id"]: r for r in records}),
        "agent_models": records[0].get("agent_models") or {"report": records[0]["agent_model"]},
        "commit": records[0].get("git_commit"),
    }


def decide(s, m):
    """The decision rule: (switch to multi?, [reasons it fails])."""
    reasons = []
    for name in CHECKS:
        a, b = s["rates"][name]["rate"], m["rates"][name]["rate"]
        if a is not None and (b is None or b < a - 1e-9):
            reasons.append(f"{name}: multi {_pct(b)} < single {_pct(a)}")
    if len(m["fallbacks"]) > len(s["fallbacks"]):
        reasons.append(f"template fallbacks: multi {len(m['fallbacks'])} > single {len(s['fallbacks'])}")
    return not reasons, reasons


def _total(xs):
    return None if xs is None else sum(xs)  # None: a model without a price in config/pricing.json


def _pct(x):
    return "—" if x is None else f"{x:.1%}"


def _f(x, digits=2):
    return "—" if x is None else f"{x:,.{digits}f}"


def build(single_path, multi_path):
    cases = load_cases()
    data_hash = dataset_hash(cases)
    ids = {c.case_id for c in cases}
    runs = []
    for path in (single_path, multi_path):
        records = load_run(path)
        check_dataset_hash(records, data_hash, path)
        runs.append([r for r in records.values() if r.get("report") and r["case_id"] in ids])
    s, m = side(runs[0]), side(runs[1])
    if s["pipeline"] != "single" or m["pipeline"] != "multi":
        raise SystemExit(f"Expected a single run and a multi run, got {s['pipeline']} and {m['pipeline']}.")
    switch, reasons = decide(s, m)

    lines = [f"# Single vs multi-agent report pipeline: `{Path(single_path).stem}` vs `{Path(multi_path).stem}`", "",
             f"- Dataset: {len(cases)} cases, hash `{data_hash[:12]}`; reports: single {s['n']}, multi {m['n']}",
             f"- Single: {', '.join(f'{a} {v}' for a, v in s['agent_models'].items())}"
             + (f" (commit `{s['commit'][:12]}`)" if s["commit"] else ""),
             f"- Multi: {', '.join(f'{a} {v}' for a, v in m['agent_models'].items())}"
             + (f" (commit `{m['commit'][:12]}`)" if m["commit"] else ""),
             f"- Judge: single {s['judge_model']}, multi {m['judge_model']} (same model, prompt and rubric)",
             "- All numbers are measured on these two runs. Latency includes waiting for Groq's per-minute token "
             "limit, so it measures throughput under the free tier as much as model speed.",
             "", "## Side by side", "", "| Metric | Single | Multi |", "| --- | --- | --- |"]
    for name in CHECKS:
        a, b = s["rates"][name], m["rates"][name]
        mark = " ⚠️" if a["rate"] is not None and (b["rate"] is None or b["rate"] < a["rate"] - 1e-9) else ""
        lines.append(f"| Check: {name} | {_pct(a['rate'])} ({a['passed']}/{a['applicable']}) | "
                     f"{_pct(b['rate'])} ({b['passed']}/{b['applicable']}){mark} |")
    for c in CRITERIA:
        lines.append(f"| Judge: {c} (mean; scores ≤ 2) | {_f(mean(s['judge'][c]))}; {sum(x <= 2 for x in s['judge'][c])} | "
                     f"{_f(mean(m['judge'][c]))}; {sum(x <= 2 for x in m['judge'][c])} |")
    lines += [
        f"| Template fallbacks | {len(s['fallbacks'])} | {len(m['fallbacks'])} |",
        f"| Guard rejections (drafts / cases) | {s['guard_drafts']} / {s['guard_cases']} | "
        f"{m['guard_drafts']} / {m['guard_cases']} |",
        f"| API-level model errors (e.g. invalid tool call) | {s['model_errors']} | {m['model_errors']} |",
        f"| Revisions (total; by guard / verifier; cases) | — (no revision step) | {m['revisions']['total']}; "
        f"{m['revisions']['by']['guard']} / {m['revisions']['by']['verifier']}; {m['revisions']['cases_revised']} |",
        f"| Tool skips (cases without all 3 tools) | {len(s['tool_skips'])} | {len(m['tool_skips'])} "
        f"(re-asks: {m['reasks']}) |",
        f"| LLM calls per report (mean) | {_f(s['calls'])} | {_f(m['calls'])} |",
    ]
    for model in sorted(set(s["tokens"]) | set(m["tokens"])):
        a, b = s["tokens"].get(model), m["tokens"].get(model)
        lines.append(f"| {model} tokens per report (mean, in + out) | {_f(a[0], 0) if a else '—'} | "
                     f"{_f(b[0], 0) if b else '—'} |")
    lines += [
        f"| Judge tokens per report (mean) | {_f(s['judge_tokens'], 0)} | {_f(m['judge_tokens'], 0)} |",
        f"| Latency per report, median | {_f(median(s['latency']), 1)} s | {_f(median(m['latency']), 1)} s |",
        f"| Latency per report, p90 | {_f(percentile(s['latency'], 90), 1)} s | {_f(percentile(m['latency'], 90), 1)} s |",
        f"| Run span, first to last record | {_f(s['span'] / 60, 1)} min | {_f(m['span'] / 60, 1)} min |",
        f"| Nominal cost per report (list price, report models) | ${_f(mean(s['nominal'] or []), 5)} | "
        f"${_f(mean(m['nominal'] or []), 5)} |",
        f"| Nominal cost, all {s['n']} / {m['n']} reports | ${_f(_total(s['nominal']), 4)} | ${_f(_total(m['nominal']), 4)} |",
        f"| Billed cost (free tier) | ${_f(_total(s['billed']), 2)} | ${_f(_total(m['billed']), 2)} |",
        "", "Fallback reasons: single " + ("; ".join(sorted({r["fallback_reason"] for r in s["fallbacks"]})) or "none")
        + "; multi " + ("; ".join(f"`{r['case_id']}` {r['fallback_reason']}" for r in m["fallbacks"]) or "none") + ".",
        "",
    ]
    multi_records = runs[1]
    lines += multi_metrics.summary_lines(multi_records, _shadow(multi_path))
    lines += audit_lines(multi_path) + sensitivity_lines()
    lines += ["## Decision rule", "",
              "Multi becomes the default only if it is at least as good as single on every deterministic check AND "
              "has no more template fallbacks.", "",
              f"**Result: {'switch to multi' if switch else 'keep single'}**"
              + ("" if switch else " — " + "; ".join(reasons)), ""]
    return "\n".join(lines) + "\n"


def _shadow(multi_path):
    path = audit_path(multi_path)
    return json.loads(path.read_text(encoding="utf-8"))["shadow_verifier"] if path.exists() else None


def audit_lines(multi_path):
    path = audit_path(multi_path)
    if not path.exists():
        return ["## Verifier audit", "", "Not run yet (python -m evals.verifier_audit).", ""]
    a = json.loads(path.read_text(encoding="utf-8"))
    checks = [c for c in a["flag_checks"] if not c.get("error")]
    likely_false = [c for c in checks if c["checker_supported"]]
    shadow = [x for x in a["shadow_verifier"] if not x.get("error")]
    lines = ["## Verifier audit", "",
             f"Post-hoc calls ({relative(path)}): verifier {a.get('verifier_model')}, claim checker "
             f"{a.get('checker_model')}. Spot-check sheet: `{relative(Path(multi_path).with_suffix('.verifier_flags.md'))}`.", "",
             f"- **Verifier false flags (per the checker): {len(likely_false)} of {len(checks)} unsupported claims** "
             "were found supported by the evidence. A human hasn't confirmed these yet; see the sheet.",
             f"- Guard-rejected drafts shown to the verifier afterwards: {len(shadow)}; it found every claim supported "
             f"in {sum(1 for x in shadow if x['unsupported'] == 0)} (the guard caught something the verifier would "
             "have let through):"]
    for x in shadow:
        lines.append(f"  - `{x['run_id']}` `{x['case_id']}` step {x['step']}: {x['unsupported']} unsupported; "
                     f"guard: {x['guard_problem'][:200]}")
    drift = [d for d in a["judge_drift"] if not d.get("error")]
    if drift:
        same = sum(d["recorded"][c] == d["now"][c] for d in drift for c in CRITERIA)
        diff = [d["now"][c] - d["recorded"][c] for d in drift for c in CRITERIA]
        lines.append(f"- Judge drift check: {len(drift)} single-run reports re-judged today; {same}/{len(diff)} scores "
                     f"identical, mean change {_f(mean(diff))} (" + ", ".join(
                         f"`{d['case_id'].split('-')[0]}` {'/'.join(str(d['recorded'][c]) for c in CRITERIA)} → "
                         f"{'/'.join(str(d['now'][c]) for c in CRITERIA)}" for d in drift) + ")")
    return lines + [""]


def sensitivity_lines():
    if not SENSITIVITY_PATH.exists():
        return []
    data = json.loads(SENSITIVITY_PATH.read_text(encoding="utf-8"))
    bad = [r for r in data["results"] if r["kind"] == "bad"]
    controls = [r for r in data["results"] if r["kind"] == "control"]
    return ["## Verifier on the planted bad reports", "",
            f"`evals/verifier_sensitivity.md` ({data['verifier_model']}, {data['recorded_at'][:10]}): caught "
            f"{sum(r['flagged_by_verifier'] for r in bad)}/{len(bad)} bad reports "
            f"({', '.join(r['id'] for r in bad if r['flagged_by_verifier']) or 'none'}); false alarms on "
            f"{sum(r['flagged_by_verifier'] for r in controls)}/{len(controls)} untouched controls. With the guard "
            f"first, the pipeline stops {sum(r['pipeline'] != 'passes' for r in bad)}/{len(bad)}.", ""]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--single", required=True)
    ap.add_argument("--multi", required=True)
    ap.add_argument("--out", help="default: evals/results/<multi run date>_single_vs_multi.md")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    md = build(Path(args.single), Path(args.multi))
    stem = Path(args.multi).stem
    out = Path(args.out) if args.out else \
        RESULTS_DIR / f"{date(int(stem[:4]), int(stem[4:6]), int(stem[6:8])).isoformat()}_single_vs_multi.md"
    notes = out.with_suffix(".notes.md")
    if notes.exists():
        md += "\n" + notes.read_text(encoding="utf-8").strip() + "\n"
    out.write_text(md, encoding="utf-8")
    print(md)
    print(f"Wrote {relative(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
