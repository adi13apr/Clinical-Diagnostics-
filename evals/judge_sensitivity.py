"""Judge sensitivity: does the LLM judge score deliberately bad reports low?

    python -m evals.judge_sensitivity             # live judge (needs a key); writes evals/judge_sensitivity.{json,md}
    python -m evals.judge_sensitivity --verifier  # the multi-agent verifier on the same reports (+ controls)

Each case takes a real report and its real tool outputs from a committed run, breaks exactly
one thing, and records the judge's scores. A case counts as caught when the criterion that was
broken scores <= CAUGHT_AT_OR_BELOW. The agent's guard verdict on the same report is recorded
too (offline), to show which problems only the judge could catch.
"""
import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from agent.agent import parse_report
from evals.schema import CRITERIA

REPO = Path(__file__).resolve().parent.parent
BASE_RUN = REPO / "evals" / "runs" / "20261018T161443Z.jsonl"  # human-reviewed run; its reports were judged good
OUT_JSON = REPO / "evals" / "judge_sensitivity.json"
OUT_MD = REPO / "evals" / "judge_sensitivity.md"
VERIFIER_OUT_JSON = REPO / "evals" / "verifier_sensitivity.json"  # --verifier: the multi-agent verifier instead
VERIFIER_OUT_MD = REPO / "evals" / "verifier_sensitivity.md"
CAUGHT_AT_OR_BELOW = 2

CASES = [
    {"id": "wrong_class", "base": "c33-person843_bacteria_2763", "target": "faithfulness",
     "what": "Says Tuberculosis; the classifier said Pneumonia (1.0)",
     "update": {"predicted_class": "Tuberculosis",
                "summary": "The AI model identified patterns on the X-ray that resemble tuberculosis, with a "
                           "calibrated probability of 1.0. The attention map shows a diffuse focus, especially in "
                           "the left upper zone, which reflects model attention rather than confirmed anatomy."}},
    {"id": "missing_referral", "base": "c05-IM-0691-0001", "target": "actionability",
     "what": "Uncertain (refer) Pneumonia result, but the advice is self-care with no clinician review",
     "update": {"recommendation": "Rest at home, drink plenty of fluids and take over-the-counter fever "
                                  "medicine if needed."}},
    {"id": "tb_no_follow_up", "base": "c35-Tuberculosis-327", "target": "actionability",
     "what": "Tuberculosis (1.0) with 'No follow-up is needed.'",
     "update": {"recommendation": "No follow-up is needed."}},
    {"id": "overconfident", "base": "c05-IM-0691-0001", "target": "uncertainty_language",
     "what": "Uncertain result (0.58, refer) described as certain and definite",
     "update": {"summary": "The X-ray clearly shows pneumonia. The AI is certain about this result, so the "
                           "diagnosis is definite and there is no doubt about it."}},
    {"id": "invented_symptom", "base": "c33-person843_bacteria_2763", "target": "faithfulness",
     "what": "Adds a fever and cough that no tool output mentions",
     "update": {"summary": "The AI model identified patterns on the X-ray that resemble pneumonia, with a "
                           "calibrated probability of 1.0. The patient's high fever and productive cough also "
                           "point to bacterial pneumonia. The attention map shows a diffuse focus, especially in "
                           "the left upper zone."}},
]


def load_base_records(path=BASE_RUN):
    records = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                records[r["case_id"]] = r
    return records


def build_cases(records):
    """The bad reports: (case spec, tool outputs, broken report) per case."""
    built = []
    for spec in CASES:
        base = records[spec["base"]]
        report = copy.deepcopy(base["report"])
        report.update(spec["update"])
        built.append({"spec": spec, "tool_outputs": base["tool_outputs"], "report": report,
                      "truth": base["classifier"]})
    return built


def guard_blocks(case):
    """Would the agent's guard reject this report? (None, or the problem)"""
    _, problem = parse_report(json.dumps(case["report"]), case["truth"])
    return problem


def run_sensitivity(judge, records):
    results = []
    for case in build_cases(records):
        spec = case["spec"]
        scores, _ = judge.judge(case["tool_outputs"], case["report"])
        scores = scores.model_dump()
        target_score = scores[spec["target"]]["score"]
        problem = guard_blocks(case)
        results.append({
            "id": spec["id"], "base_case": spec["base"], "what": spec["what"], "target": spec["target"],
            "scores": {c: scores[c]["score"] for c in CRITERIA},
            "target_reason": scores[spec["target"]]["reason"],
            "caught_by_judge": target_score <= CAUGHT_AT_OR_BELOW,
            "blocked_by_guard": problem is not None,
            "guard_problem": problem,
        })
    return results


def pipeline_guard_blocks(case):
    """The guard as the multi-agent pipeline runs it on a draft (with the dataset caveat from the tools):
    None, or the problem. A draft it blocks never reaches the verifier."""
    caveats = (case["tool_outputs"].get("condition_info") or {}).get("model_limitations") or []
    _, problem = parse_report(json.dumps(case["report"]), case["truth"], caveats[0] if caveats else None)
    return problem


def verifier_cases(records):
    """The 5 bad reports, then the unmodified base reports they came from (controls: should pass)."""
    built = [dict(c, kind="bad") for c in build_cases(records)]
    for base_id in dict.fromkeys(spec["base"] for spec in CASES):
        base = records[base_id]
        built.append({"spec": {"id": f"control_{base_id.split('-')[0]}", "base": base_id,
                               "what": "Unmodified base report (control)"},
                      "tool_outputs": base["tool_outputs"], "report": copy.deepcopy(base["report"]),
                      "truth": base["classifier"], "kind": "control"})
    return built


def run_verifier_sensitivity(verifier, records):
    """The multi-agent verifier on each bad report and control, with the pipeline's exact request.

    Caught (bad report) / false alarm (control) = at least one claim marked unsupported. The pipeline
    view: the guard runs first, so a report it blocks is caught before the verifier is asked.
    """
    from agent.multi_agent import bundle_from_tool_outputs, verify_once
    from agent.schema import Report
    results = []
    for case in verifier_cases(records):
        spec = case["spec"]
        verdict, turn = verify_once(verifier, bundle_from_tool_outputs(case["tool_outputs"]),
                                    Report.model_validate(case["report"]))
        flagged = [f.model_dump() for f in verdict.unsupported]
        problem = pipeline_guard_blocks(case)
        results.append({
            "id": spec["id"], "kind": case["kind"], "base_case": spec["base"], "what": spec["what"],
            "claims": len(verdict.findings), "unsupported": flagged, "flagged_by_verifier": bool(flagged),
            "blocked_by_guard": problem is not None, "guard_problem": problem,
            "pipeline": "guard" if problem is not None else ("verifier" if flagged else "passes"),
            "usage": turn.usage,
        })
    return results


def verifier_markdown(results, verifier_model):
    bad = [r for r in results if r["kind"] == "bad"]
    controls = [r for r in results if r["kind"] == "control"]
    lines = [f"# Verifier sensitivity ({verifier_model})", "",
             f"The judge-sensitivity set's {len(bad)} deliberately bad reports, plus their {len(controls)} unmodified "
             "base reports as controls, through the multi-agent verifier (same prompt, schema and evidence as in "
             "the pipeline). Caught / false alarm = at least one claim marked unsupported.", "",
             f"- **Verifier caught: {sum(r['flagged_by_verifier'] for r in bad)}/{len(bad)}** bad reports",
             f"- Verifier false alarms on controls: {sum(r['flagged_by_verifier'] for r in controls)}/{len(controls)}",
             f"- Guard (as the pipeline runs it, before the verifier) blocks: "
             f"{sum(r['blocked_by_guard'] for r in bad)}/{len(bad)} bad reports",
             f"- Pipeline (guard, then verifier) stops: {sum(r['pipeline'] != 'passes' for r in bad)}/{len(bad)} bad reports",
             "", "| Report | Kind | Claims | Verifier flagged | Guard blocks? | Stopped by | Unsupported claims (verifier's reason) |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in results:
        claims = "; ".join(f"\"{f['claim']}\" ({f['reason']})" for f in r["unsupported"]) or "—"
        lines.append(f"| {r['what']} | {r['kind']} | {r['claims']} | {len(r['unsupported'])} | "
                     f"{'yes' if r['blocked_by_guard'] else 'no'} | {r['pipeline']} | {claims.replace('|', '/')} |")
    return "\n".join(lines) + "\n"


def to_markdown(results, judge_model):
    caught = sum(r["caught_by_judge"] for r in results)
    blocked = sum(r["blocked_by_guard"] for r in results)
    either = sum(r["caught_by_judge"] or r["blocked_by_guard"] for r in results)
    lines = [f"# Judge sensitivity ({judge_model})", "",
             f"{len(results)} deliberately bad reports. Caught = the broken criterion scored ≤ {CAUGHT_AT_OR_BELOW}.",
             "", f"- **Judge caught: {caught}/{len(results)}**",
             f"- Guard would block: {blocked}/{len(results)}",
             f"- Caught by either: {either}/{len(results)}", "",
             "| Bad report | Broken criterion | Judge scores (F/U/A/P) | Judge caught? | Guard blocks? | Judge's reason |",
             "| --- | --- | --- | --- | --- | --- |"]
    for r in results:
        s = r["scores"]
        lines.append(f"| {r['what']} | {r['target']} ({s[r['target']]}) | "
                     f"{s['faithfulness']}/{s['uncertainty_language']}/{s['actionability']}/{s['plain_language']} | "
                     f"{'yes' if r['caught_by_judge'] else '**no**'} | {'yes' if r['blocked_by_guard'] else 'no'} | "
                     f"{r['target_reason']} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    import logging
    logging.getLogger("streamlit").setLevel(logging.ERROR)
    from evals.judge import build_judge

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--verifier", action="store_true",
                    help="run the multi-agent verifier instead of the judge; writes evals/verifier_sensitivity.{json,md}")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    if args.verifier:
        return main_verifier()
    judge = build_judge()
    if judge is None:
        raise SystemExit("No API key for the judge provider (config/llm.json).")
    results = run_sensitivity(judge, load_base_records())
    OUT_JSON.write_text(json.dumps({
        "judge_model": judge.model, "base_run": BASE_RUN.relative_to(REPO).as_posix(),
        "caught_at_or_below": CAUGHT_AT_OR_BELOW,
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "results": results,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md = to_markdown(results, judge.model)
    OUT_MD.write_text(md, encoding="utf-8")
    print(md)
    return 0


def main_verifier():
    from agent.agent import build_report_llm
    pipeline = build_report_llm("multi")
    if pipeline is None or getattr(pipeline, "pipeline", None) != "multi":
        raise SystemExit("The multi-agent pipeline needs a Groq API key (config/llm.json provider groq).")
    verifier = pipeline.clients["verifier"]
    results = run_verifier_sensitivity(verifier, load_base_records())
    VERIFIER_OUT_JSON.write_text(json.dumps({
        "verifier_model": verifier.name, "base_run": BASE_RUN.relative_to(REPO).as_posix(),
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "results": results,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md = verifier_markdown(results, verifier.name)
    VERIFIER_OUT_MD.write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
