"""Judge sensitivity: does the LLM judge score deliberately bad reports low?

    python -m evals.judge_sensitivity      # live judge (needs a key); writes evals/judge_sensitivity.{json,md}

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

    argparse.ArgumentParser(description=__doc__.split("\n")[0]).parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
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


if __name__ == "__main__":
    sys.exit(main())
