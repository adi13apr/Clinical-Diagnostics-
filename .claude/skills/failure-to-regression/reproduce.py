"""Show what went wrong in one report, and optionally save it as a regression fixture.

    python reproduce.py --case c07 [--run evals/runs/<run>.jsonl]   # eval run record (default: the baseline run)
    python reproduce.py --mcp-output saved.json                     # {"classification": ..., "report": ...}
    python reproduce.py --trace <trace_id>                          # spans in local traces/ (no report text)

    ... --write-fixture <slug> --description "what went wrong, who found it"

The fixture (tests/fixtures/<slug>.json) holds the exact report plus an "expect" block that
tests/test_regression_fixtures.py enforces: the guard must reject it (fill in expect.guard_problem
once the guard rule exists) and the checks that fail today must keep failing.
"""
import argparse
import json
import sys
from pathlib import Path

from _repo import REPO, rel

FIXTURES = REPO / "tests" / "fixtures"


def load_record(case, run=None):
    """(record, run path) for a case id or prefix ("c07") from a run file."""
    from evals.run import load_baseline, load_run
    run_path = Path(run) if run else REPO / load_baseline()["run"]
    records = load_run(run_path)
    matches = [r for cid, r in records.items() if cid == case or cid.startswith(case + "-")]
    if len(matches) != 1:
        raise SystemExit(f"{case!r} matches {len(matches)} cases in {rel(run_path)}")
    return matches[0], run_path


def record_from_mcp(data):
    """A record-shaped dict from saved MCP outputs: classify_xray as "classification", generate_report as "report"."""
    from agent.tools import ReportTools
    c, out = data["classification"], data["report"]
    classifier = {k: c.get(k) for k in ("predicted_class", "calibrated_probability", "probabilities", "status",
                                        "referral_threshold", "referral_message")}
    tools = ReportTools(None, None, prediction={"diagnosis": c["predicted_class"], "confidence": c["calibrated_probability"],
                                                "probabilities": c["probabilities"], "status": c["status"]})
    return {"case_id": data.get("case_id"), "classifier": classifier, "report": out["report"],
            "report_source": "llm" if out["source"] == "llm" else "fallback", "fallback_reason": out.get("fallback_reason"),
            "tool_calls": out.get("tool_calls"), "rejected": [], "agent_model": out.get("model"),
            "tool_outputs": {"classifier": classifier, "gradcam": data.get("gradcam"),
                             "condition_info": tools.get_condition_info()}}


def diagnose(record):
    """{"failing_checks", "guard_ok", "guard_problem"} under today's checks and guard."""
    from evals.checks import guard_verdict, run_checks
    results = run_checks(record)
    report, problem = guard_verdict(record)
    return {"failing_checks": [k for k, v in results.items() if v is False], "guard_ok": report is not None,
            "guard_problem": problem}


def show_record(record, run_path=None):
    d = diagnose(record)
    c = record["classifier"]
    print(f"Case     {record.get('case_id')}" + (f"  ({rel(run_path)})" if run_path else ""))
    print(f"Classes  {c['predicted_class']} {c['calibrated_probability']} {c['status']}")
    tool_calls = record.get("tool_calls")
    print(f"Source   {record['report_source']}, steps {record.get('steps')}, tool calls "
          f"{tool_calls if tool_calls is not None else 'not recorded (run predates the field)'}")
    for item in record.get("rejected") or []:
        print(f"Rejected step {item.get('step')}: {item.get('problem')}")
    print("Report:\n" + json.dumps(record["report"], indent=2, ensure_ascii=False))
    print(f"Failing checks today: {d['failing_checks'] or 'none'}")
    print(f"Guard today: {'ACCEPTS' if d['guard_ok'] else 'rejects: ' + d['guard_problem']}")
    return d


def show_trace(trace_id):
    files = sorted((REPO / "traces" / "mcp").glob("*.jsonl")) + sorted((REPO / "traces").glob("*.jsonl"))
    spans = []
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip() and json.loads(line).get("trace_id") == trace_id:
                spans.append(json.loads(line))
    if not spans:
        raise SystemExit(f"No spans for {trace_id} in traces/ or traces/mcp/ (Supabase traces are not read here).")
    for s in sorted(spans, key=lambda s: s["start"]):
        a = s.get("attributes") or {}
        detail = {k: a[k] for k in ("tool", "model", "step", "result", "reason", "used", "input_tokens",
                                     "output_tokens") if a.get(k) is not None}
        print(f"{s['name']:<18} {s['status']:<6} {s.get('error') or ''} {detail or ''}")
    print("\nTraces hold no report text. For the exact output, call the MCP tools on the same image "
          "(classify_xray + generate_report), save both outputs and use --mcp-output.")


def write_fixture(slug, description, record, run_path, d):
    path = FIXTURES / f"{slug}.json"
    if path.exists():
        raise SystemExit(f"{rel(path)} already exists")
    c = record["classifier"]
    fixture = {
        "description": description,
        "model": record.get("agent_model"),
        "run": rel(run_path) if run_path else None,
        "case_id": record.get("case_id"),
        "truth": {k: c[k] for k in ("predicted_class", "calibrated_probability", "status")},
        "tool_calls": record.get("tool_calls"),
        "tool_outputs": record.get("tool_outputs"),
        "report": record["report"],
        # Enforced by tests/test_regression_fixtures.py. guard_problem: a phrase of the new guard's message.
        "expect": {"guard": "reject", "guard_problem": None, "checks_fail": d["failing_checks"]},
    }
    path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"\nWrote {rel(path)}. Set expect.guard_problem once the guard rejects it.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--case", help="case id or prefix, e.g. c07")
    src.add_argument("--mcp-output", help="JSON file with the MCP classify_xray and generate_report outputs")
    src.add_argument("--trace", help="trace id (local traces only)")
    ap.add_argument("--run", help="with --case: run file (default: the baseline run)")
    ap.add_argument("--write-fixture", metavar="SLUG", help="save tests/fixtures/<SLUG>.json")
    ap.add_argument("--description", help="with --write-fixture: what went wrong and who found it")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    if args.trace:
        show_trace(args.trace)
        return 0
    if args.case:
        record, run_path = load_record(args.case, args.run)
    else:
        record, run_path = record_from_mcp(json.loads(Path(args.mcp_output).read_text(encoding="utf-8"))), None
    d = show_record(record, run_path)
    if args.write_fixture:
        if not args.description:
            raise SystemExit("--write-fixture needs --description")
        write_fixture(args.write_fixture, args.description, record, run_path, d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
