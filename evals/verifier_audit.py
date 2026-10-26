"""Post-hoc audit of the multi-agent verifier (live: Groq calls, no report regeneration).

    python -m evals.verifier_audit --run evals/runs/<multi>.jsonl [--single-run evals/runs/<single>.jsonl]

Writes evals/runs/<multi>.audit.json (resumable: items already done are skipped) and
evals/runs/<multi>.verifier_flags.md (a sheet for a human to spot-check the flags). Three parts:

1. shadow_verifier: the verifier (same model, prompt, schema and evidence as in the pipeline) on every
   draft the GUARD rejected, in the multi run and the single run. In the pipeline those drafts never
   reached the verifier, so this is the only way to see "guard flagged, verifier didn't".
2. flag_checks: every claim the verifier marked unsupported, checked by the judge model against the
   same evidence and the verifier's own definition of "supported". A claim the checker finds supported
   is a likely false flag; the sheet lets a human settle it.
3. judge_drift: the current judge re-scores a few single-run reports, to see whether the judge (a Groq
   Preview model) still scores like it did when that run was judged. Recorded only; scores aren't replaced.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from evals import multi_metrics
from evals.run import audit_path, load_run, relative, run_pipeline
from evals.schema import CRITERIA

DRIFT_EVERY = 8  # judge drift: every 8th case of the single run (5 of 36)

CLAIM_CHECK_SYSTEM = """You check ONE claim from a report about an AI chest X-ray classifier's result against the
EVIDENCE (tool outputs) the report was written from. The EVIDENCE is the only ground truth: do not use outside
medical knowledge.
- supported: true if the EVIDENCE states the claim or it follows directly from it (copying, paraphrasing, or a
  probability shown with fewer decimals is fine); false if the EVIDENCE contradicts it or says nothing about it.
  Advice to involve a clinician, to follow up, or to treat the result as a screening aid rather than a diagnosis
  is supported (the report rules require it). Hedged wording about uncertainty is supported when it matches the
  classifier's status.
- evidence: the EVIDENCE text that supports the claim, quoted, or "" if none does.
- reason: one short sentence.
Reply with JSON."""

CLAIM_CHECK_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["supported", "evidence", "reason"],
    "properties": {"supported": {"type": "boolean"}, "evidence": {"type": "string"}, "reason": {"type": "string"}},
}


def claim_check_prompt(bundle, claim):
    return f"EVIDENCE:\n{bundle.prompt_json()}\n\nCLAIM:\n{claim}"


def load_audit(path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"shadow_verifier": [], "flag_checks": [], "judge_drift": []}


def _done(items, **key):
    return any(all(i.get(k) == v for k, v in key.items()) and not i.get("error") for i in items)


def _replace(items, new, **key):
    items[:] = [i for i in items if not all(i.get(k) == v for k, v in key.items())] + [new]


def _usage(turn):
    u = turn.usage or {}
    return {"input_tokens": u.get("input_tokens"),
            "output_tokens": (u.get("output_tokens") or 0) + (u.get("thinking_tokens") or 0)}


def shadow_verify(verifier, runs, audit, save, log=print):
    """The verifier on every guard-rejected draft of each run ({run_id: records})."""
    from agent.multi_agent import _parse_draft, bundle_from_tool_outputs, verify_once
    for run_id, records in runs.items():
        by_id = {r["case_id"]: r for r in records}
        for draft in multi_metrics.guard_rejected_drafts(records):
            key = {"run_id": run_id, "case_id": draft["case_id"], "step": draft["step"]}
            if _done(audit["shadow_verifier"], **key):
                continue
            report, _ = _parse_draft(draft["text"])
            bundle = bundle_from_tool_outputs(by_id[draft["case_id"]]["tool_outputs"])
            item = dict(key, guard_problem=draft["problem"], findings=None, unsupported=None, error=None)
            try:
                verdict, turn = verify_once(verifier, bundle, report)
                item.update(findings=[f.model_dump() for f in verdict.findings], unsupported=len(verdict.unsupported),
                            usage=_usage(turn))
            except Exception as e:
                item["error"] = f"{type(e).__name__}: {e}"[:500]
            _replace(audit["shadow_verifier"], item, **key)
            save()
            log(f"shadow verifier {run_id} {draft['case_id']} step {draft['step']}: "
                f"{item['error'] or str(item['unsupported']) + ' unsupported'}")


def check_flags(checker, records, audit, save, log=print):
    """The judge model on every claim the verifier marked unsupported."""
    from agent.multi_agent import bundle_from_tool_outputs
    by_id = {r["case_id"]: r for r in records}
    for flag in multi_metrics.flagged_claims(records):
        key = {"case_id": flag["case_id"], "step": flag["step"], "claim": flag["claim"]}
        if _done(audit["flag_checks"], **key):
            continue
        bundle = bundle_from_tool_outputs(by_id[flag["case_id"]]["tool_outputs"])
        item = dict(key, verifier_source=flag["source"], verifier_reason=flag["reason"], checker_supported=None,
                    checker_evidence=None, checker_reason=None, error=None)
        try:
            turn = checker.generate_json(CLAIM_CHECK_SYSTEM, [{"role": "user", "text": claim_check_prompt(bundle, flag["claim"])}],
                                         "claim_check", CLAIM_CHECK_SCHEMA)
            out = json.loads(turn.text)
            item.update(checker_supported=bool(out["supported"]), checker_evidence=out["evidence"],
                        checker_reason=out["reason"], usage=_usage(turn))
        except Exception as e:
            item["error"] = f"{type(e).__name__}: {e}"[:500]
        _replace(audit["flag_checks"], item, **key)
        save()
        log(f"claim check {flag['case_id']}: {item['error'] or ('supported' if item['checker_supported'] else 'unsupported')}")


def judge_drift(judge, single_records, audit, save, log=print):
    """The current judge on every DRIFT_EVERY-th single-run report, next to the recorded scores."""
    for record in sorted(single_records, key=lambda r: r["case_id"])[::DRIFT_EVERY]:
        key = {"case_id": record["case_id"]}
        if _done(audit["judge_drift"], **key):
            continue
        item = dict(key, judge_model=judge.model, recorded={c: record["judge"]["scores"][c]["score"] for c in CRITERIA},
                    now=None, error=None)
        try:
            scores, usage = judge.judge(record["tool_outputs"], record["report"])
            item.update(now={c: getattr(scores, c).score for c in CRITERIA}, usage=usage)
        except Exception as e:
            item["error"] = f"{type(e).__name__}: {e}"[:500]
        _replace(audit["judge_drift"], item, **key)
        save()
        log(f"judge drift {record['case_id']}: recorded {item['recorded']} now {item['now'] or item['error']}")


def flags_sheet(audit, run_id):
    """Markdown sheet: every verifier flag with the checker's verdict and a blank column for a human."""
    checks = audit["flag_checks"]
    likely_false = sum(1 for c in checks if c.get("checker_supported"))
    lines = [f"# Verifier flags to spot-check: run `{run_id}`", "",
             f"{len(checks)} claims the verifier marked unsupported. The checker ({audit.get('checker_model')}) found "
             f"{likely_false} of them supported by the evidence (likely false flags). Fill in the last column "
             "(`false flag` / `real problem`) by reading the claim against the run's tool outputs.", "",
             "| # | Case | Step | Claim | Verifier's reason | Checker | Checker's evidence / reason | Human verdict |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for i, c in enumerate(checks, 1):
        verdict = "error" if c.get("error") else ("supported" if c["checker_supported"] else "unsupported")
        why = c.get("error") or f"\"{c['checker_evidence']}\" — {c['checker_reason']}"
        lines.append(f"| {i} | `{c['case_id']}` | {c['step']} | {_cell(c['claim'])} | {_cell(c['verifier_reason'])} | "
                     f"{verdict} | {_cell(why)} | |")
    return "\n".join(lines) + "\n"


def _cell(text):
    return str(text).replace("|", "\\|").replace("\n", " ")


def main(argv=None):
    import logging
    logging.getLogger("streamlit").setLevel(logging.ERROR)
    from agent.agent import build_llm_client, build_report_llm
    from agent.llm_config import provider_settings
    from evals.judge import build_judge

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, help="the multi-agent run")
    ap.add_argument("--single-run", help="a single-agent run: its guard-rejected drafts get the shadow verifier, "
                                         "and a few of its reports the judge drift check")
    ap.add_argument("--skip", default="", help="comma-separated parts to skip: shadow, flags, drift")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    run_path = Path(args.run)
    records = load_run(run_path)
    multi = [r for r in records.values() if r.get("report")]
    if run_pipeline(records) != "multi":
        raise SystemExit(f"{relative(run_path)} is not a multi-agent run.")
    single = [r for r in load_run(Path(args.single_run)).values() if r.get("report")] if args.single_run else []

    pipeline = build_report_llm("multi")
    _, settings = provider_settings()
    checker, judge = build_llm_client(model=settings["judge_model"]), build_judge()
    if pipeline is None or checker is None or judge is None:
        raise SystemExit("No Groq API key (st.secrets['groq']['api_key'] or GROQ_API_KEY).")
    verifier = pipeline.clients["verifier"]

    out = audit_path(run_path)
    audit = load_audit(out)
    audit.update(run=relative(run_path), single_run=relative(args.single_run) if args.single_run else None,
                 verifier_model=verifier.name, checker_model=checker.name, judge_model=judge.model,
                 updated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))

    def save():
        out.write_text(json.dumps(audit, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    if "shadow" not in skip:
        runs = {multi[0]["run_id"]: multi}
        if single:
            runs[single[0]["run_id"]] = single
        shadow_verify(verifier, runs, audit, save)
    if "flags" not in skip:
        check_flags(checker, multi, audit, save)
    if "drift" not in skip and single:
        judge_drift(judge, single, audit, save)
    save()
    sheet = run_path.with_suffix(".verifier_flags.md")
    sheet.write_text(flags_sheet(audit, run_path.stem), encoding="utf-8")
    errors = sum(1 for part in ("shadow_verifier", "flag_checks", "judge_drift") for i in audit[part] if i.get("error"))
    print(f"Wrote {relative(out)} and {relative(sheet)}" + (f"; {errors} items failed (re-run to retry them)" if errors else ""))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
