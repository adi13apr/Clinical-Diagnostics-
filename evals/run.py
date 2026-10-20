"""Eval runner.

    python -m evals.run --live [--resume [RUN]] [--limit N]             # real model + LLM agent/judge, resumable
    python -m evals.run --replay [--run RUN] [--update-baseline]        # re-score saved outputs, no network (CI)
    python -m evals.run --export-human-labels [--run RUN]               # blank sheet for human scoring
    python -m evals.run --agreement [--run RUN]                         # judge vs human agreement

Live runs append one JSON record per case to evals/runs/<timestamp>.jsonl (the last
record for a case wins, so a crash or rate limit loses nothing) and write a summary
to evals/results/<date>.md.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from evals.checks import CHECKS, pass_rates
from evals.dataset import CASES_PATH, dataset_hash, load_cases
from evals.human_labels import (
    HUMAN_LABELS_PATH, N_LABELS, LabelsAlreadyFilled, agreement, export_human_labels, labels_run, read_labels,
)
from evals.schema import CRITERIA, RECORD_VERSION

REPO = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO / "evals" / "runs"
RESULTS_DIR = REPO / "evals" / "results"
BASELINE_PATH = REPO / "evals" / "baseline.json"


class StopRun(Exception):
    """Stop a live run (e.g. rate limit) so it can be resumed later."""


# ---- Run files ----

def load_run(path):
    """{case_id: record}, last record per case wins."""
    records = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                records[record["case_id"]] = record
    return records


def latest_run():
    runs = sorted(RUNS_DIR.glob("*.jsonl"))
    return runs[-1] if runs else None


def append_record(path, record):
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def relative(path):
    try:
        return Path(path).resolve().relative_to(REPO).as_posix()
    except ValueError:
        return str(path)


def check_dataset_hash(records, expected, run_path):
    hashes = {r["dataset_hash"] for r in records.values()}
    if hashes - {expected}:
        raise SystemExit(f"Dataset hash mismatch: {relative(run_path)} was recorded against {sorted(hashes)}, "
                         f"but {relative(CASES_PATH)} hashes to {expected}. Re-run --live for this dataset.")


# ---- Live ----

def run_case(case, service, llm, roots):
    from PIL import Image

    from agent.agent import generate_report
    from agent.tools import ReportTools
    from services.tracing import Tracer, trace_span

    source, rest = case.relpath.split("/", 1)
    image = Image.open(roots[source] / rest)
    tracer = Tracer()  # in-memory only: eval runs don't write to traces/
    with trace_span(tracer, "prediction"):
        prediction = service.predict(image, tracer=tracer)
    tools = ReportTools(service, image, prediction, tracer=tracer)
    result = generate_report(tools, llm, tracer=tracer)
    if result.source == "fallback" and result.error_kind in ("rate_limit", "infrastructure"):
        # Quota, network, auth or server failure: says nothing about report quality, so stop and
        # retry later. A 'model' failure (e.g. an invalid tool call) IS a result and is recorded.
        raise StopRun(f"{result.reason} {(result.error_detail or '')[:300]}")

    truth = tools.run_classifier()
    llm_calls = [s for s in tracer.spans if s.name == "llm_call"]
    report_span = next(s for s in tracer.spans if s.name == "report")
    return {
        "classifier": truth,
        "snapshot_match": (truth["predicted_class"], truth["calibrated_probability"], truth["status"]) ==
                          (case.snapshot.predicted_class, case.snapshot.calibrated_probability, case.snapshot.status),
        "tool_outputs": {"classifier": truth, "gradcam": tools.call("get_gradcam", {}),
                         "condition_info": tools.get_condition_info()},
        "report": result.report.model_dump(),
        "report_source": result.source,
        "fallback_reason": result.reason,
        "fallback_error_kind": result.error_kind,
        "steps": result.steps,
        "tool_calls": result.tool_calls,  # which tools the model actually called (c07 skipped one)
        "rejected": result.rejected,
        "agent_usage": {
            "llm_calls": len(llm_calls),
            "input_tokens": sum(s.attributes.get("input_tokens") or 0 for s in llm_calls),
            "output_tokens": sum((s.attributes.get("output_tokens") or 0) + (s.attributes.get("thinking_tokens") or 0)
                                 for s in llm_calls),
            "report_latency_ms": report_span.latency_ms,
        },
    }


def live(args):
    import logging
    logging.getLogger("streamlit").setLevel(logging.ERROR)  # ModelService's st.cache_resource outside the app

    sys.path.insert(0, str(REPO / "scripts"))
    from agent.agent import RateLimitError, build_llm_client, describe_llm_error
    from agent.llm_config import provider_settings
    from calibrate import find_root
    from evals.judge import build_judge
    from services.model_handler import ModelService

    cases = load_cases()
    data_hash = dataset_hash(cases)
    provider, settings = provider_settings()
    # Agent and judge get per-model token-aware budgets from config/llm.json (shared if the model is the same)
    llm = build_llm_client("report")
    judge = build_judge(args.judge_model)
    if llm is None or judge is None:
        raise SystemExit(f"No API key for provider {provider!r}: set st.secrets['{settings['secret_section']}']"
                         f"['api_key'] or {settings['env_key']}.")
    model = llm.name

    if args.resume:
        run_path = Path(args.resume) if args.resume != "latest" else latest_run()
        if run_path is None or not run_path.exists():
            raise SystemExit("Nothing to resume.")
        records = load_run(run_path)
        check_dataset_hash(records, data_hash, run_path)
    else:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        run_path = RUNS_DIR / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.jsonl"
        records = {}
    run_id = run_path.stem
    print(f"Run {relative(run_path)}: {len(cases)} cases, dataset {data_hash[:12]}, provider {provider}, "
          f"reports {model}, judge {judge.model}")
    service = ModelService(str(REPO / "best_student_model_aligned_final.pth"))
    roots = {"pneumonia": find_root(REPO / "data" / "pneumonia", "chest_xray", "train"),
             "tb": find_root(REPO / "data" / "tb", "TB_Chest_Radiography_Database", "Tuberculosis")}

    todo = cases[:args.limit] if args.limit else cases
    try:
        for i, case in enumerate(todo, 1):
            record = records.get(case.case_id)
            if record and record.get("judge"):
                continue
            if record is None:
                record = {"record_version": RECORD_VERSION, "case_id": case.case_id, "run_id": run_id,
                          "dataset_hash": data_hash, "provider": provider, "agent_model": model,
                          "recorded_at": datetime.now(timezone.utc).isoformat(),
                          **run_case(case, service, llm, roots), "judge": None, "judge_error": None}
                append_record(run_path, record)
            try:
                scores, usage = judge.judge(record["tool_outputs"], record["report"])
                record = dict(record, judge={"model": judge.model, "scores": scores.model_dump(), "usage": usage},
                              judge_error=None)
            except RateLimitError as e:
                append_record(run_path, dict(record, judge_error=f"RateLimitError: {e}"))
                raise StopRun(f"judge: {describe_llm_error(e)} [{e}]")
            except Exception as e:
                record = dict(record, judge_error=f"{type(e).__name__}: {e}")
            append_record(run_path, record)
            records[case.case_id] = record
            j = record["judge"]
            print(f"[{i}/{len(todo)}] {case.case_id}: {record['report_source']}, "
                  f"judge {'-'.join(str(j['scores'][c]['score']) for c in CRITERIA) if j else record['judge_error']}"
                  f" ({record.get('agent_usage', {}).get('llm_calls', '?')} agent calls)", flush=True)
    except StopRun as e:
        remaining = sum(1 for c in todo if not (records.get(c.case_id) or {}).get("judge"))
        print(f"\nStopped: {e}\n{remaining} cases left. "
              f"Resume with: python -m evals.run --live --resume {relative(run_path)}")
        return 2

    done = [r for r in records.values() if r.get("judge")]
    if len(done) < len(todo):
        print(f"\n{len(done)}/{len(todo)} cases fully scored. Resume with: "
              f"python -m evals.run --live --resume {relative(run_path)}")
        return 2
    summary = build_summary(load_run(run_path), cases, data_hash, run_path, load_baseline(), load_agreement(run_path))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{datetime.now(timezone.utc):%Y-%m-%d}.md"
    out.write_text(summary["markdown"], encoding="utf-8")
    print(f"\nWrote {relative(out)}")
    return 0


# ---- Scoring / summary ----

def load_baseline():
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8")) if BASELINE_PATH.exists() else None


def load_agreement(run_path):
    """Judge-vs-human agreement for this run, {"other_run": ...} if the labels came from another run, or None."""
    if not HUMAN_LABELS_PATH.exists():
        return None
    source = labels_run()
    if source is not None and (REPO / source).resolve() != Path(run_path).resolve():
        return {"other_run": source}  # different reports: comparing would be meaningless
    return agreement(read_labels(), load_run(run_path))


def compare_to_baseline(rates, baseline):
    """[(check, current, baseline)] for every check whose pass rate fell below the baseline."""
    drops = []
    for name, base in (baseline or {}).get("deterministic_pass_rates", {}).items():
        current = rates.get(name, {}).get("rate")
        if base is not None and (current is None or current < base - 1e-9):
            drops.append((name, current, base))
    return drops


def mean(xs):
    return round(sum(xs) / len(xs), 2) if xs else None


def fmt(x, pct=False):
    if x is None:
        return "—"
    return f"{x:.1%}" if pct else f"{x}"


def build_summary(records_by_id, cases, data_hash, run_path, baseline=None, agreement_result=None):
    cases_by_id = {c.case_id: c for c in cases}
    records = [r for r in records_by_id.values() if r.get("report") and r["case_id"] in cases_by_id]
    llm_records = [r for r in records if r["report_source"] == "llm"]
    judged = [r for r in records if r.get("judge")]
    rates, llm_rates = pass_rates(records), pass_rates(llm_records)
    drops = compare_to_baseline(rates, baseline)

    lines = [f"# Eval results: run `{Path(run_path).stem}`", ""]
    if len(judged) < len(cases):
        lines += [f"> **Partial run: {len(records)}/{len(cases)} reports, {len(judged)} judged.** "
                  "Rates below cover only these cases. Finish with `python -m evals.run --live --resume`.", ""]
    lines += [f"- Dataset: `{relative(CASES_PATH)}`, {len(cases)} cases, hash `{data_hash}`",
             f"- Run file: `{relative(run_path)}`; reports scored: {len(records)}/{len(cases)}, judged: {len(judged)}",
             f"- Agent model: {records[0]['agent_model'] if records else '—'}; judge: "
             f"{judged[0]['judge']['model'] if judged else '—'}",
             f"- Report source: {len(llm_records)} LLM, {len(records) - len(llm_records)} template fallback; "
             f"classifier snapshot drift: {sum(not r['snapshot_match'] for r in records)}",
             "", "## Deterministic checks", "",
             "| Check | All reports | LLM-written | Baseline |", "| --- | --- | --- | --- |"]
    base_rates = (baseline or {}).get("deterministic_pass_rates", {})
    for name in CHECKS:
        a, l = rates[name], llm_rates[name]
        lines.append(f"| {name} | {fmt(a['rate'], True)} ({a['passed']}/{a['applicable']}) | "
                     f"{fmt(l['rate'], True)} ({l['passed']}/{l['applicable']}) | {fmt(base_rates.get(name), True)} |")
    lines += ["", f"**Baseline:** {'no drops' if not drops else 'DROPPED: ' + ', '.join(d[0] for d in drops)}"
              f"{'' if baseline else ' (no baseline committed yet)'}", ""]

    lines += ["## LLM judge (1-5)", "", "| Criterion | Mean, all | Mean, LLM-written | Mean, template | Scores ≤ 2 |",
              "| --- | --- | --- | --- | --- |"]
    for c in CRITERIA:
        def scores(rs):
            return [r["judge"]["scores"][c]["score"] for r in rs if r.get("judge")]
        low = [r["case_id"] for r in judged if r["judge"]["scores"][c]["score"] <= 2]
        lines.append(f"| {c} | {fmt(mean(scores(records)))} | {fmt(mean(scores(llm_records)))} | "
                     f"{fmt(mean(scores([r for r in records if r['report_source'] != 'llm'])))} | "
                     f"{', '.join(low) or '—'} |")
    lowest = sorted(judged, key=lambda r: sum(r["judge"]["scores"][c]["score"] for c in CRITERIA))[:3]
    if lowest:
        lines += ["", "Lowest-scored reports:"]
        for r in lowest:
            reasons = "; ".join(f"{c} {r['judge']['scores'][c]['score']}: {r['judge']['scores'][c]['reason']}"
                                for c in CRITERIA if r["judge"]["scores"][c]["score"] <= 3)
            lines.append(f"- `{r['case_id']}` ({r['report_source']}): {reasons or 'all criteria ≥ 4'}")

    lines += ["", "## By case category", "", "| Category | Cases | LLM-written | Mean judge score (all criteria) |",
              "| --- | --- | --- | --- |"]
    for cat in ("error", "refer", "correct"):
        rs = [r for r in records if cat in cases_by_id[r["case_id"]].categories]
        js = [r["judge"]["scores"][c]["score"] for r in rs if r.get("judge") for c in CRITERIA]
        lines.append(f"| {cat} | {len(rs)} | {sum(r['report_source'] == 'llm' for r in rs)} | {fmt(mean(js))} |")

    usage = [r["agent_usage"] for r in llm_records if r.get("agent_usage")]
    rejected = [r for r in records if r.get("rejected")]
    lines += ["", "## Agent operations", "",
              f"- Guard rejections: {sum(len(r['rejected']) for r in rejected)} drafts in {len(rejected)} reports"
              + (f" ({', '.join(sorted({x['problem'].split(':')[0] for r in rejected for x in r['rejected']}))})"
                 if rejected else ""),
              f"- Fallbacks: {len(records) - len(llm_records)}"
              + (f" ({'; '.join(sorted({r['fallback_reason'] for r in records if r['report_source'] != 'llm'}))})"
                 if len(records) > len(llm_records) else ""),
              f"- Per LLM report: {fmt(mean([u['llm_calls'] for u in usage]))} calls, "
              f"{fmt(mean([u['input_tokens'] for u in usage]))} input / {fmt(mean([u['output_tokens'] for u in usage]))} "
              f"output tokens, {fmt(mean([u['report_latency_ms'] / 1000 for u in usage]))} s"]

    lines += ["", "## Judge vs human", ""]
    if agreement_result and "other_run" in agreement_result:
        lines.append(f"The human scores in `{relative(HUMAN_LABELS_PATH)}` are for reports from "
                     f"`{agreement_result['other_run']}`, not this run, so no agreement is computed here "
                     "(see that run's results).")
    elif not agreement_result or not agreement_result["labelled_reports"]:
        lines.append(f"No human scores yet: fill in `{relative(HUMAN_LABELS_PATH)}` "
                     "(python -m evals.run --export-human-labels creates it).")
    else:
        a = agreement_result
        lines += [f"{a['labelled_reports']} reports labelled.", "",
                  "| Criterion | Exact | Within 1 | Judge − human |", "| --- | --- | --- | --- |"]
        for c, s in list(a["per_criterion"].items()) + [("overall", a["overall"])]:
            lines.append(f"| {c} | {fmt(s['exact'], True)} | {fmt(s['within_1'], True)} | "
                         f"{fmt(s['mean_judge_minus_human'])} |")
        if a["problems"]:
            lines += ["", "Problems: " + "; ".join(a["problems"])]
    # Human-written context for this run (e.g. why the setup changed), kept next to the run file
    notes = Path(run_path).with_suffix(".notes.md")
    if notes.exists():
        lines += ["", "## Run notes", "", notes.read_text(encoding="utf-8").strip()]
    return {"markdown": "\n".join(lines) + "\n", "rates": rates, "drops": drops}


def replay(args):
    cases = load_cases()
    data_hash = dataset_hash(cases)
    baseline = load_baseline()
    run_path = Path(args.run) if args.run else (REPO / baseline["run"] if baseline else latest_run())
    if run_path is None or not run_path.exists():
        raise SystemExit("No run to replay. Run python -m evals.run --live first.")
    records = load_run(run_path)
    check_dataset_hash(records, data_hash, run_path)
    summary = build_summary(records, cases, data_hash, run_path, baseline, load_agreement(run_path))
    print(summary["markdown"])
    if args.summary_out:
        Path(args.summary_out).write_text(summary["markdown"], encoding="utf-8")
    if args.update_baseline:
        n_reports = sum(1 for r in records.values() if r.get("report"))
        BASELINE_PATH.write_text(json.dumps({
            "run": relative(run_path), "dataset_hash": data_hash,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "reports": n_reports, "cases": len(cases),
            "note": "" if n_reports == len(cases) else
                    "PROVISIONAL: built from a partial run; re-run --replay --update-baseline once it completes",
            "deterministic_pass_rates": {k: v["rate"] for k, v in summary["rates"].items()},
        }, indent=2) + "\n", encoding="utf-8")
        print(f"Updated {relative(BASELINE_PATH)}")
        return 0
    if summary["drops"]:
        for name, current, base in summary["drops"]:
            print(f"FAIL: {name} pass rate {fmt(current, True)} < baseline {fmt(base, True)}")
        return 1
    return 0


def export_labels(args):
    cases = load_cases()
    run_path = Path(args.run) if args.run else latest_run()
    records = [r for r in load_run(run_path).values() if r.get("report")]
    if len(records) < N_LABELS:
        raise SystemExit(f"{relative(run_path)} has {len(records)} reports; need at least {N_LABELS} to export "
                         "the labelling sheet. Finish the live run first (--live --resume).")
    try:
        rows = export_human_labels(records, {c.case_id: c for c in cases}, run=relative(run_path))
    except LabelsAlreadyFilled as e:
        raise SystemExit(str(e))
    print(f"Wrote {len(rows)} reports to {relative(HUMAN_LABELS_PATH)} from {relative(run_path)}. "
          f"Score each with evals/rubric.md (whole numbers 1-5), then run python -m evals.run --agreement.")
    return 0


def show_agreement(args):
    # Default to the run the labels were exported from: other runs have different reports
    source = labels_run()
    run_path = Path(args.run) if args.run else (REPO / source if source else latest_run())
    a = agreement(read_labels(), load_run(run_path))
    a["run"] = relative(run_path)
    print(json.dumps(a, indent=2))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Report-agent eval suite")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true")
    mode.add_argument("--replay", action="store_true")
    mode.add_argument("--export-human-labels", action="store_true")
    mode.add_argument("--agreement", action="store_true")
    ap.add_argument("--resume", nargs="?", const="latest", help="continue a live run (default: latest)")
    ap.add_argument("--run", help="run file to replay / export / compare (default: baseline run or latest)")
    ap.add_argument("--limit", type=int, help="only the first N cases")
    ap.add_argument("--judge-model", help="judge model (default: the agent's model)")
    ap.add_argument("--update-baseline", action="store_true", help="with --replay: write evals/baseline.json")
    ap.add_argument("--summary-out", help="with --replay: also write the markdown summary to this file")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    if args.live:
        return live(args)
    if args.replay:
        return replay(args)
    if args.export_human_labels:
        return export_labels(args)
    return show_agreement(args)


if __name__ == "__main__":
    sys.exit(main())
