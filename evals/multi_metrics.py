"""Multi-agent pipeline metrics from run records (no network). Reported in summaries; they don't gate CI.

- tools: did the evidence agent call all three tools, and how many re-asks did that need;
- revisions: drafts sent back per case, and by whom (guard or verifier);
- verifier findings: claims checked and claims marked unsupported, per verdict and per case;
- guard vs verifier: drafts one flagged and the other didn't. Every draft the verifier saw had already
  passed the guard (the guard runs first), so "verifier only" comes from the records; "guard only"
  needs the verifier's opinion on guard-rejected drafts, which evals.verifier_audit asks for post hoc.
"""
from agent.multi_agent import REQUIRED_TOOLS
from evals.checks import guard_verdict


def tools_metrics(records):
    rs = list(records)
    all_called = [r for r in rs if set(REQUIRED_TOOLS) <= set(r.get("tool_calls") or [])]
    reasks = [r.get("evidence_reasks") or 0 for r in rs]
    return {"cases": len(rs), "all_three_called": len(all_called),
            "skipped": sorted(r["case_id"] for r in rs if r not in all_called),
            "reasks": sum(reasks), "cases_reasked": sum(1 for n in reasks if n)}


def revision_metrics(records):
    rs = list(records)
    per_case = {r["case_id"]: r.get("revisions") or [] for r in rs}
    by = {"guard": 0, "verifier": 0}
    reasons = []
    for case_id, revs in per_case.items():
        for rev in revs:
            by[rev["by"]] += 1
            why = rev["problems"] if rev["by"] == "guard" else [f"unsupported: {f['claim']}" for f in rev["unsupported"]]
            reasons.append({"case_id": case_id, "by": rev["by"], "why": why})
    counts = [len(v) for v in per_case.values()]
    return {"total": sum(counts), "cases_revised": sum(1 for n in counts if n),
            "max_per_case": max(counts, default=0), "by": by, "reasons": reasons}


def verdicts(record):
    """Every verifier verdict for a record, in order: [{"step", "findings", "final"}]. Older records kept
    only the unsupported findings of a flagged draft; those come back as-is."""
    out = [{"step": rev["step"], "findings": rev.get("findings") or rev["unsupported"], "final": False}
           for rev in record.get("revisions") or [] if rev["by"] == "verifier"]
    if record.get("verifier_findings"):
        out.append({"step": None, "findings": record["verifier_findings"], "final": True})
    return out


def verifier_metrics(records):
    rs = list(records)
    vs = [(r["case_id"], v) for r in rs for v in verdicts(r)]
    claims = [len(v["findings"]) for _, v in vs]
    unsupported = [sum(not f["supported"] for f in v["findings"]) for _, v in vs]
    per_case = {}
    for (case_id, _), n in zip(vs, unsupported):
        per_case[case_id] = per_case.get(case_id, 0) + n
    return {"verdicts": len(vs), "cases_verified": len({c for c, _ in vs}),
            "claims_per_verdict": round(sum(claims) / len(claims), 2) if claims else None,
            "unsupported_total": sum(unsupported),
            "cases_flagged": sorted(c for c, n in per_case.items() if n)}


def flagged_claims(records):
    """Every claim the verifier marked unsupported: [{"case_id", "step", "claim", "source", "reason"}]."""
    out = []
    for r in records:
        for v in verdicts(r):
            for f in v["findings"]:
                if not f["supported"]:
                    out.append({"case_id": r["case_id"], "step": v["step"], **f})
    return out


def guard_rejected_drafts(records):
    """Drafts the guard rejected that the verifier could read: [{"case_id", "step", "text", "problem"}].
    Skipped: cut-off replies, API-level model errors (no text), and drafts that aren't a Report at all."""
    from agent.multi_agent import _parse_draft
    out = []
    for r in records:
        for item in r.get("rejected") or []:
            if item.get("text") is None or item.get("finish_reason") or item.get("agent", "writer") != "writer":
                continue
            report, _ = _parse_draft(item["text"])
            if report is not None:
                out.append({"case_id": r["case_id"], "step": item["step"], "text": item["text"],
                            "problem": item["problem"]})
    return out


def disagreement(records, shadow=None):
    """Drafts flagged by one check and not the other.

    verifier_only: verifier-flagged drafts that today's guard accepts (by construction all of them; the
    guard is re-run here so a later guard change shows up). guard_only: guard-rejected drafts that the
    post-hoc verifier (shadow, from evals.verifier_audit) found fully supported; None without an audit.
    """
    rs = list(records)
    by_id = {r["case_id"]: r for r in rs}  # one run: case ids are unique
    flagged = [(r["case_id"], rev) for r in rs for rev in r.get("revisions") or [] if rev["by"] == "verifier"]
    verifier_only = [{"case_id": c, "step": rev["step"]} for c, rev in flagged
                     if rev.get("text") and guard_verdict(by_id[c], rev["text"])[0] is not None]
    guard_only = None
    if shadow is not None:  # the audit also covers other runs' drafts (same case ids): keep this run's
        run_ids = {r["run_id"] for r in rs}
        guard_only = [s for s in shadow if s["run_id"] in run_ids and not s.get("error") and s["unsupported"] == 0]
    return {"verifier_flagged": len(flagged), "verifier_only": verifier_only,
            "guard_rejected": len(guard_rejected_drafts(rs)), "guard_only": guard_only}


def summary_lines(records, shadow=None):
    """Markdown lines for a run summary (evals.run build_summary)."""
    rs = [r for r in records if r.get("pipeline") == "multi"]
    if not rs:
        return []
    t, rev, ver, dis = tools_metrics(rs), revision_metrics(rs), verifier_metrics(rs), disagreement(rs, shadow)
    lines = ["## Multi-agent pipeline (reported, not gating CI)", "",
             f"- All three tools called: {t['all_three_called']}/{t['cases']}"
             + (f" (missing in {', '.join(t['skipped'])})" if t["skipped"] else "")
             + f"; evidence re-asks: {t['reasks']} in {t['cases_reasked']} cases",
             f"- Revisions: {rev['total']} in {rev['cases_revised']} cases (guard {rev['by']['guard']}, "
             f"verifier {rev['by']['verifier']}; at most {rev['max_per_case']} per case)",
             f"- Verifier: {ver['verdicts']} verdicts on {ver['cases_verified']} cases, "
             f"{fmt_num(ver['claims_per_verdict'])} claims per verdict, {ver['unsupported_total']} claims unsupported"
             + (f" ({', '.join(ver['cases_flagged'])})" if ver["cases_flagged"] else ""),
             f"- Verifier flagged, guard passed: {len(dis['verifier_only'])} of {dis['verifier_flagged']} flagged drafts",
             f"- Guard rejected, verifier would pass: "
             + (f"{len(dis['guard_only'])} of {dis['guard_rejected']} rejected drafts (post-hoc verifier)"
                if dis["guard_only"] is not None else f"not measured ({dis['guard_rejected']} rejected drafts; "
                                                       "run python -m evals.verifier_audit)")]
    for item in rev["reasons"]:
        lines.append(f"  - `{item['case_id']}` sent back by the {item['by']}: {'; '.join(item['why'])[:300]}")
    return lines + [""]


def fmt_num(x):
    return "—" if x is None else f"{x}"
