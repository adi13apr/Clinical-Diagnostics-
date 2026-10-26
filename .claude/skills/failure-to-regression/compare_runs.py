"""Compare two eval runs across all cases: template fallbacks, guard rejections, checks, judge means.

    python compare_runs.py --after evals/runs/<new>.jsonl [--before evals/runs/<old>.jsonl] [--out table.md]

--before defaults to the baseline run. Prints a markdown table for the PR description. Exit 3 if the
new run has MORE template fallbacks or more guard rejections than the old one: stop there and show the
user before running --update-baseline.
"""
import argparse
import sys
from pathlib import Path

from _repo import REPO, rel


def stats(records):
    """Counts over one run's records (rejections split by cause: guard, API-level model error, truncation)."""
    from evals.checks import pass_rates
    from evals.schema import CRITERIA
    reports = [r for r in records.values() if r.get("report")]
    rejected = [(r["case_id"], item) for r in reports for item in r.get("rejected") or []]
    guard = [(cid, item) for cid, item in rejected if item.get("text") is not None and not item.get("finish_reason")]
    judged = [r for r in reports if r.get("judge")]
    return {
        "reports": len(reports),
        "llm": sum(r["report_source"] == "llm" for r in reports),
        "template": sum(r["report_source"] != "llm" for r in reports),
        "guard_rejections": len(guard),
        "cases_with_guard_rejection": len({cid for cid, _ in guard}),
        "model_errors": sum(item.get("text") is None for _, item in rejected),
        "truncated": sum(bool(item.get("finish_reason")) for _, item in rejected),
        "llm_calls": sum((r.get("agent_usage") or {}).get("llm_calls") or 0 for r in reports),
        "rates": {k: v["rate"] for k, v in pass_rates(reports).items()},
        "judge": {c: round(sum(r["judge"]["scores"][c]["score"] for r in judged) / len(judged), 2) if judged else None
                  for c in CRITERIA},
        "by_case": {r["case_id"]: r for r in reports},
    }


def pct(n, d):
    return f"{n}/{d} ({100 * n / d:.1f}%)" if d else "-"


def compare(before, after):
    """(markdown, worse) for two {case_id: record} runs."""
    b, a = stats(before), stats(after)
    rows = [("Reports", b["reports"], a["reports"]),
            ("Written by the LLM", pct(b["llm"], b["reports"]), pct(a["llm"], a["reports"])),
            ("Template fallbacks", pct(b["template"], b["reports"]), pct(a["template"], a["reports"])),
            ("Guard rejections (drafts)", b["guard_rejections"], a["guard_rejections"]),
            ("Cases with a guard rejection", b["cases_with_guard_rejection"], a["cases_with_guard_rejection"]),
            ("API-level model errors (retried)", b["model_errors"], a["model_errors"]),
            ("Truncated replies", b["truncated"], a["truncated"]),
            ("LLM calls", b["llm_calls"], a["llm_calls"])]
    rows += [(f"Check `{k}`", fmt_rate(b["rates"].get(k)), fmt_rate(v)) for k, v in a["rates"].items()]
    rows += [(f"Judge mean `{k}`", b["judge"][k], v) for k, v in a["judge"].items()]
    lines = ["| | Before | After |", "| --- | --- | --- |"] + [f"| {n} | {x} | {y} |" for n, x, y in rows]

    changes = []
    for cid in sorted(set(b["by_case"]) | set(a["by_case"])):
        old, new = b["by_case"].get(cid), a["by_case"].get(cid)
        if not old or not new:
            continue
        o_rej, n_rej = len(old.get("rejected") or []), len(new.get("rejected") or [])
        if old["report_source"] != new["report_source"] or o_rej != n_rej:
            reason = new.get("fallback_reason") or "; ".join(i.get("problem") or "" for i in new.get("rejected") or [])
            changes.append(f"- `{cid}`: {old['report_source']} ({o_rej} rejected) -> "
                           f"{new['report_source']} ({n_rej} rejected){': ' + reason[:200] if reason else ''}")
    worse = a["template"] > b["template"] or a["guard_rejections"] > b["guard_rejections"]
    md = "\n".join(lines) + "\n\nCases whose source or rejections changed:\n" + ("\n".join(changes) or "- none") + "\n"
    if a["reports"] != b["reports"]:
        md += f"\nNote: the runs have different numbers of reports ({b['reports']} vs {a['reports']}).\n"
    return md, worse


def fmt_rate(r):
    return "-" if r is None else f"{100 * r:.1f}%"


def main(argv=None):
    from evals.run import load_baseline, load_run
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--after", required=True, help="the new run file")
    ap.add_argument("--before", help="the old run file (default: the baseline run)")
    ap.add_argument("--out", help="also write the markdown here")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    before_path = Path(args.before) if args.before else REPO / load_baseline()["run"]
    md, worse = compare(load_run(before_path), load_run(args.after))
    md = f"Before: `{rel(before_path)}`, after: `{rel(args.after)}`\n\n" + md
    print(md)
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
    if worse:
        print("STOP: the new run has more template fallbacks or guard rejections. Show the user this table "
              "before running --update-baseline.")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
