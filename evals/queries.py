"""Cross-run queries over the Mongo eval store (aggregation pipelines), printed as tables.

    python -m evals.queries rejected        [--run RUN] [--since YYYY-MM-DD]
    python -m evals.queries check-trend     [--since YYYY-MM-DD]
    python -m evals.queries judge-by-model  [--agent-model MODEL]
    python -m evals.queries disagreements   [--min-diff N]
    python -m evals.queries all             # all four
    add --show-pipeline to print the pipeline that ran

Every query reads only `derived` fields (see evals/store.py), so records of any supported
record_version can be mixed. Each *_pipeline() function only builds the pipeline; run_query()
runs it, so the pipelines can be checked without a server.
"""
import argparse
import json
import sys

from evals.checks import CHECKS
from evals.schema import CRITERIA
from evals.store import LABELS, RECORDS, RUNS


def _filters(run=None, since=None, agent_model=None):
    """$match conditions on eval_records. Run ids are UTC timestamps, so --since compares them as strings."""
    match = {}
    if run:
        match["run_id"] = run
    elif since:
        match["run_id"] = {"$gte": since.replace("-", "")}
    if agent_model:
        match["derived.agent_model"] = agent_model
    return match


def rejected_pipeline(run=None, since=None):
    """Records where the guard rejected at least one draft, or the report fell back to the template."""
    match = {"$or": [{"derived.rejected_count": {"$gt": 0}}, {"derived.fallback": True}], **_filters(run, since)}
    return [
        {"$match": match},
        {"$project": {"_id": 0, "run_id": 1, "case_id": 1, "predicted": "$derived.predicted_class",
                      "status": "$derived.status", "true_label": "$derived.case.true_label",
                      "rejected": "$derived.rejected_count", "problems": "$derived.rejected_problems",
                      "source": "$derived.report_source", "fallback_reason": 1, "fallback_error_kind": 1}},
        {"$sort": {"run_id": 1, "case_id": 1}},
    ]


def check_trend_pipeline(run=None, since=None):
    """Pass rate per deterministic check per run, in run-date order (N/A outcomes excluded, like pass_rates)."""
    return [
        {"$match": {"derived.checks": {"$ne": None}, **_filters(run, since)}},
        {"$project": {"run_id": 1, "checks": {"$objectToArray": "$derived.checks"}}},
        {"$unwind": "$checks"},
        {"$group": {"_id": {"run_id": "$run_id", "check": "$checks.k"},
                    "passed": {"$sum": {"$cond": [{"$eq": ["$checks.v", True]}, 1, 0]}},
                    "applicable": {"$sum": {"$cond": [{"$ne": ["$checks.v", None]}, 1, 0]}}}},
        {"$lookup": {"from": RUNS, "localField": "_id.run_id", "foreignField": "_id", "as": "run"}},
        {"$unwind": {"path": "$run", "preserveNullAndEmptyArrays": True}},
        {"$project": {"_id": 0, "run_id": "$_id.run_id", "check": "$_id.check", "passed": 1, "applicable": 1,
                      "date": "$run.date",
                      "rate": {"$cond": [{"$gt": ["$applicable", 0]},
                                         {"$divide": ["$passed", "$applicable"]}, None]}}},
        {"$sort": {"date": 1, "run_id": 1, "check": 1}},
    ]


def judge_by_model_pipeline(agent_model=None, run=None, since=None):
    """Mean / min judge score per criterion, grouped by agent model (and judge model: judges differ)."""
    return [
        {"$match": {"derived.judge_scores": {"$ne": None}, **_filters(run, since, agent_model)}},
        {"$project": {"run_id": 1, "agent": "$derived.agent_model", "judge": "$derived.judge_model",
                      "scores": {"$objectToArray": "$derived.judge_scores"}}},
        {"$unwind": "$scores"},
        {"$group": {"_id": {"agent": "$agent", "judge": "$judge", "criterion": "$scores.k"},
                    "mean": {"$avg": "$scores.v"}, "min": {"$min": "$scores.v"}, "n": {"$sum": 1},
                    "low": {"$sum": {"$cond": [{"$lte": ["$scores.v", 2]}, 1, 0]}},
                    "runs": {"$addToSet": "$run_id"}}},
        {"$project": {"_id": 0, "agent": "$_id.agent", "judge": "$_id.judge", "criterion": "$_id.criterion",
                      "mean": 1, "min": 1, "n": 1, "low": 1, "runs": {"$size": "$runs"}}},
        {"$sort": {"agent": 1, "judge": 1, "criterion": 1}},
    ]


def disagreements_pipeline(min_diff=1, run=None):
    """Human-labelled criteria where |judge - human| >= min_diff, largest gap first.

    Starts from human_labels and joins eval_records on _id ("<run_id>/<case_id>"): labels only
    mean something against the run they were exported from, and the _id carries that run.
    """
    # One {criterion, human, judge} per human score: the judge's score for the same criterion is picked with
    # $filter. (An array literal of expressions would be shorter, but mongomock doesn't evaluate those.)
    judge_score = {"$arrayElemAt": [{"$map": {
        "input": {"$filter": {"input": {"$objectToArray": {"$ifNull": ["$rec.derived.judge_scores", {}]}},
                              "as": "j", "cond": {"$eq": ["$$j.k", "$$h.k"]}}},
        "as": "j", "in": "$$j.v"}}, 0]}
    pairs = {"$map": {"input": {"$objectToArray": "$scores"}, "as": "h",
                      "in": {"criterion": "$$h.k", "human": "$$h.v", "judge": judge_score}}}
    return [
        {"$match": {"run_id": run} if run else {}},
        {"$lookup": {"from": RECORDS, "localField": "_id", "foreignField": "_id", "as": "rec"}},
        {"$unwind": "$rec"},  # labels without a matching record drop out (counted separately)
        {"$project": {"_id": 0, "run_id": 1, "case_id": 1, "notes": 1, "pairs": pairs,
                      "predicted": "$rec.derived.predicted_class", "status": "$rec.derived.status"}},
        {"$unwind": "$pairs"},
        {"$match": {"pairs.human": {"$ne": None}, "pairs.judge": {"$ne": None}}},
        {"$addFields": {"diff": {"$subtract": ["$pairs.judge", "$pairs.human"]}}},
        {"$addFields": {"abs_diff": {"$abs": "$diff"}}},
        {"$match": {"abs_diff": {"$gte": min_diff}}},
        {"$sort": {"abs_diff": -1, "run_id": 1, "case_id": 1, "pairs.criterion": 1}},
    ]


def run_query(db, collection, pipeline):
    return list(db[collection].aggregate(pipeline))


# ---- Tables ----

def table(headers, rows):
    """Plain fixed-width table; None shows as an em dash."""
    cells = [[("—" if v is None else str(v)) for v in row] for row in rows]
    widths = [max([len(h)] + [len(r[i]) for r in cells]) for i, h in enumerate(headers)]
    line = lambda vals: "  ".join(v.ljust(w) for v, w in zip(vals, widths)).rstrip()
    out = [line(headers), line(["-" * w for w in widths])] + [line(r) for r in cells]
    if not rows:
        out.append("(no rows)")
    return "\n".join(out)


def short(text, n=70):
    return text if text is None or len(text) <= n else text[:n - 1] + "…"


def show_rejected(db, args):
    rows = run_query(db, RECORDS, rejected_pipeline(args.run, args.since))
    print("Cases the guard rejected (at least one draft) or that used the template fallback, all runs\n")
    print(table(["run", "case", "predicted", "status", "true", "rejected", "first problem", "source",
                 "fallback reason"],
                [[r["run_id"], r["case_id"], r.get("predicted"), r.get("status"), r.get("true_label"),
                  r["rejected"], short((r.get("problems") or [None])[0], 60), r.get("source"),
                  short(r.get("fallback_reason"), 50)] for r in rows]))
    print(f"\n{len(rows)} cases in {len({r['run_id'] for r in rows})} runs")
    return rows


def show_check_trend(db, args):
    rows = run_query(db, RECORDS, check_trend_pipeline(args.run, args.since))
    runs = []
    for r in rows:  # already in date order
        if r["run_id"] not in runs:
            runs.append(r["run_id"])
    cell = {(r["check"], r["run_id"]): r for r in rows}

    def fmt(r):
        if r is None:
            return None
        return "n/a" if r["rate"] is None else f"{r['rate']:.0%} ({r['passed']}/{r['applicable']})"
    names = list(CHECKS) + sorted({r["check"] for r in rows} - set(CHECKS))  # checks no longer in checks.py last
    print("Pass rate per deterministic check, per run (oldest first; checks scored by the code that imported them)\n")
    print(table(["check"] + runs, [[n] + [fmt(cell.get((n, run))) for run in runs] for n in names]))
    return rows


def show_judge_by_model(db, args):
    rows = run_query(db, RECORDS, judge_by_model_pipeline(args.agent_model, args.run, args.since))
    groups = {}
    for r in rows:
        groups.setdefault((r["agent"], r["judge"]), {})[r["criterion"]] = r
    print("Mean judge score per criterion (1-5), by agent model; min and scores <= 2 in brackets\n")
    out = []
    for (agent, judge), by_c in groups.items():
        any_r = next(iter(by_c.values()))
        out.append([agent, judge, any_r["runs"], any_r["n"]] +
                   [f"{by_c[c]['mean']:.2f} (min {by_c[c]['min']}, ≤2: {by_c[c]['low']})" if c in by_c else None
                    for c in CRITERIA])
    print(table(["agent model", "judge", "runs", "reports"] + CRITERIA, out))
    return rows


def show_disagreements(db, args):
    rows = run_query(db, LABELS, disagreements_pipeline(args.min_diff, args.run))
    label_filter = {"run_id": args.run} if args.run else {}
    labels = list(db[LABELS].find(label_filter, {"_id": 1, "run_id": 1}))
    joined = db[RECORDS].count_documents({"_id": {"$in": [l["_id"] for l in labels]}}) if labels else 0
    print(f"Judge vs human, |judge - human| >= {args.min_diff} "
          f"({len(labels)} labelled reports, {joined} joined to an eval record, "
          f"runs: {', '.join(sorted({l['run_id'] for l in labels})) or '—'})\n")
    print(table(["run", "case", "predicted", "status", "criterion", "judge", "human", "judge - human", "notes"],
                [[r["run_id"], r["case_id"], r.get("predicted"), r.get("status"), r["pairs"]["criterion"],
                  r["pairs"]["judge"], r["pairs"]["human"], f"{r['diff']:+d}", short(r.get("notes"), 40)]
                 for r in rows]))
    print(f"\n{len(rows)} criterion scores differ in {len({(r['run_id'], r['case_id']) for r in rows})} reports")
    return rows


QUERIES = {
    "rejected": (show_rejected, lambda a: (RECORDS, rejected_pipeline(a.run, a.since))),
    "check-trend": (show_check_trend, lambda a: (RECORDS, check_trend_pipeline(a.run, a.since))),
    "judge-by-model": (show_judge_by_model,
                       lambda a: (RECORDS, judge_by_model_pipeline(a.agent_model, a.run, a.since))),
    "disagreements": (show_disagreements, lambda a: (LABELS, disagreements_pipeline(a.min_diff, a.run))),
}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Cross-run eval queries (MongoDB)")
    ap.add_argument("query", choices=list(QUERIES) + ["all"])
    ap.add_argument("--run", help="only this run_id")
    ap.add_argument("--since", help="only runs from this UTC date on (YYYY-MM-DD)")
    ap.add_argument("--agent-model", help="judge-by-model: only this agent model")
    ap.add_argument("--min-diff", type=int, default=1, help="disagreements: smallest |judge - human| shown")
    ap.add_argument("--show-pipeline", action="store_true", help="print each pipeline before its table")
    return ap.parse_args(argv)


def main(argv=None, db=None):
    args = parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    if db is None:
        from evals.store import connect_or_exit
        store, where = connect_or_exit()
        db = store.db
        print(f"MongoDB: {where}\n")
    names = list(QUERIES) if args.query == "all" else [args.query]
    for i, name in enumerate(names):
        show, build = QUERIES[name]
        if i:
            print("\n" + "=" * 80 + "\n")
        if args.show_pipeline:
            collection, pipeline = build(args)
            print(f"db.{collection}.aggregate({json.dumps(pipeline, indent=2, default=str)})\n")
        show(db, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
