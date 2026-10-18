"""Human validation of the LLM judge.

export_human_labels() writes evals/human_labels.csv: 15 reports with blank score columns
for a human to fill in using evals/rubric.md. Judge scores and true labels are left out so
the rating is blind. agreement() then compares the filled-in scores with the judge.
"""
import csv
import random
from pathlib import Path

from evals.schema import CRITERIA

HUMAN_LABELS_PATH = Path(__file__).resolve().parent / "human_labels.csv"
N_LABELS = 15
SEED = 7
CONTEXT_COLUMNS = ["case_id", "classifier_result", "classifier_probabilities", "gradcam_summary",
                   "condition_info_for", "condition_info_text", "report_source", "report_summary",
                   "report_recommendation", "report_limitations"]
SCORE_COLUMNS = CRITERIA + ["notes"]


class LabelsAlreadyFilled(RuntimeError):
    pass


def read_labels(path=HUMAN_LABELS_PATH):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _has_scores(rows):
    return any((row.get(c) or "").strip() for row in rows for c in CRITERIA)


def select_for_labelling(records, cases_by_id, n=N_LABELS, seed=SEED):
    """Up to n records: a third each from error-only, refer and correct cases, preferring LLM-written reports."""
    groups = {"error": [], "refer": [], "correct": []}
    for r in sorted(records, key=lambda r: r["case_id"]):
        cats = cases_by_id[r["case_id"]].categories
        groups["refer" if "refer" in cats else cats[0]].append(r)
    rng = random.Random(seed)
    chosen = []
    per_group = n // len(groups)
    for name in ("error", "refer", "correct"):
        pool = groups[name]
        llm = [r for r in pool if r["report_source"] == "llm"]
        other = [r for r in pool if r["report_source"] != "llm"]
        rng.shuffle(llm)
        rng.shuffle(other)
        chosen += (llm + other)[:per_group]
    leftovers = [r for r in records if r not in chosen]
    rng.shuffle(leftovers)
    chosen += leftovers[:n - len(chosen)]
    return sorted(chosen, key=lambda r: r["case_id"])


def _condition_text(info):
    parts = [f"{k}: {info[k]}" for k in ("overview", "typical_xray_findings", "what_this_result_means",
                                         "typical_next_steps") if info.get(k)]
    parts += [f"model limitation: {l}" for l in info.get("model_limitations", [])]
    return " | ".join(parts)


def _row(record):
    c, g, rep = record["classifier"], record["tool_outputs"].get("gradcam") or {}, record["report"]
    gradcam = ("unavailable" if not g.get("peak_region") else
               f"peak {g['peak_region']}, {g.get('focus')}, {g.get('border_share', 0):.0%} of weight on image border")
    return {
        "case_id": record["case_id"],
        "classifier_result": f"{c['predicted_class']}, calibrated {c['calibrated_probability']}, {c['status']}",
        "classifier_probabilities": ", ".join(f"{k} {v}" for k, v in c["probabilities"].items()),
        "gradcam_summary": gradcam,
        "condition_info_for": (record["tool_outputs"].get("condition_info") or {}).get("condition", ""),
        # The same vetted text the judge sees, so faithfulness can be checked against it
        "condition_info_text": _condition_text(record["tool_outputs"].get("condition_info") or {}),
        "report_source": record["report_source"],
        "report_summary": rep["summary"],
        "report_recommendation": rep["recommendation"],
        "report_limitations": " | ".join(rep["limitations"]),
        **{col: "" for col in SCORE_COLUMNS},
    }


def export_human_labels(records, cases_by_id, path=HUMAN_LABELS_PATH):
    """Write the blank labelling sheet. Refuses to overwrite a sheet that already has scores."""
    path = Path(path)
    if path.exists() and _has_scores(read_labels(path)):
        raise LabelsAlreadyFilled(f"{path} already contains human scores; not overwriting.")
    rows = [_row(r) for r in select_for_labelling(records, cases_by_id)]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:  # utf-8-sig so Excel shows the text correctly
        writer = csv.DictWriter(f, fieldnames=CONTEXT_COLUMNS + SCORE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def agreement(label_rows, records_by_id):
    """Judge-vs-human agreement over fully labelled rows: exact, within-1 and mean (judge - human)."""
    problems, pairs = [], {c: [] for c in CRITERIA}
    labelled = 0
    for row in label_rows:
        cells = {c: (row.get(c) or "").strip() for c in CRITERIA}
        if not any(cells.values()):
            continue
        try:
            human = {c: int(v) for c, v in cells.items()}
            if not all(1 <= v <= 5 for v in human.values()):
                raise ValueError
        except ValueError:
            problems.append(f"{row['case_id']}: scores must be whole numbers 1-5 for all four criteria")
            continue
        judge = (records_by_id.get(row["case_id"]) or {}).get("judge")
        if not judge:
            problems.append(f"{row['case_id']}: no judge scores in this run")
            continue
        labelled += 1
        for c in CRITERIA:
            pairs[c].append((judge["scores"][c]["score"], human[c]))

    def stats(ps):
        if not ps:
            return {"n": 0, "exact": None, "within_1": None, "mean_judge_minus_human": None}
        return {"n": len(ps), "exact": round(sum(j == h for j, h in ps) / len(ps), 3),
                "within_1": round(sum(abs(j - h) <= 1 for j, h in ps) / len(ps), 3),
                "mean_judge_minus_human": round(sum(j - h for j, h in ps) / len(ps), 2)}

    all_pairs = [p for ps in pairs.values() for p in ps]
    return {"labelled_reports": labelled, "rows": len(label_rows), "problems": problems,
            "per_criterion": {c: stats(ps) for c, ps in pairs.items()}, "overall": stats(all_pairs)}
