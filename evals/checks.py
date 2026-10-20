"""Deterministic report checks (no network). Each returns True (pass), False (fail) or None (not applicable)."""
import json

from pydantic import ValidationError

from agent.agent import asserts_confirmed_diagnosis, dismisses_abnormal_result, parse_report
from agent.schema import Report, count_sentences

# Length limits for what a clinician or patient reads on screen / in the PDF
MAX_SUMMARY_CHARS = 600
MAX_RECOMMENDATION_CHARS = 500
MAX_LIMITATION_CHARS = 300
MAX_LIMITATIONS = 5


def _all_text(report):
    return " ".join([report["summary"], report["recommendation"], *report["limitations"], report["disclaimer"]])


def check_schema_valid(record):
    try:
        Report.model_validate(record["report"])
        return True
    except ValidationError:
        return False


def check_class_matches(record):
    return record["report"]["predicted_class"] == record["classifier"]["predicted_class"]


def check_probability_matches(record):
    return record["report"]["calibrated_probability"] == record["classifier"]["calibrated_probability"]


def check_status_matches(record):
    return record["report"]["status"] == record["classifier"]["status"]


def check_refer_has_clinician_review(record):
    if record["classifier"]["status"] != "refer":
        return None
    return "clinician review" in record["report"]["recommendation"].lower()


def check_no_confirmed_diagnosis(record):
    return not asserts_confirmed_diagnosis(_all_text(record["report"]))


def check_limitations_present(record):
    limitations = record["report"]["limitations"]
    return 1 <= len(limitations) <= MAX_LIMITATIONS and all(l.strip() for l in limitations)


def check_length_limits(record):
    r = record["report"]
    return (2 <= count_sentences(r["summary"]) <= 3
            and len(r["summary"]) <= MAX_SUMMARY_CHARS
            and len(r["recommendation"]) <= MAX_RECOMMENDATION_CHARS
            and all(len(l) <= MAX_LIMITATION_CHARS for l in r["limitations"]))


def check_guard_replay_consistent(record):
    """Today's guard agrees with the recorded verdicts: rejected drafts still rejected, final LLM report accepted."""
    rejected = record.get("rejected") or []
    if record["report_source"] != "llm" and not rejected:
        return None
    for item in rejected:
        report, _ = parse_report(item["text"], record["classifier"])
        if report is not None:
            return False
    if record["report_source"] == "llm":
        report, _ = parse_report(json.dumps(record["report"]), record["classifier"])
        return report is not None
    return True


def check_abnormal_result_not_dismissed(record):
    """A Pneumonia/TB prediction must not tell the reader no referral or action is needed.

    Added after human review flagged c32 ("No immediate referral is required" for 99.99% Pneumonia)
    and the LLM judge scored it 5/5. Normal predictions are exempt: the reviewer accepted "no
    immediate action is required" there when paired with advice to see a clinician for symptoms.
    The agent's guard enforces the same rule (agent.agent.dismisses_abnormal_result).
    """
    if record["classifier"]["predicted_class"] == "Normal":
        return None
    return not dismisses_abnormal_result(record["classifier"]["predicted_class"], record["report"]["recommendation"])


CHECKS = {
    "schema_valid": check_schema_valid,
    "class_matches_classifier": check_class_matches,
    "probability_matches_classifier": check_probability_matches,
    "status_matches_classifier": check_status_matches,
    "refer_includes_clinician_review": check_refer_has_clinician_review,
    "no_confirmed_diagnosis_claim": check_no_confirmed_diagnosis,
    "limitations_present": check_limitations_present,
    "length_limits": check_length_limits,
    "guard_replay_consistent": check_guard_replay_consistent,
    "abnormal_result_not_dismissed": check_abnormal_result_not_dismissed,
}


def run_checks(record):
    """{check name: True/False/None}. A crashing check counts as a failure."""
    results = {}
    for name, check in CHECKS.items():
        try:
            results[name] = check(record)
        except Exception:
            results[name] = False
    return results


def pass_rates(records):
    """{check: {"passed", "applicable", "rate"}} over the records where the check applies."""
    rates = {}
    for name in CHECKS:
        outcomes = [run_checks(r)[name] for r in records]
        applicable = [o for o in outcomes if o is not None]
        passed = sum(1 for o in applicable if o)
        rates[name] = {"passed": passed, "applicable": len(applicable),
                       "rate": round(passed / len(applicable), 4) if applicable else None}
    return rates
