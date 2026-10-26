"""Judge sensitivity harness (offline: MockJudge stands in for the live judge)."""
from agent.agent import MockLLMClient
from evals.judge import MockJudge
from evals.judge_sensitivity import (
    CASES, build_cases, guard_blocks, load_base_records, run_sensitivity, run_verifier_sensitivity, to_markdown,
    verifier_cases, verifier_markdown,
)
from evals.schema import CRITERIA

RECORDS = load_base_records()


def test_five_cases_each_break_one_thing():
    built = build_cases(RECORDS)
    assert [c["spec"]["id"] for c in built] == ["wrong_class", "missing_referral", "tb_no_follow_up",
                                                 "overconfident", "invented_symptom"]
    for case in built:
        base = RECORDS[case["spec"]["base"]]
        changed = {k for k in case["report"] if case["report"][k] != base["report"][k]}
        assert changed == set(case["spec"]["update"])          # only the intended fields differ
        assert case["tool_outputs"] == base["tool_outputs"]    # real tool outputs, untouched
        assert case["spec"]["target"] in CRITERIA


def test_bases_have_the_right_classifier_output():
    by_id = {c["spec"]["id"]: c for c in build_cases(RECORDS)}
    assert by_id["missing_referral"]["truth"]["status"] == "refer"
    assert by_id["tb_no_follow_up"]["truth"]["predicted_class"] == "Tuberculosis"
    assert by_id["wrong_class"]["truth"]["predicted_class"] == "Pneumonia"


def test_guard_blocks_three_and_misses_two():
    blocked = {c["spec"]["id"]: guard_blocks(c) is not None for c in build_cases(RECORDS)}
    # Structural problems are guard-detectable; overclaiming and invented facts need a judge (or a human)
    assert blocked == {"wrong_class": True, "missing_referral": True, "tb_no_follow_up": True,
                       "overconfident": False, "invented_symptom": False}


def test_caught_means_target_criterion_scored_low():
    low_everywhere = MockJudge({c: 1 for c in CRITERIA})
    high_everywhere = MockJudge({c: 5 for c in CRITERIA})
    assert all(r["caught_by_judge"] for r in run_sensitivity(low_everywhere, RECORDS))
    results = run_sensitivity(high_everywhere, RECORDS)
    assert not any(r["caught_by_judge"] for r in results)
    md = to_markdown(results, "mock-judge")
    assert "Judge caught: 0/5" in md and "Guard would block: 3/5" in md and "Caught by either: 3/5" in md
    assert len(CASES) == 5


def finding(supported):
    return {"claim": "c", "supported": supported, "source": "classifier" if supported else "none", "reason": "r"}


def test_verifier_mode_adds_the_untouched_base_reports_as_controls():
    built = verifier_cases(RECORDS)
    assert [c["kind"] for c in built] == ["bad"] * 5 + ["control"] * 3
    for control in built[5:]:
        assert control["report"] == RECORDS[control["spec"]["base"]]["report"]


def test_verifier_sensitivity_counts_catches_false_alarms_and_the_pipeline_view():
    # The verifier flags every bad report except "overconfident", and one control (a false alarm)
    verdicts = [[finding(True), finding(False)]] * 3 + [[finding(True)], [finding(False)]] + \
               [[finding(False)], [finding(True)], [finding(True)]]
    verifier = MockLLMClient([MockLLMClient.final({"findings": v}) for v in verdicts], name="verifier-m")
    results = run_verifier_sensitivity(verifier, RECORDS)
    by_id = {r["id"]: r for r in results}
    assert verifier.schemas == ["verdict"] * 8
    # The guard runs first in the pipeline (with the dataset caveat), so the structural three stop there
    assert [r["pipeline"] for r in results[:5]] == ["guard", "guard", "guard", "passes", "verifier"]
    assert by_id["invented_symptom"]["flagged_by_verifier"] and not by_id["overconfident"]["flagged_by_verifier"]
    md = verifier_markdown(results, "verifier-m")
    assert "Verifier caught: 4/5" in md and "false alarms on controls: 1/3" in md
    assert "blocks: 3/5" in md and "Pipeline (guard, then verifier) stops: 4/5" in md
