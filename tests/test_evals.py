import json

import pytest
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import ValidationError

from agent.agent import RateLimitError
from agent.schema import DISCLAIMER
from evals import run as eval_run
from evals.checks import CHECKS, pass_rates, run_checks
from evals.dataset import CASES_PATH, dataset_hash, load_cases
from evals.human_labels import LabelsAlreadyFilled, agreement, export_human_labels, read_labels
from evals.judge import GeminiJudge, MockJudge
from evals.rate_limit import RateLimiter
from evals.schema import CASE_SCHEMA_VERSION, CRITERIA, JudgeResult, SchemaVersionError

CASES = load_cases()


def make_record(case, report_source="llm", judge_scores=(5, 4, 4, 4), **report_overrides):
    snap = case.snapshot
    truth = {"predicted_class": snap.predicted_class, "calibrated_probability": snap.calibrated_probability,
             "status": snap.status, "probabilities": {snap.predicted_class: snap.calibrated_probability}}
    refer = snap.status == "refer"
    report = {"predicted_class": snap.predicted_class, "calibrated_probability": snap.calibrated_probability,
              "status": snap.status,
              "summary": f"The model suggests {snap.predicted_class}. This is a screening result, not a diagnosis.",
              "recommendation": ("Refer: uncertain, clinician review recommended." if refer
                                 else "Share this result with a clinician."),
              "limitations": ["Evaluated on two public datasets only."], "disclaimer": DISCLAIMER}
    report.update(report_overrides)
    return {"record_version": 1, "case_id": case.case_id, "run_id": "test", "dataset_hash": dataset_hash(CASES),
            "agent_model": "mock", "classifier": truth, "snapshot_match": True,
            "tool_outputs": {"classifier": truth, "gradcam": {"peak_region": "right upper zone", "focus": "diffuse",
                                                              "border_share": 0.3},
                             "condition_info": {"condition": snap.predicted_class}},
            "report": report, "report_source": report_source, "fallback_reason": None, "steps": 2, "rejected": [],
            "agent_usage": {"llm_calls": 2, "input_tokens": 2000, "output_tokens": 300, "report_latency_ms": 9000.0},
            "judge": {"model": "mock-judge", "scores": {c: {"score": s, "reason": "r"}
                                                       for c, s in zip(CRITERIA, judge_scores)}},
            "judge_error": None}


REFER_CASE = next(c for c in CASES if c.snapshot.status == "refer")
CONFIDENT_CASE = next(c for c in CASES if c.snapshot.status == "confident")


# Dataset

def test_committed_dataset_selection():
    assert len(CASES) == 36 and all(c.split == "test" for c in CASES)
    assert sum("error" in c.categories for c in CASES) == 24
    assert sum("refer" in c.categories for c in CASES) == 17
    for c in CASES:
        is_error = c.snapshot.predicted_class != c.true_label
        assert ("error" in c.categories) == is_error
        assert ("refer" in c.categories) == (c.snapshot.status == "refer")


def test_dataset_hash_ignores_line_endings(tmp_path):
    crlf = tmp_path / "cases.jsonl"
    crlf.write_bytes(CASES_PATH.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    assert dataset_hash(load_cases(crlf)) == dataset_hash(CASES)


def test_schema_version_mismatch_fails_loudly(tmp_path):
    lines = CASES_PATH.read_text(encoding="utf-8").splitlines()
    bad = json.loads(lines[0])
    bad["schema_version"] = CASE_SCHEMA_VERSION + 1
    path = tmp_path / "cases.jsonl"
    path.write_text("\n".join([json.dumps(bad)] + lines[1:]), encoding="utf-8")
    with pytest.raises(SchemaVersionError, match="schema_version"):
        load_cases(path)


def test_unknown_case_field_is_rejected(tmp_path):
    row = json.loads(CASES_PATH.read_text(encoding="utf-8").splitlines()[0])
    row["extra"] = 1
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_cases(path)


# Deterministic checks

def test_good_records_pass_every_applicable_check():
    for case in (REFER_CASE, CONFIDENT_CASE):
        results = run_checks(make_record(case))
        assert all(v in (True, None) for v in results.values()), results
    assert run_checks(make_record(CONFIDENT_CASE))["refer_includes_clinician_review"] is None


@pytest.mark.parametrize("override,failing", [
    ({"predicted_class": "Tuberculosis"}, "class_matches_classifier"),
    ({"calibrated_probability": 0.5}, "probability_matches_classifier"),
    ({"recommendation": "Rest at home."}, "refer_includes_clinician_review"),
    ({"summary": "The diagnosis is confirmed. Start treatment."}, "no_confirmed_diagnosis_claim"),
    ({"limitations": []}, "limitations_present"),
    ({"summary": "Word. " * 3 + "x" * 600 + "."}, "length_limits"),
])
def test_each_check_catches_its_failure(override, failing):
    assert run_checks(make_record(REFER_CASE, **override))[failing] is False


def test_negated_claim_passes_and_guard_replay_detects_changes():
    ok = make_record(CONFIDENT_CASE, summary="This is not a confirmed diagnosis. A clinician should review it.")
    assert run_checks(ok)["no_confirmed_diagnosis_claim"] is True
    # A draft recorded as rejected that today's guard would accept = guard behaviour changed
    accepted_now = dict(make_record(CONFIDENT_CASE), rejected=[
        {"step": 2, "text": json.dumps(make_record(CONFIDENT_CASE)["report"]), "problem": "old guard"}])
    assert run_checks(accepted_now)["guard_replay_consistent"] is False


def test_pass_rates_exclude_not_applicable():
    records = [make_record(REFER_CASE), make_record(CONFIDENT_CASE),
               make_record(REFER_CASE, recommendation="Rest.")]
    rates = pass_rates(records)
    assert rates["refer_includes_clinician_review"] == {"passed": 1, "applicable": 2, "rate": 0.5}
    assert rates["schema_valid"]["applicable"] == 3 and set(rates) == set(CHECKS)


# Replay + baseline

def write_run(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def test_replay_passes_then_fails_on_baseline_drop(tmp_path, monkeypatch, capsys):
    run_path = tmp_path / "run.jsonl"
    write_run(run_path, [make_record(c) for c in CASES[:4]])
    monkeypatch.setattr(eval_run, "BASELINE_PATH", tmp_path / "baseline.json")
    monkeypatch.setattr(eval_run, "HUMAN_LABELS_PATH", tmp_path / "missing.csv")
    monkeypatch.setattr("evals.run.HUMAN_LABELS_PATH", tmp_path / "missing.csv")
    assert eval_run.main(["--replay", "--run", str(run_path), "--update-baseline"]) == 0
    assert eval_run.main(["--replay", "--run", str(run_path)]) == 0

    # Same baseline, but a later run where one report flips the class
    bad = [make_record(c) for c in CASES[:4]]
    bad[0]["report"]["predicted_class"] = "Normal" if bad[0]["classifier"]["predicted_class"] != "Normal" else "Pneumonia"
    write_run(run_path, bad)
    assert eval_run.main(["--replay", "--run", str(run_path)]) == 1
    assert "FAIL: class_matches_classifier" in capsys.readouterr().out


def test_replay_rejects_records_from_another_dataset(tmp_path, monkeypatch):
    record = dict(make_record(CASES[0]), dataset_hash="0" * 64)
    run_path = tmp_path / "run.jsonl"
    write_run(run_path, [record])
    monkeypatch.setattr(eval_run, "BASELINE_PATH", tmp_path / "none.json")
    with pytest.raises(SystemExit, match="Dataset hash mismatch"):
        eval_run.main(["--replay", "--run", str(run_path)])


def test_load_run_last_record_wins(tmp_path):
    first = make_record(CASES[0])
    second = dict(first, judge=None, judge_error="RateLimitError")
    write_run(tmp_path / "run.jsonl", [first, second])
    assert eval_run.load_run(tmp_path / "run.jsonl")[CASES[0].case_id]["judge_error"] == "RateLimitError"


def test_committed_baseline_replays_cleanly():
    if not eval_run.BASELINE_PATH.exists():
        pytest.skip("no committed baseline yet")
    assert eval_run.main(["--replay"]) == 0


# Rate limiter

def test_rate_limiter_spaces_calls():
    now, sleeps = [0.0], []

    def sleep(s):
        sleeps.append(round(s, 3))
        now[0] += s

    limiter = RateLimiter(4, clock=lambda: now[0], sleep=sleep)
    for _ in range(3):
        limiter.wait()
    assert sleeps == [15.0, 15.0] and limiter.calls == 3


# Judge

def judge_client(response_or_error):
    def generate_content(**kwargs):
        if isinstance(response_or_error, Exception):
            raise response_or_error
        return response_or_error
    return type("C", (), {"models": type("M", (), {"generate_content": staticmethod(generate_content)})()})()


def test_gemini_judge_parses_structured_output():
    payload = {c: {"score": 4, "reason": "fine"} for c in CRITERIA}
    response = types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=[types.Part(text=json.dumps(payload))]))])
    result, _ = GeminiJudge("unused", "m", client=judge_client(response)).judge({}, {})
    assert isinstance(result, JudgeResult) and result.actionability.score == 4


def test_gemini_judge_rejects_out_of_range_scores():
    payload = {c: {"score": 7, "reason": "too high"} for c in CRITERIA}
    response = types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=[types.Part(text=json.dumps(payload))]))])
    with pytest.raises(ValidationError):
        GeminiJudge("unused", "m", client=judge_client(response)).judge({}, {})


def test_gemini_judge_long_rate_limit_raises():
    error = genai_errors.ClientError(429, {"error": {"code": 429, "message": "Please retry in 40s.",
                                                      "status": "RESOURCE_EXHAUSTED"}})
    with pytest.raises(RateLimitError):
        GeminiJudge("unused", "m", client=judge_client(error), sleep=lambda s: None).judge({}, {})


def test_judge_prompt_uses_rubric_file():
    judge = GeminiJudge("unused", "m", client=judge_client(None))
    assert "## faithfulness" in judge.rubric and "## plain_language" in judge.rubric


# Human labels

def test_export_is_blind_and_blank_and_never_overwrites_scores(tmp_path):
    records = [make_record(c) for c in CASES]
    path = tmp_path / "labels.csv"
    rows = export_human_labels(records, {c.case_id: c for c in CASES}, path)
    assert len(rows) == 15
    header = list(read_labels(path)[0])
    assert not any("judge" in h or "true" in h for h in header)  # blind
    assert all(row[c] == "" for row in read_labels(path) for c in CRITERIA)
    cats = [next(c for c in CASES if c.case_id == r["case_id"]).categories for r in rows]
    assert sum("correct" in c for c in cats) == 5 and sum("refer" in c for c in cats) == 5

    filled = read_labels(path)
    filled[0]["faithfulness"] = "4"
    import csv
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(filled)
    with pytest.raises(LabelsAlreadyFilled):
        export_human_labels(records, {c.case_id: c for c in CASES}, path)


def test_agreement_exact_within_one_and_bias():
    records = {c.case_id: make_record(c, judge_scores=(5, 4, 4, 2)) for c in CASES[:3]}
    rows = [
        {"case_id": CASES[0].case_id, "faithfulness": "5", "uncertainty_language": "4", "actionability": "3",
         "plain_language": "2"},
        {"case_id": CASES[1].case_id, "faithfulness": "3", "uncertainty_language": "4", "actionability": "4",
         "plain_language": "4"},
        {"case_id": CASES[2].case_id, "faithfulness": "", "uncertainty_language": "", "actionability": "",
         "plain_language": ""},  # not labelled yet
    ]
    a = agreement(rows, records)
    assert a["labelled_reports"] == 2
    assert a["per_criterion"]["faithfulness"] == {"n": 2, "exact": 0.5, "within_1": 0.5, "mean_judge_minus_human": 1.0}
    assert a["per_criterion"]["actionability"]["within_1"] == 1.0
    assert a["overall"]["n"] == 8 and a["overall"]["exact"] == 0.625  # 3 + 2 of 8 pairs match exactly


def test_agreement_reports_bad_scores():
    rows = [{"case_id": CASES[0].case_id, "faithfulness": "five", "uncertainty_language": "4",
             "actionability": "4", "plain_language": "4"}]
    a = agreement(rows, {CASES[0].case_id: make_record(CASES[0])})
    assert a["labelled_reports"] == 0 and "whole numbers 1-5" in a["problems"][0]


def test_mock_judge_shape():
    result, _ = MockJudge().judge({}, {})
    assert set(result.model_dump()) == set(CRITERIA)
