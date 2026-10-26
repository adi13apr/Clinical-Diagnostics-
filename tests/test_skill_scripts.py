"""The Agent Skills' helper scripts (.claude/skills/*/), offline: no model, network or dataset."""
import importlib
import json
import sys
from pathlib import Path

import pytest

from evals.schema import CASE_SCHEMA_VERSION, EvalCase

SKILLS = Path(__file__).resolve().parent.parent / ".claude" / "skills"
sys.path.insert(0, str(SKILLS / "failure-to-regression"))
sys.path.insert(0, str(SKILLS / "release-check"))

reproduce = importlib.import_module("reproduce")
add_case = importlib.import_module("add_case")
verify = importlib.import_module("verify")
compare_runs = importlib.import_module("compare_runs")
live_eval = importlib.import_module("live_eval")
release_check = importlib.import_module("release_check")

CAVEAT = "The model was trained and evaluated on two public datasets in which each disease came from a different source."
CLASSIFIER = {"predicted_class": "Pneumonia", "calibrated_probability": 0.6953, "status": "refer",
              "probabilities": {"Normal": 0.3026, "Pneumonia": 0.6953, "Tuberculosis": 0.0021},
              "referral_threshold": 0.9, "referral_message": "refer: uncertain, clinician review recommended"}


def c07_like(case_id="c07-NORMAL2-IM-0780-0001", source="llm", rejected=None, caveat_quoted=False):
    """A record like c07 in 20261020T075220Z: grounded fields, invented limitations."""
    limitations = [CAVEAT] if caveat_quoted else ["Model limitations: may be affected by image quality and positioning."]
    return {"case_id": case_id, "classifier": CLASSIFIER, "report_source": source, "rejected": rejected or [],
            "steps": 3, "tool_calls": None, "agent_model": "openai/gpt-oss-120b",
            "tool_outputs": {"condition_info": {"model_limitations": [CAVEAT]}},
            "report": {"predicted_class": "Pneumonia", "calibrated_probability": 0.6953, "status": "refer",
                       "summary": "The model leans towards pneumonia at about 70%. This is a screening result.",
                       "recommendation": "Refer: uncertain, clinician review recommended.",
                       "limitations": limitations, "disclaimer": "Not medical advice."}}


def write_run(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


# reproduce.py

def test_reproduce_finds_a_case_by_prefix_and_diagnoses_it(tmp_path, capsys):
    run = write_run(tmp_path / "run.jsonl", [c07_like(), c07_like("c08-other", caveat_quoted=True)])
    record, path = reproduce.load_record("c07", run)
    assert record["case_id"].startswith("c07-") and path == run
    d = reproduce.show_record(record, path)
    assert d == {"failing_checks": ["dataset_caveat_present"], "guard_ok": True, "guard_problem": None}
    assert "Guard today: ACCEPTS" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        reproduce.load_record("c09", run)


def test_reproduce_writes_a_fixture_the_generic_test_can_enforce(tmp_path, monkeypatch):
    monkeypatch.setattr(reproduce, "FIXTURES", tmp_path)
    record = c07_like()
    reproduce.write_fixture("gptoss_c07_test", "c07 skipped a tool", record, None, reproduce.diagnose(record))
    fixture = json.loads((tmp_path / "gptoss_c07_test.json").read_text(encoding="utf-8"))
    assert fixture["report"] == record["report"] and fixture["truth"]["status"] == "refer"
    assert fixture["expect"] == {"guard": "reject", "guard_problem": None, "checks_fail": ["dataset_caveat_present"]}
    with pytest.raises(SystemExit):  # never overwrites
        reproduce.write_fixture("gptoss_c07_test", "again", record, None, reproduce.diagnose(record))


def test_reproduce_builds_a_record_from_mcp_outputs():
    classification = {k: CLASSIFIER[k] for k in ("predicted_class", "calibrated_probability", "probabilities",
                                                 "status", "referral_threshold", "referral_message")}
    report = c07_like()["report"]
    record = reproduce.record_from_mcp({"classification": classification, "report": {
        "source": "llm", "report": report, "tool_calls": ["run_classifier", "get_gradcam"], "model": "m"}})
    assert record["report_source"] == "llm" and record["tool_calls"] == ["run_classifier", "get_gradcam"]
    assert record["tool_outputs"]["condition_info"]["model_limitations"]  # from agent/condition_info.json
    assert "dataset_caveat_present" in reproduce.diagnose(record)["failing_checks"]


# add_case.py

def make_dataset(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("relpath,label,split\npneumonia/train/NORMAL/a.jpeg,normal,test\n"
                        "pneumonia/train/NORMAL/b.jpeg,normal,test\npneumonia/train/NORMAL/c.jpeg,normal,train\n",
                        encoding="utf-8")
    cases = tmp_path / "cases.jsonl"
    case = EvalCase(schema_version=CASE_SCHEMA_VERSION, case_id="c01-a", relpath="pneumonia/train/NORMAL/a.jpeg",
                    source="pneumonia", split="test", true_label="Normal", categories=["correct"],
                    snapshot={"predicted_class": "Normal", "calibrated_probability": 0.99, "status": "confident"})
    cases.write_text(case.model_dump_json() + "\n", encoding="utf-8")
    return manifest, cases


def test_add_case_leaves_an_existing_case_alone(tmp_path):
    manifest, cases = make_dataset(tmp_path)
    before = cases.read_text(encoding="utf-8")
    case, added = add_case.add_case("data/pneumonia/chest_xray/train/NORMAL/a.jpeg", None, cases, manifest)
    assert not added and case.case_id == "c01-a" and cases.read_text(encoding="utf-8") == before


def test_add_case_refuses_training_images_and_unknown_files(tmp_path):
    manifest, cases = make_dataset(tmp_path)
    with pytest.raises(SystemExit, match="'train' split"):
        add_case.add_case("pneumonia/train/NORMAL/c.jpeg", None, cases, manifest)
    with pytest.raises(SystemExit, match="not in"):
        add_case.add_case("somewhere/else.png", None, cases, manifest)


def test_add_case_appends_a_regression_case(tmp_path, monkeypatch):
    manifest, cases = make_dataset(tmp_path)
    snapshot = {"predicted_class": "Pneumonia", "calibrated_probability": 0.95, "status": "confident"}
    monkeypatch.setattr(add_case, "classify", lambda relpath, service, data_dir: snapshot)
    case, added = add_case.add_case("pneumonia/train/NORMAL/b.jpeg", lambda: None, cases, manifest)
    assert added and case.case_id == "c02-b" and case.categories == ["error", "regression"]
    from evals.dataset import load_cases
    assert [c.case_id for c in load_cases(cases)] == ["c01-a", "c02-b"]


def test_next_case_id_follows_the_highest_number():
    from evals.dataset import next_case_id
    cases = [EvalCase(schema_version=1, case_id=cid, relpath="x", source="tb", split="test", true_label="Normal",
                      categories=["correct"], snapshot={"predicted_class": "Normal", "calibrated_probability": 1.0,
                                                        "status": "confident"}) for cid in ("c01-x", "c36-y")]
    assert next_case_id(cases, "tb/Normal/Normal-1.png") == "c37-Normal-1"


# verify.py

def test_verify_interprets_replay_failures(monkeypatch):
    assert verify.interpret_replay(0, "", [], False)[0] == "PASS"
    mismatch = "Dataset hash mismatch: evals/runs/x.jsonl was recorded against [...]"
    assert verify.interpret_replay(1, mismatch, [], True)[0] == "EXPECTED"
    assert verify.interpret_replay(1, mismatch, [], False)[0] == "FAIL"
    drop = "FAIL: guard_replay_consistent pass rate 97.2% < baseline 100.0%\n"
    monkeypatch.setattr(verify, "guard_replay_failures", lambda: ["c07-NORMAL2-IM-0780-0001"])
    assert verify.interpret_replay(1, drop, ["c07"], False)[0] == "EXPECTED"
    assert verify.interpret_replay(1, drop, ["c32"], False)[0] == "FAIL"  # a case the fix didn't target
    assert verify.interpret_replay(1, drop + "FAIL: length_limits pass rate 90% < baseline 100%", ["c07"], False)[0] == "FAIL"


def test_verify_knows_the_baseline_replay_tests():
    out = ("FAILED tests/test_evals.py::test_committed_baseline_replays_cleanly - AssertionError: assert 1 == 0\n"
           "FAILED tests/test_mcp_server.py::test_eval_summary_matches_the_committed_baseline\n2 failed, 477 passed")
    assert verify.pytest_failures(out) == verify.BASELINE_TESTS
    for test_id in verify.BASELINE_TESTS:  # renamed or moved tests would make verify.py report FAIL
        path, name = test_id.split("::")
        assert f"def {name}(" in (Path(__file__).resolve().parent.parent / path).read_text(encoding="utf-8")


# compare_runs.py

def test_compare_flags_more_fallbacks_and_rejections():
    rejection = [{"step": 4, "text": "{}", "problem": "grounding check failed"}]
    before = {"c07": c07_like("c07"), "c08": c07_like("c08", caveat_quoted=True)}
    same = {"c07": c07_like("c07", rejected=rejection, caveat_quoted=True), "c08": c07_like("c08", caveat_quoted=True)}
    md, worse = compare_runs.compare(before, same)
    assert worse and "| Guard rejections (drafts) | 0 | 1 |" in md and "`c07`: llm (0 rejected) -> llm (1 rejected)" in md
    fell_back = dict(before, c07=c07_like("c07", source="fallback", rejected=rejection * 2))
    md, worse = compare_runs.compare(before, fell_back)
    assert worse and "| Template fallbacks | 0/2 (0.0%) | 1/2 (50.0%) |" in md
    md, worse = compare_runs.compare(before, {"c07": c07_like("c07", caveat_quoted=True), "c08": before["c08"]})
    assert not worse and "| Check `dataset_caveat_present` | 50.0% | 100.0% |" in md
    # API-level model errors (text None) and truncations aren't counted as guard rejections
    errors = [{"step": 2, "text": None, "problem": "invalid tool call"}, {"step": 4, "text": "{", "finish_reason": "length"}]
    s = compare_runs.stats({"c07": c07_like("c07", rejected=errors)})
    assert (s["guard_rejections"], s["model_errors"], s["truncated"]) == (0, 1, 1)


# live_eval.py

def test_live_eval_reports_progress_and_stops(tmp_path, capsys):
    log = tmp_path / "run.log"
    log.write_text("Run evals/runs/20261026T000000Z.jsonl: 36 cases, dataset c475a686fb48\n"
                   "[1/36] c01: llm, judge 5-5-5-5 (4 agent calls)\n[2/36] c02: llm, judge 5-5-5-4 (4 agent calls)\n",
                   encoding="utf-8")
    state = {"log": str(log), "exit_file": str(tmp_path / "run.exit"), "started": 0}
    assert live_eval.report(state, live_eval.exit_code(state)) == 2
    assert "2 cases written" in capsys.readouterr().out
    Path(state["exit_file"]).write_text("2")
    assert live_eval.report(state, live_eval.exit_code(state)) == 1
    assert "start --resume evals/runs/20261026T000000Z.jsonl" in capsys.readouterr().out
    Path(state["exit_file"]).write_text("0")
    assert live_eval.report(state, 0) == 0


# release_check.py

def test_release_check_skips_the_smoke_test_without_an_api(monkeypatch):
    import httpx

    def down(*args, **kwargs):
        raise httpx.ConnectError("refused")
    monkeypatch.setattr(httpx, "get", down)
    monkeypatch.setattr(release_check, "config", lambda: (lambda name: None))
    step = release_check.smoke(None, False)
    assert step["result"] == "SKIP" and "no API answered" in step["detail"]


def test_release_check_skips_without_credentials(monkeypatch):
    assert release_check.get_token(lambda name: None)[0] is None
    assert release_check.get_token(lambda name: "tok" if name == "SUPABASE_ACCESS_TOKEN" else None) == ("tok", None)


def test_release_check_summary_never_counts_skip_as_pass():
    step = lambda name, result: {"step": name, "result": result, "seconds": 1, "detail": "x|y", "output": ""}
    assert "READY: every step passed" in release_check.summary([step("pytest", "PASS")])
    md = release_check.summary([step("pytest", "PASS"), step("smoke test", "SKIP")])
    assert "READY, with skipped steps: smoke test" in md and "x/y" in md
    assert "NOT READY: eval replay failed" in release_check.summary([step("eval replay", "FAIL"), step("smoke test", "SKIP")])
    assert release_check.replay_detail("FAIL: length_limits pass rate 90% < baseline 100%") == \
        "length_limits pass rate 90% < baseline 100%"
    assert release_check.smoke_detail("FAIL  POST /report/{id}  3.4s  KeyError\n5/6 steps passed") == \
        "5/6 steps passed (failed: POST /report/{id} 3.4s)"


# SKILL.md frontmatter (a bad one makes Claude Code skip the skill silently)

@pytest.mark.parametrize("skill", sorted(p.name for p in SKILLS.iterdir() if (p / "SKILL.md").exists()))
def test_skill_frontmatter_parses(skill):
    import yaml
    text = (SKILLS / skill / "SKILL.md").read_text(encoding="utf-8")
    assert text.startswith("---\n"), "SKILL.md must start with YAML frontmatter"
    meta = yaml.safe_load(text.split("---\n", 2)[1])
    assert meta["name"] == skill
    assert 50 < len(meta["description"]) <= 1024
    assert "Use when" in meta["description"] or "Use before" in meta["description"]  # says when to load it
