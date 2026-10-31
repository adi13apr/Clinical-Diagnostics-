"""MongoDB eval store, importer and query pipelines, offline with mongomock (no server, no URI).

mongomock 4.3 runs every stage these pipelines use ($match, $project, $objectToArray, $unwind,
$group, $lookup on a local/foreign field, $map, $filter, $addFields, $sort), so the pipelines
themselves are executed here, not just built. It does NOT support $lookup with let/pipeline,
$reduce, or bulk_write(ReplaceOne) under pymongo 4.19; the code avoids all three.
"""
import json
from pathlib import Path

import mongomock
import pytest
from pymongo.errors import DuplicateKeyError

from evals import mongo_import, queries
from evals import run as eval_run
from evals.checks import pass_rates
from evals.dataset import dataset_hash, load_cases
from evals.human_labels import HUMAN_LABELS_PATH, agreement, labels_run, read_labels
from evals.mongo_import import LiveMongoWriter, import_runs
from evals.queries import (
    check_trend_pipeline, disagreements_pipeline, judge_by_model_pipeline, rejected_pipeline, run_query,
)
from evals.schema import CRITERIA
from evals.store import (
    INDEXES, LABELS, RECORDS, MongoEvalStore, MongoNotConfigured, UnsupportedRecordVersion, label_documents,
    mongo_settings, record_document, safe_host,
)

REPO = Path(__file__).resolve().parent.parent
BASELINE_RUN = REPO / "evals" / "runs" / "20261020T075220Z.jsonl"  # fixture source; was the baseline until c07
CASES = load_cases()
DATA_HASH = dataset_hash(CASES)
CASES_BY_ID = {c.case_id: c for c in CASES}


@pytest.fixture
def store():
    s = MongoEvalStore(mongomock.MongoClient().db)
    s.ensure_indexes()
    return s


def real_records(n=3):
    """The baseline run's final records for its first n cases (real shapes, real check outcomes)."""
    return [r for _, r in sorted(mongo_import.last_records(BASELINE_RUN).values(), key=lambda x: x[0])][:n]


def judged(record, run_id, agent_model="openai/gpt-oss-120b", scores=(5, 5, 5, 4), judge_model="qwen/qwen3.8-27b"):
    r = json.loads(json.dumps(record))
    r.update(run_id=run_id, agent_model=agent_model)
    r["judge"] = {"model": judge_model, "scores": {c: {"score": s, "reason": "r"} for c, s in zip(CRITERIA, scores)}}
    return r


def write_run(path, records, unjudged_first=True):
    """A run file like the live runner writes: each case unjudged first, then judged."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in records:
            if unjudged_first:
                f.write(json.dumps(dict(r, judge=None), sort_keys=True) + "\n")
            f.write(json.dumps(r, sort_keys=True) + "\n")
    return path


def fixture_run(tmp_path, run_id="20261101T000000Z"):
    """3 cases: clean, one rejected draft then an accepted report, and a template fallback."""
    clean, rejected, fallback = [judged(r, run_id) for r in real_records(3)]
    rejected["rejected"] = [{"text": "not json", "problem": "invalid JSON: expecting value"}]
    fallback.update(report_source="fallback", fallback_reason="The AI model returned an invalid tool call.",
                    fallback_error_kind="model")
    return write_run(tmp_path / f"{run_id}.jsonl", [clean, rejected, fallback]), [clean, rejected, fallback]


# ---- Upserts and the importer ----

def test_import_is_idempotent(store, tmp_path):
    path, _ = fixture_run(tmp_path)
    first = import_runs(store, [path])
    before = sorted(store.records.find(), key=lambda d: d["_id"])
    second = import_runs(store, [path])
    assert first["runs"][0][3:6] == [3, 0, 0]   # inserted, updated, unchanged
    assert second["runs"][0][3:6] == [0, 0, 3]
    assert sorted(store.records.find(), key=lambda d: d["_id"]) == before
    assert store.records.count_documents({}) == 3 and store.runs.count_documents({}) == 1


def test_last_record_per_case_wins(store, tmp_path):
    path, records = fixture_run(tmp_path)
    import_runs(store, [path])
    doc = store.records.find_one({"_id": f"20261101T000000Z/{records[0]['case_id']}"})
    assert doc["judge"]["model"] == "qwen/qwen3.8-27b"  # line 2 (judged), not line 1 (unjudged)
    assert doc["imported"]["line"] == 2


def test_changed_record_is_updated_in_place(store, tmp_path):
    path, records = fixture_run(tmp_path)
    import_runs(store, [path])
    records[0]["report"]["summary"] = "Changed."
    write_run(path, records)
    result = import_runs(store, [path])
    assert result["runs"][0][3:6] == [0, 1, 2]
    assert store.records.find_one({"case_id": records[0]["case_id"]})["report"]["summary"] == "Changed."


def test_document_keeps_the_record_verbatim_plus_derived():
    record = real_records(1)[0]
    doc = record_document(record, CASES_BY_ID, DATA_HASH, "evals/runs/x.jsonl", 7, "abc123")
    raw = {k: v for k, v in doc.items() if k not in ("_id", "derived", "imported")}
    assert raw == record
    assert doc["_id"] == f"{record['run_id']}/{record['case_id']}"
    assert doc["imported"] == {"file": "evals/runs/x.jsonl", "line": 7, "checks_code": "abc123"}
    assert doc["derived"]["case"]["true_label"] == CASES_BY_ID[record["case_id"]].true_label


def test_id_is_the_first_field_of_every_document(tmp_path):
    """Atlas reports a replacement as modified when _id isn't first, even with identical content, so a
    re-import said 'updated' for all 180 records. mongomock ignores field order, hence this direct check."""
    record = real_records(1)[0]
    assert list(record_document(record))[0] == "_id"
    path, _ = fixture_run(tmp_path)
    docs, runs, _ = mongo_import.build_documents(path, CASES_BY_ID, DATA_HASH)
    labels = label_documents([{"case_id": record["case_id"]}], "r")
    assert all(list(d)[0] == "_id" for d in docs + list(runs.values()) + labels)


def test_case_join_only_for_the_same_dataset():
    record = dict(real_records(1)[0], dataset_hash="other")
    assert record_document(record, CASES_BY_ID, DATA_HASH)["derived"]["case"] is None


def test_unknown_record_version_is_refused(store, tmp_path):
    good, future = [judged(r, "20261101T000000Z") for r in real_records(2)]
    future["record_version"] = 99
    path = write_run(tmp_path / "20261101T000000Z.jsonl", [good, future])
    with pytest.raises(UnsupportedRecordVersion):
        record_document(future)
    result = import_runs(store, [path])
    assert len(result["refused"]) == 1 and "record_version 99" in result["refused"][0]
    assert store.records.count_documents({}) == 1


def test_v1_records_with_and_without_optional_fields_share_derived_fields():
    """Main's runs have no tool_calls; the multi-agent branch's v1 records add pipeline fields. Same contract."""
    old, new = [judged(r, "r") for r in real_records(2)]
    old.pop("tool_calls", None)
    new.update(tool_calls=["run_classifier"], pipeline="multi", agent_models={"writer": "x"}, revisions=[])
    a, b = record_document(old)["derived"], record_document(new)["derived"]
    assert a.keys() == b.keys()
    unjudged = record_document(dict(old, judge=None))["derived"]
    assert unjudged["judge_scores"] is None and unjudged["judge_model"] is None


def test_indexes_exist_and_run_case_is_unique(store):
    names = set(store.records.index_information()) | set(store.runs.index_information())
    assert {options["name"] for _, _, options, _ in INDEXES} <= names
    store.records.insert_one({"_id": "a", "run_id": "r", "case_id": "c"})
    with pytest.raises(DuplicateKeyError):
        store.records.insert_one({"_id": "b", "run_id": "r", "case_id": "c"})


def test_run_document_matches_replay_pass_rates(store):
    """eval_runs.pass_rates are the numbers --replay prints for the committed baseline run."""
    baseline = REPO / eval_run.load_baseline()["run"]
    import_runs(store, [baseline, BASELINE_RUN])
    run = store.runs.find_one({"_id": baseline.stem})
    expected = pass_rates([r for r in eval_run.load_run(baseline).values() if r.get("report")])
    assert run["pass_rates"] == expected
    assert run["is_baseline"] and run["complete"] and run["dataset_matches_cases"]
    assert store.runs.count_documents({"is_baseline": True}) == 1
    assert run["agent_model"] == "openai/gpt-oss-120b" and run["judge_model"] == "qwen/qwen3.8-27b"
    assert run["date"].year == 2026 and run["notes"]


def test_dry_run_writes_nothing(tmp_path, monkeypatch, capsys):
    path, _ = fixture_run(tmp_path)
    monkeypatch.setattr(mongo_import, "connect_or_exit", lambda: pytest.fail("dry run must not connect"))
    assert mongo_import.main([str(tmp_path), "--dry-run", "--no-labels"]) == 0
    assert "Dry run: nothing written" in capsys.readouterr().out


# ---- Settings ----

def test_env_uri_wins_over_secrets(tmp_path):
    secrets = tmp_path / "secrets.toml"
    secrets.write_text('[mongodb]\nuri = "mongodb://from-secrets"\ndb = "evals_s"\n', encoding="utf-8")
    assert mongo_settings({"MONGODB_URI": "mongodb://from-env"}, secrets) == \
        ("mongodb://from-env", "evals_s", "env MONGODB_URI")
    assert mongo_settings({}, secrets)[0] == "mongodb://from-secrets"
    with pytest.raises(MongoNotConfigured):
        mongo_settings({}, tmp_path / "missing.toml")


@pytest.mark.parametrize("error, expected", [
    (MongoNotConfigured("No MongoDB URI: set MONGODB_URI"), "MongoDB: No MongoDB URI"),
    (TimeoutError("cluster0.x.mongodb.net:27017: timed out"), "MongoDB: could not connect (TimeoutError"),
])
def test_cli_connection_errors_are_one_line(monkeypatch, error, expected):
    import evals.store as store_module

    def fail():
        raise error
    monkeypatch.setattr(store_module, "connect", fail)
    with pytest.raises(SystemExit) as exc:
        queries.main(["rejected"])
    assert str(exc.value).startswith(expected)


def test_safe_host_drops_credentials():
    assert safe_host("mongodb+srv://u:p%40ss@cluster0.ab1cd.mongodb.net/?retryWrites=true") == \
        "cluster0.ab1cd.mongodb.net"
    assert safe_host("mongodb://h1:27017,h2:27017/evals") == "h1:27017,h2:27017"


# ---- Live --mongo ----

class FlakyStore(MongoEvalStore):
    def __init__(self, db, failures):
        super().__init__(db)
        self.failures = failures

    def upsert_record(self, doc):
        if self.failures:
            self.failures -= 1
            raise ConnectionError("network down")
        return super().upsert_record(doc)


def test_live_writer_never_raises_and_finish_repairs(tmp_path):
    store = FlakyStore(mongomock.MongoClient().db, failures=1)
    path = tmp_path / "20261101T000000Z.jsonl"
    logs = []
    writer = LiveMongoWriter(store, CASES, DATA_HASH, path, log=logs.append)
    for record in [judged(r, "20261101T000000Z") for r in real_records(2)]:
        eval_run.append_record(path, record)
        writer.write(record)  # the first write fails: warned, not raised
    assert writer.failures == 1 and "WARNING: MongoDB write failed" in logs[0]
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2  # the run file is unaffected
    assert store.records.count_documents({}) == 1
    writer.finish()
    assert store.records.count_documents({}) == 2 and store.runs.count_documents({}) == 1
    assert "1 failed live writes repaired" in logs[-1]


def test_live_writer_documents_equal_imported_ones(tmp_path):
    """Writing live and importing afterwards give the same documents (same line numbers too)."""
    live, imported = MongoEvalStore(mongomock.MongoClient().db), MongoEvalStore(mongomock.MongoClient().db)
    path = tmp_path / "20261101T000000Z.jsonl"
    writer = LiveMongoWriter(live, CASES, DATA_HASH, path, log=lambda m: None)
    for record in [judged(r, "20261101T000000Z") for r in real_records(2)]:
        for line in (dict(record, judge=None), record):
            eval_run.append_record(path, line)
            writer.write(line)
    import_runs(imported, [path])
    key = lambda d: d["_id"]
    assert sorted(live.records.find(), key=key) == sorted(imported.records.find(), key=key)


def test_mongo_flag_requires_live():
    with pytest.raises(SystemExit):
        eval_run.main(["--replay", "--mongo"])


# ---- Pipelines on fixture data ----

def test_rejected_query(store, tmp_path):
    path, (clean, rejected, fallback) = fixture_run(tmp_path)
    import_runs(store, [path])
    rows = run_query(store.db, RECORDS, rejected_pipeline())
    assert [(r["case_id"], r["rejected"], r["source"]) for r in rows] == \
        [(rejected["case_id"], 1, "llm"), (fallback["case_id"], 0, "fallback")]
    assert rows[0]["problems"] == ["invalid JSON: expecting value"]
    assert rows[1]["fallback_reason"].startswith("The AI model")
    assert run_query(store.db, RECORDS, rejected_pipeline(run="other")) == []


def test_check_trend_query(store, tmp_path):
    a = [judged(r, "20261101T000000Z") for r in real_records(3)]
    b = [judged(r, "20261102T000000Z") for r in real_records(3)]
    b[0]["report"]["status"] = "refer" if b[0]["report"]["status"] == "confident" else "confident"
    import_runs(store, [write_run(tmp_path / "20261102T000000Z.jsonl", b),
                        write_run(tmp_path / "20261101T000000Z.jsonl", a)])
    rows = run_query(store.db, RECORDS, check_trend_pipeline())
    assert [r["run_id"] for r in rows][0] == "20261101T000000Z"  # date order, not import order
    status = {r["run_id"]: r for r in rows if r["check"] == "status_matches_classifier"}
    assert (status["20261101T000000Z"]["passed"], status["20261101T000000Z"]["applicable"]) == (3, 3)
    assert status["20261102T000000Z"]["passed"] == 2 and status["20261102T000000Z"]["rate"] == pytest.approx(2 / 3)
    # Not-applicable outcomes (None) are excluded from the denominator, as in pass_rates
    stored = store.runs.find_one({"_id": "20261101T000000Z"})["pass_rates"]
    for r in rows:
        if r["run_id"] == "20261101T000000Z":
            assert (r["passed"], r["applicable"]) == (stored[r["check"]]["passed"], stored[r["check"]]["applicable"])
    assert {r["run_id"] for r in run_query(store.db, RECORDS, check_trend_pipeline(since="2026-11-02"))} == \
        {"20261102T000000Z"}


def test_judge_by_model_query(store, tmp_path):
    x = [judged(r, "20261101T000000Z", "model-x", s) for r, s in zip(real_records(2), [(5, 5, 5, 5), (3, 4, 1, 5)])]
    y = [judged(r, "20261102T000000Z", "model-y", (4, 4, 4, 4)) for r in real_records(2)]
    import_runs(store, [write_run(tmp_path / "x.jsonl", x), write_run(tmp_path / "y.jsonl", y)])
    rows = run_query(store.db, RECORDS, judge_by_model_pipeline())
    by = {(r["agent"], r["criterion"]): r for r in rows}
    assert by[("model-x", "faithfulness")]["mean"] == 4.0 and by[("model-x", "actionability")]["min"] == 1
    assert by[("model-x", "actionability")]["low"] == 1 and by[("model-x", "faithfulness")]["n"] == 2
    assert by[("model-y", "plain_language")]["mean"] == 4.0 and by[("model-y", "plain_language")]["runs"] == 1
    only_y = run_query(store.db, RECORDS, judge_by_model_pipeline(agent_model="model-y"))
    assert {r["agent"] for r in only_y} == {"model-y"} and len(only_y) == len(CRITERIA)


def test_disagreements_query(store, tmp_path):
    path, (clean, rejected, _) = fixture_run(tmp_path)  # judge scores 5,5,5,4
    import_runs(store, [path])
    rows_in = [{"case_id": clean["case_id"], "faithfulness": "3", "uncertainty_language": "5",
                "actionability": "4", "plain_language": "4", "notes": "too sure"},
               {"case_id": rejected["case_id"], "faithfulness": "", "uncertainty_language": "",
                "actionability": "", "plain_language": "", "notes": ""},       # not scored yet
               {"case_id": "not-in-run", "faithfulness": "1", "uncertainty_language": "1",
                "actionability": "1", "plain_language": "1", "notes": ""}]
    for doc in label_documents(rows_in, "20261101T000000Z"):
        store.upsert_label(doc)
    rows = run_query(store.db, LABELS, disagreements_pipeline())
    assert [(r["pairs"]["criterion"], r["diff"]) for r in rows] == [("faithfulness", 2), ("actionability", 1)]
    assert rows[0]["notes"] == "too sure"
    assert [r["pairs"]["criterion"] for r in run_query(store.db, LABELS, disagreements_pipeline(min_diff=2))] == \
        ["faithfulness"]


def test_disagreements_agree_with_human_labels_module(store):
    """On the committed labels, the pipeline finds exactly the cells evals.human_labels.agreement counts as inexact."""
    source = labels_run()
    run_path = REPO / source
    import_runs(store, [run_path], HUMAN_LABELS_PATH, Path(source).stem)
    rows = run_query(store.db, LABELS, disagreements_pipeline())
    a = agreement(read_labels(), eval_run.load_run(run_path))
    inexact = sum(round(s["n"] * (1 - s["exact"])) for s in a["per_criterion"].values())
    assert len(rows) == inexact
    c32 = {r["pairs"]["criterion"]: r["diff"] for r in rows if r["case_id"].startswith("c32-")}
    assert c32 == {"faithfulness": 2, "actionability": 2, "uncertainty_language": 1, "plain_language": 1}


def test_queries_cli_prints_tables(store, tmp_path, capsys):
    path, _ = fixture_run(tmp_path)
    import_runs(store, [path])
    assert queries.main(["all", "--show-pipeline"], db=store.db) == 0
    out = capsys.readouterr().out
    for heading in ("Cases the guard rejected", "Pass rate per deterministic check", "Mean judge score",
                    "Judge vs human", "db.eval_records.aggregate(", "db.human_labels.aggregate("):
        assert heading in out
