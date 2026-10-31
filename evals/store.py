"""MongoDB store for eval records, so they can be queried across runs.

The committed evals/runs/*.jsonl files stay the source of truth (CI replays them offline); Mongo
is a queryable copy. Postgres (Supabase) stays the store for user data; nothing here touches it.

Collections (database MONGODB_DB, default "clinical_evals"):

- eval_records: one document per (run_id, case_id), the run file's LAST record for that case
  (as load_run reads it). The record's own fields are stored verbatim, plus:
    derived   fields the queries read, built by the normaliser for the record's record_version
    imported  source file, line, and the code version that computed derived.checks
  _id is "<run_id>/<case_id>", so a replace with upsert=True is idempotent and human_labels can
  join on _id with a plain $lookup.
- eval_runs: one document per run (_id = run_id): date, models, dataset hash, counts, pass rates
  per deterministic check, mean judge score per criterion.
- human_labels: evals/human_labels.csv rows, _id "<run_id>/<case_id>" of the record they scored.

Record versions: queries only read `derived`, never the raw shape. Each supported record_version
has a normaliser that produces the same `derived` fields, so older and newer shapes can sit in one
collection. A record with an unknown version is refused, not stored.
"""
import csv
import os
import subprocess
import tomllib
from datetime import datetime, timezone
from pathlib import Path

from pymongo import ASCENDING, MongoClient

from evals.checks import CHECKS, run_checks
from evals.schema import CRITERIA

REPO = Path(__file__).resolve().parent.parent
SECRETS_PATH = REPO / ".streamlit" / "secrets.toml"
DEFAULT_DB = "clinical_evals"
RECORDS, RUNS, LABELS = "eval_records", "eval_runs", "human_labels"

# (collection, keys, options, why). Small collections (hundreds of documents): these exist for
# upsert correctness and as the queries' access paths, not for speed.
INDEXES = [
    (RECORDS, [("run_id", ASCENDING), ("case_id", ASCENDING)], {"unique": True, "name": "run_case"},
     "One record per (run, case) even if a document is ever written with another _id; its run_id prefix "
     "serves --run/--since filters (all four queries)"),
    (RECORDS, [("derived.rejected_count", ASCENDING)], {"name": "rejected_count"},
     "query 'rejected': one branch of its $or (records with rejected drafts)"),
    (RECORDS, [("derived.fallback", ASCENDING)], {"name": "fallback"},
     "query 'rejected': the other $or branch (template fallbacks); MongoDB can use one index per branch"),
    (RECORDS, [("derived.agent_model", ASCENDING), ("run_id", ASCENDING)], {"name": "agent_model_run"},
     "query 'judge-by-model' with --agent-model: an index seek instead of a scan (run_id second serves --run "
     "within that model). Without a filter the query reads every judged record and no index avoids that"),
    (RUNS, [("date", ASCENDING)], {"name": "date"},
     "query 'check-trend': runs in date order, and --since"),
]


class UnsupportedRecordVersion(ValueError):
    """The record was written with a record_version this code has no normaliser for."""


class MongoNotConfigured(RuntimeError):
    pass


# ---- Normalisers: raw record -> derived fields (the queries' contract) ----

def normalize_v1(record, case=None):
    """record_version 1. Fields added since v1 (tool_calls, pipeline, agent_models, ...) are optional."""
    judge = record.get("judge") or {}
    rejected = record.get("rejected") or []
    classifier = record.get("classifier") or {}
    return {
        "agent_model": record.get("agent_model"),
        "judge_model": judge.get("model"),
        "predicted_class": classifier.get("predicted_class"),
        "status": classifier.get("status"),
        "report_source": record.get("report_source"),
        "fallback": record.get("report_source") != "llm",
        "rejected_count": len(rejected),
        "rejected_problems": [item.get("problem") for item in rejected],
        "judge_scores": {c: judge["scores"][c]["score"] for c in CRITERIA} if judge.get("scores") else None,
        # Today's checks.py, like --replay; a crashing check counts as a failure (run_checks)
        "checks": run_checks(record) if record.get("report") else None,
        "case": case,
    }


NORMALIZERS = {1: normalize_v1}


def record_id(run_id, case_id):
    return f"{run_id}/{case_id}"


def code_version():
    """`git describe --always --dirty` of the checks code, or None outside a git checkout."""
    try:
        out = subprocess.run(["git", "describe", "--always", "--dirty"], cwd=REPO, capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def record_document(record, cases_by_id=None, data_hash=None, source_file=None, line=None, checks_code=None):
    """The eval_records document for one run record. Raises UnsupportedRecordVersion."""
    version = record.get("record_version")
    normalize = NORMALIZERS.get(version)
    if normalize is None:
        raise UnsupportedRecordVersion(
            f"{record.get('run_id')}/{record.get('case_id')}: record_version {version!r} is not supported "
            f"(known: {sorted(NORMALIZERS)}). Add a normaliser to evals/store.py.")
    case = None
    if cases_by_id and record.get("dataset_hash") == data_hash and record.get("case_id") in cases_by_id:
        c = cases_by_id[record["case_id"]]  # only joined when the record was made from this exact dataset
        case = {"true_label": c.true_label, "categories": list(c.categories), "source": c.source}
    doc = {k: v for k, v in record.items() if k not in ("_id", "derived", "imported")}
    # No import timestamp: re-importing the same file with the same code leaves the document unchanged
    doc.update(_id=record_id(record["run_id"], record["case_id"]), derived=normalize(record, case),
               imported={"file": source_file, "line": line, "checks_code": checks_code})
    return doc


def run_date(run_id, records):
    """From a timestamp run_id (20261020T075220Z), else the earliest recorded_at."""
    try:
        return datetime.strptime(run_id, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        stamps = sorted(r["recorded_at"] for r in records if r.get("recorded_at"))
        return datetime.fromisoformat(stamps[0]) if stamps else None


def _one_or_list(values):
    values = sorted({v for v in values if v is not None})
    return values[0] if len(values) == 1 else (values or None)


def run_document(run_id, docs, n_cases=None, data_hash=None, source_file=None, notes=None, is_baseline=False,
                 checks_code=None):
    """The eval_runs document, from this run's eval_records documents."""
    with_report = [d for d in docs if d["derived"]["checks"] is not None]
    judged = [d for d in docs if d["derived"]["judge_scores"]]
    rates = {}
    for name in CHECKS:  # same arithmetic as evals.checks.pass_rates
        outcomes = [d["derived"]["checks"].get(name) for d in with_report]
        applicable = [o for o in outcomes if o is not None]
        passed = sum(1 for o in applicable if o)
        rates[name] = {"passed": passed, "applicable": len(applicable),
                       "rate": round(passed / len(applicable), 4) if applicable else None}
    means = {}
    for c in CRITERIA:
        scores = [d["derived"]["judge_scores"][c] for d in judged]
        means[c] = round(sum(scores) / len(scores), 3) if scores else None
    hashes = {d.get("dataset_hash") for d in docs}
    expected = n_cases if hashes == {data_hash} and n_cases else len(docs)
    return {
        "_id": run_id, "run_id": run_id, "date": run_date(run_id, docs),
        "agent_model": _one_or_list(d["derived"]["agent_model"] for d in docs),
        "judge_model": _one_or_list(d["derived"]["judge_model"] for d in docs),
        "provider": _one_or_list(d.get("provider") for d in docs),
        "dataset_hash": _one_or_list(hashes),
        "dataset_matches_cases": hashes == {data_hash},
        "records": len(docs), "reports": len(with_report), "judged": len(judged),
        "llm_written": sum(not d["derived"]["fallback"] for d in docs),
        "fallbacks": sum(d["derived"]["fallback"] for d in docs),
        "reports_with_rejections": sum(d["derived"]["rejected_count"] > 0 for d in docs),
        "complete": len(judged) >= expected,
        "pass_rates": rates, "judge_means": means,
        "is_baseline": is_baseline, "notes": notes, "source_file": source_file, "checks_code": checks_code,
    }


def label_documents(rows, run_id, labels_file=None):
    """human_labels documents from the CSV rows. Blank or invalid scores are stored as None."""
    docs = []
    for row in rows:
        scores = {}
        for c in CRITERIA:
            try:
                value = int((row.get(c) or "").strip())
                scores[c] = value if 1 <= value <= 5 else None
            except ValueError:
                scores[c] = None
        docs.append({"_id": record_id(run_id, row["case_id"]), "run_id": run_id, "case_id": row["case_id"],
                     "scores": scores, "notes": (row.get("notes") or "").strip() or None,
                     "labels_file": labels_file})
    return docs


def read_label_rows(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


# ---- Store ----

class MongoEvalStore:
    """Upserts eval records, runs and human labels into one MongoDB database (pymongo or mongomock)."""

    def __init__(self, db):
        self.db = db
        self.records, self.runs, self.labels = db[RECORDS], db[RUNS], db[LABELS]

    def ensure_indexes(self):
        """Create every index in INDEXES (a no-op when it already exists)."""
        for collection, keys, options, _why in INDEXES:
            self.db[collection].create_index(keys, **options)

    def _replace(self, collection, doc):
        """'inserted' | 'updated' | 'unchanged'."""
        result = collection.replace_one({"_id": doc["_id"]}, doc, upsert=True)
        if result.upserted_id is not None:
            return "inserted"
        return "updated" if result.modified_count else "unchanged"

    def upsert_record(self, doc):
        return self._replace(self.records, doc)

    def upsert_run(self, doc):
        return self._replace(self.runs, doc)

    def upsert_label(self, doc):
        return self._replace(self.labels, doc)

    def run_records(self, run_id):
        return list(self.records.find({"run_id": run_id}))


# ---- Connection ----

def mongo_settings(environ=None, secrets_path=SECRETS_PATH):
    """(uri, db name, where the uri came from). MONGODB_URI in the env wins over [mongodb] uri in secrets.toml."""
    env = os.environ if environ is None else environ
    section = {}
    if Path(secrets_path).is_file():
        with open(secrets_path, "rb") as f:
            section = tomllib.load(f).get("mongodb", {})
    if env.get("MONGODB_URI"):
        uri, origin = env["MONGODB_URI"], "env MONGODB_URI"
    elif section.get("uri"):
        uri, origin = section["uri"], "secrets.toml [mongodb] uri"
    else:
        raise MongoNotConfigured("No MongoDB URI: set the MONGODB_URI environment variable or [mongodb] uri in "
                                 ".streamlit/secrets.toml.")
    return uri, env.get("MONGODB_DB") or section.get("db") or DEFAULT_DB, origin


def safe_host(uri):
    """Host part of a mongodb:// or mongodb+srv:// URI, without credentials (the URI holds the password)."""
    rest = uri.split("://", 1)[-1]
    return rest.split("/", 1)[0].split("?", 1)[0].rsplit("@", 1)[-1]


def connect(environ=None, secrets_path=SECRETS_PATH, timeout_ms=10000):
    """(MongoEvalStore, description for logs). Pings the server so a bad URI or blocked IP fails here."""
    uri, db_name, origin = mongo_settings(environ, secrets_path)
    client = MongoClient(uri, serverSelectionTimeoutMS=timeout_ms, appname="clinical-evals")
    client.admin.command("ping")
    return MongoEvalStore(client[db_name]), f"{safe_host(uri)}/{db_name} (from {origin})"


def connect_or_exit(prefix="MongoDB"):
    """connect() for the command-line tools: one readable line instead of a traceback."""
    try:
        return connect()
    except MongoNotConfigured as e:
        raise SystemExit(f"{prefix}: {e}")
    except Exception as e:  # bad credentials, IP not on the Atlas access list, DNS, timeout
        raise SystemExit(f"{prefix}: could not connect ({type(e).__name__}: {e})")
