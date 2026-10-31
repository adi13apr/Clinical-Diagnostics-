"""Load eval run files (and the human labels) into MongoDB.

    python -m evals.mongo_import evals/runs/                     # every *.jsonl in the folder
    python -m evals.mongo_import evals/runs/ ../other/evals/runs/20261027T000140Z.jsonl
    python -m evals.mongo_import evals/runs/ --dry-run            # build the documents, don't connect

Idempotent: each record is replaced by _id "<run_id>/<case_id>", so importing the same files again
changes nothing ("unchanged"). Within a file the last record per case wins, as in evals.run.load_run.
The JSONL files are not modified. Records with an unsupported record_version are refused and listed.
"""
import argparse
import json
import sys
from pathlib import Path

from evals.dataset import dataset_hash, load_cases
from evals.human_labels import HUMAN_LABELS_PATH, labels_run
from evals.queries import table
from evals.store import (
    INDEXES, UnsupportedRecordVersion, code_version, connect_or_exit, label_documents, read_label_rows, record_document,
    run_document,
)

REPO = Path(__file__).resolve().parent.parent
BASELINE_PATH = REPO / "evals" / "baseline.json"


def run_files(paths):
    files = []
    for p in map(Path, paths):
        files += sorted(p.glob("*.jsonl")) if p.is_dir() else [p]
    return files


def last_records(path):
    """{case_id: (line number, record)}: the last record per case, like load_run."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                record = json.loads(line)
                out[record["case_id"]] = (n, record)
    return out


def display_path(path):
    try:
        return Path(path).resolve().relative_to(REPO).as_posix()
    except ValueError:
        return Path(path).resolve().as_posix()


def build_documents(path, cases_by_id, data_hash, baseline_run=None, checks_code=None):
    """(record docs, {run_id: run doc}, refused messages) for one run file."""
    source = display_path(path)
    docs, refused = [], []
    for case_id, (line, record) in sorted(last_records(path).items()):
        try:
            docs.append(record_document(record, cases_by_id, data_hash, source, line, checks_code))
        except UnsupportedRecordVersion as e:
            refused.append(str(e))
    notes_path = Path(path).with_suffix(".notes.md")
    notes = notes_path.read_text(encoding="utf-8").strip() if notes_path.exists() else None
    runs = {}
    for run_id in sorted({d["run_id"] for d in docs}):
        mine = [d for d in docs if d["run_id"] == run_id]
        runs[run_id] = run_document(run_id, mine, len(cases_by_id), data_hash, source, notes,
                                    is_baseline=run_id == baseline_run, checks_code=checks_code)
    return docs, runs, refused


def import_runs(store, paths, labels_path=None, labels_run_id=None, dry_run=False, log=print):
    """Import run files (+ optional labels). Returns {"runs": [...rows], "refused": [...], "labels": counts}."""
    cases = load_cases()
    cases_by_id, data_hash = {c.case_id: c for c in cases}, dataset_hash(cases)
    baseline_run = None
    if BASELINE_PATH.exists():
        baseline_run = Path(json.loads(BASELINE_PATH.read_text(encoding="utf-8"))["run"]).stem
    checks_code = code_version()
    if store is not None and not dry_run:
        store.ensure_indexes()
    rows, all_refused = [], []
    for path in run_files(paths):
        docs, runs, refused = build_documents(path, cases_by_id, data_hash, baseline_run, checks_code)
        all_refused += refused
        counts = {"inserted": 0, "updated": 0, "unchanged": 0}
        if not dry_run:
            for doc in docs:
                counts[store.upsert_record(doc)] += 1
            for run in runs.values():
                store.upsert_run(run)
        for run_id, run in runs.items():
            rows.append([run_id, display_path(path), run["records"], counts["inserted"], counts["updated"],
                         counts["unchanged"], len(refused), run["fallbacks"], run["reports_with_rejections"],
                         "yes" if run["dataset_matches_cases"] else "no"])
    label_counts = None
    if labels_path and Path(labels_path).exists() and labels_run_id:
        label_docs = label_documents(read_label_rows(labels_path), labels_run_id, display_path(labels_path))
        label_counts = {"rows": len(label_docs), "inserted": 0, "updated": 0, "unchanged": 0}
        if not dry_run:
            for doc in label_docs:
                label_counts[store.upsert_label(doc)] += 1
    return {"runs": rows, "refused": all_refused, "labels": label_counts}


class LiveMongoWriter:
    """evals.run --live --mongo: upsert each record right after it is appended to the run file.

    The JSONL file is written first and stays the source of truth; a failed Mongo write is counted
    and warned about once, never raised, so it can't stop a run that is spending LLM quota.
    finish() re-imports the whole run file, which also repairs any record whose live write failed.
    """

    def __init__(self, store, cases, data_hash, run_path, log=print):
        self.store, self.run_path, self.log = store, Path(run_path), log
        self.cases_by_id, self.data_hash = {c.case_id: c for c in cases}, data_hash
        self.checks_code = code_version()
        self.source = display_path(run_path)
        # Line numbers as the importer will see them (appends never write blank lines)
        self.lines = sum(1 for _ in open(run_path, encoding="utf-8")) if self.run_path.exists() else 0
        self.failures = 0

    @classmethod
    def connect(cls, cases, data_hash, run_path, log=print):
        """Fails before the run starts (no URI, bad credentials, IP not allowed), not halfway through it."""
        store, where = connect_or_exit("--mongo")
        try:
            store.ensure_indexes()
        except Exception as e:  # e.g. the user may read but not write
            raise SystemExit(f"--mongo: could not create indexes ({type(e).__name__}: {e})")
        log(f"MongoDB: {where}")
        return cls(store, cases, data_hash, run_path, log)

    def write(self, record):
        self.lines += 1
        try:
            self.store.upsert_record(record_document(record, self.cases_by_id, self.data_hash, self.source,
                                                     self.lines, self.checks_code))
        except Exception as e:
            self.failures += 1
            if self.failures == 1:
                self.log(f"WARNING: MongoDB write failed ({type(e).__name__}: {e}). The run continues; "
                         "the run file is unaffected.")

    def finish(self):
        """Upsert the run document (and re-import every record of this run file)."""
        command = f"python -m evals.mongo_import {self.source}"
        try:
            result = import_runs(self.store, [self.run_path], dry_run=False, log=self.log)
        except Exception as e:
            self.log(f"WARNING: MongoDB run summary failed ({type(e).__name__}: {e}). Import later with: {command}")
            return None
        repaired = f" ({self.failures} failed live writes repaired)" if self.failures else ""
        self.log(f"MongoDB: run {self.run_path.stem} stored{repaired}")
        return result


def main(argv=None):
    ap = argparse.ArgumentParser(description="Import eval run files into MongoDB")
    ap.add_argument("paths", nargs="+", help="run files or folders of *.jsonl")
    ap.add_argument("--labels", default=str(HUMAN_LABELS_PATH), help="human labels CSV (default: %(default)s)")
    ap.add_argument("--no-labels", action="store_true", help="don't import human labels")
    ap.add_argument("--dry-run", action="store_true", help="build the documents and report counts; don't connect")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    store = None
    if not args.dry_run:
        store, where = connect_or_exit()
        print(f"MongoDB: {where}")
        print(f"Indexes: {len(INDEXES)} ensured")
    labels_source = labels_run()
    result = import_runs(store, args.paths, None if args.no_labels else args.labels,
                         Path(labels_source).stem if labels_source else None, args.dry_run)
    print(("Dry run: nothing written\n" if args.dry_run else "") + table(
        ["run", "file", "records", "inserted", "updated", "unchanged", "refused", "fallbacks", "w/ rejections",
         "dataset = cases.jsonl"], result["runs"]))
    if result["labels"]:
        l = result["labels"]
        print(f"\nHuman labels: {l['rows']} rows for run {Path(labels_source).stem} "
              f"({l['inserted']} inserted, {l['updated']} updated, {l['unchanged']} unchanged)")
    for message in result["refused"]:
        print(f"REFUSED {message}")
    return 1 if result["refused"] else 0


if __name__ == "__main__":
    sys.exit(main())
