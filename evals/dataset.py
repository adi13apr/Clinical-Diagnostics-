"""Build and load evals/cases.jsonl.

    python -m evals.dataset          # (re)build from the cached test-split logits

Selection (test split of splits/split_manifest.csv only): every model error, every
'refer' case, and a stratified sample of confident correct cases (2 per class;
Normal takes one from each source dataset). Cases with the "regression" category were added
by hand (.claude/skills/failure-to-regression/add_case.py) and are carried over on a rebuild.
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

from evals.schema import CASE_SCHEMA_VERSION, EvalCase, SchemaVersionError

REPO = Path(__file__).resolve().parent.parent
CASES_PATH = REPO / "evals" / "cases.jsonl"
LOGITS_CACHE = REPO / "data" / "cache" / "calibration_logits.npz"
CLASS_NAMES = ["Normal", "Pneumonia", "Tuberculosis"]
SEED = 42


def load_cases(path=CASES_PATH):
    """Validated cases. Raises SchemaVersionError on any version mismatch and ValueError on bad rows."""
    cases = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            data = json.loads(line)
            version = data.get("schema_version")
            if version != CASE_SCHEMA_VERSION:
                raise SchemaVersionError(
                    f"{path}:{n} has schema_version={version!r}, but this code expects {CASE_SCHEMA_VERSION}. "
                    "Rebuild the dataset (python -m evals.dataset) or update evals/schema.py.")
            cases.append(EvalCase.model_validate(data))
    ids = [c.case_id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate case_id in {path}")
    return cases


def dataset_hash(cases):
    """SHA-256 over canonical JSON of the cases (independent of file line endings)."""
    h = hashlib.sha256()
    for case in cases:
        h.update(json.dumps(case.model_dump(), sort_keys=True, separators=(",", ":")).encode())
        h.update(b"\n")
    return h.hexdigest()


def build_cases(keep=()):
    """Cases selected from the cached logits, then the `keep` cases (hand-added regressions) renumbered after them."""
    import numpy as np
    sys.path.insert(0, str(REPO))
    from services.calibration import apply_temperature, load_temperature, triage

    cache = np.load(LOGITS_CACHE, allow_pickle=False)
    test = cache["splits"] == "test"
    files, labels = cache["files"][test], cache["labels"][test]
    probs = apply_temperature(cache["logits"][test], load_temperature()).numpy()
    manifest_paths = {}
    for row in __import__("csv").DictReader(open(REPO / "splits" / "split_manifest.csv", encoding="utf-8")):
        source, rest = row["relpath"].split("/", 1)
        manifest_paths[(source, Path(rest).as_posix())] = row["relpath"]

    def relpath(file):
        p = Path(str(file)).as_posix()
        for source, marker in (("pneumonia", "/chest_xray/"), ("tb", "/TB_Chest_Radiography_Database/")):
            if marker in p:
                return manifest_paths[(source, p.split(marker, 1)[1])]
        raise ValueError(f"Not a dataset path: {file}")

    rows = []
    for f, label, p in zip(files, labels, probs):
        true = CLASS_NAMES[["normal", "pneumonia", "tuberculosis"].index(str(label))]
        pred, conf = CLASS_NAMES[int(p.argmax())], round(float(p.max()), 4)
        rel = relpath(f)
        cats = (["error"] if pred != true else []) + (["refer"] if triage(float(p.max())) == "refer" else [])
        rows.append({"relpath": rel, "source": rel.split("/", 1)[0], "true_label": true, "categories": cats,
                     "snapshot": {"predicted_class": pred, "calibrated_probability": conf,
                                  "status": triage(float(p.max()))}})

    selected = [r for r in rows if r["categories"]]
    rng = random.Random(SEED)
    correct = sorted((r for r in rows if not r["categories"]), key=lambda r: r["relpath"])
    strata = [("Normal", "pneumonia", 1), ("Normal", "tb", 1), ("Pneumonia", "pneumonia", 2),
              ("Tuberculosis", "tb", 2)]
    for label, source, k in strata:
        pool = [r for r in correct if r["true_label"] == label and r["source"] == source]
        for r in rng.sample(pool, k):
            selected.append(dict(r, categories=["correct"]))

    order = {"error": 0, "refer": 1, "correct": 2}
    selected.sort(key=lambda r: (order[r["categories"][0]], r["relpath"]))
    cases = []
    for i, r in enumerate(selected, 1):
        stem = Path(r["relpath"]).stem
        cases.append(EvalCase(schema_version=CASE_SCHEMA_VERSION, case_id=f"c{i:02d}-{stem}", split="test", **r))
    built = {c.relpath for c in cases}
    for extra in keep:
        if extra.relpath not in built:
            cases.append(extra.model_copy(update={"case_id": next_case_id(cases, extra.relpath)}))
    return cases


def next_case_id(cases, relpath):
    """c<N+1>-<file stem>, after the highest existing number."""
    n = max((int(c.case_id[1:].split("-", 1)[0]) for c in cases), default=0) + 1
    return f"c{n:02d}-{Path(relpath).stem}"


def write_cases(cases, path=CASES_PATH):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for case in cases:
            f.write(json.dumps(case.model_dump(), sort_keys=True) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Build evals/cases.jsonl")
    ap.add_argument("--force", action="store_true", help="overwrite an existing cases.jsonl")
    args = ap.parse_args()
    if CASES_PATH.exists() and not args.force:
        sys.exit(f"{CASES_PATH} exists; pass --force to rebuild (this changes the dataset hash).")
    keep = [c for c in load_cases() if "regression" in c.categories] if CASES_PATH.exists() else []
    cases = build_cases(keep)
    write_cases(cases)
    counts = {k: sum(k in c.categories for c in cases) for k in ("error", "refer", "correct", "regression")}
    print(f"Wrote {len(cases)} cases to {CASES_PATH} {counts}, hash {dataset_hash(cases)[:12]}")


if __name__ == "__main__":
    main()
