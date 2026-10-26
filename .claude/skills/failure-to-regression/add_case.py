"""Add the image behind a failure to evals/cases.jsonl (category "regression").

    python add_case.py <relpath or path under data/> [--dry-run]

- Already a case: says which one and changes nothing.
- Not in the test split of splits/split_manifest.csv: refused (train/val images would leak into the eval).
- Otherwise: classifies it for the snapshot, appends it and prints the new dataset hash. Replay then
  refuses the baseline run (hash mismatch) until a new live run and --update-baseline.
"""
import argparse
import csv
import sys
from pathlib import Path

from _repo import REPO, rel

MANIFEST = REPO / "splits" / "split_manifest.csv"
MARKERS = (("pneumonia", "/chest_xray/"), ("tb", "/TB_Chest_Radiography_Database/"))


def load_manifest(path=MANIFEST):
    with open(path, encoding="utf-8") as f:
        return {row["relpath"]: row for row in csv.DictReader(f)}


def to_relpath(arg, manifest):
    """A manifest relpath from a relpath or a file path inside the downloaded datasets."""
    p = Path(arg).as_posix()
    if p in manifest:
        return p
    for source, marker in MARKERS:
        if marker in "/" + p:
            rest = ("/" + p).split(marker, 1)[1]
            for relpath in manifest:
                if relpath.startswith(source + "/") and relpath.split("/", 1)[1] == rest:
                    return relpath
    raise SystemExit(f"{arg} is not in {rel(MANIFEST)}. Only images from the two datasets can be eval cases.")


def classify(relpath, service, data_dir):
    """ClassifierSnapshot fields for one image, through ModelService (the same path as the live eval)."""
    from PIL import Image

    sys.path.insert(0, str(REPO / "scripts"))
    from calibrate import find_root
    roots = {"pneumonia": find_root(data_dir / "pneumonia", "chest_xray", "train"),
             "tb": find_root(data_dir / "tb", "TB_Chest_Radiography_Database", "Tuberculosis")}
    source, rest = relpath.split("/", 1)
    p = service.predict(Image.open(roots[source] / rest))
    return {"predicted_class": p["diagnosis"], "calibrated_probability": round(p["confidence"], 4),
            "status": p["status"]}


def new_case(relpath, row, snapshot, cases):
    from evals.dataset import next_case_id
    from evals.schema import CASE_SCHEMA_VERSION, EvalCase
    true = {"normal": "Normal", "pneumonia": "Pneumonia", "tuberculosis": "Tuberculosis"}[row["label"].lower()]
    categories = (["error"] if snapshot["predicted_class"] != true else []) + \
                 (["refer"] if snapshot["status"] == "refer" else []) + ["regression"]
    return EvalCase(schema_version=CASE_SCHEMA_VERSION, case_id=next_case_id(cases, relpath), relpath=relpath,
                    source=relpath.split("/", 1)[0], split="test", true_label=true, categories=categories,
                    snapshot=snapshot)


def add_case(arg, service_factory, cases_path=None, manifest_path=MANIFEST, data_dir=REPO / "data", dry_run=False):
    """(case, added) where added is False if the image was already a case."""
    from evals.dataset import CASES_PATH, load_cases, write_cases
    cases_path = cases_path or CASES_PATH
    manifest = load_manifest(manifest_path)
    relpath = to_relpath(arg, manifest)
    cases = load_cases(cases_path)
    existing = next((c for c in cases if c.relpath == relpath), None)
    if existing:
        return existing, False
    row = manifest[relpath]
    if row["split"] != "test":
        raise SystemExit(f"{relpath} is in the {row['split']!r} split. Only test-split images can be eval cases "
                         "(the model was trained or calibrated on the others).")
    case = new_case(relpath, row, classify(relpath, service_factory(), data_dir), cases)
    if not dry_run:
        write_cases(cases + [case], cases_path)
    return case, True


def real_service():
    import logging
    logging.getLogger("streamlit").setLevel(logging.ERROR)
    from services.model_handler import ModelService
    return ModelService(str(REPO / "best_student_model_aligned_final.pth"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("image", help="manifest relpath (pneumonia/test/...) or a path inside data/")
    ap.add_argument("--dry-run", action="store_true", help="show the case without writing cases.jsonl")
    args = ap.parse_args(argv)
    from evals.dataset import dataset_hash, load_cases
    case, added = add_case(args.image, real_service, dry_run=args.dry_run)
    if not added:
        print(f"Already in the dataset as {case.case_id}; evals/cases.jsonl unchanged "
              f"(hash {dataset_hash(load_cases())[:12]}). Go on with the check and the guard rule.")
        return 0
    print(f"{'Would add' if args.dry_run else 'Added'} {case.case_id}: {case.model_dump_json()}")
    if not args.dry_run:
        print(f"New dataset hash {dataset_hash(load_cases())[:12]}. Replay now refuses the baseline run: run "
              "python -m evals.run --live (live_eval.py), then --replay --update-baseline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
