"""Temperature-scale the student model and measure calibration + abstention.

Uses the training notebook's own split (splits/split_manifest.csv, exported on
Kaggle): the temperature is fitted on the val split and every metric is
reported on the test split only.

    python scripts/calibrate.py

Expects the two Kaggle datasets unzipped under data/ (see --pneumonia-root,
--tb-root). Writes outputs/calibration.json and outputs/reliability.png.
"""
import argparse
import csv
import hashlib
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Silence Streamlit's "no runtime" warnings from st.cache_resource / st.secrets
logging.getLogger("streamlit").setLevel(logging.ERROR)
from services.model_handler import ModelService  # noqa: E402

# Paper-reported student test metrics, used to decide how to label the results
PAPER_ACC = 97.61
PAPER_MACRO_F1 = 98.2
MATCH_TOLERANCE = 1.0  # percentage points

THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]
N_BINS = 15

CONFOUND_NOTE = (
    "Source confound: each class is tied to its source dataset. Pneumonia images come only "
    "from paultimothymooney/chest-xray-pneumonia (pediatric, JPEG), Tuberculosis only from "
    "tawsifurrahman/tuberculosis-tb-chest-xray-dataset (adult, PNG), and Normal mixes both. "
    "The model can partly separate Pneumonia from Tuberculosis by source (file format, "
    "resolution, patient age) rather than pathology, so these test-split numbers are likely "
    "optimistic for images from a new hospital or scanner."
)


def find_root(base, name, must_contain):
    """Shallowest directory called `name` under `base` that contains `must_contain` (skips __MACOSX)."""
    hits = [p for p in base.rglob(name)
            if p.is_dir() and "__MACOSX" not in p.parts and (p / must_contain).is_dir()]
    if not hits:
        sys.exit(f"Could not find a '{name}' folder containing '{must_contain}' under {base}")
    return min(hits, key=lambda p: len(p.parts))


def load_manifest(path, roots):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    for r in rows:
        source, rest = r["relpath"].split("/", 1)
        r["source"] = source
        r["file"] = roots[source] / rest
    return rows


def verify(rows, roots):
    """Every manifest file must exist, and the folders the notebook globbed must hold exactly the manifest's files."""
    missing = [r["relpath"] for r in rows if not r["file"].is_file()]
    if missing:
        sys.exit(f"{len(missing)} manifest files missing locally, e.g. {missing[:5]}")

    # Same folders and patterns as notebook cell 1
    globbed = []
    for split in ("train", "test", "val"):
        for cls in ("NORMAL", "PNEUMONIA"):
            globbed += (roots["pneumonia"] / split / cls).glob("*.jpeg")
    for cls in ("Normal", "Tuberculosis"):
        globbed += (roots["tb"] / cls).glob("*.png")
    local = {p.resolve() for p in globbed}
    listed = {r["file"].resolve() for r in rows}
    if local != listed:
        sys.exit(f"Local folders differ from the manifest: {len(local - listed)} extra, "
                 f"{len(listed - local)} not found by the notebook's globs")

    counts = Counter((r["split"], r["label"]) for r in rows)
    print(f"Verified: all {len(rows)} manifest files exist and match the notebook's folders exactly.")
    print(f"{'split':<6} {'normal':>7} {'pneumonia':>10} {'tuberculosis':>13} {'total':>6}")
    for split in ("train", "val", "test"):
        c = [counts[(split, l)] for l in ("normal", "pneumonia", "tuberculosis")]
        print(f"{split:<6} {c[0]:>7} {c[1]:>10} {c[2]:>13} {sum(c):>6}")


def load_rgb(path):
    # Same conversion as ModelService.predict / the notebook's LungDataset
    img = Image.open(path)
    return img if img.mode == "RGB" else img.convert("RGB")


@torch.no_grad()
def compute_logits(service, files, batch_size=32):
    out = []
    for i in range(0, len(files), batch_size):
        batch = torch.stack([service.transform(load_rgb(f)) for f in files[i:i + batch_size]])
        logits, _ = service.model(batch.to(service.device))
        out.append(logits.cpu().numpy())
        print(f"\r  {min(i + batch_size, len(files))}/{len(files)}", end="", flush=True)
    print()
    return np.concatenate(out).astype(np.float64)


def cached_logits(service, rows, cache_path, ckpt_hash):
    files = [str(r["file"]) for r in rows]
    if cache_path.exists():
        c = np.load(cache_path, allow_pickle=False)
        if str(c["ckpt"]) == ckpt_hash and c["files"].tolist() == files:
            print(f"Using cached logits from {cache_path}")
            return c["logits"]
    print(f"Running the model on {len(files)} val+test images ({service.device})...")
    logits = compute_logits(service, files)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, logits=logits, files=np.array(files), ckpt=ckpt_hash)
    return logits


def check_matches_app(service, rows, logits, n=5):
    """softmax(our logits) must equal what the app's predict() returns for the same images."""
    for r, z in zip(rows[:n], logits[:n]):
        app = service.predict(Image.open(r["file"]))["probabilities"]
        app = np.array([app[service.classes[i]] for i in range(len(service.classes))])
        if not np.allclose(softmax(z), app, atol=1e-4):
            sys.exit(f"Logits disagree with ModelService.predict on {r['relpath']}")
    print(f"Checked: probabilities match ModelService.predict() on {n} images.")


def softmax(z, T=1.0):
    z = np.asarray(z) / T
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def fit_temperature(logits, labels):
    """Single temperature minimising NLL on (logits, labels); optimised in log space so T > 0."""
    z = torch.tensor(logits, dtype=torch.float64)
    y = torch.tensor(labels)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=500, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(z / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp())


def nll(probs, labels):
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, None))))


def ece_bins(conf, correct, n_bins=N_BINS):
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)  # bins are (lo, hi]
    bins = []
    for b in range(n_bins):
        m = idx == b
        bins.append({"lo": edges[b], "hi": edges[b + 1], "n": int(m.sum()),
                     "acc": float(correct[m].mean()) if m.any() else None,
                     "conf": float(conf[m].mean()) if m.any() else None})
    ece = sum(b["n"] * abs(b["acc"] - b["conf"]) for b in bins if b["n"]) / len(conf)
    return float(ece), bins


def metrics(probs, labels, class_names):
    pred = probs.argmax(1)
    conf = probs.max(1)
    correct = pred == labels
    k = len(class_names)
    recall, f1 = {}, []
    for c in range(k):
        tp = np.sum((pred == c) & (labels == c))
        support, predicted = np.sum(labels == c), np.sum(pred == c)
        r = tp / support if support else 0.0
        p = tp / predicted if predicted else 0.0
        recall[class_names[c]] = float(r)
        f1.append(2 * p * r / (p + r) if p + r else 0.0)
    ece, bins = ece_bins(conf, correct)
    return {"accuracy": float(correct.mean()), "macro_f1": float(np.mean(f1)),
            "recall": recall, "ece": ece, "nll": nll(probs, labels), "_bins": bins}


def coverage_table(probs, labels):
    conf, correct = probs.max(1), probs.argmax(1) == labels
    table = []
    for t in THRESHOLDS:
        keep = conf >= t
        table.append({"threshold": t, "kept": int(keep.sum()), "coverage": float(keep.mean()),
                      "accuracy": float(correct[keep].mean()) if keep.any() else None})
    return table


def plot_reliability(before, after, temperature, label, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, m, title in [(axes[0], before, "Before (T = 1)"),
                         (axes[1], after, f"After temperature scaling (T = {temperature:.3f})")]:
        bins = [b for b in m["_bins"] if b["n"]]
        width = 1 / N_BINS
        centers = [(b["lo"] + b["hi"]) / 2 for b in bins]
        ax.bar(centers, [b["acc"] for b in bins], width=width * 0.95, color="#1f77b4",
               edgecolor="white", label="Accuracy in bin")
        ax.bar(centers, [b["conf"] for b in bins], width=width * 0.95, fill=False,
               edgecolor="#d62728", lw=1.5, label="Mean confidence in bin")
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Perfect calibration")
        for c, b in zip(centers, bins):
            ax.text(c, max(b["acc"], b["conf"]) + 0.01, str(b["n"]),
                    ha="center", va="bottom", fontsize=7, color="#444444")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.07)  # headroom for the counts above full-height bars
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_xlabel("Confidence (max probability)")
        ax.set_title(f"{title}\nECE = {m['ece'] * 100:.2f}%  ·  acc = {m['accuracy'] * 100:.2f}%")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle(f"Reliability diagram — {label}, test split ({N_BINS} bins; numbers = images per bin)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pct(x):
    return f"{'n/a':>8}" if x is None else f"{x * 100:7.2f}%"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--weights", default=REPO / "best_student_model_aligned_final.pth", type=Path)
    ap.add_argument("--manifest", default=REPO / "splits" / "split_manifest.csv", type=Path)
    ap.add_argument("--pneumonia-root", type=Path, help="folder containing train/ test/ val/ (auto-detected under data/pneumonia)")
    ap.add_argument("--tb-root", type=Path, help="folder containing Normal/ Tuberculosis/ (auto-detected under data/tb)")
    ap.add_argument("--out", default=REPO / "outputs", type=Path)
    args = ap.parse_args()

    roots = {
        "pneumonia": args.pneumonia_root or find_root(REPO / "data" / "pneumonia", "chest_xray", "train"),
        "tb": args.tb_root or find_root(REPO / "data" / "tb", "TB_Chest_Radiography_Database", "Tuberculosis"),
    }
    print(f"Pneumonia root: {roots['pneumonia']}\nTB root:        {roots['tb']}")

    rows = load_manifest(args.manifest, roots)
    verify(rows, roots)

    service = ModelService(str(args.weights))
    class_names = [service.classes[i] for i in range(len(service.classes))]
    name_to_idx = {n.lower(): i for i, n in enumerate(class_names)}  # notebook labels are lowercase

    eval_rows = [r for r in rows if r["split"] in ("val", "test")]
    ckpt_hash = sha256(args.weights)
    logits = cached_logits(service, eval_rows, REPO / "data" / "cache" / "calibration_logits.npz", ckpt_hash)
    check_matches_app(service, eval_rows, logits)

    labels = np.array([name_to_idx[r["label"]] for r in eval_rows])
    is_val = np.array([r["split"] == "val" for r in eval_rows])
    is_test = ~is_val

    T = fit_temperature(logits[is_val], labels[is_val])
    p_before = softmax(logits[is_test])
    p_after = softmax(logits[is_test], T)
    y = labels[is_test]
    before, after = metrics(p_before, y, class_names), metrics(p_after, y, class_names)

    gap_acc = before["accuracy"] * 100 - PAPER_ACC
    gap_f1 = before["macro_f1"] * 100 - PAPER_MACRO_F1
    matches = abs(gap_acc) <= MATCH_TOLERANCE and abs(gap_f1) <= MATCH_TOLERANCE
    label = "paper's held-out test set" if matches else "reconstructed split"

    # Normal is the only class drawn from both sources; break it down by source
    test_rows = [r for r, t in zip(eval_rows, is_test) if t]
    normal_by_source = {}
    for src in ("pneumonia", "tb"):
        m = np.array([r["label"] == "normal" and r["source"] == src for r in test_rows])
        if m.any():
            pred = p_before[m].argmax(1)
            normal_by_source[src] = {"n": int(m.sum()), "recall": float((pred == 0).mean()),
                                     "predicted": {class_names[c]: int((pred == c).sum()) for c in range(3)}}

    cov_before, cov_after = coverage_table(p_before, y), coverage_table(p_after, y)

    # ---- Report ----
    print(f"\n=== Results on the {label}: test split, n = {int(is_test.sum())} "
          f"(temperature fitted on val, n = {int(is_val.sum())}) ===")
    print(f"Temperature T = {T:.4f}")
    print(f"Paper: accuracy {PAPER_ACC:.2f}%, macro-F1 {PAPER_MACRO_F1:.2f}%  |  "
          f"here: accuracy {before['accuracy'] * 100:.2f}% ({gap_acc:+.2f} pt), "
          f"macro-F1 {before['macro_f1'] * 100:.2f}% ({gap_f1:+.2f} pt)  ->  "
          f"{'within' if matches else 'outside'} +/-{MATCH_TOLERANCE:g} pt")
    print(f"\n{'metric':<24}{'before':>9}{'after':>9}")
    for key, name in [("accuracy", "accuracy"), ("macro_f1", "macro-F1")]:
        print(f"{name:<24} {pct(before[key])} {pct(after[key])}")
    for c in class_names:
        print(f"{'recall ' + c:<24} {pct(before['recall'][c])} {pct(after['recall'][c])}")
    print(f"{'ECE (15 bins)':<24} {pct(before['ece'])} {pct(after['ece'])}")
    print(f"{'NLL':<24}{before['nll']:>9.4f}{after['nll']:>9.4f}")
    print("(Temperature scaling never changes the argmax, so accuracy, recall and F1 are identical.)")

    print("\nNormal test images by source (Normal is the only class drawn from both datasets):")
    for src, d in normal_by_source.items():
        print(f"  {src:<10} n={d['n']:<4} recall {d['recall'] * 100:6.2f}%   predicted as {d['predicted']}")

    print(f"\nCoverage vs accuracy (keep an image if max probability >= threshold)")
    print(f"{'threshold':<10}{'raw: kept':>11}{'coverage':>10}{'accuracy':>10}   "
          f"{'calibrated: kept':>17}{'coverage':>10}{'accuracy':>10}")
    for b, a in zip(cov_before, cov_after):
        print(f"{b['threshold']:<10}{b['kept']:>11}  {pct(b['coverage'])}  {pct(b['accuracy'])}   "
              f"{a['kept']:>17}  {pct(a['coverage'])}  {pct(a['accuracy'])}")

    print(f"\nNOTE: {CONFOUND_NOTE}")

    # ---- Save ----
    args.out.mkdir(parents=True, exist_ok=True)
    strip = lambda m: {k: v for k, v in m.items() if not k.startswith("_")}
    result = {
        "temperature": T,
        "label": label,
        "class_names": class_names,
        "fitted_on": {"split": "val", "n": int(is_val.sum())},
        "evaluated_on": {"split": "test", "n": int(is_test.sum()),
                         "per_class": {c: int((y == i).sum()) for i, c in enumerate(class_names)}},
        "paper_comparison": {"paper_accuracy": PAPER_ACC, "paper_macro_f1": PAPER_MACRO_F1,
                             "accuracy_gap_pt": round(gap_acc, 2), "macro_f1_gap_pt": round(gap_f1, 2),
                             "tolerance_pt": MATCH_TOLERANCE, "within_tolerance": matches},
        "test_metrics": {"before": strip(before), "after": strip(after), "ece_bins": N_BINS},
        "normal_by_source": normal_by_source,
        "coverage": {"raw": cov_before, "calibrated": cov_after},
        "caveats": [CONFOUND_NOTE],
        "provenance": {
            "weights": args.weights.name, "weights_sha256": ckpt_hash,
            "manifest": args.manifest.relative_to(REPO).as_posix(),
            "datasets": ["paultimothymooney/chest-xray-pneumonia",
                         "tawsifurrahman/tuberculosis-tb-chest-xray-dataset"],
            "preprocessing": "services.model_handler.ModelService.transform",
            "device": str(service.device), "torch": torch.__version__,
        },
    }
    with open(args.out / "calibration.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    plot_reliability(before, after, T, label, args.out / "reliability.png")
    print(f"\nWrote {args.out / 'calibration.json'} and {args.out / 'reliability.png'}")


if __name__ == "__main__":
    main()
