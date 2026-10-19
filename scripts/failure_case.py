"""Render the Grad-CAM overlay for a misclassified test image (README figure).

    python scripts/failure_case.py

Default: the lowest-confidence test-split image, a pneumonia X-ray the model calls
Normal (calibrated 50.15%, referred). Writes docs/failure_case_gradcam.png.
"""
import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
logging.getLogger("streamlit").setLevel(logging.ERROR)
from calibrate import find_root  # noqa: E402
from services.heatmap_handler import BORDER_FRACTION, HeatmapGenerator, summarize_cam_regions  # noqa: E402
from services.model_handler import ModelService  # noqa: E402

DEFAULT_IMAGE = "pneumonia/train/PNEUMONIA/person427_virus_875.jpeg"  # manifest relpath; in the notebook's test split
ATTRIBUTION = ("Image: Kermany et al., Cell 2018 (Kaggle chest-xray-pneumonia, CC BY 4.0), "
               "from the training notebook's held-out test split.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--image", default=DEFAULT_IMAGE, help="manifest relpath (pneumonia/... or tb/...)")
    ap.add_argument("--true-label", default="Pneumonia")
    ap.add_argument("--out", default=REPO / "docs" / "failure_case_gradcam.png", type=Path)
    args = ap.parse_args()

    source, rest = args.image.split("/", 1)
    root = (find_root(REPO / "data" / "pneumonia", "chest_xray", "train") if source == "pneumonia"
            else find_root(REPO / "data" / "tb", "TB_Chest_Radiography_Database", "Tuberculosis"))
    image = Image.open(root / rest).convert("RGB")

    service = ModelService(str(REPO / "best_student_model_aligned_final.pth"))
    pred = service.predict(image)
    cls = service.class_index(pred["diagnosis"])
    summary = summarize_cam_regions(service.compute_gradcam(image, pred["diagnosis"]))

    # Same overlay as the app (HeatmapGenerator.generate_heatmap)
    original = np.array(image)
    generator = HeatmapGenerator(service.model, service.device)
    overlay = generator.generate_heatmap(service.transform(image).unsqueeze(0).to(service.device), cls, original)
    generator.cleanup()

    probs = ", ".join(f"{k} {v:.2%}" for k, v in pred["probabilities"].items())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))
    axes[0].imshow(original)
    axes[0].set_title(f"Input (true label: {args.true_label})")
    axes[1].imshow(overlay)
    axes[1].set_title(f"Grad-CAM for predicted class: {pred['diagnosis']}")
    h, w = original.shape[:2]
    bx, by = w * BORDER_FRACTION, h * BORDER_FRACTION
    axes[1].add_patch(Rectangle((bx, by), w - 2 * bx, h - 2 * by, fill=False, ls="--", lw=1.2, ec="white"))
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"Failure case: {args.true_label} X-ray classified as {pred['diagnosis']} "
                 f"(calibrated {pred['confidence']:.2%}) → triaged {pred['status'].upper()}",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, 0.06,
             f"Calibrated probabilities: {probs}. {summary['border_share']:.0%} of the Grad-CAM weight lies outside "
             f"the dashed frame (outer {BORDER_FRACTION:.0%} of the image); attention is {summary['focus']}, "
             f"peaking in the {summary['peak_region']}.", ha="center", fontsize=9, wrap=True)
    fig.text(0.5, 0.02, ATTRIBUTION, ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.1, 1, 0.93))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    plt.close(fig)
    print(f"{args.image}: predicted {pred['diagnosis']} {pred['confidence']:.4f} ({pred['status']}), "
          f"border share {summary['border_share']}, wrote {args.out}")


if __name__ == "__main__":
    main()
