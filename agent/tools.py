"""Tools the report agent can call. The image is bound here; the LLM never sees pixels or patient details."""
import json
from pathlib import Path

from services.calibration import REFER_MESSAGE, REFER_THRESHOLD
from services.heatmap_handler import summarize_cam_regions

CONDITION_INFO_PATH = Path(__file__).resolve().parent / "condition_info.json"
CLASS_NAMES = ["Normal", "Pneumonia", "Tuberculosis"]
DECIMALS = 4  # probabilities are rounded so the report can copy them exactly

# Provider-neutral declarations (JSON Schema parameters)
TOOL_DECLARATIONS = [
    {
        "name": "run_classifier",
        "description": (
            "Run the chest X-ray classifier on the uploaded image. Returns the predicted class, the "
            "calibrated probability of that class, all class probabilities, and the triage status "
            f"('confident' if the calibrated probability is at least {REFER_THRESHOLD}, otherwise 'refer'). "
            "This output is authoritative and must be copied exactly into the report."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "get_gradcam",
        "description": (
            "Grad-CAM summary for the predicted class: which approximate image zone (3x2 grid, patient's "
            "right/left assuming a PA view) most influenced the classifier, the share of influence per "
            "zone, whether attention is focal or diffuse, and whether it sits on the image border."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "get_condition_info",
        "description": (
            "Vetted background text about a condition (overview, typical X-ray findings, what an AI "
            "result means, typical next steps). Use only this text for medical background."
        ),
        "parameters": {
            "type": "object",
            "properties": {"condition": {"type": "string", "enum": CLASS_NAMES}},
            "required": ["condition"],
        },
    },
]


class ReportTools:
    def __init__(self, model_service, image, prediction=None, condition_info_path=CONDITION_INFO_PATH):
        self.model_service = model_service
        self.image = image
        self._prediction = prediction  # reuse the dashboard's prediction instead of re-running the model
        self._classifier = None
        self._gradcam = None
        with open(condition_info_path, encoding="utf-8") as f:
            self.condition_info = json.load(f)

    def run_classifier(self):
        if self._classifier is None:
            p = self._prediction or self.model_service.predict(self.image)
            self._classifier = {
                "predicted_class": p["diagnosis"],
                "calibrated_probability": round(p["confidence"], DECIMALS),
                "probabilities": {k: round(v, DECIMALS) for k, v in p["probabilities"].items()},
                "status": p["status"],
                "referral_threshold": REFER_THRESHOLD,
                "referral_message": REFER_MESSAGE if p["status"] == "refer" else None,
            }
        return self._classifier

    def get_gradcam(self):
        if self._gradcam is None:
            predicted = self.run_classifier()["predicted_class"]
            cam = self.model_service.compute_gradcam(self.image, predicted)
            self._gradcam = {
                "explained_class": predicted,
                **summarize_cam_regions(cam),
                "caveat": ("Zones are a coarse 3x2 grid over the whole image, not a lung segmentation, and "
                           "assume a standard PA view. Grad-CAM shows what influenced the model, which "
                           "may not be clinically relevant anatomy."),
            }
        return self._gradcam

    def get_condition_info(self, condition):
        if condition not in CLASS_NAMES:
            raise ValueError(f"Unknown condition {condition!r}; expected one of {CLASS_NAMES}")
        return {"condition": condition, **self.condition_info[condition],
                "review_status": self.condition_info["_meta"]["review_status"]}

    def call(self, name, args):
        """Run a tool by name. Errors are returned to the LLM as {'error': ...} instead of raised."""
        handlers = {"run_classifier": self.run_classifier, "get_gradcam": self.get_gradcam,
                    "get_condition_info": self.get_condition_info}
        if name not in handlers:
            return {"error": f"Unknown tool {name!r}"}
        try:
            return handlers[name](**(args or {}))
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}
