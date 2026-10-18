import io

import numpy as np
import pytest
from google.genai import types
from PIL import Image
from pypdf import PdfReader

from agent.agent import (
    MAX_STEPS, GeminiClient, MockLLMClient, check_grounding, fallback_report, generate_report,
)
from agent.schema import DISCLAIMER, Report
from agent.tools import CLASS_NAMES, ReportTools
from services.calibration import REFER_MESSAGE
from services.heatmap_handler import summarize_cam_regions
from services.model_handler import ModelService, StudentEfficientNet
from services.pdf_handler import PDFReportGenerator

CONFIDENT = ([2.0, 0.0, 0.0], 0.1318)  # Normal, calibrated ~1.0
REFER = ([0.0, 0.2, 0.6], 1.0)         # Tuberculosis, calibrated ~0.44

GRADCAM = {"explained_class": None, "peak_region": "right upper zone", "focus": "diffuse",
           "region_shares": {"right upper zone": 0.22}, "border_share": 0.3, "border_attention": False}


@pytest.fixture
def make_tools(make_service, calibration_file, monkeypatch):
    """ReportTools on a stub classifier; Grad-CAM is stubbed because the stub has no conv layers."""
    def make(case):
        logits, temperature = case
        service = make_service(logits, calibration_file(temperature))
        monkeypatch.setattr(ReportTools, "get_gradcam", lambda self: GRADCAM)
        return ReportTools(service, Image.new("L", (64, 64), 128))
    return make


def report_json(truth, **overrides):
    refer = truth["status"] == "refer"
    data = {
        "predicted_class": truth["predicted_class"],
        "calibrated_probability": truth["calibrated_probability"],
        "status": truth["status"],
        "summary": f"The model's most likely category is {truth['predicted_class']}. "
                   "This is a screening result, not a diagnosis.",
        "recommendation": (f"{REFER_MESSAGE.capitalize()}. A clinician review is needed before any decision."
                           if refer else "Share this result with a clinician at the next visit."),
        "limitations": ["Evaluated on two public datasets only."],
        "disclaimer": "Not medical advice.",
    }
    data.update(overrides)
    return data


def full_script(truth, *finals):
    return [MockLLMClient.tool("run_classifier"), MockLLMClient.tool("get_gradcam"),
            MockLLMClient.tool("get_condition_info", condition=truth["predicted_class"]),
            *[MockLLMClient.final(f) for f in finals]]


# Valid report passes

def test_valid_report_passes(make_tools):
    tools = make_tools(CONFIDENT)
    truth = tools.run_classifier()
    result = generate_report(tools, MockLLMClient(full_script(truth, report_json(truth))))
    assert result.source == "llm" and result.reason is None
    assert result.report.predicted_class == truth["predicted_class"] == "Normal"
    assert result.report.calibrated_probability == truth["calibrated_probability"]
    assert result.report.status == "confident"
    assert result.report.disclaimer == DISCLAIMER  # always the fixed text
    assert result.tool_calls == ["run_classifier", "get_gradcam", "get_condition_info"]
    assert result.steps == 4


def test_tool_results_are_sent_back_to_the_llm(make_tools):
    tools = make_tools(CONFIDENT)
    truth = tools.run_classifier()
    llm = MockLLMClient(full_script(truth, report_json(truth)))
    generate_report(tools, llm)
    call, output = llm.calls[1][-1]["results"][0]
    assert call.name == "run_classifier" and output == truth


# Class flip is rejected (the LLM can't change the prediction)

def test_class_flip_is_rejected(make_tools):
    tools = make_tools(CONFIDENT)
    truth = tools.run_classifier()
    flipped = report_json(truth, predicted_class="Tuberculosis")
    result = generate_report(tools, MockLLMClient(full_script(truth, flipped, flipped)))
    assert result.source == "fallback"
    assert "predicted_class" in result.reason and len(result.problems) == 2
    assert result.report.predicted_class == "Normal"


@pytest.mark.parametrize("override", [
    {"calibrated_probability": 0.95},  # not the classifier's value
    {"status": "refer"},
])
def test_probability_or_status_mismatch_is_rejected(make_tools, override):
    tools = make_tools(CONFIDENT)
    truth = tools.run_classifier()
    bad = report_json(truth, **override)
    result = generate_report(tools, MockLLMClient(full_script(truth, bad, bad)))
    assert result.source == "fallback"
    assert result.report.calibrated_probability == truth["calibrated_probability"]


@pytest.mark.parametrize("phrase", ["The diagnosis is confirmed.", "Diagnosis confirmed: pneumonia.",
                                    "This is a confirmed diagnosis."])
def test_diagnosis_confirmed_is_rejected(make_tools, phrase):
    tools = make_tools(CONFIDENT)
    truth = tools.run_classifier()
    bad = report_json(truth, recommendation=phrase)
    result = generate_report(tools, MockLLMClient(full_script(truth, bad, bad)))
    assert result.source == "fallback" and "confirmed" in result.reason


# Invalid JSON triggers the fallback (after one retry)

def test_invalid_json_triggers_fallback(make_tools):
    tools = make_tools(CONFIDENT)
    truth = tools.run_classifier()
    result = generate_report(tools, MockLLMClient(full_script(truth, "Sure! Here is the report.", "{not json")))
    assert result.source == "fallback" and "invalid JSON" in result.reason
    Report.model_validate(result.report.model_dump())


def test_one_retry_can_recover(make_tools):
    tools = make_tools(CONFIDENT)
    truth = tools.run_classifier()
    llm = MockLLMClient(full_script(truth, "{not json", report_json(truth)))
    result = generate_report(tools, llm)
    assert result.source == "llm" and len(result.problems) == 1
    assert "rejected" in llm.calls[-1][-1]["text"]  # the retry tells the LLM what was wrong


def test_schema_violation_triggers_fallback(make_tools):
    tools = make_tools(CONFIDENT)
    truth = tools.run_classifier()
    one_sentence = report_json(truth, summary="Normal.")
    extra_field = report_json(truth, diagnosis="Normal")
    result = generate_report(tools, MockLLMClient(full_script(truth, one_sentence, extra_field)))
    assert result.source == "fallback" and "schema" in result.reason


def test_step_limit_triggers_fallback(make_tools):
    tools = make_tools(CONFIDENT)
    llm = MockLLMClient([MockLLMClient.tool("get_gradcam")] * 10)
    result = generate_report(tools, llm)
    assert result.source == "fallback" and len(llm.calls) == MAX_STEPS


def test_llm_error_triggers_fallback(make_tools):
    def boom(messages):
        raise ConnectionError("network down")
    result = generate_report(make_tools(CONFIDENT), MockLLMClient([boom]))
    assert result.source == "fallback" and "network down" in result.reason


def test_no_llm_uses_template(make_tools):
    result = generate_report(make_tools(CONFIDENT), None)
    assert result.source == "fallback" and "No LLM configured" in result.reason


# Refer case includes clinician review

def test_refer_case_includes_clinician_review(make_tools):
    tools = make_tools(REFER)
    truth = tools.run_classifier()
    assert truth["status"] == "refer" and truth["predicted_class"] == "Tuberculosis"
    result = generate_report(tools, MockLLMClient(full_script(truth, report_json(truth))))
    assert result.source == "llm"
    assert "clinician review" in result.report.recommendation.lower()


def test_refer_report_without_clinician_review_is_rejected(make_tools):
    tools = make_tools(REFER)
    truth = tools.run_classifier()
    bad = report_json(truth, recommendation="Rest and drink fluids.")
    result = generate_report(tools, MockLLMClient(full_script(truth, bad, bad)))
    assert result.source == "fallback"
    assert result.report.recommendation.startswith(REFER_MESSAGE.capitalize())


@pytest.mark.parametrize("case", [CONFIDENT, REFER])
def test_fallback_report_is_valid_and_grounded(make_tools, case):
    tools = make_tools(case)
    report = fallback_report(tools)
    Report.model_validate(report.model_dump())
    assert check_grounding(report, tools.run_classifier()) == []


def test_fallback_survives_gradcam_failure(make_service, calibration_file):
    service = make_service(REFER[0], calibration_file(REFER[1]))  # stub has no conv layer -> Grad-CAM errors
    tools = ReportTools(service, Image.new("L", (64, 64), 128))
    assert "error" in tools.call("get_gradcam", {})
    Report.model_validate(fallback_report(tools).model_dump())


# PDF

def pdf_text(data):
    return " ".join(" ".join(page.extract_text() for page in PdfReader(io.BytesIO(data)).pages).split())


def make_pdf(tools, report):
    truth = tools.run_classifier()
    return PDFReportGenerator().generate_diagnosis_report(
        patient_name="P-0001", diagnosis=truth["predicted_class"],
        confidence=truth["calibrated_probability"] * 100, probabilities=truth["probabilities"],
        radiologist_email="dr@example.com", timestamp="2026-10-18 12:00:00",
        status=truth["status"], report=report, report_source="Template report (test).")


def test_pdf_for_refer_case_contains_referral(make_tools):
    tools = make_tools(REFER)
    text = pdf_text(make_pdf(tools, fallback_report(tools)))
    assert "REFERRAL: CLINICIAN REVIEW REQUIRED" in text
    assert REFER_MESSAGE.capitalize() in text
    assert "REFER - clinician review recommended" in text  # triage row
    assert "AI-Generated Report" in text


def test_pdf_for_confident_case_has_no_referral(make_tools):
    tools = make_tools(CONFIDENT)
    text = pdf_text(make_pdf(tools, None))
    assert "REFERRAL" not in text and "Confident" in text


def test_pdf_escapes_llm_text(make_tools):
    tools = make_tools(CONFIDENT)
    report = fallback_report(tools).model_copy(update={"summary": "Risk <high> & rising. Second sentence."})
    assert "Risk <high> & rising." in pdf_text(make_pdf(tools, report))


# Tools

def test_condition_info_covers_every_class(make_tools):
    tools = make_tools(CONFIDENT)
    for name in CLASS_NAMES:
        info = tools.get_condition_info(name)
        assert info["overview"] and info["typical_next_steps"] and info["review_status"]
    assert "error" in tools.call("get_condition_info", {"condition": "Cancer"})
    assert "error" in tools.call("search_web", {})


def test_gradcam_uses_last_conv_layer(monkeypatch, calibration_file):
    # Random-init EfficientNet (no download): the CAM must have spatial structure, unlike the old 1x1 avgpool target
    monkeypatch.setattr(ModelService, "_load_model", lambda self, path: StudentEfficientNet().eval())
    service = ModelService("unused.pth", calibration_path=calibration_file(1.0))
    image = Image.fromarray(np.random.default_rng(0).integers(0, 255, (256, 256), dtype=np.uint8))
    cam = service.compute_gradcam(image, "Pneumonia")
    # Random weights give tiny gradients, so only check shape, range and spatial variation
    assert cam.shape == (7, 7) and cam.min() >= 0 and cam.max() <= 1 and cam.std() > 0


def test_cam_region_summary():
    cam = np.zeros((7, 7))
    cam[1:3, 1:3] = 1.0  # image top-left = patient's right upper zone (PA view)
    summary = summarize_cam_regions(cam)
    assert summary["peak_region"] == "right upper zone" and summary["focus"] == "focal"
    edge = np.zeros((7, 7))
    edge[0, :] = 1.0
    assert summarize_cam_regions(edge)["border_attention"] is True


# Gemini wiring, offline

class FakeModels:
    def __init__(self, responses):
        self.responses, self.requests = list(responses), []

    def generate_content(self, **kwargs):
        self.requests.append(kwargs)
        return self.responses.pop(0)


def gemini_response(*parts):
    return types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(role="model", parts=list(parts)))])


def test_gemini_client_tool_loop_offline(make_tools):
    tools = make_tools(CONFIDENT)
    truth = tools.run_classifier()
    import json
    fake = FakeModels([
        gemini_response(types.Part(function_call=types.FunctionCall(id="c1", name="run_classifier", args={}),
                                   thought_signature=b"sig-1")),
        gemini_response(types.Part(text=json.dumps(report_json(truth)))),
    ])
    client = GeminiClient(api_key="unused", model="gemini-3.8-flash", client=type("C", (), {"models": fake})())
    result = generate_report(tools, client)

    assert result.source == "llm" and result.model == "gemini-3.8-flash"
    first = fake.requests[0]
    assert first["model"] == "gemini-3.8-flash"
    config = first["config"]
    assert [d.name for d in config.tools[0].function_declarations] == ["run_classifier", "get_gradcam", "get_condition_info"]
    assert config.automatic_function_calling.disable is True and config.temperature is None
    second = fake.requests[1]["contents"]
    assert second[1].parts[0].thought_signature == b"sig-1"  # model turn replayed as received
    response = second[2].parts[0].function_response
    assert response.id == "c1" and response.name == "run_classifier"
    assert response.response["result"] == truth
