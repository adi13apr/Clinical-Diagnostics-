"""Pydantic outputs of the MCP tools (the SDK publishes them as each tool's outputSchema)."""
from typing import Literal, Optional

from pydantic import BaseModel, Field

from agent.schema import ClassName, Report, Status

SCREENING_NOTE = ("Screening aid, not a diagnosis. Metrics are in-distribution only (each disease comes from a "
                  "different public dataset), so the model may partly recognise the source.")


class Classification(BaseModel):
    trace_id: str = Field(description="Trace of this prediction; gradcam_summary/generate_report on the same "
                                      "image add to it. Read it with get_trace.")
    predicted_class: ClassName
    calibrated_probability: float = Field(description="Temperature-calibrated probability of predicted_class, 0-1, 4 dp")
    probabilities: dict[str, float] = Field(description="Calibrated probability of every class, 0-1, 4 dp")
    status: Status = Field(
        description="'confident' if calibrated_probability >= referral_threshold, else 'refer' (clinician review)")
    referral_threshold: float
    referral_message: Optional[str] = Field(description="Set when status is 'refer'")
    calibrated: bool = Field(description="False if outputs/calibration.json is missing: then every result is 'refer'")
    temperature: float
    inference_ms: int
    note: str = SCREENING_NOTE


class GradcamSummary(BaseModel):
    trace_id: str
    explained_class: ClassName = Field(description="The predicted class, whose evidence the map shows")
    peak_region: Optional[str] = Field(description="Image zone with the largest share of influence, e.g. "
                                                   "'right lower zone' (patient side, PA view assumed); None if no "
                                                   "positive evidence")
    focus: Literal["focal", "diffuse", "none"] = Field(description="'focal' if one zone holds >= 40% of the influence")
    region_shares: dict[str, float] = Field(description="Share of influence per zone of a 3x2 grid; sums to ~1")
    border_share: Optional[float] = Field(description="Share of influence in the outer 10% frame of the image")
    border_attention: Optional[bool] = Field(description="True if more than half the influence is on the border "
                                                         "(edges, markers): treat the prediction with extra caution")
    cam: list[list[float]] = Field(description="Raw 7x7 Grad-CAM map (0-1, 4 dp), row 0 at the top. It covers the "
                                               "whole image stretched to a square")
    caveat: str


class ReportOut(BaseModel):
    trace_id: str
    source: Literal["llm", "template"] = Field(
        description="'llm': written by the LLM and passed the grounding guard. 'template': the fixed template "
                    "report (no LLM key, LLM error or rate limit, or the guard rejected the LLM twice)")
    written_by: str = Field(description="One line saying who wrote the report; show it with the report")
    model: Optional[str] = Field(description="LLM that wrote it, when source is 'llm'")
    fallback_reason: Optional[str] = Field(description="Why the template was used, when source is 'template'")
    cached: bool = Field(description="True if this LLM report was reused for the same image (no LLM calls spent)")
    guard_rejections: int = Field(description="LLM drafts the guard rejected before the final report")
    tool_calls: list[str] = Field(description="Tools the LLM called while writing it (the evidence agent's, in "
                                              "the multi pipeline)")
    pipeline: Literal["single", "multi"] = Field(
        description="The pipeline that actually ran: 'single' (one tool-calling agent) or 'multi' (evidence "
                    "agent, writer and verifier). Set by REPORT_PIPELINE; multi runs as single on a provider "
                    "without strict JSON output, and pipeline_note says so")
    pipeline_note: Optional[str] = Field(None, description="Why a requested multi pipeline ran as single")
    revisions: int = Field(0, description="Drafts sent back to the writer by the guard or the verifier (multi only)")
    report: Report


class SpanOut(BaseModel):
    span_id: str
    parent_span_id: Optional[str] = None
    name: str
    start: str
    latency_ms: Optional[float] = None
    status: str
    error_type: Optional[str] = None
    error: Optional[str] = None
    attributes: dict = Field(default_factory=dict)


class TraceOut(BaseModel):
    trace_id: str
    spans: list[SpanOut] = Field(description="Every step of this trace, oldest first")


class CheckRate(BaseModel):
    check: str
    passed: int
    applicable: int
    rate: Optional[float] = Field(description="passed / applicable, or None if no case applies")
    baseline_rate: Optional[float] = Field(description="Rate in evals/baseline.json; CI fails if rate drops below it")


class EvalSummary(BaseModel):
    run_id: str
    is_baseline_run: bool
    dataset_hash: str
    cases: int
    reports: int
    judged: int
    llm_written: int
    template_fallbacks: int
    deterministic_checks: list[CheckRate]
    below_baseline: list[str] = Field(description="Checks whose rate is below the baseline (empty = no drops)")
    judge_model: Optional[str]
    judge_mean_scores: dict[str, Optional[float]] = Field(description="Mean LLM-judge score (1-5) per criterion. "
                                                                      "The judge misses subtle problems; see README")
