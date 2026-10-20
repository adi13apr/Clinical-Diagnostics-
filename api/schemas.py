"""Request and response models (Pydantic). Probabilities and confidences are 0-1."""
from datetime import date
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent.schema import ClassName, Report, Status


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    calibrated: bool
    temperature: Optional[float]
    auth_configured: bool
    llm_configured: bool = Field(description="False: reports use the fixed template")


class PredictResponse(BaseModel):
    trace_id: str = Field(description="Use it for /gradcam, /report and /traces")
    predicted_class: ClassName
    confidence: float = Field(description="Calibrated probability of the predicted class")
    probabilities: Dict[str, float] = Field(description="Calibrated probability per class")
    status: Status = Field(description="'refer' when confidence is below referral_threshold")
    referral_threshold: float
    referral_message: Optional[str]
    temperature: float
    calibrated: bool = Field(description="False: calibration file missing, so every result is 'refer'")
    inference_ms: int
    warnings: List[str] = Field(default_factory=list, description="Non-fatal problems, e.g. logging failed")


class GradcamMapResponse(BaseModel):
    trace_id: str
    explained_class: ClassName = Field(description="The predicted class; the map shows evidence for it")
    cam: List[List[float]] = Field(description="Grad-CAM map, rows top to bottom, 0-1. Stretch it over the "
                                               "whole image: the model saw the image resized to a square")
    regions: dict = Field(description="services.heatmap_handler.summarize_cam_regions: zone shares, peak "
                                      "zone, focal/diffuse, border attention (a 3x2 grid, not a lung segmentation)")


class ReportResponse(BaseModel):
    trace_id: str
    report: Report
    source: str = Field(description="'llm' or 'fallback' (fixed template)")
    model: Optional[str]
    source_line: str = Field(description="Who wrote the report, for display")
    reason: Optional[str] = Field(description="Why the template was used (user-facing)")
    cached: bool
    warnings: List[str] = Field(default_factory=list)


class HistoryQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    limit: int = Field(20, ge=1, le=100)
    offset: int = Field(0, ge=0)
    label: Optional[ClassName] = None
    status: Optional[Status] = None
    date_from: Optional[date] = Field(None, description="UTC day, inclusive")
    date_to: Optional[date] = Field(None, description="UTC day, inclusive")

    @model_validator(mode="after")
    def dates_in_order(self):
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from must not be after date_to")
        return self


class HistoryItem(BaseModel):
    id: int
    created_at: str
    predicted_label: str
    confidence: float = Field(description="Calibrated if temperature is set, else raw")
    status: Optional[Status] = Field(description="None for rows logged before triage existed")
    temperature: Optional[float]
    inference_ms: int
    trace_id: Optional[str]


class HistoryPage(BaseModel):
    items: List[HistoryItem]
    total: int
    limit: int
    offset: int


class DailyVolume(BaseModel):
    day: str
    predictions: int
    referred: int
    referral_rate: float
    by_label: Dict[str, int]
    avg_inference_ms: Optional[float]


class StepLatency(BaseModel):
    count: int
    avg_latency_ms: float
    error_rate: float


class PipelineSummary(BaseModel):
    """services.tracing.summarize_spans over the user's most recent spans."""
    traces: int
    spans: int
    steps: Dict[str, StepLatency]
    reports_with_llm: int
    avg_input_tokens_per_report: Optional[float]
    avg_output_tokens_per_report: Optional[float]
    avg_nominal_cost_per_report_usd: Optional[float]
    guard_checks: int
    guard_reject_rate: Optional[float]
    reports: int
    fallback_rate: Optional[float]


class AnalyticsResponse(BaseModel):
    days: int
    total_predictions: int
    total_referred: int
    referral_rate: Optional[float]
    daily: List[DailyVolume] = Field(description="Newest day first")
    pipeline: Optional[PipelineSummary]
    warnings: List[str] = Field(default_factory=list)


class SpanOut(BaseModel):
    span_id: str
    parent_span_id: Optional[str]
    name: str
    start: Optional[str]
    latency_ms: Optional[float]
    status: str
    error_type: Optional[str]
    error: Optional[str]
    attributes: dict


class TraceResponse(BaseModel):
    trace_id: str
    spans: List[SpanOut] = Field(description="Oldest first; parent_span_id gives the nesting")
