import json
import time
from datetime import date

import pytest
from google.genai import types
from PIL import Image

from agent.agent import GeminiClient, LLMTurn, MockLLMClient, ReportCache, generate_report, generate_report_for_image
from agent.tools import ReportTools
from services.prediction_handler import PredictionService
from services.tracing import (
    JsonlTraceSink, Tracer, estimate_cost, finish_trace, make_tracer, summarize_spans, trace_event, trace_span,
)

CONFIDENT = ([2.0, 0.0, 0.0], 0.1318)
USAGE = {"input_tokens": 1200, "output_tokens": 300, "thinking_tokens": 100}


def names(tracer):
    return [s.name for s in tracer.spans]


def by_name(tracer, name):
    return [s for s in tracer.spans if s.name == name]


# Spans nest and record latency

def test_spans_nest_and_record_latency():
    tracer = Tracer()
    with tracer.span("outer", kind="test") as outer:
        with tracer.span("inner") as inner:
            time.sleep(0.02)
    assert inner.parent_span_id == outer.span_id and outer.parent_span_id is None
    assert inner.trace_id == outer.trace_id == tracer.trace_id
    assert inner.latency_ms >= 15 and outer.latency_ms >= inner.latency_ms
    assert outer.status == inner.status == "ok" and outer.attributes == {"kind": "test"}


def test_events_are_zero_duration_children():
    tracer = Tracer()
    with tracer.span("report") as report:
        guard = tracer.event("guard", status="reject", error="bad class", result="reject")
    assert guard.parent_span_id == report.span_id and guard.latency_ms == 0.0
    assert guard.status == "error" and guard.error_type == "Rejected" and guard.error == "bad class"


# A failing step marks its span as an error

def test_failing_step_marks_span_as_error_and_still_raises():
    tracer = Tracer()
    with pytest.raises(ValueError, match="boom"):
        with trace_span(tracer, "outer"):
            with trace_span(tracer, "step"):
                raise ValueError("boom")
    step, outer = by_name(tracer, "step")[0], by_name(tracer, "outer")[0]
    assert step.status == "error" and step.error_type == "ValueError" and step.error == "boom"
    assert step.latency_ms is not None and outer.status == "error"  # error propagates to the parent


def test_failing_classifier_marks_classify_span(make_service, calibration_file):
    service = make_service(CONFIDENT[0], calibration_file(1.0))
    service.model = lambda x: (_ for _ in ()).throw(RuntimeError("CUDA out of memory"))
    tracer = Tracer()
    with pytest.raises(RuntimeError):
        service.predict(Image.new("L", (64, 64), 128), tracer=tracer)
    assert names(tracer) == ["preprocess", "classify"]
    assert by_name(tracer, "preprocess")[0].status == "ok"
    assert by_name(tracer, "classify")[0].error_type == "RuntimeError"


def test_tool_error_marks_tool_and_gradcam_spans(make_service, calibration_file):
    service = make_service(CONFIDENT[0], calibration_file(CONFIDENT[1]))  # stub has no conv layer
    tracer = Tracer()
    tools = ReportTools(service, Image.new("L", (64, 64), 128), tracer=tracer)
    with trace_span(tracer, "tool_call", tool="get_gradcam") as span:
        output = tools.call("get_gradcam", {})
        span.fail(output["error"], "ToolError")
    gradcam = by_name(tracer, "gradcam")[0]
    assert gradcam.status == "error" and gradcam.parent_span_id == by_name(tracer, "tool_call")[0].span_id


# A tracer that throws doesn't break predict()

class BrokenTracer:
    trace_id = "broken"

    def span(self, name, **attributes):
        raise RuntimeError("tracing backend down")

    def event(self, name, **kwargs):
        raise RuntimeError("tracing backend down")

    def finish(self):
        raise RuntimeError("tracing backend down")


def test_broken_tracer_does_not_break_predict(make_service, calibration_file):
    service = make_service(CONFIDENT[0], calibration_file(CONFIDENT[1]))
    image = Image.new("L", (64, 64), 128)
    assert service.predict(image, tracer=BrokenTracer()) == service.predict(image)
    trace_event(BrokenTracer(), "guard")  # helpers swallow tracer errors
    finish_trace(BrokenTracer())


class ExplodingExitTracer(Tracer):
    """Opens spans fine but fails when closing them."""

    def span(self, name, **attributes):
        class CM:
            def __enter__(cm):
                return super(ExplodingExitTracer, self).span  # any object works as a span here

            def __exit__(cm, *exc):
                raise RuntimeError("cannot close span")
        return CM()


def test_tracer_failing_on_close_does_not_break_or_mask_errors(make_service, calibration_file):
    service = make_service(CONFIDENT[0], calibration_file(CONFIDENT[1]))
    assert service.predict(Image.new("L", (64, 64), 128), tracer=ExplodingExitTracer())["status"] == "confident"
    with pytest.raises(KeyError):  # the step's own error still wins
        with trace_span(ExplodingExitTracer(), "step"):
            raise KeyError("real error")


def test_broken_tracer_does_not_break_report(make_service, calibration_file, monkeypatch):
    monkeypatch.setattr(ReportTools, "get_gradcam", lambda self: {"peak_region": None})
    service = make_service(CONFIDENT[0], calibration_file(CONFIDENT[1]))
    tools = ReportTools(service, Image.new("L", (64, 64), 128), tracer=BrokenTracer())
    truth = tools.run_classifier()
    report = {"predicted_class": truth["predicted_class"], "calibrated_probability": truth["calibrated_probability"],
              "status": truth["status"], "summary": "One sentence. Two sentences.", "recommendation": "See a clinician.",
              "limitations": ["x"], "disclaimer": "y"}
    result = generate_report(tools, MockLLMClient([MockLLMClient.final(report)]), tracer=BrokenTracer())
    assert result.source == "llm"


# predict() spans

def test_predict_records_pipeline_spans(make_service, calibration_file):
    service = make_service(CONFIDENT[0], calibration_file(CONFIDENT[1]))
    tracer = Tracer()
    with trace_span(tracer, "prediction"):
        service.predict(Image.new("L", (64, 64), 128), tracer=tracer)
    assert names(tracer) == ["prediction", "preprocess", "classify", "calibrate"]
    root = tracer.spans[0]
    assert all(s.parent_span_id == root.span_id for s in tracer.spans[1:])
    calibrate = by_name(tracer, "calibrate")[0].attributes
    assert calibrate["temperature"] == CONFIDENT[1] and calibrate["status"] == "confident"
    assert by_name(tracer, "preprocess")[0].attributes == {"input_mode": "L", "input_size": [64, 64]}


# Sinks

def test_finish_writes_jsonl_once(tmp_path):
    tracer = Tracer(sinks=[JsonlTraceSink(tmp_path)])
    with tracer.span("a"):
        pass
    tracer.finish()
    tracer.finish()  # nothing new: no duplicate lines
    with tracer.span("b"):
        pass
    tracer.finish()
    rows = JsonlTraceSink(tmp_path).read()
    assert sorted(r["name"] for r in rows) == ["a", "b"]
    assert all(r["trace_id"] == tracer.trace_id for r in rows)


class FailingStore:
    def write(self, spans):
        raise ConnectionError("supabase unreachable")


def test_failing_sink_is_recorded_not_raised(tmp_path):
    tracer = make_tracer(FailingStore(), trace_dir=tmp_path)
    with tracer.span("classify"):
        pass
    tracer.finish()
    assert len(JsonlTraceSink(tmp_path).read()) == 1  # local copy still written
    assert tracer.sink_errors and "supabase unreachable" in tracer.sink_errors[0]


def test_prediction_log_carries_trace_id():
    inserted = {}

    class Table:
        def insert(self, data):
            inserted.update(data)
            return self

        def execute(self):
            return None

    service = PredictionService(type("Client", (), {"table": lambda self, name: Table()})())
    result = {"diagnosis": "Normal", "confidence": 0.99, "status": "confident", "temperature": 0.13}
    service.log_prediction(result, 42, trace_id="1234-abcd")
    assert inserted["trace_id"] == "1234-abcd" and inserted["inference_ms"] == 42


# Agent spans

@pytest.fixture
def tools_and_truth(make_service, calibration_file, monkeypatch):
    monkeypatch.setattr(ReportTools, "get_gradcam", lambda self: {"peak_region": "right upper zone"})
    def make(tracer):
        service = make_service(CONFIDENT[0], calibration_file(CONFIDENT[1]))
        tools = ReportTools(service, Image.new("L", (64, 64), 128), tracer=tracer)
        return tools, tools.run_classifier()
    return make


def valid_report(truth, **overrides):
    data = {"predicted_class": truth["predicted_class"], "calibrated_probability": truth["calibrated_probability"],
            "status": truth["status"], "summary": "The model suggests Normal. This is a screening result.",
            "recommendation": "Share with a clinician.", "limitations": ["Two public datasets only."],
            "disclaimer": "Not advice."}
    data.update(overrides)
    return data


def test_report_spans_cover_llm_tools_guard_and_fallback(tools_and_truth):
    tracer = Tracer()
    tools, truth = tools_and_truth(tracer)
    all_tools = LLMTurn(tool_calls=[MockLLMClient.tool(n).tool_calls[0]
                                    for n in ("run_classifier", "get_gradcam", "get_condition_info")],
                        usage={"input_tokens": 900, "output_tokens": 20, "thinking_tokens": 50}, finish_reason="STOP")
    llm = MockLLMClient([all_tools, MockLLMClient.final(valid_report(truth), usage=USAGE)])
    llm.name = "gemini-3.8-flash"  # a priced model, so the span gets a nominal cost
    result = generate_report(tools, llm, tracer=tracer)

    assert result.source == "llm"
    # No precomputed prediction, so the fixture's run_classifier ran (and traced) the classifier first
    assert names(tracer) == ["preprocess", "classify", "calibrate", "report", "llm_call", "tool_call",
                             "tool_call", "tool_call", "llm_call", "guard", "fallback_used"]
    report = by_name(tracer, "report")[0]
    assert all(s.parent_span_id == report.span_id for s in tracer.spans[4:])
    second = by_name(tracer, "llm_call")[1].attributes
    assert second["input_tokens"] == 1200 and second["thinking_tokens"] == 100 and second["finish_reason"] == "STOP"
    assert second["nominal_cost_usd"] > 0 and second["billed_cost_usd"] == 0.0  # free tier
    assert [s.attributes["tool"] for s in by_name(tracer, "tool_call")] == ["run_classifier", "get_gradcam",
                                                                           "get_condition_info"]
    assert by_name(tracer, "guard")[0].attributes["result"] == "pass"
    assert by_name(tracer, "fallback_used")[0].attributes["used"] is False
    assert report.attributes["source"] == "llm"


def test_rejected_report_records_guard_reason_and_fallback(tools_and_truth):
    tracer = Tracer()
    tools, truth = tools_and_truth(tracer)
    flipped = valid_report(truth, predicted_class="Tuberculosis")
    generate_report(tools, MockLLMClient([MockLLMClient.final(flipped)] * 2), tracer=tracer)
    guards = by_name(tracer, "guard")
    assert [g.attributes["result"] for g in guards] == ["reject", "reject"]
    assert all("predicted_class" in g.attributes["reason"] and g.status == "error" for g in guards)
    fallback = by_name(tracer, "fallback_used")[0].attributes
    assert fallback["used"] is True and "rejected" in fallback["reason"]


def test_llm_error_records_error_type(tools_and_truth):
    tracer = Tracer()
    tools, _ = tools_and_truth(tracer)

    def boom(messages):
        raise TimeoutError("deadline exceeded")

    generate_report(tools, MockLLMClient([boom]), tracer=tracer)
    call = by_name(tracer, "llm_call")[0]
    assert call.status == "error" and call.error_type == "TimeoutError"
    assert by_name(tracer, "fallback_used")[0].attributes["used"] is True


def test_cache_hit_is_traced(make_service, calibration_file, monkeypatch):
    monkeypatch.setattr(ReportTools, "get_gradcam", lambda self: {"peak_region": None})
    service = make_service(CONFIDENT[0], calibration_file(CONFIDENT[1]))
    image = Image.new("L", (64, 64), 128)
    truth = ReportTools(service, image).run_classifier()
    cache, llm = ReportCache(), MockLLMClient([MockLLMClient.final(valid_report(truth))])
    generate_report_for_image(service, image, llm=llm, cache=cache)
    tracer = Tracer()
    generate_report_for_image(service, image, llm=llm, cache=cache, tracer=tracer)
    assert names(tracer) == ["report", "report_cache_hit"] and tracer.spans[0].attributes["cached"] is True


def test_gemini_usage_and_finish_reason_are_extracted():
    response = types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=[types.Part(text="{}")]),
                                    finish_reason=types.FinishReason.STOP)],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=1500, candidates_token_count=250, thoughts_token_count=80, total_token_count=1830))
    fake = type("Models", (), {"generate_content": lambda self, **kw: response})()
    client = GeminiClient(api_key="unused", client=type("C", (), {"models": fake})())
    turn = client.generate("system", [{"role": "user", "text": "hi"}], [])
    assert turn.usage == {"input_tokens": 1500, "output_tokens": 250, "thinking_tokens": 80}
    assert turn.finish_reason == "STOP" and turn.rate_limit_retries == 0


# Cost

PRICING = {"billing_tier": "free", "models": {"m": {"schedule": [
    {"until": "2026-12-31", "input_per_million": 0.75, "output_per_million": 3.75},
    {"until": None, "input_per_million": 1.50, "output_per_million": 7.50}]}}}


def test_cost_is_nominal_and_free_tier_billed_zero():
    cost = estimate_cost("m", 1_000_000, 1_000_000, on=date(2026, 10, 18), pricing=PRICING)
    assert cost == {"nominal_cost_usd": 4.5, "billed_cost_usd": 0.0}
    later = estimate_cost("m", 1_000_000, 0, on=date(2027, 1, 1), pricing=PRICING)
    assert later["nominal_cost_usd"] == 1.5  # schedule switches after the intro price ends
    paid = estimate_cost("m", 2_000_000, 0, on=date(2026, 10, 18), pricing=dict(PRICING, billing_tier="paid"))
    assert paid["billed_cost_usd"] == paid["nominal_cost_usd"] == 1.5
    assert estimate_cost("unknown", 10, 10, pricing=PRICING)["nominal_cost_usd"] is None


def test_shipped_pricing_config_is_marked_nominal_and_free():
    from services.tracing import load_pricing
    pricing = load_pricing()
    assert "NOMINAL" in pricing["_note"] and pricing["billing_tier"] == "free"
    assert "gemini-3.8-flash" in pricing["models"]


# Analytics

def span_row(trace, name, latency=10.0, parent=None, sid=None, status="ok", **attributes):
    return {"trace_id": trace, "span_id": sid or f"{trace}-{name}", "parent_span_id": parent, "name": name,
            "latency_ms": latency, "status": status, "attributes": attributes}


def test_summarize_spans_metrics():
    rows = [
        span_row("t1", "classify", 20.0), span_row("t2", "classify", 40.0, status="error"),
        span_row("t1", "report", 900.0, sid="r1"),
        span_row("t1", "llm_call", 400.0, parent="r1", sid="l1", input_tokens=1000, output_tokens=200,
                 thinking_tokens=50, nominal_cost_usd=0.002),
        span_row("t1", "llm_call", 300.0, parent="r1", sid="l2", input_tokens=1500, output_tokens=300,
                 thinking_tokens=0, nominal_cost_usd=0.003),
        span_row("t1", "guard", 0.0, parent="r1", sid="g1", result="reject"),
        span_row("t1", "guard", 0.0, parent="r1", sid="g2", result="pass"),
        span_row("t1", "fallback_used", 0.0, parent="r1", sid="f1", used=False),
        span_row("t2", "fallback_used", 0.0, sid="f2", used=True),
    ]
    m = summarize_spans(rows)
    assert m["traces"] == 2
    assert m["steps"]["classify"] == {"count": 2, "avg_latency_ms": 30.0, "error_rate": 0.5}
    assert "guard" not in m["steps"]  # events don't count toward latency
    assert m["reports_with_llm"] == 1
    assert m["avg_input_tokens_per_report"] == 2500 and m["avg_output_tokens_per_report"] == 550
    assert m["avg_nominal_cost_per_report_usd"] == pytest.approx(0.005)
    assert m["guard_reject_rate"] == 0.5 and m["fallback_rate"] == 0.5


def test_jsonl_rows_round_trip_into_summary(tmp_path, make_service, calibration_file):
    service = make_service(CONFIDENT[0], calibration_file(CONFIDENT[1]))
    tracer = make_tracer(trace_dir=tmp_path)
    service.predict(Image.new("L", (64, 64), 128), tracer=tracer)
    tracer.finish()
    rows = JsonlTraceSink(tmp_path).read()
    assert json.loads(json.dumps(rows)) == rows
    assert set(summarize_spans(rows)["steps"]) == {"preprocess", "classify", "calibrate"}
