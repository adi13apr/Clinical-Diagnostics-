"""Multi-agent report pipeline (agent/multi_agent.py): offline, with scripted fake clients."""
import asyncio
import json
from types import SimpleNamespace

import httpx
import openai
import pytest
from PIL import Image

import agent.multi_agent as multi
from agent.agent import (
    REPORT_CACHE, MockLLMClient, OpenAICompatibleClient, RateLimitError, ReportCache, build_report_llm, fallback_report,
    generate_report_for_image, run_report,
)
from agent.llm_config import load_llm_config, pipeline_mode
from agent.multi_agent import (
    REPORT_JSON_SCHEMA, VERDICT_JSON_SCHEMA, AgentTimeoutError, EvidenceBundle, MultiAgentPipeline, RevisionRequest,
    VerifierVerdict, _drive_sync, _Orchestrator, agenerate_multi_agent_report, generate_multi_agent_report,
)
from agent.schema import DISCLAIMER, Report
from agent.tools import ReportTools
from services.tracing import Tracer, summarize_spans

CONFIDENT = ([2.0, 0.0, 0.0], 0.1318)   # Normal, calibrated ~1.0
REFER = ([0.0, 0.2, 0.6], 1.0)          # Tuberculosis, calibrated ~0.44
GRADCAM = {"explained_class": None, "peak_region": "right upper zone", "focus": "diffuse",
           "region_shares": {"right upper zone": 0.22}, "border_share": 0.3, "border_attention": False}


@pytest.fixture
def make_tools(make_service, calibration_file, monkeypatch):
    def make(case=CONFIDENT, pixel=128):
        logits, temperature = case
        service = make_service(logits, calibration_file(temperature))
        monkeypatch.setattr(ReportTools, "get_gradcam", lambda self: GRADCAM)
        return ReportTools(service, Image.new("L", (64, 64), pixel))
    return make


@pytest.fixture(autouse=True)
def no_pipeline_env(monkeypatch):
    monkeypatch.delenv("REPORT_PIPELINE", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)


def evidence_script():
    return [MockLLMClient.tool("run_classifier"), MockLLMClient.tool("get_gradcam"),
            MockLLMClient.tool("get_condition_info")]


def good_report(tools, **overrides):
    """A grounded report from the tool outputs (the template is guaranteed to pass the guard)."""
    return {**fallback_report(tools).model_dump(), "disclaimer": "x", **overrides}


def unsupported_report(tools):
    """Grounded fields and schema-valid, but with a claim no tool supports (for the verifier to flag)."""
    truth = tools.run_classifier()
    return good_report(tools, summary=f"The model's most likely category is {truth['predicted_class']}. "
                                      "The patient also has a fever.")


def verdict(*supported):
    return {"findings": [{"claim": f"claim {i}", "supported": s, "source": "classifier" if s else "none",
                          "reason": "stated in the evidence" if s else "not in the evidence"}
                         for i, s in enumerate(supported)]}


def make_pipeline(evidence=None, writer=(), verifier=(), **kwargs):
    return MultiAgentPipeline(MockLLMClient(evidence_script() if evidence is None else evidence, name="evidence-m"),
                              MockLLMClient(list(writer), name="writer-m"),
                              MockLLMClient(list(verifier), name="verifier-m"), **kwargs)


def events(tracer, name):
    return [s for s in tracer.spans if s.name == name]


def states(tracer):
    return [s.attributes["state"] for s in events(tracer, "orchestrator_state")]


# ---- Each agent alone ----

def test_evidence_agent_alone_returns_a_bundle_of_tool_outputs(make_tools):
    tools = make_tools()
    pipeline = make_pipeline()
    bundle = _drive_sync(_Orchestrator(tools, pipeline, None)._evidence(), pipeline, None)
    assert isinstance(bundle, EvidenceBundle)
    assert bundle.classifier.model_dump() == tools.run_classifier()
    assert bundle.gradcam == GRADCAM and bundle.condition_info == tools.get_condition_info()
    assert bundle.dataset_caveat == tools.get_condition_info()["model_limitations"][0]
    assert bundle.tool_calls == ["run_classifier", "get_gradcam", "get_condition_info"]
    # Stops as soon as all three returned: no final-text call
    assert len(pipeline.clients["evidence"].calls) == 3


def test_writer_alone_sees_only_the_bundle_and_uses_the_strict_report_schema(make_tools):
    tools = make_tools()
    pipeline = make_pipeline(writer=[MockLLMClient.final(good_report(tools))], verifier=[MockLLMClient.final(verdict(True))])
    result = generate_multi_agent_report(tools, pipeline)
    writer = pipeline.clients["writer"]
    assert writer.schemas == ["report"] and writer.tools_allowed == [False]
    prompt = writer.calls[0][0]["text"]
    assert prompt.startswith("EVIDENCE:") and "tool_calls" not in prompt
    evidence = json.loads(prompt.split("EVIDENCE:\n", 1)[1].rsplit("\n\nWrite the report.", 1)[0])
    assert set(evidence) == {"classifier", "gradcam", "condition_info", "dataset_caveat"}
    assert result.source == "llm" and result.pipeline == "multi" and result.model == "writer-m"


def test_verifier_alone_gets_the_evidence_and_the_draft_without_disclaimer(make_tools):
    tools = make_tools()
    draft = good_report(tools, disclaimer="LLM disclaimer text")
    pipeline = make_pipeline(writer=[MockLLMClient.final(draft)], verifier=[MockLLMClient.final(verdict(True, True))])
    result = generate_multi_agent_report(tools, pipeline)
    verifier = pipeline.clients["verifier"]
    assert verifier.schemas == ["verdict"]
    text = verifier.calls[0][0]["text"]
    assert "EVIDENCE:" in text and "DRAFT REPORT:" in text and "LLM disclaimer text" not in text
    assert [f["supported"] for f in result.findings] == [True, True]


def test_strict_schemas_match_the_pydantic_models():
    assert set(REPORT_JSON_SCHEMA["required"]) == set(Report.model_fields)
    finding = VERDICT_JSON_SCHEMA["properties"]["findings"]["items"]
    assert set(finding["required"]) == set(multi.ClaimFinding.model_fields)
    for schema in (REPORT_JSON_SCHEMA, finding):
        assert schema["additionalProperties"] is False and set(schema["required"]) == set(schema["properties"])
    with pytest.raises(ValueError):
        VerifierVerdict.model_validate({"findings": []})  # at least one claim checked


def test_revision_request_prompt_names_problems_and_claims():
    text = RevisionRequest(by="verifier", unsupported=[multi.ClaimFinding(
        claim="The patient has a fever.", supported=False, source="none", reason="Not in the evidence.")]).prompt()
    assert "verifier" in text and '"The patient has a fever."' in text and "Not in the evidence." in text


# ---- Orchestrator state transitions ----

def test_happy_path_state_transitions_and_handoffs(make_tools):
    tools, tracer = make_tools(), Tracer()
    pipeline = make_pipeline(writer=[MockLLMClient.final(good_report(tools))], verifier=[MockLLMClient.final(verdict(True))])
    result = generate_multi_agent_report(tools, pipeline, tracer=tracer)
    assert states(tracer) == ["evidence", "write", "guard", "verify", "final_guard", "done"]
    assert [(h.attributes["from_agent"], h.attributes["to_agent"], h.attributes["message"])
            for h in events(tracer, "handoff")] == [("evidence", "writer", "EvidenceBundle"),
                                                    ("writer", "verifier", "Report")]
    assert result.steps == 5 and result.revisions == [] and result.rejected == []
    assert result.report.disclaimer == DISCLAIMER
    assert result.agent_models == {"evidence": "evidence-m", "writer": "writer-m", "verifier": "verifier-m"}


def test_agent_spans_record_agent_tokens_latency_and_verdict(make_tools):
    tools, tracer = make_tools(), Tracer()
    usage = {"input_tokens": 1000, "output_tokens": 200, "thinking_tokens": 50}
    pipeline = make_pipeline(writer=[MockLLMClient.final(good_report(tools), usage=usage)],
                             verifier=[MockLLMClient.final(verdict(True, True), usage=usage)])
    generate_multi_agent_report(tools, pipeline, tracer=tracer)
    by_name = {s.name: s for s in tracer.spans}
    report = by_name["report"]
    assert report.attributes["pipeline"] == "multi" and report.attributes["source"] == "llm"
    writer, verifier = by_name["writer_agent"], by_name["verifier_agent"]
    assert {"agent": "writer", "model": "writer-m", "verdict": "draft", "calls": 1, "input_tokens": 1000,
            "output_tokens": 250}.items() <= writer.attributes.items()
    assert verifier.attributes["verdict"] == "supported" and verifier.attributes["claims"] == 2
    assert writer.latency_ms is not None and by_name["evidence_agent"].attributes["verdict"] == "complete"
    llm_calls = events(tracer, "llm_call")
    assert [c.attributes["agent"] for c in llm_calls] == ["evidence"] * 3 + ["writer", "verifier"]
    assert all(c.parent_span_id != report.span_id for c in llm_calls)  # nested under their agent span
    # Analytics still count one report, with every call's tokens
    summary = summarize_spans([s.to_dict() for s in tracer.spans])
    assert summary["reports_with_llm"] == 1 and summary["avg_input_tokens_per_report"] == 2000
    assert summary["guard_checks"] == 1 and summary["reports"] == 1 and summary["fallback_rate"] == 0


# ---- Skipped tool: re-ask once, then the template ----

def test_skipped_tool_is_re_asked_once_and_the_report_continues(make_tools):
    tools, tracer = make_tools(), Tracer()
    evidence = [MockLLMClient.tool("run_classifier"), MockLLMClient.tool("get_gradcam"), MockLLMClient.final("DONE"),
                MockLLMClient.tool("get_condition_info")]
    pipeline = make_pipeline(evidence, writer=[MockLLMClient.final(good_report(tools))],
                             verifier=[MockLLMClient.final(verdict(True))])
    result = generate_multi_agent_report(tools, pipeline, tracer=tracer)
    reask = pipeline.clients["evidence"].calls[3][-1]["text"]
    assert "get_condition_info" in reask
    assert [e.attributes["missing"] for e in events(tracer, "evidence_reask")] == [["get_condition_info"]]
    assert result.source == "llm" and result.tool_calls == ["run_classifier", "get_gradcam", "get_condition_info"]


def test_tool_still_skipped_after_the_re_ask_uses_the_template(make_tools):
    tools = make_tools()
    evidence = [MockLLMClient.tool("run_classifier"), MockLLMClient.final("DONE"), MockLLMClient.final("DONE")]
    pipeline = make_pipeline(evidence)
    result = generate_multi_agent_report(tools, pipeline)
    assert result.source == "fallback" and result.error_kind == "model"
    assert "get_gradcam, get_condition_info" in result.reason and "asked twice" in result.reason
    assert pipeline.clients["writer"].calls == []  # the writer never ran


def test_condition_info_for_another_class_does_not_count(make_tools):
    tools = make_tools()  # predicted Normal
    evidence = [MockLLMClient.tool("run_classifier"), MockLLMClient.tool("get_gradcam"),
                MockLLMClient.tool("get_condition_info", condition="Tuberculosis"), MockLLMClient.final("DONE"),
                MockLLMClient.tool("get_condition_info", condition="Normal")]
    pipeline = make_pipeline(evidence, writer=[MockLLMClient.final(good_report(tools))],
                             verifier=[MockLLMClient.final(verdict(True))])
    result = generate_multi_agent_report(tools, pipeline)
    assert result.source == "llm"
    bundle = json.loads(pipeline.clients["writer"].calls[0][0]["text"].split("EVIDENCE:\n", 1)[1].rsplit("\n\n", 1)[0])
    assert bundle["condition_info"]["condition"] == "Normal"


def test_c07_like_bundle_always_carries_the_dataset_caveat_to_the_writer(make_tools):
    """c07 skipped get_condition_info and lost the caveat; here the writer can't run without it."""
    tools = make_tools()
    caveat = tools.get_condition_info()["model_limitations"][0]
    pipeline = make_pipeline(writer=[MockLLMClient.final(good_report(tools))], verifier=[MockLLMClient.final(verdict(True))])
    result = generate_multi_agent_report(tools, pipeline)
    prompt = pipeline.clients["writer"].calls[0][0]["text"]
    assert json.dumps(caveat)[1:-1] in prompt and '"dataset_caveat"' in prompt
    assert caveat in pipeline.clients["verifier"].calls[0][0]["text"]
    assert caveat in result.report.limitations


# ---- Verifier findings lead to a revision; the revision limit ----

def test_unsupported_claim_is_revised_once_then_accepted(make_tools):
    tools, tracer = make_tools(), Tracer()
    first = unsupported_report(tools)
    pipeline = make_pipeline(writer=[MockLLMClient.final(first), MockLLMClient.final(good_report(tools))],
                             verifier=[MockLLMClient.final(verdict(True, False)), MockLLMClient.final(verdict(True))])
    result = generate_multi_agent_report(tools, pipeline, tracer=tracer)
    assert result.source == "llm" and result.report.summary == fallback_report(tools).summary
    assert len(result.revisions) == 1 and result.revisions[0]["by"] == "verifier"
    assert result.revisions[0]["text"] == json.dumps(first)
    assert [f["claim"] for f in result.revisions[0]["unsupported"]] == ["claim 1"]
    # Verifier-flagged drafts are NOT guard rejections (the eval's guard replay would disagree)
    assert result.rejected == []
    revision_prompt = pipeline.clients["writer"].calls[1][-1]["text"]
    assert "Unsupported claim" in revision_prompt and "claim 1" in revision_prompt
    assert states(tracer) == ["evidence", "write", "guard", "verify", "write", "guard", "verify", "final_guard", "done"]


def test_still_unsupported_after_one_revision_uses_the_template(make_tools):
    tools = make_tools()
    pipeline = make_pipeline(writer=[MockLLMClient.final(good_report(tools))] * 3,
                             verifier=[MockLLMClient.final(verdict(False))] * 3)
    result = generate_multi_agent_report(tools, pipeline)
    assert result.source == "fallback" and "unsupported claims after 1 revision" in result.reason
    assert len(pipeline.clients["writer"].calls) == 2 and len(pipeline.clients["verifier"].calls) == 2
    assert len(result.revisions) == 2  # both drafts sent back; only the first got a rewrite


# ---- The guard: before the verifier, and the final say ----

def test_guard_rejected_draft_never_reaches_the_verifier_and_is_revised(make_tools):
    tools = make_tools()
    truth = tools.run_classifier()
    flipped = good_report(tools, predicted_class="Tuberculosis")
    pipeline = make_pipeline(writer=[MockLLMClient.final(flipped), MockLLMClient.final(good_report(tools))],
                             verifier=[MockLLMClient.final(verdict(True))])
    result = generate_multi_agent_report(tools, pipeline)
    assert result.source == "llm" and result.report.predicted_class == truth["predicted_class"]
    assert len(pipeline.clients["verifier"].calls) == 1  # only the corrected draft was verified
    assert result.rejected[0]["text"] == json.dumps(flipped) and "predicted_class" in result.rejected[0]["problem"]
    assert result.revisions[0]["by"] == "guard"


def test_guard_rejection_twice_uses_the_template_even_if_the_verifier_would_approve(make_tools):
    tools = make_tools()
    flipped = MockLLMClient.final(good_report(tools, predicted_class="Tuberculosis"))
    pipeline = make_pipeline(writer=[flipped, flipped], verifier=[MockLLMClient.final(verdict(True))] * 2)
    result = generate_multi_agent_report(tools, pipeline)
    assert result.source == "fallback" and result.reason.startswith("Report rejected after 1 revision")
    assert pipeline.clients["verifier"].calls == []


def test_the_final_guard_has_the_last_word_after_the_verifier_approves(make_tools, monkeypatch):
    tools, tracer = make_tools(), Tracer()
    real = multi.check_grounding
    calls = []

    def guard(report, truth, *args, **kwargs):
        calls.append(report)
        return real(report, truth, *args, **kwargs) or (["final check failed"] if len(calls) > 1 else [])
    monkeypatch.setattr(multi, "check_grounding", guard)
    pipeline = make_pipeline(writer=[MockLLMClient.final(good_report(tools))], verifier=[MockLLMClient.final(verdict(True))])
    result = generate_multi_agent_report(tools, pipeline, tracer=tracer)
    assert len(pipeline.clients["verifier"].calls) == 1  # the verifier approved...
    assert result.source == "fallback" and "final guard" in result.reason  # ...but the guard decided
    assert calls[-1].disclaimer == DISCLAIMER  # the guard sees exactly what would be returned
    assert events(tracer, "final_guard")[0].attributes["result"] == "reject"


def test_refer_case_goes_through_the_same_guard_rules(make_tools):
    tools = make_tools(REFER)
    assert tools.run_classifier()["status"] == "refer"
    no_review = good_report(tools, recommendation="Discuss the result with your doctor.")
    pipeline = make_pipeline(writer=[MockLLMClient.final(no_review), MockLLMClient.final(good_report(tools))],
                             verifier=[MockLLMClient.final(verdict(True))])
    result = generate_multi_agent_report(tools, pipeline)
    assert result.source == "llm" and "clinician review" in result.report.recommendation.lower()
    assert "clinician review" in result.rejected[0]["problem"]


# ---- LLM failures and timeouts ----

def test_model_failure_retried_once_infrastructure_falls_back(make_tools):
    tools = make_tools()

    def bad(messages):
        raise tool_use_failed()
    pipeline = make_pipeline(writer=[bad, MockLLMClient.final(good_report(tools))], verifier=[MockLLMClient.final(verdict(True))])
    assert generate_multi_agent_report(tools, pipeline).source == "llm"

    def down(messages):
        raise ConnectionError("network down")
    pipeline = make_pipeline(writer=[MockLLMClient.final(good_report(tools))], verifier=[down])
    result = generate_multi_agent_report(make_tools(), pipeline)
    assert result.source == "fallback" and result.error_kind == "infrastructure"
    assert result.reason.startswith("verifier agent:") and "network down" in result.error_detail


def json_validate_failed():
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    body = {"error": {"message": "Failed to generate JSON.", "type": "invalid_request_error",
                      "code": "json_validate_failed",
                      "failed_generation": "max completion tokens reached before generating a valid document"}}
    return openai.BadRequestError("Error code: 400 - json_validate_failed",
                                  response=httpx.Response(400, request=request), body=body)


def test_json_validate_failed_is_a_model_failure_retried_once(make_tools):
    """Found by the smoke run: the writer's strict JSON hit max_completion_tokens (Groq 400)."""
    from agent.agent import classify_llm_error, describe_llm_error
    assert classify_llm_error(json_validate_failed()) == "model"
    assert "JSON" in describe_llm_error(json_validate_failed())
    tools = make_tools()

    def cut(messages):
        raise json_validate_failed()
    pipeline = make_pipeline(writer=[cut, MockLLMClient.final(good_report(tools))], verifier=[MockLLMClient.final(verdict(True))])
    result = generate_multi_agent_report(tools, pipeline)
    assert result.source == "llm" and result.rejected[0]["text"] is None
    pipeline = make_pipeline(writer=[cut, cut])
    result = generate_multi_agent_report(tools, pipeline)
    assert result.source == "fallback" and result.error_kind == "model"  # recorded by the eval, not a stop


def test_rate_limit_falls_back_without_retry(make_tools):
    tools = make_tools()

    def limited(messages):
        raise RateLimitError(30, provider="Groq")
    pipeline = make_pipeline([limited])
    result = generate_multi_agent_report(tools, pipeline)
    assert result.source == "fallback" and result.error_kind == "rate_limit"
    assert len(pipeline.clients["evidence"].calls) == 1


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_agent_past_its_time_limit_falls_back(make_tools):
    tools, tracer, clock = make_tools(), Tracer(), FakeClock()

    def slow_tool(name):
        def turn(messages):
            clock.now += 50  # each evidence call takes 50 s (e.g. waiting for the shared budget)
            return MockLLMClient.tool(name)
        return turn
    evidence = [slow_tool("run_classifier"), slow_tool("get_gradcam"), slow_tool("get_condition_info")]
    pipeline = make_pipeline(evidence, timeouts={"evidence": 90}, clock=clock)
    result = generate_multi_agent_report(tools, pipeline, tracer=tracer)
    assert result.source == "fallback" and result.error_kind == "infrastructure"
    assert result.reason == "The evidence agent ran past its 90 s time limit."
    assert len(pipeline.clients["evidence"].calls) == 2  # the third call was never started
    assert states(tracer)[-1] == "fallback"


def test_each_agent_gets_its_own_time_limit_and_the_job_deadline_caps_them(make_tools):
    tools, clock = make_tools(), FakeClock()
    pipeline = make_pipeline(writer=[MockLLMClient.final(good_report(tools))], verifier=[MockLLMClient.final(verdict(True))],
                             timeouts={"evidence": 100, "writer": 60, "verifier": 30}, clock=clock)
    pipeline.deadline = 45.0  # a report job's deadline
    seen = []
    for agent, client in pipeline.clients.items():
        original = client.generate if agent == "evidence" else client.generate_json

        def wrapped(*args, _original=original, _client=client, _agent=agent, **kwargs):
            seen.append((_agent, round(_client.call_timeout_s)))
            return _original(*args, **kwargs)
        setattr(client, "generate" if agent == "evidence" else "generate_json", wrapped)
    assert generate_multi_agent_report(tools, pipeline).source == "llm"
    assert seen == [("evidence", 45)] * 3 + [("writer", 45), ("verifier", 30)]


def test_request_timeout_from_the_client_falls_back(make_tools):
    tools = make_tools()

    def hung(messages):
        raise TimeoutError()
    result = generate_multi_agent_report(tools, make_pipeline(writer=[hung]))
    assert result.source == "fallback" and result.error_kind == "infrastructure"


def test_truncated_draft_is_retried_once(make_tools):
    from agent.agent import LLMTurn
    tools = make_tools()
    cut = LLMTurn(text='{"predicted_class": "Nor', finish_reason="length")
    pipeline = make_pipeline(writer=[cut, MockLLMClient.final(good_report(tools))], verifier=[MockLLMClient.final(verdict(True))])
    result = generate_multi_agent_report(tools, pipeline)
    assert result.source == "llm" and result.rejected[0]["finish_reason"] == "length"


def test_invalid_verifier_reply_is_retried_once_then_falls_back(make_tools):
    tools = make_tools()
    pipeline = make_pipeline(writer=[MockLLMClient.final(good_report(tools))],
                             verifier=[MockLLMClient.final("not json"), MockLLMClient.final({"findings": []})])
    result = generate_multi_agent_report(tools, pipeline)
    assert result.source == "fallback" and "verifier's reply was invalid" in result.reason
    assert result.rejected == []  # a bad verdict isn't a report draft


def test_missing_client_uses_the_template(make_tools):
    pipeline = MultiAgentPipeline(MockLLMClient([]), None, MockLLMClient([]))
    result = generate_multi_agent_report(make_tools(), pipeline)
    assert result.source == "fallback" and "every agent" in result.reason and result.pipeline == "multi"


# ---- Sync and async drivers decide the same ----

def scenario(tools):
    first = unsupported_report(tools)
    return dict(evidence=[MockLLMClient.tool("run_classifier"), MockLLMClient.final("DONE"),
                          MockLLMClient.tool("get_gradcam"), MockLLMClient.tool("get_condition_info")],
                writer=[MockLLMClient.final(good_report(tools, predicted_class="Tuberculosis")),
                        MockLLMClient.final(first)],
                verifier=[MockLLMClient.final(verdict(False))])


def test_async_driver_makes_the_same_decisions(make_tools):
    tools = make_tools()
    sync = generate_multi_agent_report(tools, make_pipeline(**scenario(tools)))
    async_ = asyncio.run(agenerate_multi_agent_report(tools, make_pipeline(**scenario(tools))))
    assert sync == async_ and sync.source == "fallback" and len(sync.revisions) == 2


# ---- Entry points: run_report, the cache, build_report_llm ----

def test_run_report_dispatches_on_the_pipeline(make_tools):
    tools = make_tools()
    single = run_report(tools, MockLLMClient([MockLLMClient.tool("run_classifier"), MockLLMClient.tool("get_gradcam"),
                                              MockLLMClient.tool("get_condition_info"),
                                              MockLLMClient.final(good_report(tools))]))
    assert single.pipeline == "single" and single.agent_models == {} and single.revisions == []
    pipeline = make_pipeline(writer=[MockLLMClient.final(good_report(tools))], verifier=[MockLLMClient.final(verdict(True))])
    assert run_report(tools, pipeline).pipeline == "multi"


def test_cache_keys_include_the_pipeline(make_tools, report_cache):
    tools = make_tools()
    image, service = tools.image, tools.model_service
    pipeline = make_pipeline(writer=[MockLLMClient.final(good_report(tools))], verifier=[MockLLMClient.final(verdict(True))])
    first = generate_report_for_image(service, image, llm=pipeline, cache=report_cache)
    again = generate_report_for_image(service, image.copy(), llm=make_pipeline([]), cache=report_cache)
    assert first.pipeline == "multi" and not first.cached
    assert again.cached and again.pipeline == "multi" and again.findings == first.findings
    # Same model name for the single agent, other pipeline: not a hit
    single = MockLLMClient([MockLLMClient.tool("run_classifier"), MockLLMClient.tool("get_gradcam"),
                            MockLLMClient.tool("get_condition_info"), MockLLMClient.final(good_report(tools))],
                           name=pipeline.name)
    other = generate_report_for_image(service, image, llm=single, cache=report_cache)
    assert other.pipeline == "single" and not other.cached and len(report_cache) == 2


def groq_config(**overrides):
    config = load_llm_config()
    config.update(overrides)
    return config


def test_build_report_llm_single_by_default(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    assert pipeline_mode() == "single"
    llm = build_report_llm()
    assert llm.pipeline == "single" and llm.name == "openai/gpt-oss-120b" and llm.pipeline_note is None


def test_build_report_llm_multi_from_env_with_its_own_client_per_agent(monkeypatch):
    from agent.rate_limit import BudgetRegistry
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("REPORT_PIPELINE", "multi")
    budgets = BudgetRegistry()
    llm = build_report_llm(budgets=budgets)
    assert isinstance(llm, MultiAgentPipeline) and llm.pipeline == "multi"
    assert llm.models == {"evidence": "openai/gpt-oss-120b", "writer": "openai/gpt-oss-120b",
                          "verifier": "openai/gpt-oss-20b"}
    clients = llm.clients
    assert clients["evidence"] is not clients["writer"]  # own deadline/timeout per agent...
    assert clients["evidence"].budget is clients["writer"].budget  # ...one shared budget per model
    assert clients["verifier"].budget is budgets.get("openai/gpt-oss-20b")
    assert llm.timeouts == {"evidence": 180, "writer": 150, "verifier": 120}
    # Per-agent request options: the writer's strict JSON draft needs more room than a tool call
    caps = {agent: client.options["max_completion_tokens"] for agent, client in clients.items()}
    assert caps == {"evidence": 600, "writer": 1200, "verifier": 1600}
    assert clients["writer"].options["reasoning_effort"] == "low"  # the model's own options are kept
    assert build_report_llm("single").pipeline == "single"  # an explicit argument wins


def test_unknown_pipeline_is_an_error(monkeypatch):
    monkeypatch.setenv("REPORT_PIPELINE", "triple")
    with pytest.raises(ValueError, match="REPORT_PIPELINE"):
        pipeline_mode()


def test_multi_on_gemini_runs_single_and_records_it(monkeypatch, make_tools):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("REPORT_PIPELINE", "multi")
    monkeypatch.setattr("google.genai.Client", lambda **kwargs: object())
    llm = build_report_llm()
    assert llm.pipeline == "single" and "gemini ran the single agent" in llm.pipeline_note
    # The pipeline that ran is what the result and the trace say
    tools, tracer = make_tools(), Tracer()
    scripted = MockLLMClient([MockLLMClient.tool("run_classifier"), MockLLMClient.tool("get_gradcam"),
                              MockLLMClient.tool("get_condition_info"), MockLLMClient.final(good_report(tools))])
    scripted.pipeline_note = llm.pipeline_note
    result = run_report(tools, scripted, tracer=tracer)
    report_span = next(s for s in tracer.spans if s.name == "report")
    assert result.pipeline == "single" and result.pipeline_note == llm.pipeline_note
    assert report_span.attributes["pipeline"] == "single" and report_span.attributes["pipeline_note"] == llm.pipeline_note


def test_no_key_means_no_llm_in_either_mode(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr("agent.llm_config._secret", lambda section, key: None)
    assert build_report_llm("single") is None and build_report_llm("multi") is None


def test_streamlit_keeps_the_in_memory_cache():
    assert isinstance(REPORT_CACHE, ReportCache)


# ---- Groq client: strict json_schema requests ----

def completion(content, prompt=900, completion_tokens=120):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=None), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion_tokens,
                              total_tokens=prompt + completion_tokens, completion_tokens_details=None))


class FakeChat:
    def __init__(self, items):
        self.items, self.requests = list(items), []
        self.completions = self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.items.pop(0)


def client_for(items, **kwargs):
    chat = FakeChat(items)
    return OpenAICompatibleClient("unused", "openai/gpt-oss-20b", "https://api.groq.com/openai/v1",
                                  client=SimpleNamespace(chat=chat), sleep=lambda s: None, **kwargs), chat


def tool_use_failed():
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    body = {"error": {"message": "Failed to call a function.", "code": "tool_use_failed"}}
    return openai.BadRequestError("Error code: 400 - tool_use_failed",
                                  response=httpx.Response(400, request=request), body=body)


def test_generate_json_sends_a_strict_schema_and_no_tools():
    llm, chat = client_for([completion(content='{"findings": []}', prompt=500, completion_tokens=80)])
    llm.call_timeout_s = 12.5
    turn = llm.generate_json("system", [{"role": "user", "text": "check"}], "verdict", VERDICT_JSON_SCHEMA)
    request = chat.requests[0]
    assert "tools" not in request and request["timeout"] == 12.5
    assert request["response_format"] == {"type": "json_schema", "json_schema": {
        "name": "verdict", "strict": True, "schema": VERDICT_JSON_SCHEMA}}
    assert turn.text == '{"findings": []}' and turn.usage["input_tokens"] == 500


def test_single_agent_requests_are_unchanged():
    llm, chat = client_for([completion(content="{}")])
    llm.generate("system", [{"role": "user", "text": "hi"}], [{"name": "t", "description": "d", "parameters": {}}])
    assert "timeout" not in chat.requests[0] and "response_format" not in chat.requests[0]
    assert chat.requests[0]["tools"][0]["function"]["name"] == "t"


def test_json_requests_reserve_the_schema_in_the_budget(make_budget):
    budget = make_budget("m-json", tpm=100_000)
    llm, _ = client_for([completion(content="{}", prompt=300, completion_tokens=50)], budget=budget,
                        options={"max_completion_tokens": 1000})
    size, out = llm._reservation_size(llm._messages("s", []), {"schema": VERDICT_JSON_SCHEMA})
    assert out == 1000 and size > 1000 + len(json.dumps(VERDICT_JSON_SCHEMA)) // 4
    llm.generate_json("s", [], "verdict", VERDICT_JSON_SCHEMA)
    assert budget.used_last_minute() == (1, 350)


# ---- Eval records and the job timeout ----

def test_eval_record_says_which_pipeline_ran_and_passes_the_unchanged_checks(make_tools):
    from evals.checks import run_checks
    from evals.dataset import load_cases
    from evals.run import _finish_case
    tools, tracer = make_tools(), Tracer()
    first = unsupported_report(tools)
    pipeline = make_pipeline(writer=[MockLLMClient.final(first), MockLLMClient.final(good_report(tools))],
                             verifier=[MockLLMClient.final(verdict(False)), MockLLMClient.final(verdict(True))])
    result = generate_multi_agent_report(tools, pipeline, tracer=tracer)
    record = _finish_case(load_cases()[0], tools, tracer, result)
    assert record["pipeline"] == "multi" and len(record["revisions"]) == 1 and record["rejected"] == []
    assert set(record["agent_usage"]["by_model"]) == {"evidence-m", "writer-m", "verifier-m"}
    checks = run_checks(record)
    assert checks["guard_replay_consistent"] is True  # verifier-only revisions aren't replayed as guard rejections
    assert checks["dataset_caveat_present"] is True
    assert all(v is not False for v in checks.values()), checks


def test_job_timeout_for_the_multi_pipeline_counts_both_models(monkeypatch):
    from api.jobs import report_job_timeout, report_model_budget
    verifier = {"tpm": 8000, "rpm": 30, "max_completion_tokens": 1000}
    # Report model: 5,600 + 5 x 600 = 8,600 reserved -> 2 windows; verifier 3,000 + 2 x 1,000 -> 1 window
    assert report_job_timeout(1, tpm=8000, rpm=30, max_completion_tokens=600, pipeline="multi",
                              verifier=verifier) == 60 + 60 * 3
    assert report_job_timeout(3, tpm=8000, rpm=30, max_completion_tokens=600, pipeline="multi",
                              verifier=verifier) == 60 + 60 * (4 + 2)
    assert report_job_timeout(1, tpm=8000, rpm=30, max_completion_tokens=600) == 180  # single unchanged
    monkeypatch.setenv("REPORT_PIPELINE", "multi")
    budget = report_model_budget()
    assert budget["pipeline"] == "multi" and budget["verifier"]["max_completion_tokens"] == 1600
    assert budget["max_completion_tokens"] == 600 and budget["writer_max_completion_tokens"] == 1200
    # Report model 5,600 + 3 x 600 + 2 x 1,200 = 9,800 reserved -> 2 windows; verifier 3,000 + 2 x 1,600 = 6,200
    # -> 1 window: 240 s. 3 workers: 4 + 3 windows -> 480 s
    assert report_job_timeout(1, **budget) == 240 and report_job_timeout(3, **budget) == 480


# ---- Streamlit: "Generate report" runs whatever build_report_llm returns ----

def _single_result_app():
    import streamlit as st

    from ui.dashboard import show_single_result
    show_single_result("radiologist@example.com", st.session_state["svc"], st.session_state["img"])


def test_streamlit_generate_report_runs_the_multi_pipeline(make_tools, monkeypatch):
    from datetime import datetime

    from streamlit.testing.v1 import AppTest
    tools = make_tools(pixel=77)  # an image no other test caches in the process-wide REPORT_CACHE
    tracers = []

    def tracer(store, trace_id=None):
        tracers.append(Tracer(trace_id=trace_id))  # in memory: nothing written to traces/
        return tracers[-1]

    monkeypatch.setattr("ui.dashboard.make_tracer", tracer)
    monkeypatch.setattr("agent.agent.build_report_llm", lambda: make_pipeline(
        writer=[MockLLMClient.final(good_report(tools))], verifier=[MockLLMClient.final(verdict(True))]))
    at = AppTest.from_function(_single_result_app)
    at.session_state["svc"], at.session_state["img"] = tools.model_service, tools.image
    at.session_state["last_result"] = tools.model_service.predict(tools.image)
    at.session_state["last_patient_name"], at.session_state["last_timestamp"] = "Test Patient", datetime.now()
    at.session_state["last_trace_id"] = "00000000-0000-0000-0000-000000000077"
    at.run(timeout=60)
    next(b for b in at.button if b.label == "Generate report").click().run(timeout=60)
    assert not at.exception
    result = at.session_state["last_report"]
    assert result.source == "llm" and result.pipeline == "multi"
    assert result.source_line() in [c.value for c in at.caption]  # shown to the user
    assert "multi-agent pipeline" in result.source_line()
    report_span = next(s for s in tracers[-1].spans if s.name == "report")
    assert report_span.attributes["pipeline"] == "multi"
