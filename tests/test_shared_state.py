"""Uploads and the report cache shared between API replicas through Redis (fakeredis: no network)."""
import json
import subprocess
import sys

import pytest
from PIL import Image

from agent.agent import (
    REPORT_CACHE, MockLLMClient, RedisReportCache, ReportCache, ReportResult, fallback_report, generate_report_for_image,
    report_result_from_json, report_result_to_json,
)
from agent.tools import ReportTools
from api.state import RedisUploadStore, StoreUnavailable

ALICE, BOB = "11111111-1111-1111-1111-111111111111", "22222222-2222-2222-2222-222222222222"
TRACE = "0b0f3a5e-1c1a-4e5e-9a43-5d7c2c7d2f10"
PREDICTION = {"diagnosis": "Pneumonia", "confidence": 0.9731234, "status": "confident", "temperature": 0.1318,
              "probabilities": {"Normal": 0.0123456, "Pneumonia": 0.9731234, "Tuberculosis": 0.014531}}
CAM_MAP = ([[0.5] * 7 for _ in range(7)], {"peak_region": "right upper zone", "focus": "diffuse",
                                          "region_shares": {"right upper zone": 0.22}, "border_share": 0.3})
GRADCAM = {"peak_region": "right upper zone", "focus": "diffuse", "border_attention": False}


@pytest.fixture
def replicas(make_redis, redis_clock):
    """Two upload stores with their own clients on one Redis, like two API replicas."""
    def make(ttl_s=3600):
        return RedisUploadStore(make_redis(), ttl_s, clock=lambda: redis_clock[0])
    return make(), make()


# ---- Uploads ----

def test_two_replicas_see_each_others_uploads_but_not_other_users(replicas):
    a, b = replicas
    a.put(TRACE, ALICE, b"\x89PNG raw bytes", PREDICTION)
    upload = b.get(TRACE, ALICE)
    assert upload.data == b"\x89PNG raw bytes" and upload.prediction == PREDICTION and upload.user_id == ALICE
    assert upload.heatmap_png is None and upload.cam_map is None
    assert b.get(TRACE, BOB) is None and a.get(TRACE, BOB) is None  # same 404 as "unknown"
    assert b.get("unknown-trace", ALICE) is None


def test_gradcam_results_saved_by_one_replica_are_used_by_the_other(replicas):
    a, b = replicas
    a.put(TRACE, ALICE, b"img", PREDICTION)
    b.save_heatmap(TRACE, b"\x89PNG heatmap")
    a.save_cam_map(TRACE, CAM_MAP)
    for store in (a, b):
        upload = store.get(TRACE, ALICE)
        assert upload.heatmap_png == b"\x89PNG heatmap" and upload.cam_map == CAM_MAP
    assert b.get(TRACE, BOB) is None


def test_expired_uploads_and_their_gradcam_keys_are_gone(replicas, redis_clock):
    a, b = replicas
    a.put(TRACE, ALICE, b"img", PREDICTION)
    a.save_heatmap(TRACE, b"png")
    a.save_cam_map(TRACE, CAM_MAP)
    keys = [a.key(TRACE), a.key(TRACE, "heatmap"), a.key(TRACE, "cam")]
    assert a.client.exists(*keys) == 3
    assert all(0 < a.client.pttl(k) <= 3600 * 1000 for k in keys)

    redis_clock[0] += 3601
    assert b.get(TRACE, ALICE) is None and len(b) == 0
    assert a.client.exists(*keys) == 0  # gone from Redis itself, not just hidden

    b.save_heatmap(TRACE, b"late png")  # a Grad-CAM finishing after expiry must not bring the upload back
    b.save_cam_map(TRACE, CAM_MAP)
    assert a.client.exists(*keys) == 0


def test_new_upload_never_shows_an_older_uploads_gradcam(replicas):
    a, b = replicas
    a.put(TRACE, ALICE, b"old", PREDICTION)
    a.save_heatmap(TRACE, b"old png")
    b.put(TRACE, ALICE, b"new", PREDICTION)
    assert a.get(TRACE, ALICE).heatmap_png is None


def test_prediction_survives_the_json_round_trip_exactly(replicas, make_service, calibration_file):
    service = make_service([0.0, 0.7, 0.5], calibration_file(0.1318))
    prediction = service.predict(Image.new("L", (32, 32), 100))
    a, b = replicas
    a.put(TRACE, ALICE, b"img", prediction)
    assert b.get(TRACE, ALICE).prediction == prediction


def test_unreachable_redis_raises_store_unavailable(replicas, redis_server):
    a, _ = replicas
    redis_server.connected = False
    for call in (lambda: a.put(TRACE, ALICE, b"img", PREDICTION), lambda: a.get(TRACE, ALICE),
                 lambda: a.save_heatmap(TRACE, b"png"), lambda: len(a)):
        with pytest.raises(StoreUnavailable):
            call()


# ---- Report cache ----

@pytest.fixture
def tools(make_service, calibration_file, monkeypatch):
    monkeypatch.setattr(ReportTools, "get_gradcam", lambda self: GRADCAM)
    service = make_service([2.0, 0.0, 0.0], calibration_file(0.1318))
    return ReportTools(service, Image.new("L", (64, 64), 128))


def report_result(tools):
    """An LLM-style result with every field set, including a rejected draft with text None."""
    return ReportResult(report=fallback_report(tools), source="llm", model="openai/gpt-oss-120b", steps=4,
                        problems=["status mismatch"], tool_calls=["run_classifier", "get_gradcam"],
                        rejected=[{"step": 2, "text": '{"x": 1}', "problem": "status mismatch"},
                                  {"step": 3, "text": None, "problem": "tool_use_failed"}])


def test_report_result_json_round_trip_is_exact(tools):
    original = report_result(tools)
    text = report_result_to_json(original)
    assert report_result_from_json(text) == original
    assert "report" in json.loads(text)  # plain JSON, readable without this code


def test_redis_report_cache_is_shared_and_expires(make_redis, redis_clock, tools):
    a, b = RedisReportCache(make_redis(), ttl_s=60), RedisReportCache(make_redis(), ttl_s=60)
    key = ("ab" * 32, 0.1318, "openai/gpt-oss-120b", "single")
    a.put(key, report_result(tools))
    assert b.get(key) == report_result(tools) and len(b) == 1
    assert b.get((key[0], 0.5, key[2], key[3])) is None  # temperature is part of the key
    assert b.get((key[0], key[1], key[2], "multi")) is None  # and so is the pipeline
    assert b.key(key).startswith("cdp:report:v2:single:")
    assert b.client.pttl(b.key(key)) <= 60 * 1000

    redis_clock[0] += 61
    assert a.get(key) is None and b.get(key) is None and len(a) == 0


def test_corrupt_or_unreachable_report_cache_is_a_miss(make_redis, redis_server, tools):
    errors = []
    cache = RedisReportCache(make_redis(), on_error=lambda op, e: errors.append((op, type(e).__name__)))
    key = ("cd" * 32, 0.1318, "mock", "single")
    cache.client.set(cache.key(key), b"not json")
    assert cache.get(key) is None and errors == [("get", "JSONDecodeError")]

    redis_server.connected = False
    errors.clear()
    assert cache.get(key) is None
    cache.put(key, report_result(tools))  # never raises: the report was already written
    assert [op for op, _ in errors] == ["get", "put"] and errors[0][1] == "ConnectionError"


def test_unreachable_report_cache_still_writes_the_report(make_redis, redis_server, tools):
    redis_server.connected = False
    llm = MockLLMClient([MockLLMClient.final(fallback_report(tools).model_dump())])
    result = generate_report_for_image(tools.model_service, tools.image, llm=llm, cache=RedisReportCache(make_redis()))
    assert result.source == "llm" and not result.cached and len(llm.calls) == 1


# ---- The Streamlit app is untouched ----

def test_streamlit_app_keeps_the_in_memory_cache_and_needs_no_redis():
    assert isinstance(REPORT_CACHE, ReportCache)
    assert generate_report_for_image.__defaults__[2] is REPORT_CACHE  # cache=REPORT_CACHE
    code = ("import sys, agent.agent, ui.dashboard, services.model_handler; "
            "print('redis' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"
