"""MCP server (mcp_server/): tools, resources and path/image checks through the SDK's in-memory client.

No Groq, no Supabase, no network: the model is test_api's BrightnessModel (bright image -> confident
Normal, dark -> uncertain Tuberculosis) and reports come from MockLLMClient. The stdio test at the end
starts the real server process with the real checkpoint.
"""
import io
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
from mcp import Client, StdioServerParameters
from PIL import Image

from agent.agent import MockLLMClient, ReportCache
from mcp_server.server import create_server, load_llm_key
from test_api import BrightnessModel, multi_report_llm, png_bytes, report_script, model_service  # noqa: F401

REPO = Path(__file__).resolve().parent.parent
TOOLS = {"classify_xray", "gradcam_summary", "generate_report", "get_trace", "eval_summary"}


@pytest.fixture
def xrays(tmp_path):
    """Allowed directory with a bright (Normal) and a dark (Tuberculosis, refer) PNG, plus a sibling outside it."""
    allowed = tmp_path / "xrays"
    allowed.mkdir()
    (allowed / "bright.png").write_bytes(png_bytes(255))
    (allowed / "dark.png").write_bytes(png_bytes(0))
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.png").write_bytes(png_bytes(255))
    return allowed


@pytest.fixture
def make_server(model_service, xrays, tmp_path):  # noqa: F811
    def make(llm_factory=lambda: None, **options):
        options = {"model_service": model_service, "llm_factory": llm_factory, "report_cache": ReportCache(),
                   "allowed_dirs": [xrays], "trace_dir": tmp_path / "traces" / "mcp", **options}
        return create_server(**options)
    return make


async def call(client, tool, **args):
    """(is_error, structured result or error text)."""
    result = await client.call_tool(tool, args)
    if result.is_error:
        return True, result.content[0].text
    return False, result.structured_content


def assert_safe_error(text):
    assert "Traceback" not in text and 'File "' not in text


# ---- Listing ----

@pytest.mark.asyncio
async def test_lists_tools_with_descriptions_and_output_schemas(make_server):
    async with Client(make_server()) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
    assert set(tools) == TOOLS
    for tool in tools.values():
        assert len(tool.description) > 100, tool.name  # the calling model reads these
        assert tool.output_schema and tool.output_schema["type"] == "object", tool.name
    for name in ("classify_xray", "gradcam_summary", "generate_report"):
        assert tools[name].input_schema["required"] == ["image_path"]
    assert "ctx" not in tools["generate_report"].input_schema["properties"]
    assert tools["eval_summary"].input_schema["properties"]["run_id"]["default"] == "latest"
    assert tools["classify_xray"].annotations.read_only_hint is True
    assert tools["generate_report"].annotations.open_world_hint is True  # calls Groq


@pytest.mark.asyncio
async def test_lists_resources(make_server):
    async with Client(make_server()) as client:
        resources = {str(r.uri) for r in (await client.list_resources()).resources}
        templates = {t.uri_template for t in (await client.list_resource_templates()).resource_templates}
    assert resources == {"clinical://condition-info", "clinical://evals/latest"}
    assert templates == {"clinical://condition-info/{condition}"}


# ---- classify_xray ----

@pytest.mark.asyncio
async def test_classify_confident_and_refer(make_server, xrays):
    async with Client(make_server()) as client:
        error, normal = await call(client, "classify_xray", image_path="bright.png")
        assert not error, normal
        assert normal["predicted_class"] == "Normal" and normal["status"] == "confident"
        assert normal["referral_message"] is None and normal["calibrated"] is True
        assert normal["calibrated_probability"] == round(normal["calibrated_probability"], 4)
        assert set(normal["probabilities"]) == {"Normal", "Pneumonia", "Tuberculosis"}
        uuid.UUID(normal["trace_id"])

        error, tb = await call(client, "classify_xray", image_path=str(xrays / "dark.png"))  # absolute works too
        assert not error and tb["predicted_class"] == "Tuberculosis" and tb["status"] == "refer"
        assert tb["referral_message"] == "refer: uncertain, clinician review recommended"

        _, again = await call(client, "classify_xray", image_path="bright.png")
        assert again["trace_id"] != normal["trace_id"]  # every classify is a new prediction


# ---- Path and image checks ----

@pytest.mark.asyncio
@pytest.mark.parametrize("path, message", [
    ("../outside/secret.png", "outside the allowed image directories"),
    ("sub/../../outside/secret.png", "outside the allowed image directories"),
    ("OUTSIDE_ABS", "outside the allowed image directories"),
    (r"\\attacker.example\share\x.png", "Network and device paths"),
    ("//attacker.example/share/x.png", "Network and device paths"),
    (r"\\?\C:\Windows\x.png", "Network and device paths"),
    ("bright.png:hidden", "must not contain ':'"),
    ("missing.png", "No such file"),
    ("", "empty"),
    ("a" * 2000 + ".png", "not a valid file path"),
])
async def test_paths_outside_or_unsafe_are_rejected(make_server, xrays, path, message):
    if path == "OUTSIDE_ABS":
        path = str(xrays.parent / "outside" / "secret.png")
    (xrays / "sub").mkdir()  # POSIX resolves 'sub' before applying '..'
    async with Client(make_server()) as client:
        error, text = await call(client, "classify_xray", image_path=path)
    assert error and message in text, text
    assert_safe_error(text)


@pytest.mark.asyncio
async def test_symlink_out_of_the_allowed_directory_is_rejected(make_server, xrays):
    link = xrays / "link.png"
    try:
        os.symlink(xrays.parent / "outside" / "secret.png", link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks need developer mode or admin rights on Windows")
    async with Client(make_server()) as client:
        error, text = await call(client, "classify_xray", image_path="link.png")
    assert error and "outside the allowed image directories" in text


def gif_bytes():
    buf = io.BytesIO()
    Image.new("L", (8, 8)).save(buf, format="GIF")
    return buf.getvalue()


@pytest.mark.asyncio
@pytest.mark.parametrize("name, data, message", [
    ("notes.txt", png_bytes(255), "Only .jpg, .jpeg and .png files are accepted"),
    ("disguised.png", gif_bytes(), "Only JPEG and PNG images are accepted"),
    ("garbage.jpg", b"\xff\xd8 not really a jpeg", "not a readable JPEG or PNG image"),
    ("empty.png", b"", "Empty file"),
    ("big.png", png_bytes(255, size=(600, 600), mode="RGB") + b"\0" * 3000, "File too large (limit"),
])
async def test_files_that_are_not_acceptable_images_are_rejected(make_server, xrays, name, data, message):
    (xrays / name).write_bytes(data)
    async with Client(make_server(max_upload_bytes=2048)) as client:
        error, text = await call(client, "classify_xray", image_path=name)
    assert error and message in text, text


@pytest.mark.asyncio
async def test_a_directory_is_not_an_image(make_server, xrays):
    (xrays / "folder.png").mkdir()
    async with Client(make_server()) as client:
        error, text = await call(client, "classify_xray", image_path="folder.png")
    assert error and "Not a file" in text


@pytest.mark.asyncio
async def test_no_allowed_directory_is_a_clear_error(make_server):
    async with Client(make_server(allowed_dirs=[])) as client:
        error, text = await call(client, "classify_xray", image_path="bright.png")
    assert error and "MCP_ALLOWED_DIRS" in text


@pytest.mark.asyncio
async def test_unexpected_errors_show_only_the_exception_type(make_server, model_service, caplog):  # noqa: F811
    def broken(image, tracer=None):
        raise RuntimeError("connection string postgres://admin:hunter2@db")
    model_service.predict = broken
    async with Client(make_server()) as client:
        error, text = await call(client, "classify_xray", image_path="bright.png")
    assert error and "Internal error (RuntimeError)" in text
    assert "hunter2" not in text
    assert_safe_error(text)
    assert "hunter2" in caplog.text  # the details stay in the server's own log (stderr)


# ---- gradcam_summary ----

@pytest.mark.asyncio
async def test_gradcam_summary_joins_the_latest_prediction_trace(make_server):
    async with Client(make_server()) as client:
        _, pred = await call(client, "classify_xray", image_path="dark.png")
        error, cam = await call(client, "gradcam_summary", image_path="dark.png")
    assert not error, cam
    assert cam["trace_id"] == pred["trace_id"] and cam["explained_class"] == "Tuberculosis"
    assert len(cam["cam"]) == 7 and all(len(row) == 7 for row in cam["cam"])
    assert cam["focus"] == "diffuse" and abs(sum(cam["region_shares"].values()) - 1) < 0.01  # fake map is uniform
    assert cam["border_attention"] is False and "not a lung segmentation" in cam["caveat"]


@pytest.mark.asyncio
async def test_gradcam_summary_classifies_an_image_it_has_not_seen(make_server):
    async with Client(make_server()) as client:
        error, cam = await call(client, "gradcam_summary", image_path="bright.png")
        _, trace = await call(client, "get_trace", trace_id=cam["trace_id"])
    assert not error and cam["explained_class"] == "Normal"
    assert [s["name"] for s in trace["spans"]][:1] == ["prediction"]


# ---- generate_report (every test runs with both report pipelines) ----

def truth_for(prediction):
    return {"predicted_class": prediction["predicted_class"], "confidence": prediction["calibrated_probability"],
            "status": prediction["status"]}


@pytest.fixture(params=["single", "multi"])
def pipeline(request, monkeypatch):
    monkeypatch.setenv("REPORT_PIPELINE", request.param)
    return request.param


def report_llm(pipeline, script):
    """The single agent's mock script as that pipeline (test_api.multi_report_llm for multi)."""
    return MockLLMClient(script) if pipeline == "single" else multi_report_llm(script)


@pytest.mark.asyncio
async def test_report_by_llm_is_guarded_cached_and_traced(make_server, pipeline):
    llms = []

    def llm_factory():
        llms.append(report_llm(pipeline, report_script(truth)))
        return llms[-1]

    async with Client(make_server(llm_factory=llm_factory)) as client:
        _, pred = await call(client, "classify_xray", image_path="bright.png")
        truth = truth_for(pred)
        progress = []

        async def on_progress(value, total, message):
            progress.append(message)

        result = await client.call_tool("generate_report", {"image_path": "bright.png"}, progress_callback=on_progress)
        first = result.structured_content
        _, second = await call(client, "generate_report", image_path="bright.png")
        _, trace = await call(client, "get_trace", trace_id=pred["trace_id"])

    assert not result.is_error, result.content
    assert first["source"] == "llm" and first["model"] == "mock" and first["cached"] is False
    assert first["written_by"].startswith("Written by mock") and first["fallback_reason"] is None
    assert first["trace_id"] == pred["trace_id"] and first["guard_rejections"] == 0
    assert first["tool_calls"] == ["run_classifier", "get_gradcam", "get_condition_info"]
    assert first["report"]["calibrated_probability"] == pred["calibrated_probability"]
    assert first["report"]["disclaimer"].startswith("This report was generated by an AI system")
    assert progress and progress[-1] == "Report ready"
    assert first["pipeline"] == second["pipeline"] == pipeline and first["pipeline_note"] is None
    assert first["revisions"] == 0
    calls = llms[1].calls if pipeline == "single" else [c for x in llms[1].clients.values() for c in x.calls]
    assert second["cached"] is True and second["report"] == first["report"] and len(calls) == 0
    names = [s["name"] for s in trace["spans"]]
    assert names[0] == "prediction" and {"report", "llm_call", "tool_call", "guard", "report_cache_hit"} <= set(names)
    report_span = next(s for s in trace["spans"] if s["name"] == "report")
    assert report_span["attributes"]["pipeline"] == pipeline
    if pipeline == "multi":
        assert {"evidence_agent", "writer_agent", "verifier_agent", "handoff", "final_guard"} <= set(names)


@pytest.mark.asyncio
async def test_report_rejected_twice_by_the_guard_uses_the_template(make_server, pipeline):
    def llm_factory():
        wrong = report_script({**truth, "confidence": 0.5})
        return report_llm(pipeline, wrong + [wrong[-1]])

    async with Client(make_server(llm_factory=llm_factory)) as client:
        _, pred = await call(client, "classify_xray", image_path="bright.png")
        truth = truth_for(pred)
        _, report = await call(client, "generate_report", image_path="bright.png")
    assert report["source"] == "template" and report["model"] is None and report["guard_rejections"] == 2
    assert report["fallback_reason"].startswith("Report rejected") and report["pipeline"] == pipeline
    assert report["written_by"] == "Standard template report generated from the classifier's output."
    assert report["report"]["calibrated_probability"] == pred["calibrated_probability"]


@pytest.mark.asyncio
async def test_report_without_an_llm_key_uses_the_template(make_server, pipeline, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr("agent.llm_config._secret", lambda section, key: None)
    async with Client(make_server(llm_factory=None)) as client:  # the real build_report_llm, no key
        error, report = await call(client, "generate_report", image_path="dark.png")
    assert not error and report["source"] == "template" and report["pipeline"] == "single"  # nothing else ran
    assert report["fallback_reason"].startswith("No LLM configured")
    assert report["report"]["status"] == "refer" and "clinician review" in report["report"]["recommendation"].lower()


@pytest.mark.asyncio
async def test_llm_errors_fall_back_without_leaking_the_provider_error(make_server, pipeline):
    def boom(messages):
        raise RuntimeError("401 invalid api key gsk_live_abcdef123456")

    async with Client(make_server(llm_factory=lambda: report_llm(pipeline, [boom]))) as client:
        result = await client.call_tool("generate_report", {"image_path": "bright.png"})
    report = result.structured_content
    assert not result.is_error and report["source"] == "template" and report["pipeline"] == pipeline
    prefix = "" if pipeline == "single" else "evidence agent: "
    assert report["fallback_reason"] == f"{prefix}LLM call failed (RuntimeError)."
    everything = json.dumps(report) + "".join(getattr(c, "text", "") for c in result.content)
    assert "gsk_live" not in everything and "error_detail" not in everything


# ---- get_trace ----

@pytest.mark.asyncio
async def test_get_trace_errors(make_server):
    async with Client(make_server()) as client:
        error, text = await call(client, "get_trace", trace_id="not-a-uuid")
        assert error and "must be a UUID" in text
        error, text = await call(client, "get_trace", trace_id=str(uuid.uuid4()))
        assert error and "No trace with that id" in text


@pytest.mark.asyncio
async def test_get_trace_never_reads_the_apps_shared_traces(make_server, tmp_path):
    shared = tmp_path / "traces"  # the Streamlit app's folder: holds other sessions' spans
    shared.mkdir(parents=True, exist_ok=True)
    other = str(uuid.uuid4())
    (shared / "2026-10-26.jsonl").write_text(json.dumps({"trace_id": other, "name": "prediction"}) + "\n")
    async with Client(make_server(trace_dir=shared / "mcp")) as client:
        error, text = await call(client, "get_trace", trace_id=other)
    assert error and "No trace with that id" in text


# ---- eval_summary ----

@pytest.mark.asyncio
async def test_eval_summary_matches_the_committed_baseline(make_server):
    baseline = json.loads((REPO / "evals" / "baseline.json").read_text(encoding="utf-8"))
    async with Client(make_server()) as client:
        error, latest = await call(client, "eval_summary")
        _, named = await call(client, "eval_summary", run_id=Path(baseline["run"]).stem)
    assert not error, latest
    assert named["run_id"] == Path(baseline["run"]).stem and named["is_baseline_run"] is True
    assert named["below_baseline"] == [] and named["dataset_hash"] == baseline["dataset_hash"]
    assert {c["check"]: c["rate"] for c in named["deterministic_checks"]} == baseline["deterministic_pass_rates"]
    assert all(c["baseline_rate"] == c["rate"] for c in named["deterministic_checks"])
    assert set(named["judge_mean_scores"]) == {"faithfulness", "uncertainty_language", "actionability", "plain_language"}
    latest_file = sorted((REPO / "evals" / "runs").glob("*.jsonl"))[-1]
    assert latest["run_id"] == latest_file.stem


@pytest.mark.asyncio
@pytest.mark.parametrize("run_id, message", [
    ("../baseline", "must be 'latest' or a run id"),
    ("20261020T075220Z/../../x", "must be 'latest' or a run id"),
    ("20990101T000000Z", "No eval run 20990101T000000Z. Available: "),
])
async def test_eval_summary_rejects_bad_run_ids(make_server, run_id, message):
    async with Client(make_server()) as client:
        error, text = await call(client, "eval_summary", run_id=run_id)
    assert error and message in text


# ---- Resources ----

def resource_text(result):
    return result.contents[0].text


@pytest.mark.asyncio
async def test_condition_info_resources_carry_the_draft_review_status(make_server):
    async with Client(make_server()) as client:
        everything = json.loads(resource_text(await client.read_resource("clinical://condition-info")))
        tb = json.loads(resource_text(await client.read_resource("clinical://condition-info/Tuberculosis")))
        with pytest.raises(Exception, match="Unknown condition"):
            await client.read_resource("clinical://condition-info/Cancer")
    assert everything["_meta"]["review_status"].startswith("DRAFT")
    assert tb["condition"] == "Tuberculosis" and tb["review_status"].startswith("DRAFT")
    assert tb["model_limitations"] == everything["_model_limitations"]


@pytest.mark.asyncio
async def test_latest_eval_results_resource(make_server):
    newest = sorted((REPO / "evals" / "results").glob("20*.md"))[-1]
    async with Client(make_server()) as client:
        text = resource_text(await client.read_resource("clinical://evals/latest"))
    assert text == newest.read_text(encoding="utf-8") and text.startswith("# Eval results")


# ---- Secrets ----

def test_llm_key_comes_from_the_provider_section_only(tmp_path, monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    secrets = tmp_path / "secrets.toml"
    secrets.write_text('[supabase]\nurl = "https://x.supabase.co"\nkey = "sb_secret_never"\n'
                       '[groq]\napi_key = "gsk_from_file"\n', encoding="utf-8")
    env = {}
    load_llm_key(secrets, env)
    assert env == {"GROQ_API_KEY": "gsk_from_file"}  # nothing from [supabase]
    env = {"GROQ_API_KEY": "gsk_from_env"}
    load_llm_key(secrets, env)
    assert env == {"GROQ_API_KEY": "gsk_from_env"}  # the environment wins


def test_the_server_never_touches_supabase():
    """No Supabase client is imported and no Supabase setting is read, so no key (service-role or not) is accepted."""
    for path in (REPO / "mcp_server").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "import supabase" not in source and "SUPABASE_" not in source and "[supabase]" not in source, path.name
    code = ("import sys; import mcp_server.server, services.model_handler, evals.run; "
            "print(sorted(m for m in sys.modules if m.split('.')[0] in ('supabase', 'postgrest', 'gotrue')))")
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"


def test_starting_the_server_and_listing_tools_loads_no_torch():
    """Clients wait for initialize before they can see any tool: agent/ (torch, cv2) loads on first use."""
    code = ("import asyncio, sys; from mcp import Client; from mcp_server.server import create_server\n"
            "async def main():\n"
            "    async with Client(create_server()) as c:\n"
            "        await c.list_tools(); await c.list_resource_templates()\n"
            "asyncio.run(main())\n"
            "print(sorted(m for m in ('torch', 'cv2', 'matplotlib', 'streamlit', 'openai') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"


# ---- Real process over stdio, real model ----

@pytest.mark.asyncio
async def test_stdio_server_with_the_real_model(tmp_path):
    """Starts `python <repo>/mcp_server` from another directory, like an MCP client, and classifies the
    committed test-split X-ray. Nothing but protocol may reach stdout, or the client would fail to parse it."""
    if not (REPO / "best_student_model_aligned_final.pth").exists():
        pytest.skip("checkpoint not present")
    # Merged over the SDK's minimal default environment, as a desktop MCP client would start it
    env = {"MCP_ALLOWED_DIRS": str(REPO / "scripts"), "MCP_TRACE_DIR": str(tmp_path / "traces")}
    params = StdioServerParameters(command=sys.executable, args=[str(REPO / "mcp_server")], env=env, cwd=str(tmp_path))
    async with Client(params, read_timeout_seconds=300) as client:
        assert {t.name for t in (await client.list_tools()).tools} == TOOLS
        error, pred = await call(client, "classify_xray", image_path="smoke_xray.jpg")
        assert not error, pred
        assert pred["predicted_class"] in {"Normal", "Pneumonia", "Tuberculosis"} and pred["calibrated"] is True
        error, text = await call(client, "classify_xray", image_path=str(REPO / "README.md"))
        assert error and "outside the allowed image directories" in text
