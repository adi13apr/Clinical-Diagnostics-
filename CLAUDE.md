# CLAUDE.md

Guidance for working in this repository.

## What this is

A Streamlit app ("Clinical Diagnostic Portal") where a logged-in radiologist uploads a chest X-ray and gets a 3-class prediction (**Normal / Pneumonia / Tuberculosis**) from a distilled EfficientNet-B0. Probabilities are temperature-calibrated, and each prediction is triaged as `confident` or `refer` (clinician review). The app also shows a Grad-CAM overlay and produces a PDF report. An optional LLM tool-calling agent (Groq `openai/gpt-oss-120b` by default, Gemini as an option) writes a plain-language report; it explains the classifier's output and can never change it. Supabase handles auth and stores two things: the per-patient diagnosis history and a per-user prediction log that feeds the analytics page. The app is deployed on Streamlit Community Cloud.

## Commands

```bash
python -m venv venv
venv\Scripts\activate              # Windows; macOS/Linux: source venv/bin/activate
pip install -r requirements.txt    # app only (what Streamlit Cloud installs)
pip install -r requirements-dev.txt  # + API deps + pytest
streamlit run app.py
uvicorn api.main:create_app --factory --port 8000   # FastAPI backend; docs at /docs
docker compose up --build -d       # nginx + 2 API replicas + report worker + Redis (queue mode); settings from .env (see .env.example)
docker compose up -d --scale worker=3 --no-recreate   # more report workers (api stays at API_REPLICAS, default 2)
python scripts/multi_replica_demo.py --jobs 3   # against the running stack: shared state + shared Groq budget; writes demo_results/
python -m worker.report_worker     # report worker (Linux/Docker only); needs REDIS_URL
docker build -t clinical-api .     # API image (port 7860); see "Deployment"
.\scripts\serve_local_api.ps1      # PowerShell: API container + Cloudflare quick tunnel; updates Vercel if the URL changed
.\scripts\deploy_space.ps1 -Space <user>/<space> -DockerTest   # clean snapshot, build, check /health (-Push deploys to HF)
python scripts/smoke_test.py       # live API check; needs API_URL and SUPABASE_ACCESS_TOKEN
pip install -r requirements-mcp.txt   # MCP server (also pulled in by requirements-dev.txt)
python mcp_server                  # MCP server over stdio (what clients run: venv python + the folder path)
python .claude/skills/release-check/release_check.py   # pytest + eval replay + smoke test, PASS/FAIL/SKIP table
python .claude/skills/failure-to-regression/reproduce.py --case c07   # one failure: checks, guard verdict, report
pytest                             # no Supabase, dataset or GPU needed
$env:INTEGRATION_REDIS_URL="redis://127.0.0.1:6380/15"; pytest tests/test_integration_redis.py   # real Redis; FLUSHES that DB
python scripts/calibrate.py        # re-fit temperature; needs the datasets in data/ (below)
python -m evals.run --replay       # re-score the committed eval run (no network; what CI runs)
python -m evals.run --live --resume  # continue the live eval (real model + LLM; needs data/ and a key)
python -m evals.benchmark --simulated  # old (sync) vs new (async) runner with fake-latency clients
cd web && npm install && npm run dev   # React UI on :5173 (needs web/.env.local and the API)
cd web && npm run lint && npm run build
cd web && npx vercel deploy --prod   # React app to Vercel (project clinical-diagnostics)
```

CI (`.github/workflows/ci.yml`, on push and PR) installs CPU torch and `requirements-dev.txt` (which includes `requirements-mcp.txt`), then runs `pytest` (including `tests/test_mcp_server.py`) and `python -m evals.run --replay`. The `integration` job runs `tests/test_integration_redis.py` against a `redis:7` service container (`INTEGRATION_REDIS_URL=redis://localhost:6379/15`; Groq and Supabase still faked). That module skips itself without the variable, so the unit job shows it as skipped. A third job (`web`, Node 24) runs `npm ci`, `npm run lint` and `npm run build` in `web/`.

The app needs `.streamlit/secrets.toml` (gitignored; never commit it):

```toml
[supabase]
url = "..."
key = "..."   # anon/publishable key, NEVER the service_role/secret key (it skips RLS)

[groq]            # default provider (config/llm.json); without a key "Generate report" uses the template
api_key = "..."   # or env GROQ_API_KEY
# report_model = "openai/gpt-oss-120b"   # optional overrides of config/llm.json
# judge_model = "qwen/qwen3.8-27b"

[gemini]          # only if LLM_PROVIDER=gemini (or "provider" in config/llm.json)
api_key = "..."   # or env GEMINI_API_KEY
```

Database setup: run the files in `sql/` in order, in the Supabase SQL Editor, **before** deploying code that depends on them:
- `002` adds `status` / `temperature` to `predictions`. Without it every prediction-log insert fails, which the app reports as a warning only.
- `003` adds `status` to `diagnosis_history`. Without it every history insert fails: the single tab shows an error and batch rows show "Error".
- `004` creates `traces` and adds `predictions.trace_id`. Without it every prediction-log insert fails (warning only), and spans go to local `traces/` only.
- `005` replaces `diagnosis_history`'s old "any logged-in user can read every row" policies with owner-only ones (`radiologist_email` = the JWT's `email`). The table itself has no create migration in the repo; it already exists in the Supabase project.
- `006` adds a unique index on `traces (trace_id, span_id)` (after dropping exact duplicates). `SupabaseTraceStore.write` now upserts with `ON CONFLICT DO NOTHING`, so **without 006 every trace write fails**, in the Streamlit app and the API alike (warning only; spans stay in local JSONL). No grant or policy change: verified on Postgres 17 + PostgREST 16.5 with `authenticated` holding only INSERT/SELECT (first write, rewrite, duplicates in one batch, 8 concurrent writers, another user's conflicting span skipped, `WITH CHECK` still rejecting a forged `user_id`).

## Layout

```
app.py                      Entry point: builds services, routes login vs dashboard via st.session_state.user
api/                        FastAPI backend for the React UI (see "API"); reuses services/ and agent/
  main.py                   create_app(...) factory: endpoints, CORS, upload checks, blocking work in threads
  auth.py                   SupabaseJWTVerifier (JWKS / HS256 secret / ask Supabase), user_db_factory (PostgREST as the user)
  state.py                  UploadStore (in memory) / RedisUploadStore (shared via REDIS_URL): uploads by trace_id, per owner;
                            SerializedModel (one model call at a time)
  schemas.py                Pydantic request/response models
  settings.py               Settings.from_env(): env vars, else .streamlit/secrets.toml [supabase]
  jobs.py                   ReportJobs (RQ queue: submit/get/flush_spans), RedisSpanBuffer, write_report (both modes)
worker/report_worker.py     RQ job run_report_job + main(): loads the model once, SimpleWorker on queue "reports"
mcp_server/                 MCP server (stdio, SDK mcp 2.x MCPServer); see "MCP server"
  server.py                 create_server(...) factory, Clinic (lazy model, recent predictions), tools, resources, main()
  paths.py                  MCP_ALLOWED_DIRS, resolve_image_path (UNC/device/ADS refused, containment after resolve)
  schemas.py                Pydantic tool outputs (the SDK publishes them as outputSchema)
  __main__.py               `python <repo>/mcp_server` from any cwd (swaps the package dir on sys.path for the repo)
requirements-mcp.txt        requirements.txt + mcp>=2.3,<3
.claude/skills/             Claude Code Agent Skills (see "Agent Skills")
  failure-to-regression/    SKILL.md + reproduce, add_case, verify, live_eval, compare_runs (+ _repo paths)
  release-check/            SKILL.md + release_check.py
Dockerfile, .dockerignore   API image (CPU torch, uid 1000, port 7860); .dockerignore keeps .streamlit/ out
docker-compose.yml          redis (no persistence, healthcheck) + api x2 + worker x N (same image) + nginx in front (queue mode)
deploy/nginx/nginx.conf     Round-robin over the api replicas (Docker DNS, `resolve`); adds X-Upstream-Addr
.env.example                Placeholders for .env (gitignored): Supabase, Groq, ports, replica counts, demo login
requirements-api.txt        requirements.txt + fastapi, uvicorn, python-multipart, PyJWT[crypto], redis, rq
web/                        React UI (plain JSX) for the API; see "Web UI" and web/README.md
services/                   Non-UI logic. Each module exposes one class (calibration.py: functions).
  model_handler.py          CLAHETransform, StudentEfficientNet, ModelService (load + calibrated predict + heatmap)
  calibration.py            Temperature loading, apply_temperature, triage, ECE. No Streamlit import.
  image_validation.py       JPEG/PNG + size + pixel checks (ImageRejected.kind); shared by api/ (→ 400/413/415) and mcp_server/
  tracing.py                Tracer/Span, fail-safe helpers, JSONL + Supabase sinks, nominal cost, summarize_spans
  heatmap_handler.py        GradCAM on the last conv block + HeatmapGenerator (3-panel figure) + summarize_cam_regions
  prediction_handler.py     PredictionService: insert into `predictions`, read `prediction_daily_summary`
  data_handler.py           DataService: Supabase table `diagnosis_history` insert/select
  auth_handler.py           AuthService: Supabase email/password, sign-up, Google OAuth, logout
  pdf_handler.py            PDFReportGenerator (reportlab) → PDF bytes; referral box, triage row, optional AI report
agent/                      LLM report agent (see "Report agent")
  schema.py                 Pydantic Report + fixed DISCLAIMER
  tools.py                  ReportTools: run_classifier, get_gradcam, get_condition_info; TOOL_DECLARATIONS
  agent.py                  LLMClient / OpenAICompatibleClient / GeminiClient / MockLLMClient, tool loop, guard, fallback
  llm_config.py             Provider selection + keys from config/llm.json, st.secrets, env
  rate_limit.py             Token-aware per-model budgets (rpm, tpm, otpm, daily caps; sync + async): ModelBudget (memory),
                            RedisModelBudget (shared via REDIS_URL, API only), BudgetRegistry, budget_for()
  condition_info.json       Static condition text (DRAFT: needs clinical sign-off)
ui/                         Streamlit pages; functions named show_*(...)
  login_pages.py            show_login_ui(auth_service), handle_oauth_callback (Google sign-in return)
  privacy.py                Public privacy policy at ?page=privacy (linked from Google's consent screen)
  dashboard.py              show_dashboard(...) sidebar nav → show_diagnosis_page (single/batch/history);
                            run_prediction() = time + predict + log; show_status() = triage banner;
                            show_single_result() renders the last single run from session_state
  analytics.py              show_analytics (diagnosis_history charts) + show_prediction_analytics (summary view)
sql/
  001_predictions.sql       `predictions` table, RLS policies, `prediction_daily_summary` view
  002_prediction_status.sql `status` + `temperature` columns, `refer_count` in the view
  003_diagnosis_history_status.sql  `status` column on diagnosis_history
  004_traces.sql            `traces` table (one row per span, RLS) + predictions.trace_id
  005_diagnosis_history_owner_rls.sql  owner-only RLS on diagnosis_history (by email)
  006_traces_unique_span.sql  unique (trace_id, span_id) on traces, for idempotent span writes
config/
  pricing.json              NOMINAL per-model prices for cost estimates; billing_tier "free" → billed $0
  llm.json                  Provider (groq | gemini; env LLM_PROVIDER), report/judge models, per-model limits, request options
scripts/
  calibrate.py              Fits the temperature on the notebook's val split, reports on its test split
  failure_case.py           Renders docs/failure_case_gradcam.png (Grad-CAM of a misclassified test image)
  deploy_space.ps1          HF Docker Space deploy: history-free snapshot, Space README, *.pth in LFS, -DockerTest, -Push
  serve_local_api.ps1       API on this PC: container + cloudflared quick tunnel + Vercel VITE_API_URL update
  smoke_test.py             PASS/FAIL per endpoint against a deployed API (API_URL, SUPABASE_ACCESS_TOKEN)
  multi_replica_demo.py     Against the Compose stack: load balancing, shared uploads/jobs, shared Groq budget
  smoke_xray.jpg            13 KB test-split X-ray for the smoke test (Kermany et al., CC BY 4.0)
deploy/space/README.md      README for the HF Space only (front matter sdk: docker, app_port: 7860)
docs/
  failure_case_gradcam.png  README figure: pneumonia X-ray called Normal (50.15%, referred), attention on borders
splits/
  split_manifest.csv        The training notebook's exact train/val/test split (relpath,label,split)
notebooks/
  sta-project-ca2693.ipynb  Training notebook (Kaggle): data merge, split, multi-teacher distillation
outputs/                    Committed calibration results
  calibration.json          Temperature (read by the app at startup), test metrics, coverage table
  reliability.png           Reliability diagram before/after scaling
evals/                      Report-agent eval suite (see "Evals")
  cases.jsonl               36 versioned test-split cases (schema_version 1)
  rubric.md                 Judge + human rubric (faithfulness, uncertainty_language, actionability, plain_language)
  runs/*.jsonl              Raw live-run records (committed; replayed by CI)
  results/<date>.md         Run summaries
  baseline.json             Deterministic pass rates CI must not drop below
  human_labels.csv          Blind sheet for a HUMAN to score (never fill it in programmatically)
.github/workflows/ci.yml    pytest + eval replay
tests/                      pytest; conftest.py stubs the network so tests need no checkpoint or Supabase
  fixtures/                 Verbatim LLM outputs used as regression tests; an "expect" block makes
                            tests/test_regression_fixtures.py enforce one (guard verdict, failing checks)
data/                       Gitignored. Kaggle datasets + data/cache/calibration_logits.npz
traces/                     Gitignored. Span JSONL, one file per UTC day (ephemeral on Streamlit Cloud)
best_student_model_aligned_final.pth   state_dict for StudentEfficientNet (~16 MB, committed)
```

## Inference path

1. `ui/dashboard.py`: `st.file_uploader` (jpg/jpeg/png) → `PIL.Image.open` → `run_prediction(model_service, prediction_service, image)`. The single-diagnosis tab and the batch tab both go through it.
2. `ModelService.predict` (`services/model_handler.py`):
   - converts to RGB if needed;
   - transform: `Resize((224,224))` → `CLAHETransform(clip_limit=1.5, tile 8x8)` (CLAHE on the L channel in LAB space) → `ToTensor` → ImageNet `Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])`;
   - `StudentEfficientNet` forward returns `(logits, features)`, then `softmax(logits / T)`, then argmax.
3. Returns `{'diagnosis', 'confidence' (calibrated max, 0–1), 'probabilities' {class: calibrated 0–1}, 'status' ('confident'|'refer'), 'temperature'}`.
4. `run_prediction` logs the result to `predictions` (see below). The caller then calls `DataService.save_diagnosis` (written to `diagnosis_history`, with `status`). The single tab also calls `ModelService.generate_heatmap` and stores everything in `st.session_state` (`last_result`, `last_heatmap`, `last_report`, `last_upload_key`, …).
5. **Single tab rendering:** results are drawn by `show_single_result` from `session_state` whenever the current upload and patient match `last_upload_key`. Any button press reruns the script, and results drawn only inside the `if st.button("Run AI Diagnosis")` branch would vanish. That's why "Generate report" and the PDF download work.

**Model:** `torchvision.models.efficientnet_b0(weights=None)` wrapped as `self.model`, with `classifier[1]` replaced by `Linear(1280, 3)`. The checkpoint keys are prefixed `model.`, so the wrapper attribute name must stay `model`. `forward` also returns the backbone feature map, which was used for feature alignment during knowledge distillation; callers unpack `logits, _ = model(x)`.

**Class order:** `{0: 'Normal', 1: 'Pneumonia', 2: 'Tuberculosis'}`. This order is fixed by the trained weights and matches the notebook's `label_map` (`normal/pneumonia/tuberculosis`). Do not reorder it.

**Preprocessing must match training** (`val_test_transforms` in the notebook). Do not change it unless the model is retrained.

The model is loaded once through `@st.cache_resource` on `ModelService._load_model`. The first arg is named `_self` so Streamlit does not hash it.

## Calibration and triage

- The raw model is strongly **under**-confident; the training used focal loss γ=3, KD at T=3, and a TB class weight of 5. Raw max probabilities sit in class-specific bands: Normal ≤ 0.53, Pneumonia ≤ 0.69, TB ≈ 0.8. **Never threshold raw probabilities.**
- `outputs/calibration.json["temperature"]` (currently T ≈ 0.132) is loaded by `ModelService` at startup. If the file is missing, `T = 1`, `model_service.calibrated = False`, the dashboard shows a warning, and every prediction is `refer`. That is the intended fail-safe.
- `triage()`: calibrated max probability `>= REFER_THRESHOLD` (0.9) → `confident`, else `refer`. Constants (`REFER_THRESHOLD`, `STATUS_*`, `REFER_MESSAGE`) live in `services/calibration.py`. Import them; don't re-type the strings.
- Temperature scaling never changes the argmax, so accuracy and recall don't depend on T.
- `scripts/calibrate.py`:
  - loads the datasets via `splits/split_manifest.csv`, after verifying that every file exists and that the notebook's folders contain nothing extra;
  - fits T on **val**, reports on **test** only;
  - writes `outputs/`;
  - caches val+test logits, labels and splits in `data/cache/calibration_logits.npz`, which `tests/` also reads.
- **The split can't be rebuilt locally.** The notebook globbed files without sorting, so `train_test_split(random_state=42)` depends on Kaggle's filesystem order. The manifest was exported on Kaggle. Always use it.
- Datasets: `kaggle datasets download paultimothymooney/chest-xray-pneumonia -p data/pneumonia --unzip` and `tawsifurrahman/tuberculosis-tb-chest-xray-dataset -p data/tb --unzip`. The script auto-detects the roots and skips the nested copy / `__MACOSX`.
- Current test results ("paper's held-out test set", matches the paper's 97.61% accuracy): accuracy 97.61%, macro-F1 98.27%, ECE 39.5% → 1.2%. At 0.9, 1.7% of test images are referred, catching 11 of 24 errors.
- **Source confound:** every number above is in-distribution. Pneumonia is only from the pediatric JPEG dataset and TB only from the adult PNG dataset; Normal mixes both. The model may partly learn the source. Say so whenever these metrics are reported.

## Grad-CAM

- `GradCAM` hooks `model.model.features[-1]`, the last conv block (1280×7×7). It uses a forward hook plus a tensor hook on the output, because the block ends in an in-place SiLU and a full backward hook is fragile there. `generate_cam` enables gradients itself, so it works when called from inside `torch.no_grad()`.
- `ModelService.compute_gradcam(image, class_name)` returns the raw 7×7 map. `summarize_cam_regions(cam)` turns it into shares over a 3×2 grid of the whole image (patient right/left assumes a PA view; this is **not** a lung segmentation), the peak zone, `focus` (focal if one zone holds ≥ 40%, else diffuse), and `border_attention` (> 50% of the weight in the outer 10% frame).
- On real test images the maps are mostly diffuse, so any text about regions must hedge.

## Report agent (`agent/`)

- **Flow:** "Generate report" → `generate_report_for_image(model_service, image, prediction, build_llm_client())` → `ReportResult(report, source="llm"|"fallback", model, reason, steps, problems, tool_calls, rejected, error_detail, cached)`.
  - `rejected` holds the exact text of every LLM output the guard rejected; save real ones to `tests/fixtures/` as regression tests.
  - `reason` is short and user-facing. The raw provider error goes in `error_detail` and is never shown in the UI.
- **Cache:** `REPORT_CACHE` is a process-wide, thread-safe LRU of 256 entries, keyed by (SHA-256 of the decoded pixels + mode + size, temperature, model name).
  - The Streamlit app always uses it. The API uses `RedisReportCache` instead when `REDIS_URL` is set (same `get`/`put`, TTL 24 h, key `cdp:report:v1:{fingerprint}:{temperature!r}:{model}`).
  - Redis values are JSON (`report_result_to_json`: every `ReportResult` field, the `Report` via `model_dump`), never pickle. Reading re-validates the `Report`; a value that doesn't parse, or any Redis error, is a miss reported to `on_error` (the API logs it). Bump `RedisReportCache.VERSION` when the format changes.
  - `agent/agent.py` never imports `redis` (the client is passed in), so the Streamlit app needs no Redis package; `tests/test_shared_state.py` checks this.
  - Only `source == "llm"` results are cached. Fallbacks are free to rebuild, and caching one would block a later Gemini attempt.
  - Hits return `cached=True` with `steps=0`.
- **Providers** (`config/llm.json`, `build_llm_client(role)`): `groq` (default) uses `OpenAICompatibleClient` (Chat Completions + tools, `openai` SDK with `max_retries=0`). `gemini` uses `GeminiClient`. `LLM_PROVIDER` overrides the config. Keys come from `st.secrets[section]["api_key"]` or the env var named in the config.
  - Models (confirmed in Groq's docs, 2026-10-18): reports `openai/gpt-oss-120b` (Production; tool use, **no parallel tool calls**), judge `qwen/qwen3.8-27b` (**Preview**, "evaluation only"; strict `json_schema`).
  - Request options live in `config/llm.json`: gpt-oss `reasoning_effort: low`, `include_reasoning: false`, `max_completion_tokens: 600` (was 2000 until 2026-10-26; measured completion was 21–238 tokens per call and 266–400 per whole report, and the limiter reserves the cap on every call, see Limits). Qwen judge `reasoning_effort: none` (instruct mode), `reasoning_format: hidden`, `max_completion_tokens: 600`; with reasoning, a judgement used about 1.2k output tokens, over Qwen's 1000 output-tokens/min limit. Keys starting with `_` are comments, stripped by `request_options()`. Non-standard keys go through `extra_body` (`split_request_options`), because the openai SDK rejects unknown kwargs.
  - Usage maps to `output_tokens` (excluding reasoning) plus `thinking_tokens` (`completion_tokens_details.reasoning_tokens`), as with Gemini.
- **Request budget:** the prompt asks for each tool once (in parallel where supported); `get_condition_info` defaults to the predicted class. On Groq gpt-oss that's 3 tool turns plus the answer (4 calls, about 3.2k tokens); on Gemini, 2 calls.
  - Once every tool has been called, the next turn is sent with tool calls forbidden (`tools_allowed=False` → Groq `tool_choice: "none"`, Gemini function-calling mode NONE), so it can only be the answer.
- **Limits** (`agent/rate_limit.py`): each client holds its model's shared budget from a `BudgetRegistry`: `budget_for(model)` / `BUDGETS` (in memory) unless `build_llm_client(budgets=...)` gets another one.
  - Before a call it reserves an estimate (prompt chars / 3.5 + `max_completion_tokens`) and waits until requests/min and tokens/min in the last 60 s fit. After the call, the real `total_tokens` replaces the estimate. The prompt estimate runs 26–32% above Groq's real input count, and a call can't produce more than `max_completion_tokens`, so the reservation is an upper bound and the window can't go over the limit. **Never reserve less than prompt + cap**: a smaller guess could be exceeded by a legal reply.
  - **Deadline:** `acquire`/`aacquire(..., max_wait=)` raise `BudgetWaitTooLong` instead of sleeping past `max_wait` (a request that fits right away is always reserved). Clients take `LLMClient.deadline` (`time.monotonic()`, None by default) and pass `budget_wait_left()`; the error becomes `BudgetDeadlineError` (a `RateLimitError`, kind `rate_limit`), so the report falls back to the template without calling the LLM. Only report jobs set it (`write_report(..., deadline=)` from the worker).
  - **Cut-off replies:** a turn with `finish_reason` in `TRUNCATED_FINISH_REASONS` (`length`, Gemini `MAX_TOKENS`) is never replayed to the model. It is recorded in `rejected` (with `finish_reason`) and as an `llm_truncated` event, then retried once (shared `MAX_RETRIES`), then the template (`error_kind` `model`). `guard_replay_consistent` skips these drafts, since the guard didn't reject them.
  - Daily request/token caps raise `DailyCapReached`, which becomes `RateLimitError.daily`.
  - Groq free tier on this account: 1000 requests/day and 8000 tokens/min per model (from the `x-ratelimit-limit-*` headers); 30/min and 200k tokens/day from the docs. Daily counters reset at midnight UTC; in memory they are per process.
  - **Output tokens/min (`otpm`):** `qwen/qwen3.8-27b` also has an undocumented 1000 OTPM limit, found from a 429 "Request too large ... (OTPM): Limit 1000". Groq checks `max_completion_tokens` against it before running, so a request whose `max_completion_tokens` exceeds it can never succeed. `ModelBudget` reserves `max_completion_tokens` against `otpm`, then records the real `completion_tokens`.
  - **Two backends, one interface** (`BaseBudget`: `acquire`/`aacquire`/`slot`/`arecord`; subclasses implement `_try_reserve`, `record`, `used_last_minute`). `ModelBudget` keeps counters in memory: the Streamlit app and the eval runner always use it. `RedisModelBudget` keeps them in Redis, so API replicas share one budget per model. Only the API uses it, and only with `REDIS_URL` (`shared_state()` → `BudgetRegistry(client)` → `partial(build_llm_client, budgets=...)`). Nothing reads `REDIS_URL` implicitly.
  - **Redis keys** (`cdp:rate:{model}:…`; `{model}` is a cluster hash tag): sorted sets `requests` (id), `tokens` and `output` (`id|n`), scored by time, with a 120 s TTL; `day:YYYY-MM-DD:requests` / `:tokens` counters with `EXPIREAT` the next UTC midnight. A reservation is **one Lua script** (`RESERVE_SCRIPT`: prune, check every limit and daily cap, add), so two replicas can't both take the last tokens. `record` is another script; it corrects the reservation's own day, even after the request left the window, and leaves an expired day alone.
  - Scores come from `clock`, which defaults to `time.time` (wall clock: `time.monotonic` isn't comparable between processes); replica clocks only need to agree to well under a second. `slot()` (`max_concurrent`) and the asyncio.Lock stay **per process**. Async calls run the Redis round trips in `asyncio.to_thread`. A Redis error propagates, and the report falls back to the template (`infrastructure`) without calling the LLM.
  - Tests: `make_budget` (conftest) runs every budget test on both backends with `budget_clock`, one fake wall clock that also drives fakeredis's expiry. `tests/test_shared_budget.py` races 40 threads/tasks over two replicas (a non-atomic check-then-add version let 40 of 40 through a limit of 10), and covers record corrections and the midnight reset. Lua needs `fakeredis[lua]` (lupa). Checked by hand against Redis 7.4.11.
- **429 handling:** Gemini reads `RetryInfo`; Groq reads `retry-after` (else "try again in 1m2.5s" in the message). If the wait is under `RATE_LIMIT_MAX_RETRY_DELAY` (10 s), sleep and retry **once**; otherwise raise `RateLimitError` immediately, with the daily quota recognised from the message ("tokens per day (TPD)").
- **Error kinds** (`classify_llm_error`, stored in `ReportResult.error_kind`): `rate_limit`, `infrastructure` (network, auth, server, unknown), or `model`. A model failure is Groq's 400 `tool_use_failed`, a malformed tool call that gpt-oss produces intermittently. A model failure gets **one retry** (sharing `MAX_RETRIES` with guard rejections, and recorded in `rejected` with `text: None`) before falling back.
- **Tools** (`ReportTools`): the image and the dashboard's prediction are bound server-side. The LLM sees only tool outputs: no pixels, no patient name or email.
  - `run_classifier` rounds probabilities to 4 dp, so the report can copy them exactly.
  - `get_gradcam` covers the predicted class.
  - `get_condition_info` reads `condition_info.json` only; there are no web lookups. `condition` is optional and defaults to the predicted class. It also returns `model_limitations` (from `_model_limitations`), which holds the dataset/source-confound caveat. That caveat used to be required by the system prompt without being in any tool output, and the pilot judge flagged it as ungrounded. Report limitations must come from this tool; the template report uses them verbatim.
  - Tool exceptions are returned to the LLM as `{"error": ...}`.
- **Loop:** at most `MAX_STEPS = 5` LLM calls. The final text must be JSON → `Report.model_validate` → `check_grounding`. On failure the LLM gets the problem and `MAX_RETRIES = 1` more try. After that, or on any LLM exception, it falls back to `fallback_report`.
- **One loop, two drivers.** All decisions live in `_report_loop`, a generator that yields an `LLMRequest` and receives the call's outcome (an `LLMTurn` or the exception). `generate_report` drives it synchronously (the app). `agenerate_report` drives it with `await llm.agenerate(...)` (the eval runner; clients without `agenerate`, e.g. Gemini, are called inline, not in a thread). **Change agent behaviour only in `_report_loop`**, so both paths stay identical (`tests/test_async_eval.py` checks this).
- **Async clients:** `OpenAICompatibleClient.agenerate` and `OpenAICompatibleJudge.ajudge` share `_prepare`/`_parse` with their sync methods.
  - They use `AsyncOpenAI` (created lazily, `max_retries=0`), wait with `asyncio.sleep`, and reserve with `budget.aacquire`.
  - Each request holds `budget.slot()` (the model's `asyncio.Semaphore`, `max_concurrent` in `config/llm.json`, 1 by default) and is wrapped in `asyncio.wait_for(..., timeout_s)` (`request_timeout_s`, 120). The timeout covers the network request only, not rate-limit waits.
  - A timeout raises `TimeoutError`, which is classified `infrastructure`, so the eval stops and can be resumed.
- **`ModelBudget` serves both:** `_try_reserve` never sleeps. `acquire` loops with `time.sleep`; `aacquire` loops with `asyncio.sleep` under an `asyncio.Lock`. Sync and async share the same counters. asyncio primitives are kept per event loop (`_primitives`).
- **Grounding guard** (`check_grounding`):
  - `predicted_class`, `calibrated_probability` and `status` must `==` `run_classifier`'s output;
  - no assertion of a confirmed diagnosis (`asserts_confirmed_diagnosis`, using `BANNED_PATTERNS`). A match is allowed when a negation cue (not, no, never, without, cannot, n't, until, unless, before) appears in the 4 words before it within the same clause (clauses split on punctuation, "but", "however"). "No doubt" and "not only" don't count as negations. This was added after Gemini wrote "a screening aid, not a confirmed diagnosis" (fixture `gemini_rejected_negated_confirmed.json`);
  - a `refer` case's recommendation must contain "clinician review";
  - a Pneumonia/TB recommendation must not say no referral, action or follow-up is needed (`dismisses_abnormal_result` / `DISMISSIVE_NEXT_STEP`; Normal exempt). Added after the 2026-10-20 human review: the reviewer marked down `c32` ("No immediate referral is required" for 99.99% Pneumonia), the judge scored it 5/5, and it's now the fixture `gptoss_c32_dismissed_referral.json`. The system prompt states the rule too.
  - the limitations must quote the dataset/source-confound caveat (`get_condition_info`'s first `model_limitations` item) word for word, ignoring case, whitespace and hyphen variants (`includes_caveat`, also used by the eval check `dataset_caveat_present`). `check_grounding(report, truth, caveat)`: `_report_loop` reads the caveat from `tools` itself, so the rule holds even when the model skips `get_condition_info`; the rejection message quotes the caveat for the retry. Added after `c07` in `20261020T075220Z` skipped the tool and invented its limitations (judge faithfulness 5; fixture `gptoss_c07_missing_dataset_caveat.json`). Without `caveat` (e.g. `evals/judge_sensitivity.py`) the rule is skipped.
  - The disclaimer is always replaced with `schema.DISCLAIMER`.
- **`fallback_report`:** a deterministic template built only from tool outputs. It is always schema-valid and grounded (asserted), and it works even when Grad-CAM fails. Pneumonia/TB recommendations always end with `CLINICIAN_FOLLOW_UP` ("Arrange clinician follow-up for this result.").
- **`GeminiClient`:** `models.generate_content` with manual function calling. It is stateless: the history is resent each call, and nothing is stored server-side, unlike the Interactions API.
  - `FunctionDeclaration(parameters_json_schema=...)`, AFC disabled, `thinking_level="low"`, and **no temperature** (deprecated for Gemini 3.8).
  - Model turns are replayed as received, which keeps thought signatures. Each `FunctionResponse` carries the call's `id` and `name`.
  - The final answer is plain JSON text, not `response_json_schema`: combining structured output with tools wasn't confirmed in the docs.
- **Tests** use `MockLLMClient` (scripted `LLMTurn`s / callables). The Gemini and Groq wiring is tested offline with fake clients (`tests/test_providers.py` fakes `chat.completions.create` and builds real `openai` errors with `httpx` responses). Never call the network in tests.

## Tracing (`services/tracing.py`)

- **One trace per prediction.** The dashboard creates `make_tracer(trace_store)` on "Run AI Diagnosis" (and per batch image), stores `last_trace_id`, and reuses that id for "Generate report". A prediction and its report therefore share one `trace_id`, which is also written to `predictions.trace_id`.
- **Spans:** `prediction` → `preprocess` / `classify` / `calibrate` (inside `ModelService.predict`), `log_prediction`, `save_history`, `gradcam` (`generate_heatmap` and `compute_gradcam`), and `report` → `llm_call` / `tool_call` (→ `gradcam`) / `guard` / `fallback_used`. A cache hit records `report` → `report_cache_hit`.
  - `llm_call` attributes: model, step, `input_tokens`, `output_tokens`, `thinking_tokens`, `nominal_cost_usd`, `billed_cost_usd`, `finish_reason`, `tool_calls`, `rate_limit_retries`. On failure: status error plus `error_type`.
  - `guard`: zero-duration event, `result` pass/reject plus `reason`. A reject has status error and `error_type` "Rejected".
  - `fallback_used`: `used` bool plus `reason`, one per generated report.
- **Fail-safe rule:** app code only calls `trace_span`, `trace_event`, `safe_set` and `finish_trace`. These swallow and log tracer errors, while the step's own exceptions still propagate. Pass `tracer=None` to disable tracing. Never call `tracer.span()` directly from app code.
- **Sinks:** `finish()` writes new spans to every sink. A failing sink is logged and appended to `tracer.sink_errors`, never raised.
  - `JsonlTraceSink` writes `traces/YYYY-MM-DD.jsonl` and always runs.
  - `SupabaseTraceStore` writes the `traces` table and needs the logged-in auth client. It upserts with `ignore_duplicates` on `(trace_id, span_id)` (needs `sql/006`), so writing the same spans twice is harmless.
  - `RedisSpanBuffer` (`api/jobs.py`) is the report worker's sink: a Redis list the API later writes to Supabase as the user.
- **Cost:** `estimate_cost(model, input, output+thinking)` uses `config/pricing.json`. Prices are dated (`until`), because the intro price ends 2026-12-31. With `billing_tier: "free"`, billed cost is 0. Unpriced models (e.g. the mock) get `None`.
- **UI and analytics:**
  - "View trace" under the AI report shows `trace_table(spans)`.
  - Analytics `show_trace_analytics` reads Supabase, falling back to local JSONL, and uses `summarize_spans` for average latency per step (events excluded), tokens and nominal cost per LLM report, guard reject rate and fallback rate (cache hits excluded).
- **Measured (live, CPU, one report):** preprocess 130 ms, classify 47 ms, Grad-CAM figure 1.15 s (raw map 0.1 s), LLM calls 10.8 s + 3.5 s, about 2.2k input and 350 output tokens, about $0.003 nominal.

## Evals (`evals/`)

- **Dataset:** `python -m evals.dataset` builds `cases.jsonl` from the cached test-split logits. It includes every model error (24), every refer case (17; 11 overlap with errors), and 6 confident correct cases (2 per class; Normal takes one from each source), for 36 in total.
  - Each case snapshots the classifier output; live runs record `snapshot_match` to catch drift.
  - `load_cases` raises `SchemaVersionError` on a `schema_version` mismatch. `EvalCase` forbids unknown fields.
  - `dataset_hash` is computed over canonical JSON, independent of CRLF/LF, and is stored in every run record. Replay refuses records from a different hash.
  - Rebuilding needs `--force` and changes the hash, which means a new live run and a new baseline.
  - Category `regression`: a case added by hand after a failure (`add_case.py`, test split only). `build_cases(keep)` carries these over on a rebuild, renumbered after the built cases (`next_case_id`). Adding one changes the hash too.
- **Deterministic checks** (`checks.py`): schema valid, class/probability/status match the classifier, refer → "clinician review", no confirmed-diagnosis claim (`asserts_confirmed_diagnosis`), limitations present (1–5), length limits (summary 2–3 sentences and ≤ 600 chars, recommendation ≤ 500, each limitation ≤ 300), `guard_replay_consistent` (today's guard still rejects the recorded rejected drafts and accepts the final LLM report; it calls the guard through `guard_verdict(record)`, the one place the eval does, so a guard needing more tool outputs changes only there), and `abnormal_result_not_dismissed`. The last one fails if a Pneumonia/TB recommendation says no referral, action or follow-up is needed (`DISMISSIVE_NEXT_STEP`; tolerates Unicode hyphens and one word in between). Normal predictions are exempt, per the human reviewer. Pass rates exclude not-applicable cases (`None`).
  - `abnormal_result_not_dismissed` uses the same rule as the agent's guard (`agent.agent.dismisses_abnormal_result`). **Any guard change makes `guard_replay_consistent` fail** on recorded drafts the new guard would reject; that's intended. Re-run `--live` and `--update-baseline` after changing the guard.
  - `confident_not_called_uncertain` mirrors the guard's `labels_confident_as_refer`: only refer cases may use "Refer: uncertain…". An ambiguous prompt rule once made 5 of 8 confident Pneumonia/TB reports use it; see `evals/runs/20261020T071107Z.notes.md` and the fixture `gptoss_confident_called_uncertain.json`. **After any prompt change, re-run the eval before trusting it.**
  - `dataset_caveat_present`: the first `model_limitations` item must appear verbatim in the limitations, with whitespace and hyphens normalised (`includes_caveat`, the same rule as the guard). It failed on `c07` in `20261020T075220Z` (35/36), where gpt-oss skipped `get_condition_info` and invented limitations; the guard has enforced it since. `guard_verdict` passes the recorded `tool_outputs.condition_info.model_limitations[0]` to the guard (records without it are checked without the caveat). Run records include `tool_calls`.
- **Current baseline run:** `20261026T040807Z` (12 of 12 checks at 100%, 0 fallbacks; 2 drafts rejected by `labels_confident_as_refer` on `c35`/`c36` and fixed on retry; the caveat rule didn't fire, since no case skipped `get_condition_info`). The human labels belong to `20261018T161443Z`.
- **Judge** (`judge.py`, `build_judge()`): Groq `qwen/qwen3.8-27b`, a different model family from the report model to avoid self-preference. It uses strict `json_schema` (`JUDGE_JSON_SCHEMA`: closed objects, scores as an integer enum 1–5, and Pydantic validates the result). `GeminiJudge` (`response_schema`) is used when the provider is gemini. `rubric.md` is sent verbatim. The judge sees tool outputs and the report only, never the true label. `--judge-model` overrides the model.
- **Async runner (default; `--sync` = old):** `run_pipeline_async`, in one event loop, writes reports in order and puts each new record on a queue for a single judge coroutine, so judging case N overlaps the report for case N+1. No threads.
  - `run_pipeline_sync` is the original loop. Both take plain callables (`make_record`/`amake_record`, `judge`, `append`), so tests and `evals/benchmark.py` drive them with fake-latency clients.
  - **Same records:** `run_case`/`arun_case` share `_prepare_case`/`_finish_case`; only line order in the run file can differ, and the last record per case wins.
  - **Stops:** a report-side `StopRun` stops new reports, but queued records are still judged. A judge `RateLimitError` records the `judge_error` and stops; any report already in progress still gets appended, and resume judges it.
  - Records now include `judge.latency_ms`.
  - `--run-file PATH` writes a side run. `--limit` and `--run-file` runs never write `evals/results/<date>.md`; before this fix, `--limit` overwrote it.
- **Benchmark** (`evals/benchmark.py`): `--simulated` uses real timings scaled down 50× (report = baseline run mean, judge ≈ 1 s) and writes `evals/results/benchmark_simulated.md`. Result: **1.01×** at the real ratio (the judge's time is all that can overlap; the report model's own token limit is the bottleneck), 1.19× with a 15 s judge, 1.65× when both take equal time. `--live` runs 5 cases each way into scratch files, with a cooldown between them, and writes `benchmark_live.md`.
- **Live runs** (`run.py --live`):
  - Agent and judge come from `build_llm_client("report")` / `build_judge()` with per-model budgets from `config/llm.json`; a model used by both shares one budget. A google-genai `Client` closes its HTTP connection when garbage-collected, so always keep a reference to it.
  - Records are appended; the last record per case wins. `--resume` skips judged cases and re-judges unjudged ones.
  - A `rate_limit` or `infrastructure` failure in the agent step raises `StopRun` (retry later) instead of recording a template report as a result. A `model` failure (e.g. repeated `tool_use_failed`) is recorded as a real fallback outcome (`fallback_error_kind`).
  - Context for a run (setup changes, why) goes in `evals/runs/<run>.notes.md`; `build_summary` appends it as "Run notes".
  - Rate-limit stops print the resume command.
- **Quotas:** a full run is 36 × (about 4 gpt-oss calls + 1 Qwen call), roughly 115k + 105k tokens, which fits Groq's free daily limits (200k tokens/day per model) at about 1–2 cases/min under 8k tokens/min. Gemini's free tier (20 requests/day for `gemini-3.8-flash`) could not run it in under about 6 days, which is why the provider moved to Groq.
- **Replay and baseline:** `--replay` re-scores the baseline's run (or `--run`), prints the summary, and exits 1 if any deterministic pass rate is below `baseline.json`. `--update-baseline` rewrites it, marked PROVISIONAL while the run is partial. CI runs `--replay`.
- **Human validation:** `--export-human-labels` writes 15 blind rows (5 error-only, 5 refer, 5 correct; LLM-written preferred) with empty score columns. It needs ≥ 15 reports and refuses to overwrite a sheet that has scores. `--agreement` gives exact, within-1 and mean (judge − human) per criterion. **Only a human fills in the scores.**
  - **Result (2026-10-20, owner-scored):** 81.7% exact, 96.7% within 1 point, but both raters gave almost all 5s, so agreement is inflated. The judge gave 5/5 on faithfulness and actionability to the one report the human marked down (`c32`, "no referral needed" for 99.99% Pneumonia). Don't rely on the judge to catch problems; add deterministic checks for issues humans find.
  - **Labels belong to one run.** `evals/human_labels.meta.json` records the run the sheet was exported from (`labels_run()`). `load_agreement` returns `{"other_run": …}` for any other run, so summaries never compare human scores with a different run's judge, and `--agreement` defaults to the labels' run. The current labels are for `20261018T161443Z`.
- **Judge sensitivity** (`python -m evals.judge_sensitivity`, live): 5 deliberately broken reports (wrong class, missing referral, TB "no follow-up needed", overconfident wording, invented symptom), each built on real tool outputs from `20261018T161443Z`. It records whether the judge scores the broken criterion ≤ 2, and whether the guard blocks the report (offline). Writes `evals/judge_sensitivity.{json,md}`. The guard blocks 3 of 5 by construction; overclaiming and invented facts can only be caught by the judge or a human.

## Supabase

There is one client, `AuthService.client`, built with the anon/publishable key from `st.secrets`. It lives for one browser session (`st.session_state.auth_provider`) and carries the logged-in user's JWT after `login_email` or Google sign-in. `AuthService`, `PredictionService`, `SupabaseTraceStore` and `DataService` all use it, so RLS applies to `predictions`, `prediction_daily_summary`, `traces` and `diagnosis_history`.

- **Never use the service_role/secret key in the app.** It skips RLS. Until 2026-10 `DataService` built its own client with whatever `[supabase] key` held, and that was a service_role key, which was also committed in the first commit (`744ea86`) and is public in git history. It was rotated and the legacy keys disabled; history access now depends on `sql/005`.

- Login state for routing lives in `st.session_state.user` (the Supabase `User` object). The dashboard only uses `.email`.
- `PredictionService` and `DataService` **must** be given `auth_provider.client`. Their tables depend on RLS seeing the user's session.

### Google sign-in (Supabase OAuth, PKCE)

- **The button is a real link.** "Continue with Google" is an `st.link_button` pointing at `AuthService.google_sign_in_url(app_base_url())`: Supabase `/auth/v1/authorize?provider=google&code_challenge=…&redirect_to=<app>?oauth_state=<random>`. A plain `st.button` can't work, because the server can't redirect the browser.
- **The code verifier lives server-side.** Returning from Google starts a **new Streamlit session**, so the PKCE verifier can't live in `session_state` or in the Supabase client's memory (supabase-py's own `sign_in_with_oauth` stores it there). It is generated with `supabase_auth.helpers` and kept in the process-wide `PendingOAuthStore`, keyed by a random single-use `oauth_state` that rides in `redirect_to`. Entries expire after 10 min. This works because Streamlit Cloud runs one process; multiple replicas would need shared storage.
- **The callback:** `app.py` calls `handle_oauth_callback(auth_provider)` while logged out.
  - It reads `?code=&oauth_state=` (or `?error=`) and clears the query params right away, since the code is single-use.
  - Then it calls `complete_google_sign_in` → `exchange_code_for_session({"auth_code", "code_verifier"})` on this session's auth client, so RLS sees the new session.
  - Finally it sets `st.session_state.user` and reruns.
- **App URL:** `app_base_url()` uses `[supabase] redirect_url` if set, else `st.context.url`. If neither is known, the button is shown disabled instead of breaking the login page.
- **Run one server only.** The verifier is in the memory of the process that made the link, so the same process must handle the return. With two `streamlit run` processes (8501 and 8502), Google returns to whichever `redirect_url` names, and the sign-in fails.
- **Privacy page:** `app.py` serves `ui/privacy.py` for `?page=privacy` **before** building any service, so it needs no secrets or login. Google's Branding page needs this public URL to publish the app beyond test users. Keep its claims true to the code: images aren't stored, Groq gets tool outputs only, and history visibility is enforced by app filtering (not proven RLS).
- **Tests and secrets:** AppTest reads the developer's real `.streamlit/secrets.toml`, so auth UI tests monkeypatch `app_base_url` instead of relying on secrets being absent.
- **External setup** (not in code):
  - Supabase → Authentication → Providers → Google: enabled, with a Google Cloud OAuth client ID and secret.
  - Supabase → Authentication → URL Configuration → Redirect URLs: `https://clinical-diagnosis.streamlit.app/**` and `http://localhost:8501/**`. The `/**` is needed because `redirect_to` carries `?oauth_state=…`; without a match, Supabase falls back to the Site URL and drops the state.
  - Google Cloud OAuth client: authorized redirect URI `https://<project-ref>.supabase.co/auth/v1/callback`.

### `diagnosis_history` (per-patient record)

- Columns, as used in code: `radiologist_email`, `patient_name`, `prediction` (class name string), `confidence_score` (percent, 0–100, 2 dp), `status` (`confident`/`refer`, migration 003; NULL on older rows, shown as "—"), `created_at` (server default). Probabilities are **not** stored.
- `confidence_score` is calibrated from the calibration commit onward. Older rows hold raw, much lower values.
- It is read with `.eq("radiologist_email", email)` through the logged-in client. RLS (`sql/005`) enforces the same rule in the database: a user can insert and select only rows whose `radiologist_email` matches their JWT `email` (case-insensitive). There is no `user_id` column.

### `predictions` (prediction log)

- **When a row is written:** inside `run_prediction`, immediately after `model_service.predict` succeeds and before `save_diagnosis`. That gives one row per image, single or batch.
- **Columns:**
  - App sends `predicted_label`, `confidence` (calibrated, 0–1), `inference_ms` (int, `perf_counter` around `predict`), `status` (`confident`/`refer`), `temperature`, and `trace_id` (migration 004).
  - Server fills `id`, `user_id` (`default auth.uid()`), and `created_at`.
  - `status`/`temperature` are NULL on rows logged before migration 002; NULL `temperature` means `confidence` is raw.
- **RLS:** `authenticated` can insert and select only their own rows (`user_id = auth.uid()`).
- **Failure handling:** `log_prediction` is wrapped in `try/except` → `st.warning`, and the result is returned anyway.
- Patient name is deliberately not logged here.

### `prediction_daily_summary` (view)

- `security_invoker = true`, so RLS applies and each user sees only their own aggregates.
- It is grouped by UTC day and `predicted_label`. Columns: `day`, `predicted_label`, `prediction_count`, `avg_confidence`, `low_confidence_count` (`confidence < 0.6`), `avg_inference_ms`, `refer_count` (002).

## Analytics page (`📊 Analytics` in the sidebar)

1. `show_analytics(data_service, email)` → `diagnosis_history`. It shows metrics, pie, histogram, daily bars, a per-diagnosis table, and recent rows. A failed query raises and stops the page.
2. `show_prediction_analytics(prediction_service)` → `select * from prediction_daily_summary order by day desc`. It renders a stacked bar chart per day by label, plus the summary table (`refer_count` shown as "Referred"). A query error becomes `st.warning`.

## API (`api/`)

- **Run:** `uvicorn api.main:create_app --factory`. The factory takes injectable `model_service`, `verifier`, `db_factory`, `llm_factory`, `report_cache`, `uploads` and `report_jobs`; tests pass fakes, production builds them from `Settings`. The checkpoint is loaded in the lifespan (only when no `model_service` was injected), so importing `api.main` loads nothing.
- **Endpoints:** `POST /predict`, `GET /gradcam/{trace_id}`, `GET /gradcam/{trace_id}/map`, `POST /report/{trace_id}`, `GET /report/jobs/{job_id}`, `GET /history`, `GET /analytics`, `GET /traces/{trace_id}`, and `GET /health` (the only one without auth). `trace_id`/`job_id` path params are `uuid.UUID` (422 if malformed).
- **Reports, two modes.** Both return `ReportJobResponse`: `job_id`, `trace_id`, `status` (queued|running|done|failed), `result` (a `ReportResponse` when done), `error`, `warnings`.
  - **Inline (no `REDIS_URL`):** `POST /report` writes the report during the request on the `REPORT_WORKERS` pool, as before, and answers **200** with `status: "done"`, `job_id: null` and the result. `GET /report/jobs/*` is always 404. One container, no worker.
  - **Queue (`REDIS_URL` set, or `report_jobs` injected):** `POST /report` answers **202** at once with a job; `worker/report_worker.py` writes the report; the client polls `GET /report/jobs/{job_id}`. `/health` shows `report_mode` and `report_workers` (`rq.Worker.count`); the lifespan warns when no worker is listening.
  - **Same report code in both:** `api.jobs.write_report` → `generate_report_for_image` (cache, guard, retry, template fallback), then `finish_trace` in a `finally`.
- **Report jobs (`api/jobs.py`, RQ 2.x, `JSONSerializer`):**
  - Job args are `(trace_id, user_id)` only; `meta` holds `user_id`/`trace_id` (ownership) and, on a known failure, a user-facing `error`. No report text, image or token is ever in the job. The return value is `report_result_to_json(result)`.
  - **Job timeout, derived per job** (`report_job_timeout`, called in `submit` unless `ReportJobs(job_timeout_s=...)` fixes it): `JOB_BASE_S` (60) + 60 s × ceil(workers × one report's reservation ÷ tpm), also bounded by rpm, within 180–900 s. A report's reservation is `REPORT_PROMPT_TOKENS` (6,200, the 4 measured prompt estimates) + `REPORT_CALLS` (4) × `max_completion_tokens`; workers = `Worker.count` now; limits from `report_model_budget()` (config/llm.json). With the current config: 1 worker 180 s, 3 → 300 s, 5 → 420 s. The worker sets the LLM deadline to start + timeout − `DEADLINE_MARGIN_S` (30), so a job that can't get budget ends as the template rather than being killed (`JobTimeoutException`). The sync Groq client also has `timeout=request_timeout_s`, so a hung request is cut long before that. Tests: `tests/test_report_budget.py`.
  - Retention: queued `ttl` 600 s (no worker → the job expires and the poll gets 404), `result_ttl` 1 h, `failure_ttl` 24 h (RQ's default is a year). The tests check these TTLs.
  - **Idempotency:** `cdp:report-job:active:{user_id}:{trace_id}` → newest job id. `submit` reuses it only while **queued or running**; failed, expired and finished jobs get a new job (a finished LLM report then comes back as a cache hit). The key and the new job (saved as queued) are written in one MULTI under WATCH, then enqueued, so concurrent POSTs share one job (12-thread test).
  - **Ownership:** `get` checks `meta.user_id` (and `func_name`) **before** anything else; other users' jobs, unknown and expired ones are all 404, and their spans are never read or written.
  - **No Supabase credentials in the worker** (option (c), decided 2026-10-21). The worker never talks to Supabase. Its tracer writes to local JSONL and to `cdp:report-job:spans:{job_id}` (a Redis list, TTL 24 h). When a poll sees the job ended (done or failed), the API writes those spans with **that request's** user client (`SupabaseTraceStore(db)`; RLS fills `user_id`), then `LTRIM`s exactly the spans it wrote. A failed write keeps them for the next poll (warning in the response); `sql/006` makes repeated or concurrent writes no-ops. If nobody polls the finished job, its spans reach only the worker's JSONL. Rejected: the user's access token in the job (a 1-hour bearer credential in Redis, and RQ keeps job data around), and the service-role key in the worker (bypasses RLS on every table).
  - **Worker crash:** `SimpleWorker` registers a started job until start + timeout + 60 s, with no heartbeat during the job. RQ marks an expired entry failed (`AbandonedJobError`) only when someone cleans the started registry (workers do it every 10 min), so `_job_and_status` runs `started_job_registry.cleanup()` whenever it sees a started job. A dead worker's job therefore turns **failed** within about 4 min, and `failure_reason` says the worker stopped. A timeout (SIGALRM) fails the job normally (`JobTimeoutException`). An upload that expired before the job ran → failed with `UPLOAD_EXPIRED`.
  - **Worker process:** `python -m worker.report_worker` (Linux/Docker only; it exits on Windows, which has no SIGALRM). It loads the model once (`SerializedModel`) and uses `RedisUploadStore`, `RedisReportCache` and a Redis `BudgetRegistry` on one Redis client, plus a second client for RQ (RQ raises its socket timeout for the blocking dequeue). `SimpleWorker` doesn't fork: the model stays loaded and PyTorch never runs in a forked child. One job at a time per process, so scale with more processes (`docker compose up --scale worker=N`). Groq limits are shared through Redis, so more workers mean more reports waiting, not more tokens per minute. Needs `REDIS_URL` (+ `GROQ_API_KEY`), no Supabase settings. `configure(WorkerContext(...))` sets what jobs use; tests call it with fakes. **`python -m` trap:** the `__main__` block must call `main` from the imported `worker.report_worker`, because RQ imports that module (a second copy) to run the job; calling `main()` directly configured `__main__` and every job failed "not configured" (found in Docker, guarded by `test_python_m_configures_the_module_rq_imports`).
  - Tests (`tests/test_report_jobs.py`): fakeredis + RQ sync mode (`ReportJobs(..., is_async=False)` runs the real job function inside the POST). Async-queue tests with no worker cover idempotency; `start_like_a_worker` (a `SimpleWorker`'s `prepare_execution` + `prepare_job_execution`) plus a patched `rq.registry.current_timestamp` covers the crash. Redis down → 503.
- **Auth:** `current_user` reads the bearer token and calls `SupabaseJWTVerifier.verify` in a thread (it may fetch JWKS or call Supabase).
  - ES256/RS256 are checked locally against `{url}/auth/v1/.well-known/jwks.json` (cached by `PyJWKClient`).
  - HS256 is checked with `SUPABASE_JWT_SECRET` if set; otherwise `GET /auth/v1/user` is called.
  - Required claims: `aud == "authenticated"`, `iss == {url}/auth/v1`, `exp`, `sub`. `AuthError` → 401, `AuthUnavailable` → 503.
  - **Claims that need no key are checked first** (`_check_claims` on the unverified payload: `iss`, `exp`, `aud`, `sub`), so a token that is expired, malformed or from **another Supabase project** gets 401 even when the JWKS can't be fetched. Only a token that could be valid gets 503 for unreachable keys. The wrong-project 401 names both issuers.
  - **Settings precedence trap:** env `SUPABASE_URL`/`SUPABASE_KEY` win over `secrets.toml` (needed for deploys). A stale Windows user variable once pointed the API at a deleted project, so every login got 503 "Could not fetch Supabase signing keys" (DNS `getaddrinfo failed`). `Settings.from_env` now warns when the two differ, and the lifespan logs the issuer, where the URL came from, and the result of a one-off JWKS fetch (`check_keys`; `check_keys_on_startup` is only set by `from_env`, so tests never touch the network).
  - **Logging (API only):** `api/` uses `logging` (`configure_logging()` in `create_app` adds a root handler only when none exists; `api.*` at INFO). Key-fetch failures log the URL and the full cause chain (`describe()`); rejected tokens log the reason at WARNING. Never log the token. The Streamlit app still has no logging framework.
- **Own data only:** `user_db` builds a `SyncPostgrestClient` per request with `Authorization: Bearer <user token>` (same `.table()` API the services take), so RLS applies. Queries also filter `user_id` (`PredictionService.get_history`, `SupabaseTraceStore.read/read_trace`). The view `prediction_daily_summary` relies on RLS alone (`security_invoker`).
  - **Never fall back to local `traces/` JSONL in the API.** It holds every user's spans. `/analytics` returns `pipeline: null` plus a warning instead; `/traces` returns 404.
  - `diagnosis_history` is not used by the API. `/history` reads `predictions`, which has no patient names.
- **Uploads:** content type must be `image/jpeg`/`image/png` **and** Pillow must decode JPEG/PNG (else 415); empty → 400; over `max_upload_bytes` → 413 (early from `Content-Length` in middleware, exact after reading); over 40M pixels → 413. CORS is added last so it's outermost (the early 413 still gets CORS headers).
- **UploadStore:** the validated bytes plus the prediction, keyed by `trace_id`, owned by the uploader. It lives in memory only (never on disk, matching the privacy page), with TTL 1 h and at most 100 items. `get()` returns None for "unknown", "expired" and "another user's" alike, so all three are 404. The Grad-CAM PNG (`heatmap_png`) and the raw map (`cam_map`: 7×7 values rounded to 4 dp, plus `summarize_cam_regions`) are cached with the upload through `save_heatmap` / `save_cam_map`. **Never cache by setting attributes on the returned `Upload`**: with Redis it's a copy.
- **Shared state (`REDIS_URL`, env or `[redis] url` in secrets.toml):** `shared_state()` in `api/main.py` picks `RedisUploadStore` + `RedisReportCache` + a Redis `BudgetRegistry` for LLM rate limits (one client, `connect_redis`, 2 s connect / 5 s timeouts); without it, `UploadStore` + `REPORT_CACHE` + `BUDGETS` as before. Stores injected into `create_app` always win (budgets stay in Redis even then); an injected `llm_factory` ignores the registry.
  - Keys: `cdp:upload:{trace_id}` (hash: `user_id`, `data` raw bytes, `prediction` JSON, `created`; `PEXPIRE` ttl in the same MULTI), `cdp:upload:{trace_id}:heatmap` (PNG bytes) and `:cam` (JSON `{cam, regions}`), both written with `PX` = the upload's remaining `PTTL`, so a late save can't revive an expired upload; `cdp:uploads` (sorted set trace_id → created) enforces `max_items` across replicas and gives `len()`. Rate-limit keys: see "Limits" under Report agent.
  - All store calls run in `asyncio.to_thread` (`use_store`). `redis.RedisError` becomes `StoreUnavailable` → **503**; there is no fallback to memory (it would give random 404s across replicas). Report-cache errors are only logged. Startup sends one `PING` and logs `host:port/db` (never the URL: it can hold a password); a failure doesn't stop the API.
  - **Redis must run without persistence** (`save ""`, `appendonly no`), or the X-rays reach Redis's disk and the "never on disk" claim is false. Use `rediss://` with a password for a remote Redis.
  - Tests use fakeredis (`tests/conftest.py`: `backend` param, `make_redis`, `make_upload_store`, `report_cache`; `redis_clock` moves fakeredis's expiry clock by patching `fakeredis._socket._base.time`). Every endpoint test runs on both backends.
- **`/gradcam/{id}/map`** exists for the web UI's overlay and opacity slider: it returns the predicted class's map from `compute_gradcam` (one `gradcam` span with `output=map`). Clients must stretch it over the whole image, because the model saw the image resized to a square. Without `REDIS_URL` it is one process only, like `PendingOAuthStore`.
- **Threads:** PyTorch, Grad-CAM, image decoding and Supabase calls run in `asyncio.to_thread`. Inline reports run `write_report` (the same cache, guard and fallback as the app) on a separate 4-thread pool (`REPORT_WORKERS`), because one report can wait about 60 s on Groq's limit and must not starve the default pool. In queue mode the API does no report work. Store and job calls go through `use_store` (thread, and `StoreUnavailable` → 503). `tests/test_api.py::test_prediction_does_not_block_the_event_loop` fails if model work runs on the loop.
- **SerializedModel:** wraps `ModelService` with one lock around `predict`, `generate_heatmap` and `compute_gradcam`. Grad-CAM hooks the shared network, so a concurrent forward pass would corrupt its activations. Always pass the wrapper (not the raw service) to `generate_report_for_image`, because its `get_gradcam` tool runs the model too.
- **Tracing:** same span names as the dashboard. `/predict` makes the trace (`prediction` → `preprocess`/`classify`/`calibrate`, `log_prediction`), and `/gradcam` and `/report` add to the same `trace_id`. Sinks are local JSONL plus `SupabaseTraceStore(user db)`; sink errors become a response warning.
- **matplotlib:** `api/__init__.py` sets the `Agg` backend. Outside Streamlit, Windows picked Tk, and the Grad-CAM figure crashed in worker threads (found by a real-model smoke test).
- **Type hints:** the "no type hints" rule doesn't apply to the API's FastAPI parameters and Pydantic models, which need annotations. Plain helpers in `api/` still have none.
- **Deploy:** see "Deployment". Env vars: `SUPABASE_URL`, `SUPABASE_KEY` (publishable), optionally `SUPABASE_JWT_SECRET`, `FRONTEND_URL` (comma-separated), `GROQ_API_KEY`, `MAX_UPLOAD_MB`, `REDIS_URL` (needed for more than one replica; it also switches reports to the worker queue, so **run at least one worker** with it).

## MCP server (`mcp_server/`)

- **Design (chosen 2026-10-26, option (b)):** in-process on local files. No Supabase at all: no client, no key read (`test_the_server_never_touches_supabase` checks the source and `sys.modules`), so no RLS and nothing in History/Analytics. Option (a), calling the API with the user's Supabase token, was rejected: a stdio server would need a long-lived refresh token or password in a plain-text client config, and the API is only up while the owner's PC and tunnel are.
- **SDK:** `mcp` 2.x. `FastMCP` was renamed `mcp.server.mcpserver.MCPServer`; `mcp.server.fastmcp` only raises an error. Tests use `mcp.Client(server)` (in memory) and `Client(StdioServerParameters(...))`.
- **Tools** (`create_server` in `server.py`): `classify_xray`, `gradcam_summary`, `generate_report`, `get_trace`, `eval_summary`. Tool bodies are blocking and run in `anyio.to_thread` via `run_blocking`. The model behind `SerializedModel` (from `api/state.py`) is loaded on the first call, so `initialize`/`tools/list` need no torch work.
  - Output fields reuse `ReportTools.run_classifier()` (4 dp) and `GRADCAM_CAVEAT`, so they match what the report agent sees.
  - `generate_report` calls `generate_report_for_image` (process-wide `REPORT_CACHE`, guard, retry, template). It maps `source` "fallback" → **"template"**, and never returns `error_detail`.
  - **One trace per prediction:** each `classify_xray` is a new trace, remembered by `image_fingerprint` (LRU 100, no pixels or paths). `gradcam_summary`/`generate_report` on the same image add to it, classifying first if the image is new.
  - `get_trace` reads only `MCP_TRACE_DIR` (default `traces/mcp/`), **never** the app's shared `traces/*.jsonl`.
  - `eval_summary` accepts only `latest` or `^\d{8}T\d{6}Z$` that exists in `evals/runs/`, and refuses a dataset-hash mismatch.
- **Resources:** `clinical://condition-info`, `clinical://condition-info/{condition}` (both carry `review_status`: DRAFT), `clinical://evals/latest`.
- **Errors:** expected problems raise `ToolError` with one plain sentence; the SDK prefixes "Error executing tool X: ". Anything else is logged (stderr, with traceback) and returned as "Internal error (TypeName)".
- **stdout:** the SDK points fd 1 at stderr while it serves stdio, so stray prints from torch/Streamlit can't corrupt the protocol (the stdio test would fail if they did).
- **Keys and cwd:** MCP clients start the server in any cwd, where `st.secrets` wouldn't find the repo's file. So `load_llm_key()` copies only the provider section's `api_key` from `<repo>/.streamlit/secrets.toml` into the env (the env wins). Relative image paths resolve against the first allowed directory, never the cwd. Register the server as `<venv python> <repo>\mcp_server`, not `-m`.
- **No HTTP transport yet** (README TODO): it would need auth, since anyone reaching the port could read the allowed folders.

## Agent Skills (`.claude/skills/`)

- Claude Code loads them from the repo (`/failure-to-regression <case|trace|file>`, `/release-check`). Keep each `SKILL.md` short and put mechanical work in its scripts; tests in `tests/test_skill_scripts.py` (offline) and `tests/test_regression_fixtures.py`.
- Scripts import the repo through `_repo.py` (`REPO` = 3 levels up; `main_checkout()` finds the main worktree for gitignored `data/`, `.env`, `.streamlit/secrets.toml`, since a worktree has none).
- **failure-to-regression:** `reproduce.py` (case id from a run, `--trace` from local `traces/` only, `--mcp-output` = `{"classification", "report"}` from the MCP tools; `--write-fixture` adds `expect` with `guard_problem: null` to fill in), `add_case.py`, `verify.py` (exit 0 pass, 2 = replay fails only as a guard change or new case must, 1 fail), `live_eval.py start|wait|status` (detached `evals.run --live`; `wait` blocks ≤ 540 s so it fits one shell call; links `data/` and copies only the provider's API key into the child's env from the main checkout), `compare_runs.py` (all cases vs the baseline run: fallbacks, guard rejections split from API-level model errors and truncations, checks, judge means; **exit 3 if fallbacks or guard rejections rose: stop and ask before `--update-baseline`**).
- **release-check:** `release_check.py` runs pytest, `evals.run --replay` and `scripts/smoke_test.py`. Smoke target: `--api-url`, `API_URL`, else local 7860/7861; token: `SUPABASE_ACCESS_TOKEN`, else the `.env` demo account (password grant). Missing either → SKIP, never PASS. It doesn't rebuild the API: the local container is the live one behind Vercel.

## Web UI (`web/`)

- **Stack:** Vite 8, React 19, Tailwind 4 (`@tailwindcss/vite`, theme tokens in `src/index.css`), shadcn/ui as **JSX** (`components.json` `tsx: false`; generated files in `src/components/ui/`), Framer Motion, Recharts 3, TanStack Query 5, React Router 8 (`RouterProvider` comes from `react-router/dom`), supabase-js 2. ESLint 9 flat config with `react-hooks`, `react-refresh` and `jsx-a11y`. **ESLint 10 isn't used:** `eslint-plugin-jsx-a11y` doesn't support it yet. Node >= 22.22.
- **Plain JavaScript, no TypeScript.** The API client (`src/lib/api.js`) has JSDoc typedefs for every response; keep them in step with `api/schemas.py`.
- **Env:** `VITE_API_URL`, `VITE_SUPABASE_URL`, `VITE_SUPABASE_ANON_KEY` in `web/.env.local` (gitignored). Without the Supabase values the login page explains what to set.
- **Auth:** supabase-js with `flowType: "pkce"`. Google sign-in redirects to `window.location.origin`, so Supabase's Redirect URLs need `http://localhost:5173/**` and the deployed site. This is separate from the Streamlit `PendingOAuthStore`: the browser keeps the verifier.
- **State:** server data through the hooks in `src/hooks/queries.js` (keys include the user id; the cache is cleared on sign-out). The Diagnose page's upload, prediction and report live in `DiagnosisProvider` above the router, so they survive navigation; a different user resets it.
- **Numbers shown to users:** model probabilities go through `probability()` in `src/lib/format.js`. It never shows 100% or 0% (99.99% / <0.1%) and rounds to 4 dp like `run_classifier`, so the page agrees with the report text. Rates use `percent()`.
- **Grad-CAM overlay:** `src/lib/colormap.js` reproduces the Python overlay (jet colormap, cv2-style bilinear resize from 7×7, stretched over the whole image). The slider sets the canvas opacity.
- **PDF:** `src/lib/pdf.js` (jsPDF, dynamically imported). The patient name is typed in the browser and only goes into the file; the API never receives it. Helvetica only covers Latin-1, so text goes through `plain()`.
- **Charts:** class colours are `--chart-normal/-pneumonia/-tuberculosis`, validated for colour-blind separation in both themes (TB is a raspberry red, because the red/amber pair failed in dark mode). Light Normal/Pneumonia are under 3:1 contrast, so every chart has a legend and a table view (`ChartCard`). Recharts gets colours through `useCssVars`, which watches the `<html>` class, because SVG attributes don't resolve `var()` reliably.
- **Layout:** `AppShell` shows a bottom tab bar under `md`, an icon rail at `md`, and the full sidebar at `lg`. History, Analytics and Traces are lazy routes; the auth route has a `HydrateFallback`.
- **Referral wording:** the banner and PDF use `referralReason()`, not the API's `referral_message` (a terse status string). When calibration is missing, everything is referred even above 0.9, and the text says so.
- **Reports are jobs:** `createReport()` returns a `ReportJob`. If it has no result yet, `useReportJob` polls `getReportJob()` every 2 s (`refetchInterval`; also in a background tab, because the server stores the trace on the poll that sees the job end) until done or failed. A 404 isn't retried. `DiagnosisProvider` turns this into `report`, `reporting`, `reportStatus` (submitting|queued|running|done|failed), `reportStartedAt` and `reportError`. `ReportPanel` shows Queued → Writing → Checked with an elapsed timer (outside the live region), and an error box whose "Try again" queues a new job.

## Deployment

| Part | Host | Update |
| --- | --- | --- |
| Streamlit app | Streamlit Community Cloud, `main`, `app.py` | push to `main`. A module whose signature changed may stay cached after the pull (a `TypeError` from the old class): **Manage app → Reboot** |
| React app | Vercel project `clinical-diagnostics` (account adi13apr, Hobby), https://clinical-diagnostics.vercel.app | `cd web; npx vercel deploy --prod` (deploys the local folder; not connected to Git) |
| API | Docker `clinical-api` container on the owner's PC + `cloudflared` quick tunnel (`D:\tools\cloudflared`) | `.\scripts\serve_local_api.ps1` (`-Rebuild` for code changes) |
| Supabase | migrations `sql/001`-`006` applied (006 confirmed 2026-10-26: the compose demo's worker spans were upserted and read back through `/traces`) | new numbered SQL file, run before deploying dependent code |
| Report worker | not deployed; `docker compose up` runs nginx + 2 API replicas + workers + Redis on one machine | `serve_local_api.ps1` still runs the single container in inline mode |

- **Why not a cloud API host:** the API peaks at ~660 MB RSS (436 MB after model load; measured in the image). Free 512 MB tiers (Render, Koyeb) are too small; HF Docker Spaces need PRO (2026); Leapcell's free tier is serverless only (multi-instance breaks the in-memory UploadStore) and its sign-up failed; Tailscale Funnel couldn't log in (its service only tried IPv6, which this network doesn't route). Cloud Run (card required) or HF PRO are the realistic upgrades; `deploy_space.ps1` is ready for HF.
- **Keys:** all apps use the Supabase **publishable** key (`sb_publishable_...`). Legacy anon/service_role keys are disabled: a service_role key was committed in `744ea86` (public history) and was the app's `[supabase] key` until 2026-10-20. Never use the secret key in an app.
- **Vercel env vars** (`VITE_API_URL`, `VITE_SUPABASE_URL`, `VITE_SUPABASE_ANON_KEY`) are baked in at build time. **Windows PowerShell can prepend a UTF-8 BOM when piping to a native command:** a BOM ended up inside all three values, `fetch()` refused the apikey header (non-Latin-1 character), and Google sign-in silently dropped back to /login (the PKCE code was never exchanged). Set values from Git Bash with `printf '%s' "$v" | npx vercel env add ...`, or set `$OutputEncoding = New-Object System.Text.UTF8Encoding $false` first (as `serve_local_api.ps1` does). To check a deploy, count `\ufeff` in the served JS: it must be 0.
- `web/.vercelignore` keeps `.env*` out of uploads; `vercel link` also writes `VERCEL_OIDC_TOKEN` into `web/.env.local` (gitignored, not `VITE_`-prefixed, so never bundled).
- **Supabase Redirect URLs:** Streamlit, `https://clinical-diagnostics.vercel.app/**` and the localhost URLs. The Site URL stays on Streamlit (the React app always passes `redirectTo`). Google OAuth needs no change: Google returns to Supabase's `/auth/v1/callback`, and Supabase forwards to the app.
- **CORS:** the container gets `FRONTEND_URL=https://clinical-diagnostics.vercel.app`; other origins get 400 on preflight.
- **PowerShell scripts** must run on Windows PowerShell 5.1: ASCII only, and native commands are judged by `$LASTEXITCODE` with `$ErrorActionPreference` relaxed around them (5.1 turns git/docker stderr into errors). `docker info` exits 0 even when the engine answers 500, so check `docker version --format "{{.Server.Version}}"`.
- `auth.flow_state` (SQL Editor) shows each OAuth attempt: "code issued, never exchanged" means the browser didn't call `/token?grant_type=pkce`.

## Compose stack (multiple replicas)

- **Services:** `redis` (healthcheck; published on `127.0.0.1:${REDIS_PORT:-6380}` for the demo, 6380 so it doesn't clash with a local Redis), `api` (`deploy.replicas: ${API_REPLICAS:-2}`, no host port, healthcheck on `/health` with a 60 s start period), `worker` (`${WORKER_REPLICAS:-1}`, `--scale worker=N`), `nginx` (`nginx:1.28-alpine`, the only published API port, `127.0.0.1:${API_PORT:-7860}`; `.env.example` sets 7861 because `serve_local_api.ps1` uses 7860).
- **nginx:** `upstream { zone; server api:7860 resolve; }` with Docker's resolver `127.0.0.11 valid=10s`, so scaling `api` is picked up without restarting nginx. Round-robin (the default). `client_max_body_size 11m`, `proxy_read_timeout 300s`, and `X-Upstream-Addr` (replica IP) on every response.
- **X-Served-By:** `SERVED_BY_HEADER=1` → `Settings.served_by = socket.gethostname()` (the container ID) → a middleware in `create_app` sets the header on every response. Off by default, so the tunnel-served API doesn't expose hostnames. Not added to CORS `expose_headers` (the browser doesn't need it).
- **Demo (`scripts/multi_replica_demo.py`):** auth from `SUPABASE_ACCESS_TOKEN`, else `DEMO_EMAIL`/`DEMO_PASSWORD` via Supabase's password grant (env first, then `.env`; never printed). It checks: (A) `/health` answered by ≥ 2 replicas; (B) predict on replica A → Grad-CAM map/PNG from another replica, `POST /report` on two replicas → same `job_id`, written by a worker (worker hostname from `rq:job:{id}` → `rq:worker:{name}`); (C) N concurrent reports on pixel-distinct images (random per run, so the 24 h report cache can't answer), while a thread samples `cdp:rate:{model}:requests|tokens` every 0.25 s. Then: peak requests and tokens in any 60 s window vs `config/llm.json`, the `llm_call` spans from `/traces` (tokens, `rate_limit_retries`, `RateLimitError`), and "written by the LLM" with fallback reasons. A check that couldn't run is **SKIP**, never PASS (e.g. this run reserved nothing, or `/traces` unreadable: it needs real Supabase and `sql/006`). Raw numbers go to `demo_results/<UTC>.json` (gitignored); **README numbers come only from those files of a real run.**
- **Verified on 2026-10-25** with a throwaway project (`-p cdp-verify`, fake Supabase URL, HS256 test secret, so `/traces` was 500 and that check SKIPped): strict alternation between 2 replicas, every shared-state check passed, jobs spread over 3 workers. One run with real Groq: 14 calls, peak 5,483 tokens / 5 requests in 60 s (limit 8,000 / 30), `rate_limit_retries` 0 on every call; **2 of 3 concurrent reports fell back** (`JobTimeoutException`): see Known issues.
- **Real run, 2026-10-25 21:37 UTC** (`demo_results/20261025T213758Z.json`, real Supabase via a dedicated demo account (`DEMO_EMAIL`/`DEMO_PASSWORD` in the gitignored `.env` only), real Groq; the README's "What we measured" quotes it): 14 PASS, 1 FAIL. 15 calls over 3 workers, peak 5 requests / 5,478 tokens in 60 s, Redis and trace totals both 18,742 tokens, 0 rate-limit retries; 2 of 3 concurrent reports by the LLM, 1 `JobTimeoutException` fallback. This is the README's **before** run.
- **After the budget fix, 2026-10-25 23:59 UTC** (`demo_results/20261025T235938Z.json` in the `fix/report-budget-timeout` worktree; README "after"): 15 PASS. Cap 600, derived timeout 300 s for 3 workers: 16 calls, peak 6 requests / 7,124 tokens in 60 s, 20,511 tokens (Redis = traces), 0 retries, **3 of 3 by the LLM**, all done in 176 s. 7 of 16 calls still waited 52–65 s in the budget, and the slowest report took 175 s: throughput is bounded by real use (~5.1k tokens per report against 8k/min), so the longer timeout is what saved the third report.
- An earlier run (`20261025T205804Z`) was cut by Windows Modern Standby (about 35 min): 2 jobs ended `failed` "worker stopped". Run the demo under `SetThreadExecutionState(ES_CONTINUOUS|ES_SYSTEM_REQUIRED|ES_DISPLAY_REQUIRED)` (PowerShell `Add-Type` kernel32) so the PC can't sleep mid-run.
- **Docker Desktop DNS:** in one run every Groq call failed with `APIConnectionError` (`getaddrinfo` EAI_NODATA inside the Compose network; the host resolved fine). Cold-name probes failed 0–8% of lookups with or without `dns_opt: [single-request]` or explicit `dns:` servers, so no setting is in the compose file. The demo shows such failures as fallbacks with the reason.
- **Docker Desktop pauses:** if the PC sleeps (Modern Standby; check Kernel-Power events 506/507), the VM freezes (no container logs at all), in-flight requests time out, and running jobs come back `failed` (abandoned). That isn't an app bug.

## Conventions

- **Services in `services/`, Streamlit pages in `ui/`.** `app.py` builds the services and passes them into `show_*` functions as arguments. A new service gets built in `app.py` and threaded through the `show_dashboard` / `show_diagnosis_page` signatures.
- Code that must run outside Streamlit (scripts, tests) goes in Streamlit-free modules like `services/calibration.py`. `scripts/` reuses `ModelService` and `services.calibration` rather than duplicating model or preprocessing code.
- A service that needs the user's Supabase session takes a `client` in its constructor.
- Heavy or circular imports are done lazily inside functions (`HeatmapGenerator` in `generate_heatmap`, `ui.analytics` in `show_dashboard`).
- Non-critical Supabase calls (logging, analytics) are wrapped in `try/except` → `st.warning` and do not block the main flow.
- **Confidence units:** model output, `predictions.confidence` and `avg_confidence` are 0–1. `diagnosis_history.confidence_score` and the PDF `confidence` argument are percent (0–100).
- **Class names are duplicated** as string keys. If a class is added or renamed, update all of them:
  `ModelService.classes`, `get_severity_badge` in `ui/dashboard.py`, both `colors_map`s and `severity_map` in `ui/analytics.py`, `severity_colors` in `services/pdf_handler.py`, the placeholder probabilities in the history download, `LABELS` in `tests/test_calibration.py`, the `Literal` in `agent/schema.py`, `CLASS_NAMES` in `agent/tools.py`, and the keys of `agent/condition_info.json`. Then retrain and recalibrate.
- Text from an LLM that goes into a reportlab `Paragraph` must be `escape()`d.
- Severity colours: Normal `#00D084`, Pneumonia `#FFA500`, Tuberculosis `#FF0000`, unknown `#808080`.
- Schema changes go in a new numbered file under `sql/`, written to be re-runnable (`if not exists`, `drop policy if exists`, `create or replace`; views can only gain columns at the end).
- Tests: `tests/conftest.py`'s `make_service` monkeypatches `ModelService._load_model` with a fixed-logits stub. Use it rather than loading the checkpoint. Tests that need the dataset or cache must `pytest.skip` when it's absent.
- Plain classes, docstrings on non-trivial methods, short comments. No type hints. No logging framework; errors go to the user through `st.error` / `st.warning`.
- `requirements.txt` is unpinned and is what Streamlit Cloud installs. Dev-only tools go in `requirements-dev.txt`.

## Known issues (not yet fixed)

- `agent/condition_info.json` is a draft that hasn't had clinical review. It must be signed off before clinical use.
- The judge (`qwen/qwen3.8-27b`) is a Groq **Preview** model; Groq may change or remove it. Re-confirm the ID before relying on old scores.
- The grounding guard checks the structured fields and banned phrases, not the meaning of the free text. A summary could still describe the wrong condition in words while keeping `predicted_class` correct.
- The PDF still labels the prediction "Primary Diagnosis" and the title emoji renders as a box (Helvetica has no emoji).
- The batch tab still marks a row "Error" if the history save fails, even though the prediction and log succeeded.
- History PDF downloads use made-up probabilities (0.33/0.33/0.34).
- Google sign-in opens in a new browser tab (`st.link_button`), so the user is signed in there; the original tab stays on the login page.
- API: without `REDIS_URL`, uploads and the report cache live in one process's memory, so a restart loses uploads (404 → predict again) and replicas can't share them. With Redis, a Redis restart does the same. The Redis path is tested with fakeredis and was checked by hand against a local Redis 7.4, but has not run in a deployment. The `max_items` cap can be off by one briefly when replicas write at once, and `/gradcam/{id}/map` reads the whole image from Redis even when the map is cached. The report cache is keyed by image, not user, so two users uploading the identical image share one LLM report (it contains no patient data).
- Report worker: checked end to end with `docker compose` on 2026-10-25 (real Redis 7, RQ 2.12, model and Groq; login and Supabase faked, since a real token needs a browser sign-in): 202 in 0.02 s, the same job on a repeat POST, other users 404, queued → running → done in 4 s with an LLM report, spans stored once on the poll, a cache hit on the next job, a SIGKILLed worker's job failed after 222 s with the "worker stopped" message, a restarted worker drained the queue, and Redis's `/data` stayed empty. Not yet run with a real Supabase token (`scripts/smoke_test.py` against a queue-mode API). In queue mode a finished job's spans reach Supabase only if someone polls it, and a dead worker's job shows "running" for up to about 4 min before it turns failed.
- API hosting: it runs on the owner's PC behind a Cloudflare quick tunnel, so the React app works only while that PC is on, and the tunnel URL changes on restart (`serve_local_api.ps1` re-points Vercel). Needs a real host (HF PRO, Cloud Run, ...) to be always on.
- Web UI: History's search box filters only the loaded page (the API has no text search); class, triage and date filters are server-side.
- Web UI: no automated frontend tests yet; CI runs lint and build only. It was checked by hand in Edge (light, dark, phone, tablet, keyboard) against the real model with faked login, and deployed on Vercel (Google and email sign-in confirmed on a phone, 2026-10-21).
- Shared Groq budget under concurrency: reports still queue in the budget (about 1.5 reports per minute at 8k tokens/min), so N workers mean slower reports, not more of them per minute. The job timeout now grows with the worker count, but it is read when the job is queued: workers added later don't extend it, and a worse than fair share ends as the template (deadline). Whether Groq counts `max_completion_tokens` against tokens/min up front (as it does for Qwen's OTPM) is unknown; reserving the full cap keeps either way safe.
- `use_container_width` is deprecated in current Streamlit (warning says removal after 2025-12-31), and requirements are unpinned.
