# CLAUDE.md

Guidance for working in this repository.

## What this is

A Streamlit app ("Clinical Diagnostic Portal") where a logged-in radiologist uploads a chest X-ray and gets a 3-class prediction (**Normal / Pneumonia / Tuberculosis**) from a distilled EfficientNet-B0. Probabilities are temperature-calibrated, and each prediction is triaged as `confident` or `refer` (clinician review). The app also shows a Grad-CAM overlay and produces a PDF report. An optional Gemini tool-calling agent writes a plain-language report; it explains the classifier's output and can never change it. Supabase handles auth and stores two things: the per-patient diagnosis history and a per-user prediction log that feeds the analytics page. The app is deployed on Streamlit Community Cloud.

## Commands

```bash
python -m venv venv
venv\Scripts\activate              # Windows; macOS/Linux: source venv/bin/activate
pip install -r requirements.txt    # app only (what Streamlit Cloud installs)
pip install -r requirements-dev.txt  # + pytest
streamlit run app.py
pytest                             # no Supabase, dataset or GPU needed
python scripts/calibrate.py        # re-fit temperature; needs the datasets in data/ (below)
python -m evals.run --replay       # re-score the committed eval run (no network; what CI runs)
python -m evals.run --live --resume  # continue the live eval (real model + Gemini; needs data/ and a key)
```

CI (`.github/workflows/ci.yml`, on push and PR) installs CPU torch and `requirements-dev.txt`, then runs `pytest` and `python -m evals.run --replay`.

The app needs `.streamlit/secrets.toml` (gitignored; never commit it):

```toml
[supabase]
url = "..."
key = "..."   # anon key

[gemini]          # optional; without it "Generate report" uses the template report
api_key = "..."   # or env GEMINI_API_KEY
# model = "gemini-3.8-flash"   # optional override (or env GEMINI_MODEL)
```

Database setup: run the files in `sql/` in order, in the Supabase SQL Editor, **before** deploying code that depends on them:
- `002` adds `status` / `temperature` to `predictions`. Without it every prediction-log insert fails, which the app reports as a warning only.
- `003` adds `status` to `diagnosis_history`. Without it every history insert fails: the single tab shows an error and batch rows show "Error".
- `004` creates `traces` and adds `predictions.trace_id`. Without it every prediction-log insert fails (warning only), and spans go to local `traces/` only. The `diagnosis_history` table has no migration in the repo; it already exists in the Supabase project.

## Layout

```
app.py                      Entry point: builds services, routes login vs dashboard via st.session_state.user
services/                   Non-UI logic. Each module exposes one class (calibration.py: functions).
  model_handler.py          CLAHETransform, StudentEfficientNet, ModelService (load + calibrated predict + heatmap)
  calibration.py            Temperature loading, apply_temperature, triage, ECE. No Streamlit import.
  tracing.py                Tracer/Span, fail-safe helpers, JSONL + Supabase sinks, nominal cost, summarize_spans
  heatmap_handler.py        GradCAM on the last conv block + HeatmapGenerator (3-panel figure) + summarize_cam_regions
  prediction_handler.py     PredictionService: insert into `predictions`, read `prediction_daily_summary`
  data_handler.py           DataService: Supabase table `diagnosis_history` insert/select
  auth_handler.py           AuthService: Supabase email/password, sign-up, Google OAuth, logout
  pdf_handler.py            PDFReportGenerator (reportlab) → PDF bytes; referral box, triage row, optional AI report
agent/                      LLM report agent (see "Report agent")
  schema.py                 Pydantic Report + fixed DISCLAIMER
  tools.py                  ReportTools: run_classifier, get_gradcam, get_condition_info; TOOL_DECLARATIONS
  agent.py                  LLMClient / GeminiClient / MockLLMClient, tool loop, grounding guard, fallback report
  condition_info.json       Static condition text (DRAFT: needs clinical sign-off)
ui/                         Streamlit pages; functions named show_*(...)
  login_pages.py            show_login_ui(auth_service)
  dashboard.py              show_dashboard(...) sidebar nav → show_diagnosis_page (single/batch/history);
                            run_prediction() = time + predict + log; show_status() = triage banner;
                            show_single_result() renders the last single run from session_state
  analytics.py              show_analytics (diagnosis_history charts) + show_prediction_analytics (summary view)
sql/
  001_predictions.sql       `predictions` table, RLS policies, `prediction_daily_summary` view
  002_prediction_status.sql `status` + `temperature` columns, `refer_count` in the view
  003_diagnosis_history_status.sql  `status` column on diagnosis_history
  004_traces.sql            `traces` table (one row per span, RLS) + predictions.trace_id
config/
  pricing.json              NOMINAL per-model prices for cost estimates; billing_tier "free" → billed $0
scripts/
  calibrate.py              Fits the temperature on the notebook's val split, reports on its test split
  failure_case.py           Renders docs/failure_case_gradcam.png (Grad-CAM of a misclassified test image)
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
  fixtures/                 Verbatim LLM outputs used as regression tests
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
  - Only `source == "llm"` results are cached. Fallbacks are free to rebuild, and caching one would block a later Gemini attempt.
  - Hits return `cached=True` with `steps=0`.
- **Request budget:** the prompt asks for all three tools in the first turn, and `get_condition_info` defaults to the predicted class. A report is then 2 Gemini requests, 3 with a retry. Measured live: 2 requests, 6.5 s, versus 5 requests and 16.6 s calling one tool per turn. The free tier allows 5 requests/minute per model.
- **429 handling** (`GeminiClient.generate`): read Google's retry delay (the `RetryInfo` detail, else "retry in Ns" in the message). If it's under `RATE_LIMIT_MAX_RETRY_DELAY` (10 s), sleep and retry **once**; otherwise raise `RateLimitError` right away, which falls back with "Gemini rate limit reached (retry in N s)."
- **Tools** (`ReportTools`): the image and the dashboard's prediction are bound server-side. The LLM sees only tool outputs: no pixels, no patient name or email.
  - `run_classifier` rounds probabilities to 4 dp, so the report can copy them exactly.
  - `get_gradcam` covers the predicted class.
  - `get_condition_info` reads `condition_info.json` only; there are no web lookups. `condition` is optional and defaults to the predicted class.
  - Tool exceptions are returned to the LLM as `{"error": ...}`.
- **Loop:** at most `MAX_STEPS = 5` LLM calls. The final text must be JSON → `Report.model_validate` → `check_grounding`. On failure the LLM gets the problem and `MAX_RETRIES = 1` more try. After that, or on any LLM exception, it falls back to `fallback_report`.
- **Grounding guard** (`check_grounding`):
  - `predicted_class`, `calibrated_probability` and `status` must `==` `run_classifier`'s output;
  - no assertion of a confirmed diagnosis (`asserts_confirmed_diagnosis`, using `BANNED_PATTERNS`). A match is allowed when a negation cue (not, no, never, without, cannot, n't, until, unless, before) appears in the 4 words before it within the same clause (clauses split on punctuation, "but", "however"). "No doubt" and "not only" don't count as negations. This was added after Gemini wrote "a screening aid, not a confirmed diagnosis" (fixture `gemini_rejected_negated_confirmed.json`);
  - a `refer` case's recommendation must contain "clinician review".
  - The disclaimer is always replaced with `schema.DISCLAIMER`.
- **`fallback_report`:** a deterministic template built only from tool outputs. It is always schema-valid and grounded (asserted), and it works even when Grad-CAM fails.
- **`GeminiClient`:** `models.generate_content` with manual function calling. It is stateless: the history is resent each call, and nothing is stored server-side, unlike the Interactions API.
  - `FunctionDeclaration(parameters_json_schema=...)`, AFC disabled, `thinking_level="low"`, and **no temperature** (deprecated for Gemini 3.8).
  - Model turns are replayed as received, which keeps thought signatures. Each `FunctionResponse` carries the call's `id` and `name`.
  - The final answer is plain JSON text, not `response_json_schema`: combining structured output with tools wasn't confirmed in the docs.
- **Model ID:** `GEMINI_MODEL = "gemini-3.8-flash"` (stable). Override with `st.secrets["gemini"]["model"]` or `GEMINI_MODEL`.
- **Tests** use `MockLLMClient` (scripted `LLMTurn`s / callables). The Gemini wiring is tested offline with a fake `client.models`. Never call the network in tests.

## Tracing (`services/tracing.py`)

- **One trace per prediction.** The dashboard creates `make_tracer(trace_store)` on "Run AI Diagnosis" (and per batch image), stores `last_trace_id`, and reuses that id for "Generate report". A prediction and its report therefore share one `trace_id`, which is also written to `predictions.trace_id`.
- **Spans:** `prediction` → `preprocess` / `classify` / `calibrate` (inside `ModelService.predict`), `log_prediction`, `save_history`, `gradcam` (`generate_heatmap` and `compute_gradcam`), and `report` → `llm_call` / `tool_call` (→ `gradcam`) / `guard` / `fallback_used`. A cache hit records `report` → `report_cache_hit`.
  - `llm_call` attributes: model, step, `input_tokens`, `output_tokens`, `thinking_tokens`, `nominal_cost_usd`, `billed_cost_usd`, `finish_reason`, `tool_calls`, `rate_limit_retries`. On failure: status error plus `error_type`.
  - `guard`: zero-duration event, `result` pass/reject plus `reason`. A reject has status error and `error_type` "Rejected".
  - `fallback_used`: `used` bool plus `reason`, one per generated report.
- **Fail-safe rule:** app code only calls `trace_span`, `trace_event`, `safe_set` and `finish_trace`. These swallow and log tracer errors, while the step's own exceptions still propagate. Pass `tracer=None` to disable tracing. Never call `tracer.span()` directly from app code.
- **Sinks:** `finish()` writes new spans to every sink. A failing sink is logged and appended to `tracer.sink_errors`, never raised.
  - `JsonlTraceSink` writes `traces/YYYY-MM-DD.jsonl` and always runs.
  - `SupabaseTraceStore` writes the `traces` table and needs the logged-in auth client.
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
- **Deterministic checks** (`checks.py`): schema valid, class/probability/status match the classifier, refer → "clinician review", no confirmed-diagnosis claim (`asserts_confirmed_diagnosis`), limitations present (1–5), length limits (summary 2–3 sentences and ≤ 600 chars, recommendation ≤ 500, each limitation ≤ 300), and `guard_replay_consistent` (today's guard still rejects the recorded rejected drafts and accepts the final LLM report). Pass rates exclude not-applicable cases (`None`).
- **Judge** (`judge.py`): Gemini structured output (`response_schema=JudgeResult`), with `rubric.md` sent verbatim. It sees tool outputs and the report only, never the true label. The judge model defaults to the agent's model (self-preference bias is possible; `--judge-model` changes it).
- **Live runs** (`run.py --live`):
  - One `RateLimiter` (default 4 RPM) is shared by the agent and the judge through `rate_limited_client`. It holds a reference to the genai `Client`, which closes its HTTP connection when garbage-collected. Never wrap an inline `genai.Client(...)` without keeping a reference.
  - Records are appended; the last record per case wins. `--resume` skips judged cases and re-judges unjudged ones.
  - Any LLM *exception* during the agent step raises `StopRun` instead of recording a template report as a result.
  - Rate-limit stops print the resume command.
- **Free-tier quota:** `gemini-3.8-flash` allows **20 requests/day per model** (`GenerateRequestsPerDayPerProjectPerModel-FreeTier`; resets at midnight Pacific) and 5/min. A full run is about 108 requests (36 × 2 agent + 1 judge), so it takes about 6 days on the free tier, or minutes on a paid key. `RateLimitError.daily` and `describe_llm_error` report daily exhaustion explicitly.
- **Replay and baseline:** `--replay` re-scores the baseline's run (or `--run`), prints the summary, and exits 1 if any deterministic pass rate is below `baseline.json`. `--update-baseline` rewrites it, marked PROVISIONAL while the run is partial. CI runs `--replay`.
- **Human validation:** `--export-human-labels` writes 15 blind rows (5 error-only, 5 refer, 5 correct; LLM-written preferred) with empty score columns. It needs ≥ 15 reports and refuses to overwrite a sheet that has scores. `--agreement` gives exact, within-1 and mean (judge − human) per criterion. **Only a human fills in the scores.**

## Supabase

There are two kinds of client, both using the anon key from `st.secrets`:

| Client | Lifetime | Session | Used by | Tables |
| --- | --- | --- | --- | --- |
| `AuthService.client` | one per browser session (`st.session_state.auth_provider`) | logged-in user's JWT after `login_email` | `AuthService`, `PredictionService`, `SupabaseTraceStore` | `predictions`, `prediction_daily_summary`, `traces` |
| `DataService.client` | rebuilt every rerun | none (anon role) | `DataService` | `diagnosis_history` |

- Login state for routing lives in `st.session_state.user` (the Supabase `User` object). The dashboard only uses `.email`.
- `PredictionService` **must** be given `auth_provider.client`. Its tables depend on RLS seeing the user's session.

### `diagnosis_history` (per-patient record)

- Columns, as used in code: `radiologist_email`, `patient_name`, `prediction` (class name string), `confidence_score` (percent, 0–100, 2 dp), `status` (`confident`/`refer`, migration 003; NULL on older rows, shown as "—"), `created_at` (server default). Probabilities are **not** stored.
- `confidence_score` is calibrated from the calibration commit onward. Older rows hold raw, much lower values.
- It is read with `.eq("radiologist_email", email)` from an anon client. Access control depends entirely on that table's RLS policies, which are not in the repo.

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
- The live eval run is partial (2/36 reports; the free-tier daily quota ran out), so `evals/baseline.json` is provisional.
- The judge docks faithfulness for the "two public datasets" limitation because that fact isn't in any tool output, yet the agent's system prompt requires it. Consider exposing it through a tool (e.g. in `run_classifier`), so it's grounded.
- The grounding guard checks the structured fields and banned phrases, not the meaning of the free text. A summary could still describe the wrong condition in words while keeping `predicted_class` correct.
- The PDF still labels the prediction "Primary Diagnosis" and the title emoji renders as a box (Helvetica has no emoji).
- The batch tab still marks a row "Error" if the history save fails, even though the prediction and log succeeded.
- `diagnosis_history` is accessed through the anon `DataService` client; its protection depends on RLS policies not in the repo.
- History PDF downloads use made-up probabilities (0.33/0.33/0.34).
- `login_google()` discards the OAuth URL, so the Google button does nothing visible.
- `use_container_width` is deprecated in current Streamlit (warning says removal after 2025-12-31), and requirements are unpinned.
