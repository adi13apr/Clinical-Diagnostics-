# CLAUDE.md

Guidance for working in this repository.

## What this is

A Streamlit app ("Clinical Diagnostic Portal") where a logged-in radiologist uploads a chest X-ray and gets a 3-class prediction (**Normal / Pneumonia / Tuberculosis**) from a distilled EfficientNet-B0. Probabilities are temperature-calibrated, and each prediction is triaged as `confident` or `refer` (clinician review). The app also shows a Grad-CAM overlay and produces a PDF report. Supabase handles auth and stores two things: the per-patient diagnosis history and a per-user prediction log that feeds the analytics page. The app is deployed on Streamlit Community Cloud.

## Commands

```bash
python -m venv venv
venv\Scripts\activate              # Windows; macOS/Linux: source venv/bin/activate
pip install -r requirements.txt    # app only (what Streamlit Cloud installs)
pip install -r requirements-dev.txt  # + pytest
streamlit run app.py
pytest                             # no Supabase, dataset or GPU needed
python scripts/calibrate.py        # re-fit temperature; needs the datasets in data/ (below)
```

The app needs `.streamlit/secrets.toml` (gitignored; never commit it):

```toml
[supabase]
url = "..."
key = "..."   # anon key
```

Database setup: run the files in `sql/` in order, in the Supabase SQL Editor. `002` must be applied before deploying code that sends `status` / `temperature`; until then every prediction-log insert fails, which the app reports as a warning only. The `diagnosis_history` table has no migration in the repo; it already exists in the Supabase project.

## Layout

```
app.py                      Entry point: builds services, routes login vs dashboard via st.session_state.user
services/                   Non-UI logic. Each module exposes one class (calibration.py: functions).
  model_handler.py          CLAHETransform, StudentEfficientNet, ModelService (load + calibrated predict + heatmap)
  calibration.py            Temperature loading, apply_temperature, triage, ECE. No Streamlit import.
  heatmap_handler.py        GradCAM (hooks) + HeatmapGenerator (matplotlib 3-panel figure → PIL image)
  prediction_handler.py     PredictionService: insert into `predictions`, read `prediction_daily_summary`
  data_handler.py           DataService: Supabase table `diagnosis_history` insert/select
  auth_handler.py           AuthService: Supabase email/password, sign-up, Google OAuth, logout
  pdf_handler.py            PDFReportGenerator (reportlab) → PDF bytes
ui/                         Streamlit pages; functions named show_*(...)
  login_pages.py            show_login_ui(auth_service)
  dashboard.py              show_dashboard(...) sidebar nav → show_diagnosis_page (single/batch/history);
                            run_prediction() = time + predict + log; show_status() = triage banner
  analytics.py              show_analytics (diagnosis_history charts) + show_prediction_analytics (summary view)
sql/
  001_predictions.sql       `predictions` table, RLS policies, `prediction_daily_summary` view
  002_prediction_status.sql `status` + `temperature` columns, `refer_count` in the view
scripts/
  calibrate.py              Fits the temperature on the notebook's val split, reports on its test split
splits/
  split_manifest.csv        The training notebook's exact train/val/test split (relpath,label,split)
notebooks/
  sta-project-ca2693.ipynb  Training notebook (Kaggle): data merge, split, multi-teacher distillation
outputs/                    Committed calibration results
  calibration.json          Temperature (read by the app at startup), test metrics, coverage table
  reliability.png           Reliability diagram before/after scaling
tests/                      pytest; conftest.py stubs the network so tests need no checkpoint or Supabase
data/                       Gitignored. Kaggle datasets + data/cache/calibration_logits.npz
best_student_model_aligned_final.pth   state_dict for StudentEfficientNet (~16 MB, committed)
```

## Inference path

1. `ui/dashboard.py`: `st.file_uploader` (jpg/jpeg/png) → `PIL.Image.open` → `run_prediction(model_service, prediction_service, image)`. The single-diagnosis tab and the batch tab both go through it.
2. `ModelService.predict` (`services/model_handler.py`):
   - converts to RGB if needed;
   - transform: `Resize((224,224))` → `CLAHETransform(clip_limit=1.5, tile 8x8)` (CLAHE on the L channel in LAB space) → `ToTensor` → ImageNet `Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])`;
   - `StudentEfficientNet` forward returns `(logits, features)`, then `softmax(logits / T)`, then argmax.
3. Returns `{'diagnosis', 'confidence' (calibrated max, 0–1), 'probabilities' {class: calibrated 0–1}, 'status' ('confident'|'refer'), 'temperature'}`.
4. `run_prediction` logs the result to `predictions` (see below). The caller then calls `DataService.save_diagnosis` (written to `diagnosis_history`), `ModelService.generate_heatmap` (Grad-CAM, single tab only), and `PDFReportGenerator` (single tab only).

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

## Supabase

There are two kinds of client, both using the anon key from `st.secrets`:

| Client | Lifetime | Session | Used by | Tables |
| --- | --- | --- | --- | --- |
| `AuthService.client` | one per browser session (`st.session_state.auth_provider`) | logged-in user's JWT after `login_email` | `AuthService`, `PredictionService` | `predictions`, `prediction_daily_summary` |
| `DataService.client` | rebuilt every rerun | none (anon role) | `DataService` | `diagnosis_history` |

- Login state for routing lives in `st.session_state.user` (the Supabase `User` object). The dashboard only uses `.email`.
- `PredictionService` **must** be given `auth_provider.client`. Its tables depend on RLS seeing the user's session.

### `diagnosis_history` (per-patient record)

- Columns, as used in code: `radiologist_email`, `patient_name`, `prediction` (class name string), `confidence_score` (percent, 0–100, 2 dp), `created_at` (server default). Probabilities and status are **not** stored.
- `confidence_score` is calibrated from the calibration commit onward. Older rows hold raw, much lower values.
- It is read with `.eq("radiologist_email", email)` from an anon client. Access control depends entirely on that table's RLS policies, which are not in the repo.

### `predictions` (prediction log)

- **When a row is written:** inside `run_prediction`, immediately after `model_service.predict` succeeds and before `save_diagnosis`. That gives one row per image, single or batch.
- **Columns:**
  - App sends `predicted_label`, `confidence` (calibrated, 0–1), `inference_ms` (int, `perf_counter` around `predict`), `status` (`confident`/`refer`), and `temperature`.
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
  `ModelService.classes`, `get_severity_badge` in `ui/dashboard.py`, both `colors_map`s and `severity_map` in `ui/analytics.py`, `severity_colors` in `services/pdf_handler.py`, the placeholder probabilities in the history download, and `LABELS` in `tests/test_calibration.py`. Then retrain and recalibrate.
- Severity colours: Normal `#00D084`, Pneumonia `#FFA500`, Tuberculosis `#FF0000`, unknown `#808080`.
- Schema changes go in a new numbered file under `sql/`, written to be re-runnable (`if not exists`, `drop policy if exists`, `create or replace`; views can only gain columns at the end).
- Tests: `tests/conftest.py`'s `make_service` monkeypatches `ModelService._load_model` with a fixed-logits stub. Use it rather than loading the checkpoint. Tests that need the dataset or cache must `pytest.skip` when it's absent.
- Plain classes, docstrings on non-trivial methods, short comments. No type hints. No logging framework; errors go to the user through `st.error` / `st.warning`.
- `requirements.txt` is unpinned and is what Streamlit Cloud installs. Dev-only tools go in `requirements-dev.txt`.

## Known issues (not yet fixed)

- Grad-CAM targets `model.model.avgpool`, whose output is 1×1. The resulting CAM is therefore uniform. A meaningful map needs a spatial layer such as `model.model.features[-1]`.
- The PDF report and the `diagnosis_history` table don't include the triage status.
- In the single tab, `save_diagnosis` is not wrapped in `try`. If it fails, the page errors after the `predictions` row was written.
- `diagnosis_history` is accessed through the anon `DataService` client; its protection depends on RLS policies not in the repo.
- History PDF downloads use made-up probabilities (0.33/0.33/0.34).
- `login_google()` discards the OAuth URL, so the Google button does nothing visible.
- `use_container_width` is deprecated in current Streamlit (warning says removal after 2025-12-31), and requirements are unpinned.
