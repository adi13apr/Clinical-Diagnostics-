# Clinical Diagnostic Portal

A Streamlit app that classifies chest X-rays as **Normal**, **Pneumonia** or **Tuberculosis**. It reports a calibrated probability, flags uncertain cases for clinician review, shows where the model looked (Grad-CAM), and can generate a plain-language report with an LLM.

> **Not a medical device.** This is a research/portfolio project. Its output is a screening aid, not a diagnosis.

**Live demo:** [clinical-diagnosis.streamlit.app](https://clinical-diagnosis.streamlit.app/). The demo is deployed from `main`, so features added on a branch appear only after merging, setting the secrets, and running the SQL migrations.

## How it works

1. **Classifier:** an EfficientNet-B0 "student", distilled from DenseNet121 and Swin-T teachers (`notebooks/sta-project-ca2693.ipynb`). Input is 224×224 with CLAHE contrast enhancement.
2. **Calibration and triage:** raw probabilities are rescaled with a single temperature fitted on the validation split. If the top calibrated probability is below **0.9**, the result is marked **refer: uncertain, clinician review recommended**.
3. **Grad-CAM:** computed on the last convolutional layer, summarised as the most influential zone of the image.
4. **Report agent:** an LLM (Groq `openai/gpt-oss-120b` by default) gathers facts through three tools: classifier output, Grad-CAM summary, and vetted condition text.
   - A guard rejects any report whose class, probability or status differs from the classifier, or that claims a confirmed diagnosis.
   - After one retry, a fixed template report is used instead.
   - The LLM never sees the image or patient details, and can't change the prediction.
5. **Logging:** predictions, triage status and step-level traces (latency, tokens, guard verdicts) go to Supabase, with per-user row-level security, and to local JSONL.

## Results and failure cases

### Classifier and calibration

The test split is the training notebook's own held-out set (1,006 images), rebuilt from the exported manifest in `splits/`. Accuracy matches the paper's reported 97.61%.

| Metric | Value |
| --- | --- |
| Accuracy / macro-F1 | 97.61% / 98.27% |
| Recall: Normal / Pneumonia / TB | 98.6% / 96.0% / 100% |
| Calibration error (ECE, 15 bins) | **39.5% → 1.2%** after temperature scaling (T = 0.132) |
| Referred at the 0.9 threshold | 1.7% of images; catches 11 of the 24 errors |

The raw model was strongly **under**-confident: its probabilities never reached 0.9. Calibration fixes the numbers but not the decisions, because accuracy is unchanged by temperature scaling. Details are in [`outputs/calibration.json`](outputs/calibration.json) and [`outputs/reliability.png`](outputs/reliability.png).

### Report agent eval

36 test cases, made up of every model error, every refer case, and 6 correct ones. Latest run: [`evals/results/2026-10-20.md`](evals/results/2026-10-20.md).

- **Deterministic checks: 11 of 12 at 100%.** These are: class, probability and status match the classifier; refer cases recommend clinician review; confident results are never called uncertain; Pneumonia/TB reports never say no follow-up is needed; no confirmed-diagnosis claim; limitations present; length limits; schema valid; and the guard still agrees with the decisions recorded in the run.
  - The 12th check, that the dataset caveat is quoted from the tool, fails on 1 of 36 (`c07`): the model skipped the condition-info tool and invented its own limitations.
  - CI replays this run on every push and fails if any rate drops.
- All 36 reports were written by the LLM: no template fallbacks and no guard rejections.
- An LLM judge (`qwen/qwen3.8-27b`, a different model family from the report writer) scored 4.58–4.97 out of 5 on faithfulness, uncertainty language, actionability and plain language.
- **Judge vs human:** I scored 15 reports blind (from the 18 Oct run) with the same rubric. They agree on 82% of scores exactly and 97% within one point, but both of us rated almost everything 5, so that number is flattering.
  - The one report I marked down (`c32`: "No immediate referral is required" for a 99.99% Pneumonia result) got 5/5 from the judge on faithfulness and actionability.
- **Judge sensitivity:** given 5 deliberately bad reports (wrong class, missing referral, TB "no follow-up needed", overconfident wording, invented symptom), the judge scored the broken part 1/5 in **all 5** ([`evals/judge_sensitivity.md`](evals/judge_sensitivity.md)). The guard blocks 3 of them.
- **Overall: the judge catches blatant errors but misses subtle ones** (`c32`, and `c07`'s invented limitations). So every problem a human finds becomes a deterministic check, and where possible a guard rule.

### Failure case: confidently wrong

![Grad-CAM failure case](docs/failure_case_gradcam.png)

This pneumonia X-ray is called Normal at 50.15%, so it is correctly **referred**. But 61% of the Grad-CAM weight lies on the image border, around the shoulders and corners, not in the lungs.

**13 of the 24 errors are confident** (not referred). For example, a Normal X-ray is called Pneumonia at 99.97%. The report for such a case faithfully says "very high confidence", and the eval scores it 5/5, because the eval checks the report against the classifier, not the truth. No report-level check can catch these.

### Things the evals caught

- **Guard false positive.** Gemini wrote "a screening aid, *not* a confirmed diagnosis", and the guard rejected it as a diagnosis claim. The guard now ignores negated mentions in the same clause. That exact output is a regression test (`tests/fixtures/`).
- **Ungrounded caveat.** In the pilot, the judge marked down a report for saying the model was evaluated on "two public datasets". The fact was true, but it came from the agent's instructions rather than any tool output. It was moved into a tool, so it's now grounded.
- **Dismissed follow-up (found by human review, missed by the judge).** `c32` said "No immediate referral is required" for a 99.99% Pneumonia result. The guard now rejects any Pneumonia/TB report that says no referral, action or follow-up is needed, and the template always recommends clinician follow-up. In the latest run, `c32` reads "Clinician follow-up is advised…".
- **My own regression (found by the judge, missed by the guard).** The first prompt wording for that rule made 5 of 8 confident Pneumonia/TB reports open with "Refer: uncertain…", calling 99.99% results uncertain. The judge scored them 1–2. The prompt was fixed and the guard now rejects it; that run is kept as evidence (`evals/runs/20261020T071107Z`).

### Known limits

- **Source confound:** each disease comes from a different dataset (pneumonia: pediatric JPEGs; TB: adult PNGs; Normal: both). The model may partly learn the source. Normal recall is 100% on images from the TB dataset but 95.1% on images from the pneumonia dataset, and every metric above is in-distribution.
- **Slow reports:** about 58 s each on Groq's free tier, mostly waiting to stay under the 8k tokens/min limit (4 sequential LLM calls per report).
- **Grad-CAM** is coarse (a 3×2 grid, not a lung segmentation) and usually diffuse.
- **The model can skip a tool.** Nothing forces the report model to call all three tools (`c07` skipped one). The eval flags it, but the guard doesn't block it yet.
- **Unreviewed medical text:** the condition text in `agent/condition_info.json` is a draft that hasn't been reviewed by a clinician. The judge model is a Groq *Preview* model.

## Why the eval runner is async (and why that isn't threads)

Each eval case has two steps: the **report model** writes a report (about 58 s, almost all of it *waiting*, for the network and for Groq's 8,000-tokens-per-minute limit), then the **judge model** scores it (about 1 s, also waiting). The old runner did one thing at a time: report 1, judge 1, report 2, judge 2, and so on.

The new runner uses Python's `asyncio`. While report 2 is being written, judge 1 is already being scored. Think of one cook who puts the pasta on to boil and chops the salad while the water heats: there's still only one cook, but no time is lost standing at the stove. In the code:

- **One worker, two jobs.** One loop writes reports; one judge job scores whatever is ready. Whenever either is waiting for a reply or a rate-limit timer, the other runs.
- **Not threads.** Nothing runs at the same instant. The work is waiting, not computing, so threads would add risk (two threads writing the same results file or sharing one rate-limit budget) without making it faster.
- **Limits still hold.** Each model keeps its own token-aware budget. An `asyncio.Lock` makes waiters queue in order, an `asyncio.Semaphore` allows one request per model at a time, and every request has a timeout.
- **Same results.** Tests with fake clients show the records are identical to the old runner's; only the order of lines in the run file can differ. Resume, stop and fallback behave as before. `--sync` still runs the old way.

Simulated benchmark, 5 cases, fake clients with real timings scaled down ([`evals/results/benchmark_simulated.md`](evals/results/benchmark_simulated.md)):

| Scenario | Old (sync) | New (async) | Speed-up |
| --- | --- | --- | --- |
| Real ratio: report ~58 s, judge ~1 s | 293 s | 291 s | **1.01×** |
| Slow judge: 15 s | 363 s | 305 s | 1.19× |
| Equal times (best case) | 576 s | 348 s | 1.65× |

The honest result: with today's timings, async saves about 1%, because overlap can only hide the judge's time. The slow part is the report model waiting on its own token limit, and that can't overlap with itself. Async pays off when the judge is slow (a reasoning judge, or one hitting its own limits). The live benchmark is `python -m evals.benchmark --live`; it hasn't been run yet.

## Run it locally

```bash
git clone https://github.com/adi13apr/Clinical-Diagnostics-.git
cd Clinical-Diagnostics-
python -m venv venv && venv\Scripts\activate      # macOS/Linux: source venv/bin/activate
pip install -r requirements-dev.txt               # app + tests
streamlit run app.py
```

`.streamlit/secrets.toml` (never commit it):

```toml
[supabase]
url = "..."
key = "..."        # anon key
redirect_url = "https://clinical-diagnosis.streamlit.app/"   # where Google sign-in returns (local: http://localhost:8501/)

[groq]             # optional: without it, reports use the template
api_key = "..."
```

Run `sql/001`–`005` in the Supabase SQL Editor before deploying.

**Google sign-in** needs three settings outside the code:
1. Supabase → Authentication → Providers → **Google**: enable it, and paste a Google Cloud OAuth client ID and secret.
2. Supabase → Authentication → URL Configuration → **Redirect URLs**: add `https://clinical-diagnosis.streamlit.app/**` and `http://localhost:8501/**`.
3. Google Cloud Console → that OAuth client → **Authorized redirect URIs**: add `https://<project-ref>.supabase.co/auth/v1/callback`.

| Task | Command |
| --- | --- |
| Tests (no network, data or GPU) | `pytest` |
| Re-score the committed eval run (what CI runs) | `python -m evals.run --replay` |
| Live eval (needs `data/` and a key) | `python -m evals.run --live` (async; add `--sync` for the old runner) |
| Old vs new runner, simulated / live | `python -m evals.benchmark --simulated` / `--live` |
| Re-fit calibration (needs the Kaggle datasets in `data/`) | `python scripts/calibrate.py` |

`CLAUDE.md` describes the code layout and conventions in detail.

## API (backend for the React UI)

`api/` is a FastAPI server built on the same services as the Streamlit app: same model, calibration, report agent (cache, guard, fallback), tracing and Supabase tables. The Streamlit app is unchanged.

| Endpoint | What it does |
| --- | --- |
| `POST /predict` | Upload a JPEG/PNG (max 10 MB). Returns class, calibrated probabilities, `confident`/`refer`, and a `trace_id` |
| `GET /gradcam/{trace_id}` | Grad-CAM figure (PNG) for that upload |
| `GET /gradcam/{trace_id}/map` | The raw 7×7 Grad-CAM map plus a region summary (JSON), so a client can draw its own overlay |
| `POST /report/{trace_id}` | AI report, with the same guard and template fallback as the app |
| `GET /history` | Your past predictions: `limit`, `offset`, `label`, `status`, `date_from`, `date_to` |
| `GET /analytics` | Daily volume, referral rate, average latency per pipeline step |
| `GET /traces/{trace_id}` | Every step of one prediction, for the trace viewer |
| `GET /health` | Is the model loaded and calibrated; is the LLM configured. No login needed |

- **Login:** every endpoint except `/health` needs `Authorization: Bearer <Supabase access token>`. The server checks the token, then queries Supabase *as that user*, so row-level security returns only their own rows.
- **Uploads stay in memory.** They're never written to disk, and `/gradcam` and `/report` can use them for an hour. After a restart, call `/predict` again.
- **History** comes from the `predictions` log, so it has no patient names.

```bash
pip install -r requirements-api.txt
uvicorn api.main:create_app --factory --port 8000     # docs at http://localhost:8000/docs
```

Locally it reads `.streamlit/secrets.toml`. On a server, it uses environment variables:

| Variable | Meaning |
| --- | --- |
| `SUPABASE_URL`, `SUPABASE_KEY` | Project URL and anon key (required) |
| `SUPABASE_JWT_SECRET` | Only for projects still on the legacy HS256 JWT secret. Without it, those tokens are checked by asking Supabase (one extra request each) |
| `FRONTEND_URL` | Deployed React app's origin(s), comma-separated. `localhost` on any port is always allowed |
| `GROQ_API_KEY` | Optional. Without it, reports use the template |
| `MAX_UPLOAD_MB` | Default 10 |

**Hugging Face Spaces:** create a Docker Space and push this repo to it. The Space's `README.md` needs `sdk: docker` and `app_port: 7860` in its front matter. Set the variables above as Space secrets. The checkpoint is over 10 MB, so the Space repo has to store `*.pth` with Git LFS.

## React UI (`web/`)

A browser front end for the API, in plain JavaScript: Vite, React, Tailwind, shadcn/ui, Framer Motion, Recharts, TanStack Query, React Router and supabase-js. Details are in [`web/README.md`](web/README.md).

- **Diagnose:** drag-and-drop upload, a result card with an animated confidence gauge, a Confident/Refer banner, a Grad-CAM overlay with an opacity slider, the AI report, and a PDF you download.
- **History, Analytics, Traces:** a searchable and filterable table, charts (each with a table view), and a step-by-step timeline of one prediction.
- **Design:** calm blue/teal, light and dark, works on phone, tablet and desktop, keyboard accessible.
- **Privacy:** the PDF is built in the browser, so the optional patient name never reaches the server.

```bash
cd web && cp .env.example .env.local   # set VITE_API_URL, VITE_SUPABASE_URL, VITE_SUPABASE_ANON_KEY
npm install && npm run dev             # http://localhost:5173 (needs the API running)
```

## License

This project is licensed under the MIT License.
