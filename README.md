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

36 test cases, made up of every model error, every refer case, and 6 correct ones ([`evals/results/2026-10-18.md`](evals/results/2026-10-18.md)):

- **Deterministic checks: 100% on all 9.** These are: class, probability and status match the classifier; refer cases recommend clinician review; no confirmed-diagnosis claim; limitations present; length limits; schema valid; and the guard still agrees with the decisions recorded in the run. CI replays this run on every push and fails if any rate drops.
- All 36 reports were written by the LLM: no template fallbacks and no guard rejections.
- An LLM judge (`qwen/qwen3.8-27b`, a different model family from the report writer) scored 4.78–4.94 out of 5 on faithfulness, uncertainty language, actionability and plain language. **The judge is not validated:** the 15-report human scoring sheet (`evals/human_labels.csv`) hasn't been filled in yet, and the scores sit near the ceiling.

### Failure case: confidently wrong

![Grad-CAM failure case](docs/failure_case_gradcam.png)

This pneumonia X-ray is called Normal at 50.15%, so it is correctly **referred**. But 61% of the Grad-CAM weight lies on the image border, around the shoulders and corners, not in the lungs.

**13 of the 24 errors are confident** (not referred). For example, a Normal X-ray is called Pneumonia at 99.97%. The report for such a case faithfully says "very high confidence", and the eval scores it 5/5, because the eval checks the report against the classifier, not the truth. No report-level check can catch these.

### Two things the evals caught

- **Guard false positive.** Gemini wrote "a screening aid, *not* a confirmed diagnosis", and the guard rejected it as a diagnosis claim. The guard now ignores negated mentions in the same clause. That exact output is a regression test (`tests/fixtures/`).
- **Ungrounded caveat.** In the pilot, the judge marked down a report for saying the model was evaluated on "two public datasets". The fact was true, but it came from the agent's instructions rather than any tool output. It was moved into a tool, so it's now grounded; all 36 reports cite it.

### Known limits

- **Source confound:** each disease comes from a different dataset (pneumonia: pediatric JPEGs; TB: adult PNGs; Normal: both). The model may partly learn the source. Normal recall is 100% on images from the TB dataset but 95.1% on images from the pneumonia dataset, and every metric above is in-distribution.
- **Slow reports:** about 58 s each on Groq's free tier, mostly waiting to stay under the 8k tokens/min limit (4 sequential LLM calls per report).
- **Grad-CAM** is coarse (a 3×2 grid, not a lung segmentation) and usually diffuse.
- **Unreviewed medical text:** the condition text in `agent/condition_info.json` is a draft that hasn't been reviewed by a clinician. The judge model is a Groq *Preview* model.

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

[groq]             # optional: without it, reports use the template
api_key = "..."
```

Run `sql/001`–`004` in the Supabase SQL Editor before deploying.

| Task | Command |
| --- | --- |
| Tests (no network, data or GPU) | `pytest` |
| Re-score the committed eval run (what CI runs) | `python -m evals.run --replay` |
| Live eval (needs `data/` and a key) | `python -m evals.run --live` |
| Re-fit calibration (needs the Kaggle datasets in `data/`) | `python scripts/calibrate.py` |

`CLAUDE.md` describes the code layout and conventions in detail.

## License

This project is licensed under the MIT License.
