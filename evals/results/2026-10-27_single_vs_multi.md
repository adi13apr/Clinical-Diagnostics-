# Single vs multi-agent report pipeline: `20261026T040807Z` vs `20261027T000140Z`

- Dataset: 36 cases, hash `c475a686fb48`; reports: single 36, multi 36
- Single: report openai/gpt-oss-120b
- Multi: evidence openai/gpt-oss-120b, verifier openai/gpt-oss-20b, writer openai/gpt-oss-120b (commit `61d670c907e7`)
- Judge: single qwen/qwen3.8-27b, multi qwen/qwen3.8-27b (same model, prompt and rubric)
- All numbers are measured on these two runs. Latency includes waiting for Groq's per-minute token limit, so it measures throughput under the free tier as much as model speed.

## Side by side

| Metric | Single | Multi |
| --- | --- | --- |
| Check: schema_valid | 100.0% (36/36) | 100.0% (36/36) |
| Check: class_matches_classifier | 100.0% (36/36) | 100.0% (36/36) |
| Check: probability_matches_classifier | 100.0% (36/36) | 100.0% (36/36) |
| Check: status_matches_classifier | 100.0% (36/36) | 100.0% (36/36) |
| Check: refer_includes_clinician_review | 100.0% (17/17) | 100.0% (17/17) |
| Check: no_confirmed_diagnosis_claim | 100.0% (36/36) | 100.0% (36/36) |
| Check: limitations_present | 100.0% (36/36) | 100.0% (36/36) |
| Check: length_limits | 100.0% (36/36) | 100.0% (36/36) |
| Check: guard_replay_consistent | 100.0% (36/36) | 100.0% (36/36) |
| Check: abnormal_result_not_dismissed | 100.0% (13/13) | 100.0% (13/13) |
| Check: confident_not_called_uncertain | 100.0% (19/19) | 100.0% (19/19) |
| Check: dataset_caveat_present | 100.0% (36/36) | 100.0% (36/36) |
| Judge: faithfulness (mean; scores ≤ 2) | 5.00; 0 | 4.86; 0 |
| Judge: uncertainty_language (mean; scores ≤ 2) | 4.81; 0 | 4.81; 1 |
| Judge: actionability (mean; scores ≤ 2) | 5.00; 0 | 4.81; 1 |
| Judge: plain_language (mean; scores ≤ 2) | 4.64; 0 | 4.61; 0 |
| Template fallbacks | 0 | 3 |
| Guard rejections (drafts / cases) | 2 / 2 | 0 / 0 |
| API-level model errors (e.g. invalid tool call) | 0 | 9 |
| Revisions (total; by guard / verifier; cases) | — (no revision step) | 2; 0 / 2; 2 |
| Tool skips (cases without all 3 tools) | 0 | 0 (re-asks: 2) |
| LLM calls per report (mean) | 4.06 | 5.42 |
| openai/gpt-oss-120b tokens per report (mean, in + out) | 5,249 | 3,614 |
| openai/gpt-oss-20b tokens per report (mean, in + out) | — | 1,795 |
| Judge tokens per report (mean) | 1,792 | 1,836 |
| Latency per report, median | 58.8 s | 54.7 s |
| Latency per report, p90 | 60.0 s | 59.8 s |
| Run span, first to last record | 29.1 min | 21.5 min |
| Nominal cost per report (list price, report models) | $0.00094 | $0.00106 |
| Nominal cost, all 36 / 36 reports | $0.0337 | $0.0382 |
| Billed cost (free tier) | $0.00 | $0.00 |

Fallback reasons: single none; multi `c06-NORMAL2-IM-0554-0001` verifier agent: The model's JSON output didn't match the schema (or was cut off).; `c13-person21_bacteria_73` verifier agent: The model's JSON output didn't match the schema (or was cut off).; `c14-person293_bacteria_1379` verifier agent: The model's JSON output didn't match the schema (or was cut off)..

## Multi-agent pipeline (reported, not gating CI)

- All three tools called: 36/36; evidence re-asks: 2 in 2 cases
- Revisions: 2 in 2 cases (guard 0, verifier 2; at most 1 per case)
- Verifier: 35 verdicts on 33 cases, 7.86 claims per verdict, 2 claims unsupported (c12-person1300_virus_2240, c34-Normal-1113)
- Verifier flagged, guard passed: 2 of 2 flagged drafts
- Guard rejected, verifier would pass: 0 of 0 rejected drafts (post-hoc verifier)
  - `c12-person1300_virus_2240` sent back by the verifier: unsupported: Otherwise, no immediate further imaging is required
  - `c34-Normal-1113` sent back by the verifier: unsupported: No specific follow‑up is required based on this result.

## Verifier audit

Post-hoc calls (evals/runs/20261027T000140Z.audit.json): verifier openai/gpt-oss-20b, claim checker qwen/qwen3.8-27b. Spot-check sheet: `evals/runs/20261027T000140Z.verifier_flags.md`.

- **Verifier false flags (per the checker): 0 of 2 unsupported claims** were found supported by the evidence. A human hasn't confirmed these yet; see the sheet.
- Guard-rejected drafts shown to the verifier afterwards: 2; it found every claim supported in 2 (the guard caught something the verifier would have let through):
  - `20261026T040807Z` `c35-Tuberculosis-327` step 4: 0 unsupported; guard: grounding check failed: status is 'confident', so the report must not call the result uncertain or use the refer sentence "Refer: uncertain, clinician review recommended"
  - `20261026T040807Z` `c36-Tuberculosis-337` step 4: 0 unsupported; guard: grounding check failed: status is 'confident', so the report must not call the result uncertain or use the refer sentence "Refer: uncertain, clinician review recommended"
- Judge drift check: 5 single-run reports re-judged today; 12/20 scores identical, mean change 0.00 (`c01` 5/5/5/5 → 5/5/5/5, `c09` 5/4/5/4 → 5/5/4/5, `c17` 5/5/5/4 → 5/5/5/5, `c25` 5/5/5/4 → 4/4/5/4, `c33` 5/4/5/5 → 5/5/5/4)

## Verifier on the planted bad reports

`evals/verifier_sensitivity.md` (openai/gpt-oss-20b, 2026-10-26): caught 5/5 bad reports (wrong_class, missing_referral, tb_no_follow_up, overconfident, invented_symptom); false alarms on 0/3 untouched controls. With the guard first, the pipeline stops 5/5.

## Decision rule

Multi becomes the default only if it is at least as good as single on every deterministic check AND has no more template fallbacks.

**Result: keep single** — template fallbacks: multi 3 > single 0


## What we learned

**Decision: keep `single` as the default.** The rule was: switch only if multi is at least as good on every
deterministic check AND has no more template fallbacks. The checks tie (12 of 12 at 100% in both), but multi
had **3 template fallbacks** (`c06`, `c13`, `c14`) and single had 0. Multi stays available behind
`REPORT_PIPELINE=multi`.

**Worse in multi**
- **The verifier's JSON is the weak link.** gpt-oss-20b failed Groq's strict `json_schema` check on 12 of its
  47 calls (400 `json_validate_failed`). Each time, one `findings[i].source` (the 5th to 9th finding) was a
  value outside `classifier | gradcam | condition_info | none`. 9 of the 36 first checks failed; the single
  retry fixed 6 and failed again on 3, which became the template. All 3 fallbacks come from this, none from a
  report-quality problem. A post-hoc re-check of 5 of the affected reports (2 calls each, 2026-10-27) didn't
  reproduce it (10 of 10 valid), so it is intermittent. The run records keep only the first 500 characters of
  the error, which cuts off Groq's `failed_generation`, so we can't say which value it invented.
- **Duplicated limitation in 9 of 33 LLM reports** (single: 0 of 36). The writer prompt asks it to copy every
  `model_limitations` item *and* says `dataset_caveat` must be one of them, so it sometimes lists the caveat
  twice. The judge took a point off faithfulness for it in 4 cases (`c01`, `c02`, `c24`, `c35`).
- **`c19` (Normal, confident):** multi recommended "If the patient has concerning symptoms, a clinician should
  evaluate them regardless of this result." The judge scored actionability 1 ("no next step provided"); the
  single report (which also says no referral is needed) got 5. The recommendation does give a conditional
  next step, so this looks harsh, but one judge score moving 4 points is beyond the noise measured below.
- More calls per report (5.42 vs 4.06) and a slightly higher nominal cost ($0.0382 vs $0.0337 for 36 reports;
  billed $0 on the free tier). Failed verifier calls report no usage, so multi's cost is a lower bound.

**Better in multi, or what only multi gives**
- **The verifier catches what the guard can't.** On the 5 planted bad reports it flagged 5 of 5, each time on
  the planted claim, with 0 false alarms on the 3 untouched originals. The guard alone stops 3 of 5 (it
  can't see overclaiming or invented symptoms). In the live run it sent back 2 drafts, both Normal results
  saying no follow-up or imaging is needed (`c12`, `c34`). The guard deliberately allows that wording for
  Normal results, and the claim checker agreed both claims are unsupported by the evidence (0 of 2 false flags).
- **Tools always called:** 36 of 36, with 2 re-asks (`c03`, `c18`). Single also called all three tools in
  this run, but nothing forces it to (`c07` skipped one on 20 Oct).
- 120b tokens per report fell from 5,249 to 3,614 (plus 1,795 on 20b's separate budget), and the 36 cases took
  21.5 min instead of 29.1 min. Per-report latency barely moved (median 54.7 s vs 58.8 s, p90 59.8 s vs
  60.0 s): both runs mostly wait for the 8k tokens/min window.
- **The guard still matters:** the single run's 2 guard-rejected drafts (`c35`, `c36`: confident TB opening
  with "Refer:") were judged fully supported by the post-hoc verifier. It would have let both through.

**How much to trust the judge numbers.** Re-judging 5 single-run reports a day later gave identical scores on
12 of 20 criteria (mean change 0.00). The judge is noisy by ±1 but not drifting. Of the 144 multi scores, 23
are below single's and 13 above, mostly by 1 point. Apart from `c19` and the duplicated limitation, the
judge differences are within that noise.

**To fix before trying multi again** (each needs a new full run and its own baseline):
1. Verifier schema: let the retry tell 20b the allowed values, or make `source` a free string validated in
   Pydantic. Also keep `failed_generation` in the recorded error detail.
2. Writer prompt: copy `model_limitations` once; the caveat is already in it.
3. Add a deterministic check for duplicated limitations, so this shows up in CI rather than only in the judge's
   reasons.
