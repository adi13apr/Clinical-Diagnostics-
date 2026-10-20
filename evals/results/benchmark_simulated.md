# Eval runner benchmark: **simulated** (2026-10-20)

Fake clients that only sleep, 5 cases, no network. Per-case times are real measurements scaled down 50x: report time is the mean of `evals/runs/20261020T075220Z.jsonl`; judge time is ~1 s as observed live (older run records don't store it). The pipelines are the real ones from `evals/run.py`.

| Scenario | Sync (old) | Async (new) | Speed-up | Expected | Same records |
| --- | --- | --- | --- | --- | --- |
| Measured ratio (report ~58 s, judge ~1 s) | 293 s | 291 s | 1.01x | 1.01x | yes |
| Slow judge (report ~58 s, judge 15 s) | 363 s | 305 s | 1.19x | 1.20x | yes |
| Equal times (best case for overlap) | 576 s | 348 s | 1.65x | 1.67x | yes |

Times are shown scaled back up to real seconds (measured simulated time × 50).
Live numbers: run `python -m evals.benchmark --live` (5 real cases each way).
