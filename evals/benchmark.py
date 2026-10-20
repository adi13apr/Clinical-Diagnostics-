"""Old (sync) vs new (async) eval runner on 5 cases.

    python -m evals.benchmark --simulated   # fake latency clients, no network (labelled "simulated")
    python -m evals.benchmark --live        # real model + Groq, 5 cases each way (uses your daily quota)

Simulated mode drives the real pipelines (run_pipeline_sync / run_pipeline_async) with clients
that only sleep. Their per-case report and judge times are real measurements scaled down 50x so
the benchmark takes seconds. Live mode runs `evals.run --live --limit 5` twice (sync, then async)
into scratch run files that are never committed or used as a baseline.
"""
import argparse
import asyncio
import json
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "evals" / "results"
N_CASES = 5
SCALE = 50  # simulated times = real seconds / SCALE
# Real per-case report time: mean report_latency_ms of the baseline run (gpt-oss, paced by 8k tokens/min).
# Real judge time isn't in the older run records (judge latency is recorded from now on); ~1 s per call
# was observed live for qwen in instruct mode.
JUDGE_REAL_S = 1.0
SLOW_JUDGE_REAL_S = 15.0  # e.g. a reasoning judge, or one waiting on its own output-tokens-per-minute limit


def baseline_report_seconds():
    baseline = json.loads((REPO / "evals" / "baseline.json").read_text(encoding="utf-8"))
    latencies = []
    with open(REPO / baseline["run"], encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("agent_usage", {}).get("report_latency_ms"):
                latencies.append(r["agent_usage"]["report_latency_ms"] / 1000)
    return sum(latencies) / len(latencies), baseline["run"]


def scenarios():
    report_s, run = baseline_report_seconds()
    return run, [
        {"name": "Measured ratio (report ~58 s, judge ~1 s)", "report_real_s": report_s, "judge_real_s": JUDGE_REAL_S},
        {"name": "Slow judge (report ~58 s, judge 15 s)", "report_real_s": report_s, "judge_real_s": SLOW_JUDGE_REAL_S},
        {"name": "Equal times (best case for overlap)", "report_real_s": report_s, "judge_real_s": report_s},
    ]


def simulate(report_s, judge_s, n=N_CASES):
    """Wall time of the sync and async pipelines with clients that just sleep."""
    from evals import run as eval_run
    from evals.dataset import load_cases
    from evals.judge import MockJudge

    cases = load_cases()[:n]

    def record(case):
        return {"case_id": case.case_id, "report": {}, "tool_outputs": {}, "report_source": "llm",
                "agent_usage": {"llm_calls": 4}, "judge": None, "judge_error": None}

    def make_sync(case):
        time.sleep(report_s)
        return record(case)

    async def make_async(case):
        await asyncio.sleep(report_s)
        return record(case)

    quiet = dict(log=lambda s: None)
    sync_records, async_records = {}, {}
    started = time.perf_counter()
    eval_run.run_pipeline_sync(cases, sync_records, make_sync, MockJudge(delay_s=judge_s), lambda r: None, **quiet)
    sync_s = time.perf_counter() - started
    started = time.perf_counter()
    asyncio.run(eval_run.run_pipeline_async(cases, async_records, make_async, MockJudge(delay_s=judge_s),
                                            lambda r: None, **quiet))
    async_s = time.perf_counter() - started
    same = {k: {**v, "judge": {**v["judge"], "latency_ms": None}} for k, v in sync_records.items()} == \
           {k: {**v, "judge": {**v["judge"], "latency_ms": None}} for k, v in async_records.items()}
    return sync_s, async_s, same


def run_simulated():
    run, rows = scenarios()
    lines = [f"# Eval runner benchmark: **simulated** ({datetime.now(timezone.utc):%Y-%m-%d})", "",
             f"Fake clients that only sleep, {N_CASES} cases, no network. Per-case times are real measurements "
             f"scaled down {SCALE}x: report time is the mean of `{run}`; judge time is ~1 s as observed live "
             "(older run records don't store it). The pipelines are the real ones from `evals/run.py`.", "",
             "| Scenario | Sync (old) | Async (new) | Speed-up | Expected | Same records |",
             "| --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        r, j = row["report_real_s"] / SCALE, row["judge_real_s"] / SCALE
        sync_s, async_s, same = simulate(r, j)
        # Sync: n(R+J). Async: about nR + J when R >= J (the judge only adds time after the last report)
        expected = (N_CASES * (r + j)) / (N_CASES * max(r, j) + min(r, j))
        lines.append(f"| {row['name']} | {sync_s * SCALE:.0f} s | {async_s * SCALE:.0f} s | "
                     f"{sync_s / async_s:.2f}x | {expected:.2f}x | {'yes' if same else '**no**'} |")
    lines += ["", f"Times are shown scaled back up to real seconds (measured simulated time × {SCALE}).",
              "Live numbers: run `python -m evals.benchmark --live` (5 real cases each way)."]
    md = "\n".join(lines) + "\n"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "benchmark_simulated.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


def run_live(cooldown_s):
    """Real model + Groq: 5 cases with the sync runner, then 5 with the async runner (scratch run files)."""
    from evals import run as eval_run
    times = {}
    with tempfile.TemporaryDirectory() as tmp:
        for i, mode in enumerate(("sync", "async")):
            if i:
                print(f"Cooling down {cooldown_s} s so the first run's tokens leave the 1-minute windows...")
                time.sleep(cooldown_s)
            argv = ["--live", "--limit", str(N_CASES), "--run-file", str(Path(tmp) / f"{mode}.jsonl")]
            started = time.perf_counter()
            code = eval_run.main(argv + (["--sync"] if mode == "sync" else []))
            times[mode] = time.perf_counter() - started
            if code != 0:
                raise SystemExit(f"{mode} run stopped early (exit {code}); resume tomorrow or check quota.")
    md = (f"# Eval runner benchmark: **live** ({datetime.now(timezone.utc):%Y-%m-%d})\n\n"
          f"{N_CASES} real cases each way (model + Groq), scratch run files.\n\n"
          f"| Sync (old) | Async (new) | Speed-up |\n| --- | --- | --- |\n"
          f"| {times['sync']:.0f} s | {times['async']:.0f} s | {times['sync'] / times['async']:.2f}x |\n\n"
          "Report text differs between any two live runs (the LLM isn't deterministic), so content equality is "
          "proven by the tests with fake clients, not here.\n")
    (RESULTS_DIR / "benchmark_live.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--simulated", action="store_true")
    mode.add_argument("--live", action="store_true")
    ap.add_argument("--cooldown", type=float, default=65.0, help="seconds between the live sync and async runs")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    return run_simulated() if args.simulated else run_live(args.cooldown)


if __name__ == "__main__":
    sys.exit(main())
