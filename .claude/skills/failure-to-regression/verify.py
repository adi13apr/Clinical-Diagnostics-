"""Run pytest and the eval replay, and say whether a replay failure is the expected one.

    python verify.py [--case c07 ...]

Exit 0: both pass. Exit 2: pytest passes and the replay fails only in the way a guard change or a new
case must (the new guard rejects the recorded final report of the --case cases, or cases.jsonl changed
so the hash no longer matches): run the live eval next. Exit 1: anything else.
"""
import argparse
import re
import subprocess
import sys

from _repo import REPO


def run(cmd):
    p = subprocess.run([sys.executable, *cmd], cwd=REPO, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=None)
    return p.returncode, p.stdout + p.stderr


def guard_replay_failures():
    """Case ids in the baseline run whose recorded verdicts today's guard no longer agrees with."""
    from evals.checks import check_guard_replay_consistent
    from evals.run import load_baseline, load_run
    records = load_run(REPO / load_baseline()["run"])
    return sorted(cid for cid, r in records.items() if r.get("report") and check_guard_replay_consistent(r) is False)


def interpret_replay(code, output, targets, dataset_changed):
    """("PASS" | "EXPECTED" | "FAIL", explanation)."""
    if code == 0:
        return "PASS", "replay matches the baseline"
    if "Dataset hash mismatch" in output:
        if dataset_changed:
            return "EXPECTED", "cases.jsonl changed, so the baseline run no longer matches: a new live run is needed"
        return "FAIL", "dataset hash mismatch, but cases.jsonl has no uncommitted change"
    drops = set(re.findall(r"^FAIL: (\w+) pass rate", output, re.M))
    if drops == {"guard_replay_consistent"} and targets:
        failing = guard_replay_failures()
        if failing and all(any(cid == t or cid.startswith(t + "-") for t in targets) for cid in failing):
            return "EXPECTED", (f"the new guard rejects the recorded final report of {', '.join(failing)} "
                                "(intended): a new live run and baseline are needed")
        return "FAIL", f"guard_replay_consistent dropped on {', '.join(failing) or '?'}, not only on {targets}"
    return "FAIL", f"pass rates below baseline: {', '.join(sorted(drops)) or output.strip()[-300:]}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--case", action="append", default=[], help="case(s) this fix targets, e.g. c07")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    code, out = run(["-m", "pytest", "-q", "-p", "no:cacheprovider"])
    tail = [l for l in out.strip().splitlines() if l.strip()][-1:]
    pytest_ok = code == 0
    print(f"pytest: {'PASS' if pytest_ok else 'FAIL'}  {tail[0] if tail else ''}")
    if not pytest_ok:
        print(out[-3000:])

    diff = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "evals/cases.jsonl"], cwd=REPO)
    code, out = run(["-m", "evals.run", "--replay"])
    verdict, why = interpret_replay(code, out, args.case, diff.returncode != 0)
    print(f"replay: {verdict}  {why}")
    if verdict == "FAIL":
        print(out[-2000:])
    if not pytest_ok or verdict == "FAIL":
        return 1
    if verdict == "EXPECTED":
        print("Next: python .claude/skills/failure-to-regression/live_eval.py start")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
