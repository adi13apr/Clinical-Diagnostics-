"""Pre-merge check: pytest, the eval replay and the API smoke test, summarised as PASS / FAIL / SKIP.

    python release_check.py [--api-url URL] [--skip-smoke] [--out summary.md]

Smoke test target: --api-url, else env API_URL, else the first of http://127.0.0.1:7860 / :7861 whose
/health answers. Token: env SUPABASE_ACCESS_TOKEN, else a password sign-in with DEMO_EMAIL /
DEMO_PASSWORD + SUPABASE_URL / SUPABASE_KEY (env, then .env in this checkout or the main one). The
token and password are never printed. A step that can't run is SKIP, never PASS. Exit 1 on any FAIL.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LOCAL_APIS = ("http://127.0.0.1:7860", "http://127.0.0.1:7861")


def main_checkout():
    try:
        out = subprocess.run(["git", "-C", str(REPO), "worktree", "list", "--porcelain"],
                             capture_output=True, text=True, check=True).stdout
        return Path(out.splitlines()[0].split(" ", 1)[1])
    except (OSError, subprocess.CalledProcessError, IndexError):
        return REPO


def read_env_file(path):
    values = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                values[k.strip()] = v.strip().strip('"').strip("'")
    return values


def config():
    """env first, then .env here, then the main checkout's .env (a worktree has none)."""
    merged = {**read_env_file(main_checkout() / ".env"), **read_env_file(REPO / ".env")}
    return lambda name: os.environ.get(name) or merged.get(name)


def last_line(out):
    return next((l for l in reversed(out.splitlines()) if l.strip()), "")


def replay_detail(out):
    fails = [l.removeprefix("FAIL: ") for l in out.splitlines() if l.startswith("FAIL: ")]
    hash_error = next((l for l in out.splitlines() if "Dataset hash mismatch" in l), None)
    return "; ".join(fails) or hash_error or "no deterministic pass rate below evals/baseline.json"


def smoke_detail(out):
    failed = [" ".join(l.split()[1:4]) for l in out.splitlines() if l.startswith("FAIL ")]
    return last_line(out) + (f" (failed: {'; '.join(failed)})" if failed else "")


def run_step(name, cmd, env=None, detail=last_line):
    started = time.perf_counter()
    p = subprocess.run([sys.executable, *cmd], cwd=REPO, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env)
    out = (p.stdout + p.stderr).strip()
    return {"step": name, "result": "PASS" if p.returncode == 0 else "FAIL", "seconds": time.perf_counter() - started,
            "detail": detail(out)[:200], "output": out}


def find_api(explicit, get):
    import httpx
    candidates = [explicit or get("API_URL")] if (explicit or get("API_URL")) else list(LOCAL_APIS)
    for url in candidates:
        try:
            if httpx.get(f"{url.rstrip('/')}/health", timeout=5).status_code == 200:
                return url, None
        except httpx.HTTPError:
            pass
    return None, f"no API answered /health at {', '.join(candidates)}"


def get_token(get):
    import httpx
    if get("SUPABASE_ACCESS_TOKEN"):
        return get("SUPABASE_ACCESS_TOKEN"), None
    email, password, url, key = (get(k) for k in ("DEMO_EMAIL", "DEMO_PASSWORD", "SUPABASE_URL", "SUPABASE_KEY"))
    if not (email and password and url and key):
        return None, "no SUPABASE_ACCESS_TOKEN and no DEMO_EMAIL/DEMO_PASSWORD + SUPABASE_URL/SUPABASE_KEY"
    try:
        r = httpx.post(f"{url.rstrip('/')}/auth/v1/token", params={"grant_type": "password"},
                       headers={"apikey": key}, json={"email": email, "password": password}, timeout=30)
    except httpx.HTTPError as e:
        return None, f"Supabase sign-in failed ({type(e).__name__})"
    if r.status_code != 200:
        return None, f"Supabase sign-in failed (HTTP {r.status_code})"
    return r.json()["access_token"], None


def smoke(api_url, skip):
    if skip:
        return {"step": "smoke test", "result": "SKIP", "seconds": 0, "detail": "--skip-smoke", "output": ""}
    get = config()
    url, why = find_api(api_url, get)
    token, why_token = (get_token(get) if url else (None, None))
    if not url or not token:
        return {"step": "smoke test", "result": "SKIP", "seconds": 0, "detail": why or why_token, "output": ""}
    env = dict(os.environ, API_URL=url, SUPABASE_ACCESS_TOKEN=token, PYTHONIOENCODING="utf-8")
    step = run_step("smoke test", ["scripts/smoke_test.py"], env, smoke_detail)
    step["detail"] = f"{url}: {step['detail']}"
    return step


def summary(steps):
    lines = ["| Step | Result | Time | Detail |", "| --- | --- | --- | --- |"]
    lines += [f"| {s['step']} | {s['result']} | {s['seconds']:.0f} s | {s['detail'].replace('|', '/')} |" for s in steps]
    failed = [s for s in steps if s["result"] == "FAIL"]
    skipped = [s for s in steps if s["result"] == "SKIP"]
    verdict = ("NOT READY: " + ", ".join(s["step"] for s in failed) + " failed") if failed else \
        ("READY, with skipped steps: " + ", ".join(s["step"] for s in skipped)) if skipped else "READY: every step passed"
    return "\n".join(lines) + f"\n\n**{verdict}**\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--api-url", help="API for the smoke test (default: API_URL, else a local one)")
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--out", help="also write the markdown summary here")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    steps = [run_step("pytest", ["-m", "pytest", "-q", "-p", "no:cacheprovider"]),
             run_step("eval replay", ["-m", "evals.run", "--replay"], detail=replay_detail),
             smoke(args.api_url, args.skip_smoke)]
    for s in steps:
        if s["result"] == "FAIL":
            print(f"--- {s['step']} output (last 40 lines) ---\n" + "\n".join(s["output"].splitlines()[-40:]) + "\n")
    md = summary(steps)
    print(md)
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
    return 1 if any(s["result"] == "FAIL" for s in steps) else 0


if __name__ == "__main__":
    sys.exit(main())
