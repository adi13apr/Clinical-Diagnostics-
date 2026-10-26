"""Run the live eval (python -m evals.run --live) in the background and wait for it in short chunks.

    python live_eval.py start [--resume evals/runs/<run>.jsonl]
    python live_eval.py wait [--timeout 540]    # exit 0 finished, 2 still running, 1 stopped or failed
    python live_eval.py status

A full run is 36 cases at ~1 case/min (Groq free tier), longer than one shell command may block, so
`wait` returns after --timeout seconds and is called again. A stop (rate limit, network) prints the
resume command from the runner's log. In a worktree, data/ is linked to the main checkout's copy and
the provider's API key is read from the main checkout's .streamlit/secrets.toml (never printed).
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import tomllib
from pathlib import Path

from _repo import REPO, main_checkout, rel

STATE = Path(tempfile.gettempdir()) / "cdp-live-eval" / (re.sub(r"\W", "_", str(REPO)) + ".json")


def link_data():
    """Junction/symlink data/ to the main checkout's data/ when this checkout has none."""
    data, main_data = REPO / "data", main_checkout() / "data"
    if data.exists() or not main_data.is_dir() or main_data == data:
        return
    if os.name == "nt":
        subprocess.run(["cmd", "/c", "mklink", "/J", str(data), str(main_data)], check=True, capture_output=True)
    else:
        data.symlink_to(main_data, target_is_directory=True)
    print(f"Linked data/ -> {main_data}")


def llm_env():
    """os.environ plus the provider's key from the main checkout's secrets.toml, if the env lacks it."""
    from agent.llm_config import load_llm_config
    config = load_llm_config()
    settings = config["providers"][os.environ.get("LLM_PROVIDER") or config["provider"]]
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    secrets = main_checkout() / ".streamlit" / "secrets.toml"
    if not env.get(settings["env_key"]) and secrets.is_file():
        with open(secrets, "rb") as f:
            key = tomllib.load(f).get(settings["secret_section"], {}).get("api_key")
        if key:
            env[settings["env_key"]] = key
    if not env.get(settings["env_key"]):
        raise SystemExit(f"No {settings['env_key']} in the environment or {secrets}")
    return env


def start(resume=None):
    if STATE.exists() and not exit_code(load_state()):
        raise SystemExit(f"A live eval is already running (log {load_state()['log']}). Use wait or status.")
    link_data()
    STATE.parent.mkdir(parents=True, exist_ok=True)
    log = STATE.with_suffix(f".{time.strftime('%Y%m%dT%H%M%S')}.log")
    cmd = [sys.executable, "-m", "evals.run", "--live"] + (["--resume", resume] if resume else [])
    state = {"cmd": cmd, "log": str(log), "exit_file": str(log.with_suffix(".exit")), "started": time.time()}
    STATE.write_text(json.dumps(state), encoding="utf-8")
    # A small detached child runs the eval and writes its exit code, so `wait` needs no PID checks
    child = [sys.executable, __file__, "_child"]
    kwargs = {"cwd": REPO, "env": llm_env(), "stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL,
              "stderr": subprocess.DEVNULL}
    if os.name == "nt":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        try:
            subprocess.Popen(child, creationflags=flags | subprocess.CREATE_BREAKAWAY_FROM_JOB, **kwargs)
        except OSError:  # the parent's job object forbids breakaway
            subprocess.Popen(child, creationflags=flags, **kwargs)
    else:
        subprocess.Popen(child, start_new_session=True, **kwargs)
    print(f"Started: {' '.join(cmd[1:])}\nLog: {log}\nNext: python {rel(__file__)} wait")


def child():
    state = load_state()
    with open(state["log"], "w", encoding="utf-8") as log:
        code = subprocess.run(state["cmd"], cwd=REPO, stdout=log, stderr=subprocess.STDOUT).returncode
    Path(state["exit_file"]).write_text(str(code), encoding="utf-8")


def load_state():
    if not STATE.exists():
        raise SystemExit("No live eval started from this checkout.")
    return json.loads(STATE.read_text(encoding="utf-8"))


def exit_code(state):
    path = Path(state["exit_file"])
    return int(path.read_text()) if path.exists() else None


def report(state, code):
    lines = Path(state["log"]).read_text(encoding="utf-8", errors="replace").splitlines() \
        if Path(state["log"]).exists() else []
    run = next((m.group(1) for l in lines if (m := re.match(r"Run (\S+\.jsonl):", l))), None)
    done = [l for l in lines if re.match(r"\[\d+/\d+\]", l)]
    print(f"Run {run or '?'}: {len(done)} cases written, {(time.time() - state['started']) / 60:.0f} min")
    print("\n".join((done[-3:] if code is None else lines[-12:])))
    if code is None:
        print("Still running: call wait again.")
        return 2
    if code == 0:
        print(f"Finished. Next: compare_runs.py --after {run}")
        return 0
    print(f"Stopped (exit {code}). Resume after the cause clears (rate limits reset within a minute; "
          f"the daily cap at 00:00 UTC): python {rel(__file__)} start --resume {run}")
    return 1


def wait(timeout):
    state = load_state()
    deadline = time.time() + timeout
    while exit_code(state) is None and time.time() < deadline:
        time.sleep(10)
    return report(state, exit_code(state))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("action", choices=["start", "wait", "status", "_child"])
    ap.add_argument("--resume", help="with start: run file to continue")
    ap.add_argument("--timeout", type=int, default=540, help="with wait: seconds before returning (default 540)")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    if args.action == "start":
        start(args.resume)
        return 0
    if args.action == "_child":
        child()
        return 0
    return wait(args.timeout if args.action == "wait" else 0)


if __name__ == "__main__":
    sys.exit(main())
