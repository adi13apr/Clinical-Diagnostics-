"""Prove the Compose stack's distributed behaviour against the running stack, and print what was observed.

    docker compose up --build -d --scale worker=3
    python scripts/multi_replica_demo.py --jobs 3

What it checks, all through nginx (one port, round-robin over the API replicas):
  A. Topology: /health answered by more than one replica (X-Served-By), queue mode, workers listening.
  B. Shared state: an upload predicted on replica A is found by another replica (Grad-CAM map and PNG),
     and POST /report on two different replicas returns the same job, which a worker then writes.
  C. Shared Groq budget: N reports at once (pixel-distinct images, so the report cache can't answer),
     spread over the workers. Meanwhile the report model's rate-limit sorted sets in Redis are sampled,
     and afterwards the peak requests/tokens in any 60 s window are compared with config/llm.json.
     The llm_call spans from /traces give the tokens Groq reported and whether Groq ever answered 429.

Settings come from the shell, then from .env (the same file docker compose reads):
  API_URL                 default http://127.0.0.1:$API_PORT (API_PORT default 7860)
  REDIS_URL               default redis://127.0.0.1:$REDIS_PORT/0 (REDIS_PORT default 6380)
  SUPABASE_ACCESS_TOKEN   a signed-in user's token; or else
  DEMO_EMAIL + DEMO_PASSWORD with SUPABASE_URL + SUPABASE_KEY: signs in with Supabase's password grant
The token and password are never printed. Results also go to demo_results/<UTC time>.json (gitignored).
Exit code 1 if any check fails; a check that couldn't run is SKIP, never PASS.

Cost: about 5k gpt-oss tokens per LLM report (4 calls), so --jobs 3 is roughly 20k of the 200k/day quota.
Each call reserves about 3k tokens of the 8k/min budget up front (prompt estimate + max_completion_tokens),
so concurrent jobs queue behind each other; one that waits past the RQ job timeout (180 s) falls back
to the template, and the "written by the LLM" check shows it.
"""
import argparse
import io
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import httpx
import redis
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
IMAGE = REPO / "scripts" / "smoke_xray.jpg"
LLM_CONFIG = REPO / "config" / "llm.json"
RESULTS_DIR = REPO / "demo_results"
WINDOW_S = 60.0            # agent/rate_limit.py WINDOW_S
SAMPLE_EVERY_S = 0.25      # reservations stay in Redis at least 60 s, so none is missed
POLL_EVERY_S = 1.0
JOB_TIMEOUT_S = 900
MAX_TRIES = 10             # requests until a different replica answers (2 replicas: normally 1-2)


# ---- Settings ----

def read_env_file(path):
    values = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip('"').strip("'")
    return values


class Config:
    def __init__(self):
        self.file = read_env_file(REPO / ".env")

    def get(self, name, default=None):
        return os.environ.get(name) or self.file.get(name) or default


def sign_in(config):
    """(access token, how it was obtained). Never prints the token or the password."""
    token = config.get("SUPABASE_ACCESS_TOKEN")
    if token:
        return token.strip(), "SUPABASE_ACCESS_TOKEN"
    email, password = config.get("DEMO_EMAIL"), config.get("DEMO_PASSWORD")
    url, key = config.get("SUPABASE_URL"), config.get("SUPABASE_KEY")
    if not (email and password and url and key):
        sys.exit("Set SUPABASE_ACCESS_TOKEN, or DEMO_EMAIL + DEMO_PASSWORD (+ SUPABASE_URL, SUPABASE_KEY) in .env")
    r = httpx.post(f"{url.rstrip('/')}/auth/v1/token", params={"grant_type": "password"},
                   headers={"apikey": key}, json={"email": email, "password": password}, timeout=30)
    if r.status_code != 200:
        try:
            body = r.json()
            why = body.get("error_description") or body.get("msg") or body.get("error_code")
        except ValueError:
            why = r.text[:200]
        sys.exit(f"Supabase sign-in failed for {email}: HTTP {r.status_code} {why}")
    return r.json()["access_token"], f"password sign-in as {email}"


def report_model_limits():
    config = json.loads(LLM_CONFIG.read_text(encoding="utf-8"))
    provider = os.environ.get("LLM_PROVIDER") or config["provider"]
    model = config["providers"][provider]["report_model"]
    return model, config["limits"].get(model, {})


# ---- HTTP through the load balancer ----

class Api:
    def __init__(self, url, token):
        self.http = httpx.Client(base_url=url, timeout=120, headers={"Authorization": f"Bearer {token}"})

    def call(self, method, path, **kwargs):
        """(response, replica): replica is X-Served-By (the API container's hostname)."""
        r = self.http.request(method, path, **kwargs)
        return r, r.headers.get("x-served-by")

    def predict(self, png):
        r, replica = self.call("POST", "/predict", files={"file": ("xray.png", png, "image/png")})
        expect(r, "/predict")
        return r.json(), replica

    def until_other(self, method, path, avoid):
        """Repeat a request until a replica other than `avoid` answers. [(replica, status)], last response."""
        seen = []
        for _ in range(MAX_TRIES):
            r, replica = self.call(method, path)
            seen.append((replica, r.status_code))
            if replica != avoid:
                break
        return seen, r


def expect(r, what, ok=(200,)):
    if r.status_code not in ok:
        raise RuntimeError(f"{what}: HTTP {r.status_code} {r.text[:300]}")


def xray_variants(n, seed):
    """n PNGs of the test X-ray, each with a few changed pixels: same picture, different report-cache key.

    Random per run, so a second run isn't answered from the 24 h report cache either.
    """
    base = Image.open(IMAGE).convert("L")
    rng = random.Random(seed)
    images = []
    for _ in range(n):
        image = base.copy()
        for x in range(8):
            image.putpixel((x, 0), rng.randrange(256))
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        images.append(buf.getvalue())
    return images


# ---- Redis: rate-limit counters and RQ job records ----

class BudgetSampler(threading.Thread):
    """Reads the report model's requests/tokens sorted sets every SAMPLE_EVERY_S.

    Keeps every reservation it sees: id -> (time, tokens). record() swaps the reserved estimate for
    the real total_tokens, so the last value seen is the real count once the call has finished.
    """

    def __init__(self, client, model):
        super().__init__(daemon=True)
        self.client = client
        key = f"cdp:rate:{{{model}}}"
        self.requests_key, self.tokens_key = f"{key}:requests", f"{key}:tokens"
        self.reservations = {}
        self.samples = 0
        self.errors = []
        self.stop_event = threading.Event()

    def sample(self):
        times = dict(self.client.zrange(self.requests_key, 0, -1, withscores=True))
        tokens = {}
        for member in self.client.zrange(self.tokens_key, 0, -1):
            rid, n = member.rsplit(b"|", 1)
            tokens[rid] = int(n)
        for rid, at in times.items():
            if rid in tokens:
                self.reservations[rid.decode()] = (at, tokens[rid])
        self.samples += 1

    def run(self):
        while not self.stop_event.is_set():
            try:
                self.sample()
            except redis.RedisError as e:
                self.errors.append(f"{type(e).__name__}: {e}")
            self.stop_event.wait(SAMPLE_EVERY_S)

    def stop(self):
        self.stop_event.set()
        self.join()
        self.sample()


def peak_window(reservations, since=None):
    """Most requests and most tokens in any 60 s window, as the budget counts them.

    The budget prunes reservations at least 60 s old, so a window is [t, t + 60). The maximum is
    reached by a window starting at some reservation, so only those are tried.
    """
    events = sorted(reservations.values())
    best = {"requests": 0, "tokens": 0, "requests_at": None, "tokens_at": None}
    for i, (start, _) in enumerate(events):
        inside = [tok for at, tok in events[i:] if at < start + WINDOW_S]
        if len(inside) > best["requests"]:
            best["requests"], best["requests_at"] = len(inside), start
        if sum(inside) > best["tokens"]:
            best["tokens"], best["tokens_at"] = sum(inside), start
    if events:
        best["first"], best["last"] = events[0][0], events[-1][0]
    return best


def worker_of(client, job_id):
    """Hostname of the worker container that ran the job (RQ job hash -> worker hash), or None."""
    name = client.hget(f"rq:job:{job_id}", "worker_name")
    if not name:
        return None
    host = client.hget(f"rq:worker:{name.decode()}", "hostname")
    return host.decode() if host else f"worker {name.decode()[:8]}"


# ---- Reports ----

def wait_for_jobs(api, jobs, timeout_s=JOB_TIMEOUT_S):
    """Poll every job until done/failed. jobs: {job_id: first response}. Returns ({job_id: final}, poll replicas)."""
    final, replicas = {}, []
    pending = {job_id: job for job_id, job in jobs.items() if job["status"] in ("queued", "running")}
    final.update({job_id: job for job_id, job in jobs.items() if job_id not in pending})
    deadline = time.monotonic() + timeout_s
    while pending:
        if time.monotonic() > deadline:
            raise RuntimeError(f"{len(pending)} report jobs not finished after {timeout_s} s (are workers running?)")
        time.sleep(POLL_EVERY_S)
        for job_id in list(pending):
            r, replica = api.call("GET", f"/report/jobs/{job_id}")
            expect(r, "/report/jobs")
            replicas.append(replica)
            job = r.json()
            if job["status"] in ("done", "failed"):
                final[job_id] = job
                del pending[job_id]
    return final, replicas


def llm_spans(api, trace_id):
    """(llm_call spans of a trace or None if /traces can't return it, HTTP status)."""
    r, _ = api.call("GET", f"/traces/{trace_id}")
    if r.status_code != 200:
        return None, r.status_code
    return [s for s in r.json()["spans"] if s["name"] == "llm_call"], r.status_code


# ---- Output ----

class Checks:
    def __init__(self):
        self.rows = []

    def add(self, name, outcome, detail):
        """outcome: True (PASS), False (FAIL) or None (SKIP: couldn't be checked)."""
        label = {True: "PASS", False: "FAIL", None: "SKIP"}[outcome]
        self.rows.append({"check": name, "result": label, "detail": detail})
        print(f"  {label}  {name}: {detail}", flush=True)

    def failed(self):
        return any(r["result"] == "FAIL" for r in self.rows)


def short(host):
    return host[:12] if host else "?"


def utc(ts):
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%H:%M:%S") if ts else None


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--jobs", type=int, default=3, help="concurrent report jobs for the budget check")
    args = parser.parse_args()

    config = Config()
    api_url = config.get("API_URL") or f"http://127.0.0.1:{config.get('API_PORT', '7860')}"
    redis_url = config.get("REDIS_URL") or f"redis://127.0.0.1:{config.get('REDIS_PORT', '6380')}/0"
    model, limits = report_model_limits()
    token, auth_source = sign_in(config)
    api = Api(api_url.rstrip("/"), token)
    store = redis.Redis.from_url(redis_url, socket_connect_timeout=3, socket_timeout=5)
    try:
        store.ping()
    except redis.RedisError as e:
        sys.exit(f"Redis at {redis_url} is unreachable ({type(e).__name__}); is the stack up with its Redis port published?")

    started = datetime.now(timezone.utc)
    out = {"started": started.isoformat(), "api_url": api_url, "auth": auth_source, "report_model": model,
           "limits": {k: limits.get(k) for k in ("rpm", "tpm", "rpd", "tpd")}, "jobs_requested": args.jobs}
    checks = Checks()
    print(f"API {api_url} (through nginx)   Redis {redis_url}   auth: {auth_source}")
    print(f"Report model {model}: rpm {limits.get('rpm')}, tpm {limits.get('tpm')} (config/llm.json)\n")

    sampler = BudgetSampler(store, model)
    sampler.start()  # from the start, so step B's report counts toward the window too
    images = xray_variants(args.jobs + 1, seed=started.timestamp())

    # ---- A. Topology ----
    print("A. Topology")
    seen = []
    for _ in range(6):
        r, replica = api.call("GET", "/health")
        expect(r, "/health")
        seen.append(replica)
    health = r.json()
    replicas = sorted({s for s in seen if s})
    out["health"] = {"replicas_seen": replicas, "sequence": seen, "report_mode": health["report_mode"],
                     "report_workers": health["report_workers"], "llm_configured": health["llm_configured"]}
    if None in seen:
        checks.add("X-Served-By", False, "missing on some responses (SERVED_BY_HEADER not set, or not via nginx?)")
    checks.add("load balancing", len(replicas) >= 2,
               f"6 x /health answered by {len(replicas)} replicas, in order {' '.join(short(s) for s in seen)}")
    checks.add("queue mode", health["report_mode"] == "queue",
               f"report_mode={health['report_mode']}, {health['report_workers']} workers listening, "
               f"llm_configured={health['llm_configured']}")

    # ---- B. Shared state ----
    print("\nB. Shared state")
    prediction, replica_a = api.predict(images[0])
    trace_id = prediction["trace_id"]
    out["shared_state"] = {"trace_id": trace_id, "predict_replica": replica_a,
                           "prediction": f"{prediction['predicted_class']} {prediction['confidence']:.4f} ({prediction['status']})"}
    checks.add("predict", True, f"replica A = {short(replica_a)}: {out['shared_state']['prediction']}, trace {trace_id}")
    checks.add("upload in Redis", bool(store.exists(f"cdp:upload:{trace_id}")),
               f"key cdp:upload:{trace_id} exists, TTL {store.ttl(f'cdp:upload:{trace_id}')} s")

    for path in (f"/gradcam/{trace_id}/map", f"/gradcam/{trace_id}"):
        tries, r = api.until_other("GET", path, avoid=replica_a)
        other, status = tries[-1]
        out["shared_state"][path.replace(trace_id, "{id}")] = tries
        checks.add(f"GET {path.replace(trace_id, '{id}')}", other != replica_a and status == 200,
                   f"{len(tries)} request(s): " + ", ".join(f"{short(h)} -> {s}" for h, s in tries))

    r, first_replica = api.call("POST", f"/report/{trace_id}")
    expect(r, "POST /report", ok=(200, 202))
    first = r.json()
    tries, r = api.until_other("POST", f"/report/{trace_id}", avoid=first_replica)
    expect(r, "POST /report", ok=(200, 202))
    second, second_replica = r.json(), tries[-1][0]
    same_job = first["job_id"] is not None and first["job_id"] == second["job_id"]
    checks.add("POST /report on two replicas", second_replica != first_replica and same_job,
               f"{short(first_replica)} -> job {str(first['job_id'])[:8]} ({first['status']}), "
               f"{short(second_replica)} -> job {str(second['job_id'])[:8]} ({second['status']})")
    final, poll_replicas = wait_for_jobs(api, {first["job_id"]: first} if first["job_id"] else {})
    job = final.get(first["job_id"], first)
    worker = worker_of(store, first["job_id"]) if first["job_id"] else None
    result = job.get("result") or {}
    out["shared_state"]["report"] = {
        "job_id": first["job_id"], "post_replicas": [first_replica, second_replica], "status": job["status"],
        "poll_replicas": sorted({p for p in poll_replicas if p}), "polls": len(poll_replicas), "worker": worker,
        "source": result.get("source"), "cached": result.get("cached"), "reason": result.get("reason")}
    checks.add("report written by a worker", job["status"] == "done" and worker is not None,
               f"status {job['status']}, worker {short(worker)}, source={result.get('source')}, "
               f"polled {len(poll_replicas)}x on {len(set(poll_replicas))} replica(s)")

    # ---- C. Shared budget under concurrent reports ----
    print(f"\nC. {args.jobs} concurrent reports, one shared budget")
    uploads = [api.predict(png)[0]["trace_id"] for png in images[1:]]
    fired = time.time()
    with ThreadPoolExecutor(len(uploads)) as pool:
        posted = list(pool.map(lambda t: api.call("POST", f"/report/{t}"), uploads))
    jobs = {}
    for r, _ in posted:
        expect(r, "POST /report", ok=(200, 202))
        jobs[r.json()["job_id"]] = r.json()
    final, poll_replicas = wait_for_jobs(api, jobs)
    took = time.time() - fired
    sampler.stop()

    per_job = []
    for job_id, job in final.items():
        result = job.get("result") or {}
        per_job.append({"job_id": job_id, "trace_id": job["trace_id"], "status": job["status"],
                        "worker": worker_of(store, job_id), "source": result.get("source"),
                        "cached": result.get("cached"), "reason": result.get("reason"), "error": job.get("error")})
    workers = {}
    for j in per_job:
        workers[j["worker"]] = workers.get(j["worker"], 0) + 1
    sources = {}
    for j in per_job:
        key = f"{j['status']}/{j['source']}" + ("/cached" if j["cached"] else "")
        sources[key] = sources.get(key, 0) + 1
    out["concurrent"] = {"jobs": per_job, "seconds_until_all_done": round(took, 1), "by_worker": workers,
                         "outcomes": sources, "poll_replicas": sorted({p for p in poll_replicas if p})}
    checks.add("jobs finished", all(j["status"] == "done" for j in per_job),
               f"{len(per_job)} jobs in {took:.0f} s: {sources}")
    # A template fallback still "finishes"; say why, e.g. JobTimeoutException when a job waited on the
    # shared budget past the RQ job timeout (180 s), or APIConnectionError
    fallbacks = {}
    for j in per_job:
        if j["source"] != "llm":
            why = j["reason"] or j["error"] or j["status"]
            fallbacks[why] = fallbacks.get(why, 0) + 1
    written = len(per_job) - sum(fallbacks.values())
    checks.add("written by the LLM", (written == len(per_job)) if health["llm_configured"] else None,
               f"{written} of {len(per_job)}" + (f"; fallbacks: {fallbacks}" if fallbacks else "")
               + ("" if health["llm_configured"] else " (no GROQ_API_KEY: template reports)"))
    checks.add("spread over workers", len([w for w in workers if w]) >= min(2, health["report_workers"] or 1),
               ", ".join(f"{short(w)}: {n}" for w, n in workers.items()))

    # Budget as the limiter saw it (Redis). The peak counts every reservation seen, including any made
    # just before this run (they share the window); the totals count only this run's.
    peak = peak_window(sampler.reservations)
    ours = [t for at, t in sampler.reservations.values() if at >= started.timestamp()]
    total_tokens = sum(ours)
    out["budget_redis"] = {"reservations": len(sampler.reservations), "reservations_this_run": len(ours),
                           "tokens_this_run": total_tokens,
                           "samples": sampler.samples, "sampler_errors": sampler.errors[:5],
                           "peak_requests_60s": peak["requests"], "peak_tokens_60s": peak["tokens"],
                           "peak_tokens_window_start": utc(peak["tokens_at"]),
                           "first": utc(peak.get("first")), "last": utc(peak.get("last"))}
    if not ours:
        checks.add("budget (Redis)", None, "this run reserved nothing: no LLM calls (GROQ_API_KEY empty, or all cached)")
    else:
        rpm, tpm = limits.get("rpm"), limits.get("tpm")
        span_s = peak["last"] - peak["first"]
        checks.add("requests/min (Redis)", rpm is None or peak["requests"] <= rpm,
                   f"peak {peak['requests']} in any 60 s (limit {rpm}); {len(sampler.reservations)} LLM calls seen "
                   f"over {span_s:.0f} s, {sampler.samples} samples")
        checks.add("tokens/min (Redis)", tpm is None or peak["tokens"] <= tpm,
                   f"peak {peak['tokens']} in any 60 s (limit {tpm}), window from {utc(peak['tokens_at'])} UTC; "
                   f"{total_tokens} tokens in this run's {len(ours)} calls")

    # Budget as Groq reported it (llm_call spans)
    trace_ids = [trace_id] + uploads
    spans, missing, statuses = [], [], {}
    for t in trace_ids:
        found, status = llm_spans(api, t)
        statuses[status] = statuses.get(status, 0) + 1
        if found is None:
            missing.append(t)
        else:
            spans.extend(found)
    def span_tokens(s):
        return ((s["attributes"].get("input_tokens") or 0) + (s["attributes"].get("output_tokens") or 0)
                + (s["attributes"].get("thinking_tokens") or 0))

    tokens = [span_tokens(s) for s in spans]
    # The Redis numbers are the report model's budget only; with REPORT_PIPELINE=multi the verifier's
    # calls (another model, its own budget) are in the traces too, so compare like with like
    by_model = {}
    for s in spans:
        m = by_model.setdefault(s["attributes"].get("model"), {"llm_calls": 0, "tokens": 0})
        m["llm_calls"] += 1
        m["tokens"] += span_tokens(s)
    report_tokens = by_model.get(out["report_model"], {}).get("tokens", 0)
    retries = sum(s["attributes"].get("rate_limit_retries") or 0 for s in spans)
    errors = {}
    for s in spans:
        if s["status"] == "error":
            errors[s["error_type"]] = errors.get(s["error_type"], 0) + 1
    out["budget_traces"] = {"traces_read": len(trace_ids) - len(missing), "traces_missing": len(missing),
                            "llm_calls": len(spans), "tokens": sum(tokens), "by_model": by_model,
                            "groq_429_retries": retries, "llm_call_errors": errors, "traces_http_status": statuses}
    if missing == trace_ids:
        checks.add("Groq 429s (traces)", None, f"no trace readable through /traces (HTTP status: count {statuses}; "
                   "404 usually means the spans never reached Supabase: is sql/006 applied?)")
    else:
        rate_limited = retries + errors.get("RateLimitError", 0)
        partial = f" ({len(missing)} of {len(trace_ids)} traces missing)" if missing else ""
        checks.add("Groq 429s (traces)", rate_limited == 0,
                   f"{len(spans)} llm_call spans, {retries} rate-limit retries, errors {errors or 'none'}{partial}")
        others = {m: v["tokens"] for m, v in by_model.items() if m != out["report_model"]}
        checks.add("tokens: traces vs Redis", None if missing else report_tokens == total_tokens,
                   f"{out['report_model']}: traces {report_tokens}, Redis {total_tokens}"
                   + (f"; other models (own budgets, not sampled): {others}" if others else "")
                   + (" (a call that failed keeps its reserved estimate in Redis)" if report_tokens != total_tokens else ""))

    out["checks"] = checks.rows
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{started.strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    passed = sum(r["result"] == "PASS" for r in checks.rows)
    skipped = sum(r["result"] == "SKIP" for r in checks.rows)
    print(f"\n{passed} passed, {len(checks.rows) - passed - skipped} failed, {skipped} skipped. "
          f"Raw numbers: {path.relative_to(REPO)}")
    print("Note: the Redis numbers are what the shared limiter reserved (estimates replaced by real usage once "
          "each call finished); Groq's own verdict is the 429 count. Other users of the same Groq key "
          "(the tunnel API, Streamlit Cloud) aren't in this budget.")
    sys.exit(1 if checks.failed() else 0)


if __name__ == "__main__":
    main()
