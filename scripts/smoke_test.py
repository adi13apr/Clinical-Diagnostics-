"""Smoke-test a deployed API end to end and print PASS/FAIL per step.

    $env:API_URL = "https://<user>-<space>.hf.space"
    $env:SUPABASE_ACCESS_TOKEN = "<access token from the signed-in web app>"
    python scripts/smoke_test.py

Steps: /health, POST /predict, /gradcam/{id}/map, POST /report/{id} (then GET /report/jobs/{job_id}
until done when the API queues reports for a worker), /history, /traces/{id}.
The token is sent only to API_URL and never printed. Exit code 1 if any step fails.

Test image: scripts/smoke_xray.jpg, a downsized test-split X-ray from Kermany et al.,
"Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification"
(Mendeley Data, CC BY 4.0). SMOKE_IMAGE=path uses another JPEG/PNG.
"""
import os
import sys
import time
from pathlib import Path

import httpx

DEFAULT_IMAGE = Path(__file__).resolve().parent / "smoke_xray.jpg"
WAKE_TIMEOUT_S = 300     # a sleeping free Space takes a minute or two to start
REPORT_TIMEOUT_S = 240   # a report can wait ~60 s on Groq's token limit
REPORT_POLL_S = 2


class Smoke:
    def __init__(self, api_url, token):
        self.client = httpx.Client(base_url=api_url, timeout=60, follow_redirects=True,
                                   headers={"Authorization": f"Bearer {token}"} if token else {})
        self.results = []

    def step(self, name, fn):
        """Run one step; a step returns a short detail string or raises."""
        start = time.perf_counter()
        try:
            detail = fn()
            ok = True
        except Exception as e:
            detail, ok = f"{type(e).__name__}: {e}", False
        took = time.perf_counter() - start
        self.results.append(ok)
        print(f"{'PASS' if ok else 'FAIL'}  {name:<22} {took:6.1f}s  {detail}", flush=True)
        return ok


def check(response, what):
    if response.status_code != 200:
        raise AssertionError(f"{what}: HTTP {response.status_code} {response.text[:200]}")
    return response.json()


def main():
    api_url = os.environ.get("API_URL", "").rstrip("/")
    token = os.environ.get("SUPABASE_ACCESS_TOKEN", "").strip()
    image = Path(os.environ.get("SMOKE_IMAGE") or DEFAULT_IMAGE)
    if not api_url:
        sys.exit("Set API_URL, e.g. https://<user>-<space>.hf.space")
    if not token:
        sys.exit("Set SUPABASE_ACCESS_TOKEN (see the README's Deployment section)")
    print(f"API: {api_url}   image: {image.name}\n")

    smoke = Smoke(api_url, token)
    state = {}

    def health():
        deadline = time.monotonic() + WAKE_TIMEOUT_S
        while True:
            try:
                r = smoke.client.get("/health", timeout=30)
                if r.status_code == 200:
                    break
            except httpx.TransportError:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError(f"no 200 from /health within {WAKE_TIMEOUT_S} s")
            time.sleep(5)
        h = r.json()
        for field in ("model_loaded", "calibrated", "auth_configured"):
            if not h.get(field):
                raise AssertionError(f"{field} is {h.get(field)}: {h}")
        return f"T={h['temperature']:.4f}, llm_configured={h['llm_configured']}"

    def predict():
        mime = "image/png" if image.suffix.lower() == ".png" else "image/jpeg"
        with open(image, "rb") as f:
            r = smoke.client.post("/predict", files={"file": (image.name, f, mime)})
        p = check(r, "/predict")
        state["trace_id"], state["predicted_class"] = p["trace_id"], p["predicted_class"]
        warn = f", warnings: {p['warnings']}" if p["warnings"] else ""
        return f"{p['predicted_class']} {p['confidence']:.2%} ({p['status']}), trace {p['trace_id']}{warn}"

    def gradcam_map():
        m = check(smoke.client.get(f"/gradcam/{state['trace_id']}/map"), "/gradcam/map")
        shape = (len(m["cam"]), len(m["cam"][0]) if m["cam"] else 0)
        if shape != (7, 7):
            raise AssertionError(f"expected a 7x7 map, got {shape}")
        return f"7x7 map for {m['explained_class']}, peak zone {m['regions'].get('peak_zone')}"

    def report():
        r = smoke.client.post(f"/report/{state['trace_id']}", timeout=REPORT_TIMEOUT_S)
        if r.status_code not in (200, 202):
            raise AssertionError(f"/report: HTTP {r.status_code} {r.text[:200]}")
        job = r.json()
        deadline = time.monotonic() + REPORT_TIMEOUT_S
        while job["status"] in ("queued", "running") or (job["status"] == "done" and job["result"] is None):
            if time.monotonic() > deadline:
                raise AssertionError(f"job {job['job_id']} still {job['status']} after {REPORT_TIMEOUT_S} s "
                                     "(is a report worker running?)")
            time.sleep(REPORT_POLL_S)
            job = check(smoke.client.get(f"/report/jobs/{job['job_id']}"), "/report/jobs")
        if job["status"] == "failed":
            raise AssertionError(f"report job failed: {job['error']}")
        rep = job["result"]
        if rep["report"]["predicted_class"] != state["predicted_class"]:
            raise AssertionError("report class differs from the prediction")
        why = f" ({rep['reason']})" if rep["reason"] else ""
        mode = f"job {job['job_id'][:8]}" if job["job_id"] else "inline"
        return f"{mode}, source={rep['source']}, model={rep['model']}, cached={rep['cached']}{why}"

    def history():
        h = check(smoke.client.get("/history", params={"limit": 20}), "/history")
        if not any(item["trace_id"] == state["trace_id"] for item in h["items"]):
            raise AssertionError(f"trace {state['trace_id']} not in the latest {len(h['items'])} rows")
        return f"{h['total']} predictions in total, the new one is listed"

    def trace():
        t = check(smoke.client.get(f"/traces/{state['trace_id']}"), "/traces")
        names = {s["name"] for s in t["spans"]}
        missing = {"prediction", "classify"} - names
        if missing:
            raise AssertionError(f"missing spans {sorted(missing)}; got {sorted(names)}")
        return f"{len(t['spans'])} spans: {', '.join(sorted(names))}"

    if smoke.step("GET /health", health) and smoke.step("POST /predict", predict):
        smoke.step("GET /gradcam/{id}/map", gradcam_map)
        smoke.step("POST /report/{id}", report)
        smoke.step("GET /history", history)
        smoke.step("GET /traces/{id}", trace)

    passed = sum(smoke.results)
    print(f"\n{passed}/{len(smoke.results)} steps passed")
    sys.exit(0 if passed == len(smoke.results) == 6 else 1)


if __name__ == "__main__":
    main()
