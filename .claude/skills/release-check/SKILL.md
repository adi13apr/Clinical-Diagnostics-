---
name: release-check
description: Pre-merge check for this repo. Runs pytest, the eval replay (CI's baseline gate) and the API smoke test, then summarises PASS / FAIL / SKIP. Use before merging a branch or PR, or when asked whether a branch is ready to merge or release.
---

# Release check

1. In the branch's checkout, run with the venv python (in a worktree with no venv, use the main checkout's):
   `python .claude/skills/release-check/release_check.py --out <scratchpad>/release.md`.
   It takes 1 to 2 minutes. Exit 1 means at least one step failed.
2. **Smoke test.** It needs a running API: `--api-url`, `API_URL`, or a local one on port 7860/7861. It also needs a token: `SUPABASE_ACCESS_TOKEN`, or the demo account in `.env`. Without them the step is **SKIP**, and SKIP is not PASS: say what was missing.
   - If it fails because the running API is older than the branch (for example a response field is missing), say so. Don't rebuild without asking: the local container is the live API behind the Vercel app (`.\scripts\serve_local_api.ps1 -Rebuild`).
3. **Web UI.** If the branch changes `web/`, also run `cd web && npm run lint && npm run build` (CI's web job).
4. **Report** the table. For each FAIL, give the cause from the output (the failing test, the check below baseline, or the endpoint), and end with the verdict line. Don't fix or merge anything unless asked.
