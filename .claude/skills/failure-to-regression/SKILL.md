---
name: failure-to-regression
description: "Turn a report-agent failure in this repo into a permanent regression check (eval case, deterministic check or guard rule, fixture test, README entry, PR). Use when a bad AI report has been found, given as an eval case id (e.g. c07), a trace id, a saved MCP generate_report output, or a human-review or judge note. Not for classifier errors (a wrong class with a faithful report), which no report-level check can catch."
argument-hint: "<case id | trace id | MCP output file> [what went wrong]"
---

# Failure to regression check

Target: $ARGUMENTS

Scripts live in `.claude/skills/failure-to-regression/`. Run them with the repo's venv python
(`venv\Scripts\python.exe`; in a worktree with no venv, use the main checkout's). Each has `--help`.

## 0. Branch
Work in a new worktree, never `git checkout` in a shared folder:
`git worktree add ../Clinical-Diagnostics-<slug> -b fix/<slug> <base branch>`.

## a. Reproduce
- Eval case: `reproduce.py --case c07` (the baseline run; `--run` for another).
- Trace id: `reproduce.py --trace <id>` lists the steps (local traces only). Traces hold no report text. To get the exact output, call the MCP tools `classify_xray` and `generate_report` on the image and save `{"classification": ..., "report": ...}` to a file for `--mcp-output`.
- `generate_report` is cached per image and the LLM varies between runs. A recorded bad output is the evidence: a clean retry doesn't clear it.
- Note which checks fail, whether today's guard accepts the report, and the cause. If no check fails and the guard accepts it, nothing catches it yet; step c fixes that.

## b. Eval dataset
`add_case.py <relpath or path under data/>`.
- If the image is already a case, nothing changes.
- Train/val images are refused.
- A new case changes the dataset hash, so step d will need a live run.

## c. Check, guard rule, regression test
1. **Fixture** from the exact failing output: `reproduce.py ... --write-fixture <model>_<case>_<what> --description "<what went wrong; who found it>"`. Never edit the report text.
2. **Check**: add a deterministic one to `evals/checks.py` (`CHECKS`; `None` when not applicable; the docstring names the run and case) if none flags it, plus a unit test in `tests/test_evals.py`.
3. **Guard rule** in `check_grounding` (`agent/agent.py`), when the tool outputs decide it.
   - The rejection message tells the model how to fix the report, because it gets one retry.
   - `fallback_report` must still pass.
   - If the rule needs more than the classifier output, pass it through `evals.checks.guard_verdict`, the only place the eval calls the guard.
   - Change only the agent code the rule needs.
4. **Fixture expectation**: set `expect.guard_problem` to a phrase of the new message. `tests/test_regression_fixtures.py` enforces it.
5. **Agent tests** in `tests/test_agent.py`: a rejected draft that recovers on retry, and two rejections that fall back to the template.

## d. Verify
`verify.py --case <id>` exits:
- **0**: pytest and the replay pass.
- **1**: something is broken. Fix it; never weaken a check or the baseline to get green.
- **2**: expected. The new guard rejects the recorded report, or the dataset changed. Run the live eval:
  1. `live_eval.py start`, then `live_eval.py wait` until it exits 0. A full run takes about 45 min and about 220k Groq tokens. `wait` blocks up to 9 min, so give that shell call a 10-minute timeout. Exit 2 means still running (call `wait` again); exit 1 means stopped (resume as printed).
  2. `compare_runs.py --after <new run> --out <scratchpad>/compare.md`. **Exit 3** (more template fallbacks or guard rejections than the baseline run) means **stop**: show the user the table and don't update the baseline.
  3. `python -m evals.run --replay --run <new run> --update-baseline`. Write `evals/runs/<run>.notes.md` (why this run exists), then `verify.py` again, which must exit 0. Commit the run, its notes, `evals/results/<date>.md` and `evals/baseline.json`.

## e. Docs
- Add one bullet under README "Things the evals caught", in the style of the others: `**<Name> (found by <who>, missed by <what>).** <what the report said>. <what blocks it now>. <evidence>.` At most 4 sentences.
- Fix any README or CLAUDE.md text the change made untrue: known limits, guard bullets, the baseline run id, check counts. Numbers come from run files only.

## f. PR
Commit, `git push -u origin <branch>` (never `--force`), then `gh pr create`. The PR body covers:
- the failure, with the quoted output;
- the fix;
- the pytest and replay results;
- the `compare_runs.py` table (all cases, not just the target);
- the README entry.

Never merge. Human scores (`evals/human_labels.csv`) are never written by code.
