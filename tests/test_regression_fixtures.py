"""Every fixture with an "expect" block is a regression test: no new test code per failure.

Written by .claude/skills/failure-to-regression/reproduce.py --write-fixture. The guard is called
through evals.checks.guard_verdict, the same way the eval replay calls it.
"""
import json
from pathlib import Path

import pytest

from evals.checks import guard_verdict, run_checks

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def load_fixtures():
    found = []
    for path in sorted(FIXTURES.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if "expect" in data:
            found.append(pytest.param(data, id=path.stem))
    return found


def as_record(fixture):
    """The run-record fields the checks and the guard read."""
    report = fixture.get("report") or json.loads(fixture["llm_output"])
    return {"classifier": fixture["truth"], "report": report, "report_source": "llm", "rejected": [],
            "tool_outputs": fixture.get("tool_outputs") or {}}


@pytest.mark.parametrize("fixture", load_fixtures())
def test_fixture_is_still_handled(fixture):
    expect = fixture["expect"]
    record = as_record(fixture)
    report, problem = guard_verdict(record, fixture.get("llm_output"))
    if expect["guard"] == "reject":
        assert expect.get("guard_problem"), "set expect.guard_problem to a phrase of the guard's rejection message"
        assert report is None, "the guard must reject this exact output"
        assert expect["guard_problem"] in problem
    else:
        assert report is not None, f"the guard must accept this output, but said: {problem}"
    results = run_checks(record)
    for name in expect.get("checks_fail", []):
        assert results[name] is False, f"check {name} must flag this output"
