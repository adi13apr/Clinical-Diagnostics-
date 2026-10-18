"""LLM judge: scores a report against the tool outputs with the rubric in evals/rubric.md.

Uses Gemini structured output (response_schema = JudgeResult). The judge never sees
the true label: it judges the report against what the tools returned.
"""
import json
import time
from pathlib import Path

from agent.agent import RATE_LIMIT_MAX_RETRY_DELAY, RateLimitError, quota_info, retry_delay_seconds
from evals.schema import JudgeResult

RUBRIC_PATH = Path(__file__).resolve().parent / "rubric.md"

JUDGE_SYSTEM = """You are a strict evaluator of AI-written chest X-ray screening reports.
Score the REPORT against the TOOL OUTPUTS using the rubric. The tool outputs are the only
ground truth: do not use outside medical knowledge to reward or penalise content, except to
notice claims the tools do not support. Give whole-number scores 1-5 and one short line of
reason per criterion. Do not reward length."""


def judge_prompt(tool_outputs, report, rubric):
    return (f"RUBRIC:\n{rubric}\n\nTOOL OUTPUTS (ground truth):\n{json.dumps(tool_outputs, indent=1)}\n\n"
            f"REPORT TO SCORE:\n{json.dumps(report, indent=1)}")


class GeminiJudge:
    def __init__(self, api_key, model, client=None, sleep=time.sleep, rubric_path=RUBRIC_PATH):
        from google import genai
        from google.genai import errors, types
        self.types, self.errors = types, errors
        self.model = model
        self.client = client or genai.Client(api_key=api_key)
        self.sleep = sleep
        self.rubric = Path(rubric_path).read_text(encoding="utf-8")

    def judge(self, tool_outputs, report):
        config = self.types.GenerateContentConfig(
            system_instruction=JUDGE_SYSTEM,
            response_mime_type="application/json",
            response_schema=JudgeResult,
            thinking_config=self.types.ThinkingConfig(thinking_level="low"),
        )
        contents = judge_prompt(tool_outputs, report, self.rubric)
        for attempt in (1, 2):
            try:
                response = self.client.models.generate_content(model=self.model, contents=contents, config=config)
                break
            except self.errors.ClientError as e:
                if e.code != 429:
                    raise
                delay = retry_delay_seconds(e)
                if attempt == 1 and delay is not None and delay < RATE_LIMIT_MAX_RETRY_DELAY:
                    self.sleep(delay)
                    continue
                raise RateLimitError(delay, *quota_info(e)) from e
        parsed = response.parsed
        if not isinstance(parsed, JudgeResult):  # SDK couldn't parse: validate the raw text (raises if invalid)
            parsed = JudgeResult.model_validate_json(response.text)
        meta = response.usage_metadata
        usage = None if meta is None else {"input_tokens": meta.prompt_token_count,
                                           "output_tokens": meta.candidates_token_count,
                                           "thinking_tokens": meta.thoughts_token_count}
        return parsed, usage


class MockJudge:
    """Returns fixed scores (tests)."""
    model = "mock-judge"

    def __init__(self, scores=None):
        self.scores = scores or {"faithfulness": 5, "uncertainty_language": 4, "actionability": 4,
                                 "plain_language": 3}
        self.calls = 0

    def judge(self, tool_outputs, report):
        self.calls += 1
        return JudgeResult.model_validate(
            {k: {"score": v, "reason": f"mock {k}"} for k, v in self.scores.items()}), None
