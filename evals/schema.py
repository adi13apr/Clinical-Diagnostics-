"""Versioned schemas for eval cases and judge output."""
from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field

CASE_SCHEMA_VERSION = 1   # bump when EvalCase changes; the loader refuses other versions
RECORD_VERSION = 1        # raw run records (evals/runs/*.jsonl)


class SchemaVersionError(ValueError):
    """cases.jsonl was written for a different schema version."""


class ClassifierSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")
    predicted_class: Literal["Normal", "Pneumonia", "Tuberculosis"]
    calibrated_probability: float = Field(ge=0, le=1)  # 4 dp, as run_classifier reports it
    status: Literal["confident", "refer"]


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: int
    case_id: str
    relpath: str                     # splits/split_manifest.csv relpath
    source: Literal["pneumonia", "tb"]
    split: Literal["test"]
    true_label: Literal["Normal", "Pneumonia", "Tuberculosis"]
    categories: List[Literal["error", "refer", "correct"]]
    snapshot: ClassifierSnapshot     # classifier output when the dataset was built (drift check)


CRITERIA = ["faithfulness", "uncertainty_language", "actionability", "plain_language"]


class CriterionScore(BaseModel):
    score: int = Field(ge=1, le=5)
    reason: str = Field(description="One line explaining the score")


class JudgeResult(BaseModel):
    faithfulness: CriterionScore
    uncertainty_language: CriterionScore
    actionability: CriterionScore
    plain_language: CriterionScore
