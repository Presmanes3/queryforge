from __future__ import annotations

from pydantic import BaseModel, Field


class EvaluationMetrics(BaseModel):
    """Metrics produced by the evaluate component for one model checkpoint."""

    execution_accuracy: float = Field(
        description="Fraction of samples where the predicted SQL is an exact match "
                    "after normalization. Aliased from exact_match for ConditionStep "
                    "compatibility. Range 0.0–1.0."
    )
    exact_match: float = Field(
        description="Fraction of samples where the predicted SQL is character-identical "
                    "to the reference SQL after lowercasing and whitespace normalization. "
                    "Primary promotion metric."
    )
    invalid_query_rate: float = Field(
        description="Fraction of samples where the predicted SQL raises a syntax error "
                    "when parsed by SQLite against an empty table. Lower is better."
    )
    n_samples: int = Field(description="Total number of evaluation samples.")
    schema_name: str = Field(description="Name of the schema used for evaluation.")
    schema_version: str = Field(description="Version of the schema used for evaluation.")
