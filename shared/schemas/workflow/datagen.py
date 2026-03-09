from __future__ import annotations

from pydantic import BaseModel, Field

from shared.schemas.dataset import TrainingSample


class DatagenState(BaseModel):
    """Mutable state threaded through the datagen LangGraph workflow."""

    schema_name: str = Field(
        description="Logical name of the schema being processed."
    )
    schema_version: str = Field(
        description="Version string of the schema being processed."
    )
    ddl: str = Field(
        description="CREATE TABLE DDL derived from the schema class."
    )
    n_samples: int = Field(
        default=500,
        description="Number of question-SQL pairs to generate."
    )
    output_dir: str = Field(
        default="datasets",
        description="Directory where the output JSONL file will be written."
    )
    pairs: list[dict[str, str]] = Field(
        default_factory=list,
        description="Raw question-SQL dicts returned by the LLM before wrapping in TrainingSample."
    )
    samples: list[TrainingSample] = Field(
        default_factory=list,
        description="Accumulated TrainingSample instances produced by the workflow."
    )
    output_path: str = Field(
        default="",
        description="Absolute or relative path to the written JSONL file."
    )
