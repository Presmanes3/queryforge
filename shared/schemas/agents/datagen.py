from __future__ import annotations

from pydantic import BaseModel, Field


class DatagenInput(BaseModel):
    """Input contract for the DatasetGenerator."""

    schema_name: str = Field(
        description="Logical name of the schema to process."
    )
    schema_version: str = Field(
        description="Version string of the schema to process."
    )
    n_samples: int = Field(
        default=500,
        description="Number of question-SQL pairs to generate."
    )
    output_dir: str = Field(
        default="datasets",
        description="Directory where the output JSONL file is written."
    )


class DatagenOutput(BaseModel):
    """Output contract returned by the DatasetGenerator after a successful run."""

    output_path: str = Field(
        description="Path to the written JSONL file."
    )
    n_written: int = Field(
        description="Number of TrainingSample records written to the JSONL file."
    )
