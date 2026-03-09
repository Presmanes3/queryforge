from __future__ import annotations

from pydantic import BaseModel, Field


class TrainingSample(BaseModel):
    """One question-SQL pair used for fine-tuning."""

    schema_name: str = Field(
        description="Name of the Pydantic schema that generated this sample."
    )
    schema_version: str = Field(
        description="Version of the schema at generation time."
    )
    ddl: str = Field(
        description="CREATE TABLE DDL string for the target database."
    )
    question: str = Field(
        description="Natural language question to be answered by SQL."
    )
    sql: str = Field(
        description="Correct SQL query that answers the question."
    )
    text: str = Field(
        description=(
            "Full instruction-response string consumed by SFTTrainer. "
            "Formatted as ### Instruction / ### Response."
        )
    )
