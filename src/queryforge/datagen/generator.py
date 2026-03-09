from __future__ import annotations

from pydantic import BaseModel

from shared.schemas.agents.datagen import DatagenOutput
from shared.schemas.workflow.datagen import DatagenState
from queryforge.datagen.workflow import graph
from queryforge.schema.ddl import derive_ddl


class DatasetGenerator:
    """Generates synthetic question-SQL pairs from a Pydantic schema."""

    def run(
        self,
        schema_cls: type[BaseModel],
        n_samples: int = 500,
        output_dir: str = "datasets",
    ) -> DatagenOutput:
        """Run the full datagen workflow for a given schema class.

        DDL is derived from the class's model_fields before the graph runs.
        The schema_name and schema_version are read from the class's field defaults.

        Args:
            schema_cls: Pydantic model class defining the table structure.
            n_samples: Number of question-SQL pairs to generate.
            output_dir: Directory where the JSONL file is written.

        Returns:
            DatagenOutput with the path and count of written samples.

        Raises:
            ValueError: When schema_name or schema_version defaults are missing.
        """
        schema_name = _extract_default(schema_cls, "schema_name")
        schema_version = _extract_default(schema_cls, "schema_version")
        ddl = derive_ddl(schema_cls, table_name=schema_name)

        initial_state = DatagenState(
            schema_name=schema_name,
            schema_version=schema_version,
            ddl=ddl,
            n_samples=n_samples,
            output_dir=output_dir,
        )

        # LangGraph returns a plain dict when the state schema is a Pydantic BaseModel.
        result: dict = graph.invoke(initial_state)
        final_state = DatagenState(**result)

        return DatagenOutput(
            output_path=final_state.output_path,
            n_written=len(final_state.samples),
        )


def _extract_default(schema_cls: type[BaseModel], field_name: str) -> str:
    """Extract a string default value from a Pydantic model class field.

    Args:
        schema_cls: The Pydantic model class.
        field_name: Name of the field to read the default from.

    Returns:
        The default value as a string.

    Raises:
        ValueError: When the field is missing or has no default.
    """
    field_info = schema_cls.model_fields.get(field_name)
    if field_info is None:
        raise ValueError(
            f"{schema_cls.__name__} must define a '{field_name}' field."
        )
    default = field_info.default
    if default is None:
        raise ValueError(
            f"'{field_name}' in {schema_cls.__name__} must have a string default."
        )
    return str(default)
