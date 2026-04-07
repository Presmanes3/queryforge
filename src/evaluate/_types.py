"""Schema types for evaluation samples and metrics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Sample:
    """A single evaluation record loaded from a JSONL dataset file."""

    schema_name: str
    schema_version: str
    ddl: str
    sql: str
    text: str

    @classmethod
    def from_dict(cls, d: dict) -> "Sample":
        """Construct a Sample from a raw JSONL-parsed dictionary."""
        return cls(
            schema_name=d["schema_name"],
            schema_version=d["schema_version"],
            ddl=d["ddl"],
            sql=d["sql"],
            text=d["text"],
        )


@dataclass
class Metrics:
    """Aggregated evaluation metrics for a single run."""

    execution_accuracy: float
    exact_match: float
    invalid_query_rate: float
    n_samples: int
    schema_name: str
    schema_version: str
