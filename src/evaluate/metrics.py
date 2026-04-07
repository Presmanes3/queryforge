"""SQL prediction quality metrics for QueryForge evaluation jobs."""

from __future__ import annotations

import json
import re
import sqlite3

from shared.schemas.dataset import TrainingSample
from shared.schemas.metrics import EvaluationMetrics


def normalize_sql(sql: str) -> str:
    """Normalize SQL for textual comparison by collapsing whitespace and lowercasing."""
    return re.sub(r"\s+", " ", sql.strip().lower())


def is_valid_sql(ddl: str, sql: str) -> bool:
    """Check whether *sql* is syntactically valid against an empty table defined by *ddl*.

    Creates an in-memory SQLite database, executes the DDL, then attempts to
    execute *sql*. No rows are inserted; this validates syntax only.

    Args:
        ddl: CREATE TABLE statement for the target schema.
        sql: SQL query to validate.

    Returns:
        True if SQLite accepts the query without a syntax or operational error.
    """
    try:
        conn = sqlite3.connect(":memory:")
        conn.execute(ddl)
        conn.execute(sql)
        conn.close()
        return True
    except Exception:
        return False


def compute_metrics(
    samples: list[TrainingSample],
    predictions: list[str],
) -> EvaluationMetrics:
    """Compute evaluation metrics by comparing predictions to reference SQL strings.

    Exact Match is the primary metric. Execution Accuracy is set equal to Exact
    Match for ConditionStep compatibility (no live database is used).

    Args:
        samples: Ground-truth TrainingSample objects with reference SQL and DDL.
        predictions: Predicted SQL strings from the model, aligned with samples.

    Returns:
        Populated EvaluationMetrics instance.
    """
    if not samples:
        return EvaluationMetrics(
            execution_accuracy=0.0,
            exact_match=0.0,
            invalid_query_rate=0.0,
            n_samples=0,
            schema_name="",
            schema_version="",
        )

    exact = 0
    invalid = 0
    ddl = samples[0].ddl

    for sample, pred in zip(samples, predictions):
        if normalize_sql(pred) == normalize_sql(sample.sql):
            exact += 1
        if not is_valid_sql(ddl, pred):
            invalid += 1

    n = len(samples)
    exact_match = exact / n
    return EvaluationMetrics(
        execution_accuracy=exact_match,
        exact_match=exact_match,
        invalid_query_rate=invalid / n,
        n_samples=n,
        schema_name=samples[0].schema_name,
        schema_version=samples[0].schema_version,
    )


def write_metrics(metrics: EvaluationMetrics, output_dir: str) -> None:
    """Write evaluation metrics to a JSON file readable by SageMaker ConditionStep.

    The file is written to ``output_dir/metrics.json``. SageMaker Pipeline
    ConditionStep reads the ``execution_accuracy`` key directly via JsonGet.

    Args:
        metrics: Populated evaluation metrics.
        output_dir: Directory where ``metrics.json`` will be written.
    """
    path = f"{output_dir}/metrics.json"
    with open(path, "w") as f:
        json.dump(metrics.model_dump(), f, indent=2)
