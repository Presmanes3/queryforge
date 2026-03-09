---
name: evaluation_metrics
description: >
  Canonical evaluation patterns for QueryForge, including Execution Accuracy,
  Exact Match, and invalid query rate. Use this skill whenever writing or
  modifying evaluation code in src/queryforge/evaluate/ or shared/schemas/metrics.py.
---

# Evaluation Metrics Skill

## Principles

- **Execution Accuracy is the primary metric.** Promotion decisions in the SageMaker
  Pipeline `ConditionStep` compare only `execution_accuracy`. Textual metrics are
  recorded but do not block promotion.
- SQL is tested by executing both the predicted and reference queries against an
  ephemeral, in-memory SQLite database and comparing result sets.
- The ephemeral database is recreated from scratch for every evaluation run to
  guarantee isolation.

---

## 1. Metrics schema

```python
# shared/schemas/metrics.py
from pydantic import BaseModel, Field

class EvaluationMetrics(BaseModel):
    """Metrics produced by the evaluate component for one model checkpoint."""

    execution_accuracy: float = Field(
        description="Fraction of samples where the predicted SQL produces the same "
                    "result set as the reference SQL when executed. Range 0.0–1.0."
    )
    exact_match: float = Field(
        description="Fraction of samples where the predicted SQL is character-identical "
                    "to the reference SQL after normalization. Secondary metric only."
    )
    invalid_query_rate: float = Field(
        description="Fraction of samples where the predicted SQL raises a SQL syntax "
                    "or execution error. Lower is better."
    )
    n_samples: int = Field(description="Total number of evaluation samples.")
    schema_name: str = Field(description="Name of the schema used for evaluation.")
    schema_version: str = Field(description="Version of the schema used for evaluation.")
```

---

## 2. Ephemeral database setup

```python
# src/queryforge/evaluate/db.py
import sqlite3
from contextlib import contextmanager

@contextmanager
def ephemeral_db(ddl: str, fixtures: list[dict], table_name: str):
    """Create an in-memory SQLite database, populate it, and tear it down.

    Args:
        ddl: CREATE TABLE statement derived from the Pydantic schema.
        fixtures: List of row dicts used to populate the table for realistic queries.
        table_name: Name of the table matching the DDL.

    Yields:
        sqlite3.Connection ready for query execution.
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(ddl)
        if fixtures:
            cols = ", ".join(fixtures[0].keys())
            placeholders = ", ".join("?" * len(fixtures[0]))
            conn.executemany(
                f"INSERT INTO {table_name} ({cols}) VALUES ({placeholders})",
                [list(row.values()) for row in fixtures],
            )
        conn.commit()
        yield conn
    finally:
        conn.close()
```

---

## 3. Execution Accuracy computation

```python
# src/queryforge/evaluate/metrics.py
import sqlite3
from src.queryforge.evaluate.db import ephemeral_db

def execute_sql(conn: sqlite3.Connection, sql: str) -> list | None:
    """Execute a SQL query and return result rows, or None on error.

    Returns:
        Sorted list of tuples, or None if execution raised an exception.
    """
    try:
        cursor = conn.execute(sql)
        rows = cursor.fetchall()
        # Sort for order-independent comparison
        return sorted([tuple(row) for row in rows])
    except Exception:
        return None


def compute_execution_accuracy(
    samples: list,           # list of TrainingSample
    predictions: list[str],  # predicted SQL strings, same order as samples
    ddl: str,
    fixtures: list[dict],
    table_name: str,
) -> float:
    """Compute Execution Accuracy over a set of predicted SQL queries.

    Args:
        samples: Ground-truth TrainingSample objects with reference SQL.
        predictions: Predicted SQL strings from the model, aligned with samples.
        ddl: DDL string for the ephemeral database.
        fixtures: Representative rows for the ephemeral database.
        table_name: Table name matching the DDL.

    Returns:
        Fraction of samples where prediction and reference produce identical result sets.
    """
    correct = 0
    for sample, pred_sql in zip(samples, predictions):
        with ephemeral_db(ddl, fixtures, table_name) as conn:
            ref_result  = execute_sql(conn, sample.sql)
            pred_result = execute_sql(conn, pred_sql)
        if ref_result is not None and pred_result == ref_result:
            correct += 1
    return correct / len(samples) if samples else 0.0
```

---

## 4. Metrics output file

The evaluate component must write a JSON file at the SageMaker output path that can be
read by a `JsonGet` function in the `ConditionStep`:

```python
import json
from shared.schemas.metrics import EvaluationMetrics

def write_metrics(metrics: EvaluationMetrics, output_dir: str) -> None:
    """Write evaluation metrics to a JSON file readable by SageMaker ConditionStep.

    The file is written to output_dir/metrics.json. The ConditionStep reads
    the 'execution_accuracy' key directly via JsonGet.
    """
    path = f"{output_dir}/metrics.json"
    with open(path, "w") as f:
        json.dump(metrics.model_dump(), f, indent=2)
```

The `JsonGet` in the pipeline reads:
```python
json_path="execution_accuracy"
```

---

## 5. Exact Match normalization

Before computing Exact Match, normalize both prediction and reference:

```python
import re

def normalize_sql(sql: str) -> str:
    """Normalize SQL for textual comparison by collapsing whitespace and lowercasing."""
    return re.sub(r"\s+", " ", sql.strip().lower())
```
