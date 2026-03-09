---
name: pydantic_to_ddl
description: >
  Rules for deriving SQL DDL and JSONL training datasets from Pydantic models.
  Use this skill whenever creating or modifying schema definitions in
  src/queryforge/schema/, generating datasets in src/queryforge/datagen/,
  or working with shared/schemas/dataset.py.
---

# Pydantic-to-DDL Skill

## Principles

- DDL is derived from Pydantic models. Never write raw SQL DDL by hand.
- JSONL training samples are serialized from Pydantic `BaseModel` instances. Never
  construct raw dicts for training data.
- Every schema model carries an explicit version string used in S3 paths.

---

## 1. Schema definition — Pydantic model

```python
# src/queryforge/schema/orders.py
from typing import Optional
from pydantic import BaseModel, Field

class OrderSchema(BaseModel):
    """Defines the structure of an orders table for DDL and dataset generation."""

    schema_name: str = Field(default="orders", description="Logical name of this schema.")
    schema_version: str = Field(default="v1", description="Semver-style version string.")

    # --- Table columns ---
    order_id: int = Field(description="Primary key, auto-incremented.")
    customer_id: int = Field(description="Foreign key referencing the customers table.")
    amount: float = Field(description="Total order amount in USD.")
    status: str = Field(description="Order lifecycle status: pending, shipped, delivered, cancelled.")
    created_at: str = Field(description="ISO 8601 timestamp of order creation.")
```

### Field conventions

| Pydantic type | SQL type |
|---|---|
| `int` | `INTEGER` |
| `float` | `REAL` |
| `str` | `TEXT` |
| `bool` | `INTEGER` (SQLite) / `BOOLEAN` |
| `Optional[T]` | nullable column |
| `Field(description="Primary key...")` | maps to `PRIMARY KEY` |
| `Field(description="Foreign key referencing...")` | maps to `FOREIGN KEY` |

Use the `description` field to encode SQL semantics. The DDL derivation function reads
descriptions to detect primary keys, foreign keys, and index hints.

---

## 2. DDL derivation

```python
# src/queryforge/schema/ddl.py
from pydantic import BaseModel

def derive_ddl(schema: BaseModel, table_name: str) -> str:
    """Derive a CREATE TABLE statement from a Pydantic model.

    Primary keys are identified by the string "Primary key" in the field description.
    Foreign keys are identified by "Foreign key referencing <table>" in the description.

    Args:
        schema: Pydantic model instance whose fields describe the table structure.
        table_name: Name to use in the CREATE TABLE statement.

    Returns:
        SQL DDL string (CREATE TABLE IF NOT EXISTS ...).
    """
```

### Output format

```sql
CREATE TABLE IF NOT EXISTS orders (
    order_id    INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    amount      REAL    NOT NULL,
    status      TEXT    NOT NULL,
    created_at  TEXT    NOT NULL
);
```

---

## 3. Training sample schema

```python
# shared/schemas/dataset.py
from pydantic import BaseModel, Field

class TrainingSample(BaseModel):
    """One question-SQL pair used for fine-tuning."""

    schema_name: str = Field(description="Name of the Pydantic schema that generated this sample.")
    schema_version: str = Field(description="Version of the schema at generation time.")
    ddl: str = Field(description="CREATE TABLE DDL string for the target database.")
    question: str = Field(description="Natural language question to be answered by SQL.")
    sql: str = Field(description="Correct SQL query that answers the question.")
```

### JSONL serialization

```python
def write_jsonl(samples: list[TrainingSample], path: str) -> None:
    """Write training samples to a JSONL file, one JSON object per line."""
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(sample.model_dump_json() + "\n")
```

Never serialize with `json.dumps(sample.__dict__)`. Always use `model_dump_json()`.

---

## 4. Dataset generation — structure

```python
# src/queryforge/datagen/generator.py
from src.queryforge.schema.ddl import derive_ddl
from shared.schemas.dataset import TrainingSample

class DatasetGenerator:
    """Generates synthetic question-SQL pairs from a Pydantic schema."""

    def generate(self, schema, n_samples: int) -> list[TrainingSample]:
        """Generate n_samples question-SQL pairs for the given schema.

        Args:
            schema: Pydantic model instance describing the table structure.
            n_samples: Number of pairs to generate.

        Returns:
            List of validated TrainingSample instances ready for JSONL serialization.
        """
        ddl = derive_ddl(schema, table_name=schema.schema_name)
        samples = []
        # --- generate questions and SQL using an LLM or template engine ---
        return samples
```

---

## 5. Prompt format for fine-tuning

Training samples must follow this instruction-response format:

```
### Instruction:
Given the following SQL table schema:
{ddl}

Write a SQL query to answer the following question:
{question}

### Response:
{sql}
```

This format is defined in `src/queryforge/datagen/prompt_template.py` and must not be
duplicated elsewhere. Import from that module.
