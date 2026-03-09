---
name: documentation
description: >
  Applies minimalist, professional documentation standards to QueryForge.
  Use this skill whenever writing, updating, or reviewing Python docstrings,
  module headers, README files, CLI help text, architecture notes, or inline comments.
---

# Documentation Skill

## Principles

- Write only what adds value. If a reader can infer it from the code, omit it.
- No emojis. No filler phrases ("Please note that…", "This function is responsible for…").
- Prefer short sentences and direct language.
- Use tables and code blocks to convey structure, not prose.

---

## Python Docstrings

Use **Google style**. Every public class, method, and function must have a docstring.

### Format

```python
def method(self, param: Type) -> ReturnType:
    """One-line summary ending with a period.

    Args:
        param: Description of what it represents, not its type.

    Returns:
        Description of the returned value.

    Raises:
        ValueError: When the input violates a constraint.
    """
```

### Rules

- The first line is a single sentence, imperative mood, no subject ("Run…" not "This method runs…").
- Omit `Args` / `Returns` / `Raises` sections when the signature is self-explanatory.
- For abstract methods, document the contract, not the implementation.
- For `__init__`, document only if construction has non-obvious side effects.

### Class docstrings

```python
class DatasetGenerator:
    """Generates synthetic question-SQL pairs from a Pydantic schema."""
```

Document the class purpose in one line. List invariants or ownership semantics only when
they are non-obvious.

---

## Module Headers

Place a single docstring at the top of every module when the module purpose is not
obvious from its name.

```python
"""Derives SQL DDL statements from Pydantic model field definitions."""
```

Omit the header for modules whose name fully describes their purpose (e.g., `s3.py`,
`metrics.py`).

---

## Inline Comments

Use inline comments only to explain **why**, never **what**.

```python
# Sort rows before comparison to make result-set equality order-independent
return sorted([tuple(row) for row in rows])
```

---

## README Structure

Every README must include these sections in order:

1. **Project description** — one paragraph, no marketing language.
2. **Prerequisites** — Python version, AWS account requirements, local tools.
3. **Quick start** — minimum commands to run something.
4. **Component overview** — table mapping each `src/queryforge/` package to its responsibility.
5. **Configuration** — how to populate `config/pipeline.yaml` and required env vars.
6. **Running the pipeline** — commands to execute each phase.

---

## CLI Help Text (argparse / Click)

```python
parser.add_argument(
    "--schema-name",
    help="Logical name of the Pydantic schema to use for DDL and dataset generation.",
)
```

- Use imperative mood, no period.
- Be specific: name the Pydantic model, the S3 path format, or the expected value type.
- Avoid generic descriptions like "the schema" or "a string value".

---

## Config YAML Comments

```yaml
# Minimum Execution Accuracy (0.0–1.0) required to register a model version.
accuracy_threshold: 0.75
```

Comment every non-obvious config key with the valid range and its effect on the pipeline.
