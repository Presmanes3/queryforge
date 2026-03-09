# QueryForge — Copilot Instructions

These rules apply to every Copilot interaction in this repository, regardless of which
agent is active. They encode the non-negotiable architectural invariants. Read them
before proposing any code change.

---

## Project topology

```
src/queryforge/
├── schema/        Pydantic schema definitions (source of truth for DDL and datasets)
├── datagen/       Synthetic question-SQL pair generation (schema → JSONL)
├── train/         QLoRA fine-tuning via HuggingFace PEFT + SageMaker Training Job
├── evaluate/      Execution Accuracy evaluation (ephemeral SQLite + SQL execution)
├── merge/         LoRA adapter fusion with the base model
├── quantize/      GGUF weight conversion (llama.cpp)
├── publish/       Ollama Modelfile generation and local publishing
├── pipeline/      SageMaker Pipeline DAG definitions
├── orchestration/ AWS Step Functions state machine definitions (ASL)
├── registry/      SageMaker Model Registry operations
└── utils/         S3 helpers, config loading, logging

shared/
├── schemas/
│   ├── dataset.py     Training sample Pydantic schema
│   ├── config.py      Pipeline config Pydantic schema
│   ├── metrics.py     Evaluation metrics Pydantic schema
│   └── registry.py    Model registry metadata Pydantic schema
└── enums/             Shared enumerated types

config/            YAML configuration files (no hardcoded values anywhere in src/)
scripts/           Entry-point scripts (parse args → call src/queryforge/)
```

---

## Rule 1 — Dependency direction (CLI layer)

The only valid import direction inside `src/cli/` is:

```
commands/ → interactors/ → client/ → (API over HTTP/WebSocket)
                        ↘ screens/ → views/
                                   → components/
```

A command file contains **only** arg parsing and a single interactor call.
An interactor file contains the use-case flow and nothing else.
A view file is a pure function: data in, Rich `RenderableType` out.

---

## Rule 2 — HTTP client is the only backend interface

The CLI communicates with the backend **exclusively** through:

- `src/cli/client/http_client.py` — typed `httpx` wrapper
- `src/cli/client/ws_client.py` — WebSocket transcription protocol

No code in `src/cli/` may import from:

| Forbidden import | Reason |
|---|---|
| `src/repository/` | DB layer; backend only |
| `src/services/` | Singleton services; backend only |
| `src/agents/` | LangGraph nodes; backend only |
| `src/workflows/` | LangGraph graphs; backend only |
| `src/registry/` (for `repos`/`service_registry`) | Registry singletons; backend only |

If an endpoint does not yet exist in the API, **create the endpoint first**, then call it
from the CLI via `http_client`.

---

## Rule 3 — Shared schemas usage

| Sub-namespace | Backend | CLI |
|---|---|---|
| `shared/schemas/api/` | Owns/defines (wire format) | Import for request/response types |
| `shared/schemas/models/` | Full read/write (SQLModel tables) | `TYPE_CHECKING` imports only |
| `shared/schemas/agents/` | Full read/write (agent I/O) | Do not import |
| `shared/schemas/workflow/` | Full read/write (LangGraph state) | `TYPE_CHECKING` imports only |
| `shared/enums/` | Full read | Full read |

The CLI's wire format is defined in `shared/schemas/api/`.

---

## Rule 4 — Rich/Textual is frontend-only

No import of `rich` or `textual` may appear anywhere outside `src/cli/`.
Backend code (routers, agents, repositories, services, workflows) must not produce any
terminal-formatted output.

---

## Rule 5 — Singletons only via registry

All repository and service singletons are obtained from `src/registry/`:

```python
from src.registry import repos, service_registry, agent_registry
```

Never instantiate a repository or service class directly with `ClassName()`.
The single exception is test code, which may construct instances with explicit
dependencies for isolation.

---

## Rule 6 — Standardized CLI error handling

Every Typer command function wraps its interactor call:

```python
try:
    interactor.run()
except ValueError as e:
    Console().print(Panel(f"[red]{e}[/red]", title="[bold]Error[/bold]", border_style="red"))
    raise SystemExit(1)
except Exception as e:
    Console().print(Panel(f"[red]Unexpected error:[/red] {e}", border_style="red"))
    raise SystemExit(1)
```

Never use bare `print()` for errors in command files. Never swallow exceptions silently.

---

## Rule 7 — One StateGraph per workflow file

Each file in `src/workflows/` exports exactly one compiled `StateGraph`.
All `*State` models used by workflows are Pydantic `BaseModel` (not `TypedDict`), defined
in `shared/schemas/workflow/`.

---

## Rule 8 — No business logic in routers

Router functions in `src/api/routers/` do only three things:
1. Validate the request body (Pydantic does this automatically).
2. Delegate to a repository or invoke a workflow.
3. Return a typed response from `src/api/schemas.py`.

Any logic beyond that belongs in a workflow node or a service method.

---

## Rule 9 — Imports at the top of every file

All imports must appear at the top of the file, in standard Python order:
1. Standard library
2. Third-party packages
3. `shared/` and `src/` project imports

```python
# 1. stdlib
from typing import TYPE_CHECKING, Optional, List

# 2. third-party
from fastapi import APIRouter, Depends
from sqlmodel import Session, select

# 3. project
from src.registry import repos
from shared.schemas.api.notes import NoteResponse

if TYPE_CHECKING:
    from shared.schemas.models.note import Note
```

**Lazy imports** (imports inside function or method bodies) are forbidden unless one of
these conditions holds:
- The import creates an **unavoidable circular dependency** that cannot be resolved by
  restructuring (document why with a `# noqa: PLC0415 — circular: <reason>` comment).
- The import is a **heavy optional dependency** loaded only on a specific code path
  (e.g., a model library loaded only when a service is initialized).

In all other cases, move the import to the top of the file.

---

## Rule 10 — Documentation standard

All documentation in this project follows the rules in the `documentation` skill.
Apply it whenever writing or modifying:
- Python docstrings (modules, classes, functions, methods)
- Inline comments
- `README.md`, `TODO.md`, architecture notes
- CLI `help` strings in Typer commands and options

Key rules (full detail in the skill):
- Google-style docstrings. First line: imperative mood, no subject.
- Omit `Args`/`Returns` when the signature is self-explanatory.
- Inline comments explain **why**, never **what**.
- No emojis. No filler phrases ("This class is responsible for…").
- CLI `Typer()` help: one sentence, no period.
- Command help: imperative, one sentence with a period.

Skill path: `.github/skills/documentation/SKILL.md`

---

## Rule 11 — Mandatory Schema Documentation (Pydantic)

All fields in Pydantic models within `shared/schemas/api/`, `shared/schemas/agents/`, and
`shared/schemas/workflow/` must use `pydantic.Field` with a `description`.

1.  **Requirement**: Avoid plain type hints. Use `field: type = Field(..., description="...")`.
2.  **Context**: Descriptions must explain the business logic or expected format (e.g., "ISO 639-1 code").
3.  **Defaults**: Explicitly define defaults using `default=...` or `default_factory=...` within `Field`.

---

## Naming conventions

| Artifact | Convention | Example |
|---|---|---|
| CLI command file | `<verb>.py` | `add.py`, `find.py` |
| Interactor file | `<domain>_interactor.py` | `note_list_interactor.py` |
| Screen file | `<use_case>_screen.py` | `search_screen.py` |
| View file | `<domain>_views.py` | `note_views.py` |
| Agent file | `<name>_agent.py` | `normalizer_agent.py` |
| Router file | `<resource>.py` | `notes.py`, `search.py` |
| Workflow file | `<name>_workflow.py` | `ingest_workflow.py` |
| Workflow state | `<Name>State` in `shared/schemas/workflow/` | `IngestState` |
| Agent I/O schema | `<Name>Input` / `<Name>Output` in `shared/schemas/agents/` | `NormalizerInput` |
