from __future__ import annotations

import json
import os
import re
from math import ceil
from pathlib import Path
from typing import Any

from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from shared.schemas.dataset import TrainingSample
from shared.schemas.workflow.datagen import DatagenState
from queryforge.utils.config import load_config

_PROMPT_PATH = Path(__file__).parent / "prompts" / "generate_pair.md"

# Maximum samples to request in a single Bedrock call. Large requests cause
# the model to truncate mid-JSON, producing unparseable output.
_BATCH_SIZE = 20


def _load_prompt() -> str:
    """Read the generate_pair prompt template from disk."""
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _validate_state(state: DatagenState) -> dict[str, Any]:
    """Validate that required state fields are populated before generation."""
    if not state.ddl:
        raise ValueError("DatagenState.ddl must be set before the workflow runs.")
    if not state.schema_name:
        raise ValueError("DatagenState.schema_name must be set before the workflow runs.")
    # Return an empty update — this node is a guard only.
    return {}


def _extract_json_array(raw: str) -> list[dict[str, str]]:
    """Extract a JSON array from a model response that may contain extra text.

    Strips markdown code fences first, then finds the outermost JSON array
    using bracket matching to handle trailing prose or cut-off responses.

    Args:
        raw: Raw string returned by the LLM.

    Returns:
        Parsed list of question-SQL dicts.

    Raises:
        ValueError: When no valid JSON array can be extracted.
    """
    text = raw.strip()

    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    text = re.sub(r"^```[a-z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text.strip())
    text = text.strip()

    # Find the first '[' and walk to its matching ']'
    start = text.find("[")
    if start == -1:
        raise ValueError("No JSON array found in model response.")

    depth = 0
    end = -1
    for i, ch in enumerate(text[start:], start):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end == -1:
        raise ValueError("JSON array in model response is not properly closed.")

    return json.loads(text[start : end + 1])


# SQL pattern groups rotated across batches to force coverage diversity.
_PATTERN_GROUPS = [
    "SELECT *, WHERE filters, comparison operators (=, >, <, BETWEEN, IN)",
    "ORDER BY, LIMIT, OFFSET, NULL checks (IS NULL, IS NOT NULL)",
    "GROUP BY, aggregate functions (COUNT, SUM, AVG, MIN, MAX)",
    "HAVING clauses, CASE expressions, string functions (UPPER, LOWER, LIKE)",
    "Subqueries in WHERE and SELECT, DISTINCT, date/time filtering on text columns",
]


def _generate_pairs(state: DatagenState) -> dict[str, Any]:
    """Call AWS Bedrock in batches to produce deduplicated question-SQL pairs.

    Each batch rotates through a different SQL pattern group to maximise
    coverage. Temperature is set high to reduce repetition across calls.
    Duplicate questions (case-insensitive) are dropped after all batches.
    """
    config = load_config()
    # High temperature increases output diversity across batches.
    llm = ChatBedrockConverse(
        model=config.bedrock_model_id,
        region_name=config.aws_region,
        temperature=0.9,
    )
    prompt_template = _load_prompt()

    all_pairs: list[dict[str, str]] = []
    seen_questions: set[str] = set()
    n_batches = ceil(state.n_samples / _BATCH_SIZE)

    for batch_idx in range(n_batches):
        remaining = state.n_samples - len(all_pairs)
        batch_size = min(_BATCH_SIZE, remaining)
        pattern_focus = _PATTERN_GROUPS[batch_idx % len(_PATTERN_GROUPS)]

        prompt = prompt_template.format(
            ddl=state.ddl,
            n_samples=batch_size,
            pattern_focus=pattern_focus,
            batch_index=batch_idx + 1,
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        content: str = response.content  # type: ignore[assignment]

        try:
            pairs = _extract_json_array(content)
        except (ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Batch {batch_idx + 1}/{n_batches} returned unparseable JSON: {exc}\n"
                f"Raw response (first 500 chars): {content[:500]}"
            ) from exc

        # Deduplicate by question text (case-insensitive).
        for pair in pairs:
            key = pair.get("question", "").strip().lower()
            if key and key not in seen_questions:
                seen_questions.add(key)
                all_pairs.append(pair)

    return {"pairs": all_pairs[: state.n_samples]}


def _format_samples(state: DatagenState) -> dict[str, Any]:
    """Convert raw LLM pairs to validated TrainingSample instances."""
    samples = [
        TrainingSample(
            schema_name=state.schema_name,
            schema_version=state.schema_version,
            ddl=state.ddl,
            question=pair["question"],
            sql=pair["sql"],
            text=(
                f"### Instruction:\n"
                f"Given the following SQL table schema:\n{state.ddl}\n\n"
                f"Write a SQL query to answer the following question:\n{pair['question']}\n\n"
                f"### Response:\n{pair['sql']}"
            ),
        )
        for pair in state.pairs
    ]
    return {"samples": samples}


def _write_jsonl(state: DatagenState) -> dict[str, Any]:
    """Serialize samples to JSONL and write to the configured output directory."""
    os.makedirs(state.output_dir, exist_ok=True)

    filename = f"{state.schema_name}_{state.schema_version}.jsonl"
    output_path = os.path.join(state.output_dir, filename)

    with open(output_path, "w", encoding="utf-8") as fh:
        for sample in state.samples:
            fh.write(sample.model_dump_json() + "\n")

    return {"output_path": output_path}


def _build_graph() -> StateGraph:
    """Construct and compile the datagen StateGraph."""
    builder = StateGraph(DatagenState)
    builder.add_node("validate_state", _validate_state)
    builder.add_node("generate_pairs", _generate_pairs)
    builder.add_node("format_samples", _format_samples)
    builder.add_node("write_jsonl", _write_jsonl)

    builder.add_edge(START, "validate_state")
    builder.add_edge("validate_state", "generate_pairs")
    builder.add_edge("generate_pairs", "format_samples")
    builder.add_edge("format_samples", "write_jsonl")
    builder.add_edge("write_jsonl", END)

    return builder.compile()


# Single compiled graph exported from this module (Rule 7).
graph = _build_graph()
