from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from shared.schemas.dataset import TrainingSample
from shared.schemas.workflow.datagen import DatagenState
from queryforge.utils.config import load_config

_PROMPT_PATH = Path(__file__).parent / "prompts" / "generate_pair.md"


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


def _generate_pairs(state: DatagenState) -> dict[str, Any]:
    """Call AWS Bedrock via ChatBedrockConverse to produce question-SQL pairs."""
    config = load_config()
    llm = ChatBedrockConverse(
        model=config.bedrock_model_id,
        region_name=config.aws_region,
    )

    prompt = _load_prompt().format(ddl=state.ddl, n_samples=state.n_samples)
    response = llm.invoke([HumanMessage(content=prompt)])
    content: str = response.content  # type: ignore[assignment]

    # Strip markdown code fences when the model wraps its response.
    content = content.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        content = "\n".join(lines[1:-1])

    pairs: list[dict[str, str]] = json.loads(content)
    return {"pairs": pairs}


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
