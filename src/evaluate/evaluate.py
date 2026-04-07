"""SageMaker Processing Job entry point for QueryForge model evaluation.

Reads three input channels supplied by SageMaker (or SM_CHANNEL_* env vars
for local execution): ''model'' (base model weights), ''adapter'' (LoRA adapter
files or model.tar.gz), and ''dataset'' (JSONL test file). Writes metrics.json
and predictions.jsonl to the output channel.

Support modules (_types, _sql, _io, _model) are pre-copied into the Docker
image at /opt/queryforge/evaluate/ by the Dockerfile COPY instructions. When
running locally (--local flag via run_evaluation.py) the same modules are
loaded from the src/evaluate/ directory alongside this script.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Resolve the support-module directory. In the container the modules are baked
# into /opt/queryforge/evaluate/ by the Dockerfile. Locally they sit alongside
# this script in src/evaluate/.
_CONTAINER_LIB = Path("/opt/queryforge/evaluate")
_LOCAL_LIB     = Path(__file__).resolve().parent
sys.path.insert(0, str(_CONTAINER_LIB if _CONTAINER_LIB.exists() else _LOCAL_LIB))

from _data  import load_samples, compute_metrics, write_metrics, write_predictions
from _model import load_model_and_tokenizer, generate_predictions

_MODEL_CHANNEL   = os.getenv("SM_CHANNEL_MODEL",   "/opt/ml/processing/input/model")
_ADAPTER_CHANNEL = os.getenv("SM_CHANNEL_ADAPTER", "/opt/ml/processing/input/adapter")
_DATASET_CHANNEL = os.getenv("SM_CHANNEL_DATASET", "/opt/ml/processing/input/dataset")
_OUTPUT_DIR      = os.getenv("SM_OUTPUT_DIR",      "/opt/ml/processing/output")


def main() -> None:
    """Run evaluation inside a SageMaker Processing Job container."""
    print("\n=== QueryForge Evaluation ===")

    print("Loading dataset...")
    samples = load_samples(_DATASET_CHANNEL)
    if not samples:
        raise FileNotFoundError(
            f"No JSONL files found under {_DATASET_CHANNEL}. "
            "Ensure the test dataset was uploaded to the correct S3 URI."
        )
    print(f"  Loaded {len(samples)} samples.")

    print("Loading base model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(_MODEL_CHANNEL, _ADAPTER_CHANNEL)

    print("Generating predictions...")
    predictions = generate_predictions(model, tokenizer, samples)

    print("Computing metrics...")
    metrics = compute_metrics(samples, predictions)

    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    write_metrics(metrics, _OUTPUT_DIR)
    write_predictions(samples, predictions, _OUTPUT_DIR)

    print("\n=== Evaluation results ===")
    print(f"  Accuracy           : {metrics.execution_accuracy:.4f} ({metrics.execution_accuracy * 100:.2f}%)")
    print(f"  Exact Match        : {metrics.exact_match:.4f}")
    print(f"  Invalid Query Rate : {metrics.invalid_query_rate:.4f}")
    print(f"  Samples            : {metrics.n_samples}")
    print(f"  Schema             : {metrics.schema_name} {metrics.schema_version}")
    print("==========================")
    print(f"Metrics written to: {_OUTPUT_DIR}/metrics.json")


if __name__ == "__main__":
    main()
