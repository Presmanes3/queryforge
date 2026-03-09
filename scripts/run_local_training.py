"""QueryForge Local Training Launcher.

Runs a QLoRA fine-tuning job in a local Docker container via the SageMaker V3
ModelTrainer LOCAL_CONTAINER mode.  Accepts the same parameters as
run_finetuning.py so workflows remain consistent, but reads model weights and
datasets from local directories instead of S3.

Prerequisites
-------------
- Docker Desktop must be running.
- The SageMaker training image must be pullable (set --training-image).

Local path conventions
----------------------
- Models  : models/<model_name>/            (e.g. models/Llama-3.2-1B-Instruct/)
- Datasets: datasets/<schema_name>_<schema_version>.jsonl
            OR datasets/<schema_name>_<schema_version>/<any files>

Output is written to ./local_output/ relative to the workspace root.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

_WORKSPACE_ROOT = Path(__file__).parent.parent


def _resolve_model_dir(model_name: str, model_version: str) -> Path:
    """Return the local model directory, checking it exists.

    Args:
        model_name: Subdirectory name under models/ (e.g. 'Llama-3.2-1B-Instruct').
        model_version: Version tag — used only for log messages; the local
            convention stores one version per named directory.

    Returns:
        Absolute Path to the model directory.

    Raises:
        SystemExit: If the directory does not exist.
    """
    candidate = _WORKSPACE_ROOT / "models" / model_name
    if not candidate.is_dir():
        print(
            f"Error: Local model directory not found: {candidate}\n"
            f"Download the model first with: python scripts/download_hf_model.py"
        )
        sys.exit(1)
    return candidate


def _resolve_dataset_dir(schema_name: str, schema_version: str) -> Path:
    """Return the local dataset path for a given schema and version.

    Looks for, in order:
    1. datasets/<schema_name>_<schema_version>/   (directory)
    2. datasets/<schema_name>_<schema_version>.jsonl  (single file — wrapped in a
       sibling temp-dir so SageMaker local mode can mount a directory channel)

    Args:
        schema_name: Domain schema name (e.g. 'orders').
        schema_version: Schema version string (e.g. 'v1').

    Returns:
        Absolute Path to a *directory* containing the dataset file(s).

    Raises:
        SystemExit: If neither form is found.
    """
    slug = f"{schema_name}_{schema_version}"
    datasets_root = _WORKSPACE_ROOT / "datasets"

    # Preferred: versioned sub-directory
    dir_path = datasets_root / slug
    if dir_path.is_dir():
        return dir_path

    # Fallback: single .jsonl file → expose its parent directory
    jsonl_path = datasets_root / f"{slug}.jsonl"
    if jsonl_path.is_file():
        # The parent directory (datasets/) is returned.  The channel will
        # mount the whole directory; train.py reads any *.jsonl it finds.
        return datasets_root

    print(
        f"Error: No local dataset found for schema '{schema_name}' version '{schema_version}'.\n"
        f"Expected one of:\n"
        f"  {dir_path}\n"
        f"  {jsonl_path}\n"
        f"Generate a dataset first with: python scripts/run_datagen.py"
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Train action
# ---------------------------------------------------------------------------

def _train(args) -> None:
    """Launch a QLoRA Training Job in a local Docker container."""
    # noqa: PLC0415 — heavy optional dependencies, only needed for train action
    from sagemaker.train.model_trainer import Mode
    from sagemaker.train.configs import InputData
    from queryforge.train.estimator import TrainingJobBuilder
    from queryforge.utils.config import load_config

    config = load_config(args.config)

    model_dir = _resolve_model_dir(args.model_name, args.model_version)
    dataset_dir = _resolve_dataset_dir(args.schema_name, args.schema_version)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve instance type before building so it can be displayed in the
    # summary printed before the potentially long container startup.
    if args.instance_type:
        instance_type = args.instance_type
    else:
        try:
            import torch  # noqa: PLC0415 — heavy optional, only for GPU auto-detection
            instance_type = "local_gpu" if torch.cuda.is_available() else "local_cpu"
        except ImportError:
            instance_type = "local_cpu"

    trainer = (
        TrainingJobBuilder(config)
        .with_mode(Mode.LOCAL_CONTAINER)
        .with_training_image(args.training_image)
        .with_instance_type(instance_type)
        .with_input_data([
            InputData(channel_name="model", data_source=str(model_dir)),
            InputData(channel_name="training", data_source=str(dataset_dir)),
        ])
        .build()
    )

    print(f"Model     : {model_dir}")
    print(f"Dataset   : {dataset_dir}")
    print(f"Output    : {output_dir}")
    print(f"Instance  : {instance_type}")
    print(f"Image     : {args.training_image}")
    print("\nStarting local container training...")

    trainer.train(wait=args.wait)

    if args.wait:
        print(f"\nLocal training complete. Adapter saved to: {output_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse arguments and launch local container training."""
    parser = argparse.ArgumentParser(
        description=(
            "Run QLoRA fine-tuning in a local Docker container via SageMaker "
            "LOCAL_CONTAINER mode. Reads model and dataset from local directories."
        )
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Model subdirectory under models/ (e.g. Llama-3.2-1B-Instruct).",
    )
    parser.add_argument(
        "--model-version",
        default="v1",
        help="Model version label used in log messages (default: v1).",
    )
    parser.add_argument(
        "--schema-name",
        required=True,
        help="Dataset schema name (e.g. orders).",
    )
    parser.add_argument(
        "--schema-version",
        default="v1",
        help="Schema version (default: v1).",
    )
    parser.add_argument(
        "--training-image",
        required=True,
        help=(
            "Docker image URI to use for the training container "
            "(e.g. 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.0.0-gpu-py310)."
        ),
    )
    parser.add_argument(
        "--instance-type",
        default=None,
        help="Override instance type: 'local_cpu' or 'local_gpu'. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--output-dir",
        default="local_output",
        help="Local directory for training artifacts (default: local_output/).",
    )
    parser.add_argument(
        "--no-wait",
        dest="wait",
        action="store_false",
        default=True,
        help="Return immediately without waiting for the container to finish.",
    )
    parser.add_argument(
        "--config",
        default="config/pipeline.yaml",
        help="Path to pipeline.yaml (default: config/pipeline.yaml).",
    )

    args = parser.parse_args()
    _train(args)


if __name__ == "__main__":
    main()
