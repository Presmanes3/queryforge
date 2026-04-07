"""CLI entry point for launching the QueryForge training pipeline.

Usage
-----
    # Run with default parameter values defined in pipeline/parameters.py
    python scripts/run_pipeline.py

    # Override specific parameters
    python scripts/run_pipeline.py --model-name Llama-3.2-3B-Instruct --schema-name orders --schema-version v2 --accuracy-threshold 0.80
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Detect --local before importing pipeline modules. session.py reads QF_LOCAL_MODE
# at import time to pick LocalPipelineSession vs PipelineSession, so this flag
# must be set before any pipeline.* import occurs.
if "--local" in sys.argv:
    os.environ["QF_LOCAL_MODE"] = "1"

# Add src/ to sys.path so that top-level packages (pipeline, config, utils, etc.)
# are importable when running this script directly from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Set region before pipeline imports — the SageMaker SDK instantiates LocalSession
# unconditionally inside ProcessingStep.__init__ to resolve a method name string,
# and that constructor raises if AWS_DEFAULT_REGION is unset.
from utils.config import ConfigLoader as _ConfigLoader  # noqa: E402
os.environ.setdefault("AWS_DEFAULT_REGION", _ConfigLoader.load().aws_region)

from config import config
from pipeline.definition import pipeline
from pipeline.parameters import (
    accuracy_threshold,
    model_name,
    schema_name,
    schema_version,
    training_dataset_uri,
    validation_dataset_uri,
)
from pipeline.steps.deploy import deploy_endpoint


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the QueryForge SageMaker training pipeline.",
    )
    parser.add_argument(
        "--model-name",
        default=model_name.default_value,
        help=f"Base model identifier (default: {model_name.default_value})",
    )
    parser.add_argument(
        "--schema-name",
        default=schema_name.default_value,
        help=f"Schema name to fine-tune for (default: {schema_name.default_value})",
    )
    parser.add_argument(
        "--schema-version",
        default=schema_version.default_value,
        help=f"Schema version (default: {schema_version.default_value})",
    )
    parser.add_argument(
        "--accuracy-threshold",
        type=float,
        default=accuracy_threshold.default_value,
        help=f"Minimum execution accuracy to register the model (default: {accuracy_threshold.default_value})",
    )
    parser.add_argument(
        "--training-dataset-uri",
        default=training_dataset_uri.default_value,
        help=(
            "S3 URI for the training JSONL file or prefix "
            f"(default: {training_dataset_uri.default_value})"
        ),
    )
    parser.add_argument(
        "--validation-dataset-uri",
        default=validation_dataset_uri.default_value,
        help=(
            "S3 URI for the validation/test JSONL file or prefix "
            f"(default: {validation_dataset_uri.default_value})"
        ),
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Deploy the registered model to a SageMaker endpoint after the pipeline completes.",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Return immediately after starting the execution without waiting.",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run pipeline locally using Docker containers (no AWS costs, requires Docker).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    mode = "local" if args.local else "managed"
    print("=== QueryForge Training Pipeline ===")
    print(f"  mode                  : {mode}")
    print(f"  model_name            : {args.model_name}")
    print(f"  schema_name           : {args.schema_name}")
    print(f"  schema_version        : {args.schema_version}")
    print(f"  accuracy_threshold    : {args.accuracy_threshold}")
    print(f"  training_dataset_uri  : {args.training_dataset_uri}")
    print(f"  validation_dataset_uri: {args.validation_dataset_uri}")
    print(f"  deploy                : {args.deploy}")
    print("====================================\n")

    if args.local:
        pipeline.create(role_arn=config.execution_role_arn)
    else:
        pipeline.upsert(
            role_arn=config.execution_role_arn,
            tags=[{"Key": "project", "Value": "queryforge"}],
        )

    execution = pipeline.start(
        parameters={
            "ModelName":              args.model_name,
            "SchemaName":             args.schema_name,
            "SchemaVersion":          args.schema_version,
            "AccuracyThreshold":      args.accuracy_threshold,
            "TrainingDatasetUri":     _s3_prefix(args.training_dataset_uri),
            "ValidationDatasetUri":   args.validation_dataset_uri,
        }
    )

    print(f"Execution started: {execution.arn}\n")

    if args.no_wait:
        print("--no-wait set. Exiting without waiting for completion.")
        return

    print("Waiting for pipeline execution to complete...")
    try:
        execution.wait()
    except Exception as e:
        print(f"\nPipeline execution failed: {e}")
        _print_failed_steps(execution)
        sys.exit(1)

    print("\nPipeline execution completed successfully.")

    if args.deploy:
        print(f"\nDeploying model to endpoint 'queryforge-{args.schema_name}'...")
        deploy_endpoint(execution, args.schema_name, config)
        print("Deployment request submitted. Monitor the endpoint in the SageMaker console.")


def _print_failed_steps(execution) -> None:
    """Print failure reasons for all failed steps."""
    try:
        steps = execution.list_steps()
        for step in steps.get("PipelineExecutionSteps", []):
            if step["StepStatus"] == "Failed":
                print(f"\n  Step '{step['StepName']}' failed:")
                print(f"    {step.get('FailureReason', '(no reason provided)')}")
    except Exception:
        pass


def _s3_prefix(uri: str) -> str:
    """Normalize an S3 URI to a prefix (directory) with a trailing slash.

    SageMaker Training and Processing jobs only accept S3Prefix data types,
    not individual object URIs. When the caller passes a full object URI
    (e.g. ending in .jsonl), this function strips the filename so that
    SageMaker downloads the containing directory.

    Args:
        uri: An S3 URI — either a prefix (with or without trailing slash)
            or a full object URI pointing to a specific file.

    Returns:
        The URI with a trailing slash, suitable for use as an S3Prefix input.
    """
    uri = uri.rstrip("/")
    # Treat URIs whose last path segment contains a dot as object URIs.
    last_segment = uri.rsplit("/", 1)[-1]
    if "." in last_segment:
        uri = uri.rsplit("/", 1)[0]
    return uri + "/"


if __name__ == "__main__":
    main()