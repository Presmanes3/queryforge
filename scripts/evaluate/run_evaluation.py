"""CLI launcher for standalone QueryForge evaluation Processing Jobs.

Usage
-----
    # List available adapters in S3
    python scripts/evaluate/run_evaluation.py --action list-adapters

    # Evaluate with auto-resolved adapter (picks the latest run for the schema)
    python scripts/evaluate/run_evaluation.py --action evaluate \\
        --model-name Llama-3.2-1B-Instruct \\
        --schema-name orders \\
        --schema-version v1 \\
        --dataset-s3-uri s3://bucket/prefix/datasets/orders/v1/<run_id>/orders_v1_test.jsonl

    # Evaluate with explicit adapter
    python scripts/evaluate/run_evaluation.py --action evaluate \\
        --model-name Llama-3.2-1B-Instruct \\
        --schema-name orders \\
        --schema-version v1 \\
        --dataset-s3-uri s3://... \\
        --adapter-s3-uri s3://...
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from utils.config import ConfigLoader
from utils.s3 import S3Repository, generate_run_id, list_s3_objects

_EVAL_SCRIPT = Path(__file__).resolve().parents[2] / "src" / "evaluate" / "evaluate.py"


# ---------------------------------------------------------------------------
# Listing helpers
# ---------------------------------------------------------------------------

def _list_adapters(config) -> None:
    """Print a table of LoRA adapter artifacts available in S3."""
    import boto3
    session = boto3.Session(region_name=config.aws_region, profile_name=config.aws_profile)
    adapters_prefix = S3Repository.resolve_component_prefix(config, "adapters")
    all_keys = S3Repository(session).list(config.s3_bucket, adapters_prefix)

    # Keys follow: <adapters_prefix>/<schema_name>/<version>/<run_id>/<filename>
    seen: dict[tuple[str, str, str], int] = defaultdict(int)
    for key in all_keys:
        relative = key[len(adapters_prefix):].lstrip("/")
        parts = relative.split("/")
        if len(parts) >= 3:
            schema, version, run_id = parts[0], parts[1], parts[2]
            seen[(schema, version, run_id)] += 1

    if not seen:
        print("No adapters found in S3.")
        return

    print(f"\n{'Schema':<20} {'Version':<10} {'Run ID':<20} {'Files'}")
    print("-" * 65)
    for (schema, version, run_id), count in sorted(seen.items()):
        print(f"{schema:<20} {version:<10} {run_id:<20} {count}")
    print()


def _resolve_adapter_uri(config, schema_name: str, schema_version: str) -> str:
    """Return the S3 URI of the most recent adapter for a given schema/version.

    When the run contains a SageMaker Training Job output (model.tar.gz) that
    object URI is returned. Otherwise the run S3 prefix is returned so the
    container reads the extracted files directly.
    """
    import boto3
    session = boto3.Session(region_name=config.aws_region, profile_name=config.aws_profile)
    adapters_prefix = S3Repository.resolve_component_prefix(config, "adapters")
    prefix = f"{adapters_prefix}/{schema_name}/{schema_version}/"
    keys = S3Repository(session).list(config.s3_bucket, prefix)

    run_ids: set[str] = set()
    for key in keys:
        relative = key[len(prefix):].lstrip("/")
        parts = relative.split("/")
        if parts and parts[0]:
            run_ids.add(parts[0])

    if not run_ids:
        print(
            f"Error: No adapter found in S3 for schema '{schema_name}' "
            f"version '{schema_version}'."
        )
        sys.exit(1)

    latest_run = sorted(run_ids)[-1]
    run_prefix = f"{adapters_prefix}/{schema_name}/{schema_version}/{latest_run}"

    # Prefer the Training Job tar.gz artefact when it exists.
    for key in keys:
        if key.startswith(run_prefix) and key.endswith("/output/model.tar.gz"):
            return f"s3://{config.s3_bucket}/{key}"

    return f"s3://{config.s3_bucket}/{run_prefix}"


def _to_file_uri(path_or_uri: str) -> str:
    """Return a file:// URI if *path_or_uri* is a local path; S3 URIs pass through unchanged."""
    if path_or_uri.startswith("s3://") or path_or_uri.startswith("file://"):
        return path_or_uri
    return Path(path_or_uri).resolve().as_uri()


# ---------------------------------------------------------------------------
# Local evaluation (direct in-process execution, no Docker)
# ---------------------------------------------------------------------------

def _evaluate_local(args) -> None:
    """Run evaluate.py directly in this process using local file paths.

    Sets SM_CHANNEL_* environment variables to the local paths supplied via
    CLI args, then imports and calls main() from evaluate.py. Output goes to
    ./eval_output/ next to the project root.
    """
    model_path   = str(Path(__file__).resolve().parents[2] / "models" / args.model_name)
    adapter_path = args.adapter_s3_uri
    dataset_path = args.dataset_s3_uri
    output_path  = str(Path("eval_output").resolve())

    if not adapter_path:
        print("Error: --adapter-s3-uri (local path) is required in local mode.")
        sys.exit(1)
    if adapter_path.startswith("s3://"):
        print("Error: --adapter-s3-uri must be a local path in --local mode. Download the adapter first.")
        sys.exit(1)
    if dataset_path.startswith("s3://"):
        print("Error: --dataset-s3-uri must be a local path in --local mode.")
        sys.exit(1)

    os.environ["SM_CHANNEL_MODEL"]   = model_path
    os.environ["SM_CHANNEL_ADAPTER"] = adapter_path
    os.environ["SM_CHANNEL_DATASET"] = dataset_path
    os.environ["SM_OUTPUT_DIR"]      = output_path

    Path(output_path).mkdir(parents=True, exist_ok=True)

    print("=== QueryForge Local Evaluation ===")
    print(f"  model   : {model_path}")
    print(f"  adapter : {adapter_path}")
    print(f"  dataset : {dataset_path}")
    print(f"  output  : {output_path}")
    print("===================================\n")

    import importlib.util  # noqa: PLC0415 — only used in local path
    spec = importlib.util.spec_from_file_location("evaluate", _EVAL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["evaluate"] = module  # required: @dataclass resolves types via sys.modules
    spec.loader.exec_module(module)
    module.main()


# ---------------------------------------------------------------------------
# Evaluate action
# ---------------------------------------------------------------------------

def _evaluate(args, config) -> None:
    """Launch a SageMaker Processing Job (remote or local-mode) to evaluate a fine-tuned model."""
    os.environ.setdefault("AWS_DEFAULT_REGION", config.aws_region)
    if config.aws_profile:
        os.environ.setdefault("AWS_PROFILE", config.aws_profile)

    import boto3  # noqa: PLC0415 — only needed for evaluate action
    from sagemaker.core.processing import ProcessingInput, ProcessingOutput, ScriptProcessor, ProcessingS3Input
    from sagemaker.core.shapes.shapes import ProcessingS3Output

    boto_session = boto3.Session(
        region_name=config.aws_region,
        profile_name=config.aws_profile,
    )

    local_mode = getattr(args, "local", False)
    local_gpu  = getattr(args, "local_gpu", False)

    if local_mode or local_gpu:
        # sagemaker v3 removed the local Docker mode. Run evaluate.py directly in-process
        # instead — faster, free, and identical behaviour to the container (same script,
        # same env-var channel convention).
        _evaluate_local(args)
        return

    from sagemaker.core.helper.session_helper import Session  # noqa: PLC0415 — remote only
    sm_session    = Session(boto_session=boto_session)
    instance_type = config.evaluation_instance_type
    model_uri     = (
        f"{S3Repository.component_uri(config, 'models')}"
        f"/{args.model_name}/{args.schema_version}"
    )
    adapter_uri = args.adapter_s3_uri or _resolve_adapter_uri(
        config, args.schema_name, args.schema_version
    )
    dataset_uri = args.dataset_s3_uri
    run_id      = generate_run_id()
    output_uri  = args.output_s3_uri or (
        f"{S3Repository.component_uri(config, 'evaluation')}"
        f"/{args.schema_name}/{args.schema_version}/{run_id}"
    )

    print("=== QueryForge Standalone Evaluation ===")
    print(f"  mode          : remote")
    print(f"  model_name    : {args.model_name}")
    print(f"  schema        : {args.schema_name} {args.schema_version}")
    print(f"  model_uri     : {model_uri}")
    print(f"  adapter_uri   : {adapter_uri}")
    print(f"  dataset_uri   : {dataset_uri}")
    print(f"  output_uri    : {output_uri}")
    print("========================================\n")

    processor = ScriptProcessor(
        image_uri         = config.processing_image_uri,
        command           = ["python3"],
        instance_type     = instance_type,
        instance_count    = 1,
        role              = config.execution_role_arn,
        sagemaker_session = sm_session,
        base_job_name     = "queryforge-evaluate",
    )

    processor.run(
        code    = _EVAL_SCRIPT.as_uri(),
        inputs  = [
            ProcessingInput(
                input_name = "model",
                s3_input   = ProcessingS3Input(
                    s3_uri       = model_uri,
                    local_path   = "/opt/ml/processing/input/model",
                    s3_data_type = "S3Prefix",
                ),
            ),
            ProcessingInput(
                input_name = "adapter",
                s3_input   = ProcessingS3Input(
                    s3_uri       = adapter_uri,
                    local_path   = "/opt/ml/processing/input/adapter",
                    s3_data_type = "S3Prefix",
                ),
            ),
            ProcessingInput(
                input_name = "dataset",
                s3_input   = ProcessingS3Input(
                    s3_uri       = dataset_uri,
                    local_path   = "/opt/ml/processing/input/dataset",
                    s3_data_type = "S3Prefix",
                ),
            ),
        ],
        outputs = [
            ProcessingOutput(
                output_name = "metrics",
                s3_output   = ProcessingS3Output(
                    s3_uri         = output_uri,
                    local_path     = "/opt/ml/processing/output",
                    s3_upload_mode = "EndOfJob",
                ),
            ),
        ],
        wait = True,
        logs = True,
    )

    print(f"\nEvaluation complete. Metrics saved at: {output_uri}/metrics.json")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse arguments and dispatch to the requested action."""
    parser = argparse.ArgumentParser(
        description="Launch a standalone QueryForge model evaluation Processing Job on SageMaker."
    )
    parser.add_argument(
        "--action",
        choices=["list-adapters", "evaluate"],
        required=True,
        help="Action to perform.",
    )
    parser.add_argument("--model-name", help="Base model folder name in S3 (required for evaluate).")
    parser.add_argument("--schema-name", help="Schema name for adapter and dataset resolution.")
    parser.add_argument("--schema-version", default="v1", help="Schema version (default: v1).")
    parser.add_argument(
        "--adapter-s3-uri",
        default=None,
        help=(
            "S3 URI or local path to the LoRA adapter. Accepts an S3 prefix (extracted files) "
            "or a full object URI ending in model.tar.gz. "
            "Auto-resolved to the latest run when omitted (remote mode only)."
        ),
    )
    parser.add_argument(
        "--dataset-s3-uri",
        default=None,
        help="S3 URI or local path to the test JSONL file (required for evaluate).",
    )
    parser.add_argument(
        "--output-s3-uri",
        default=None,
        help="S3 URI where metrics.json will be written. Auto-generated if omitted.",
    )
    parser.add_argument("--config", default="config/pipeline.yaml", help="Path to pipeline.yaml.")
    local_group = parser.add_mutually_exclusive_group()
    local_group.add_argument(
        "--local",
        action="store_true",
        default=False,
        help="Run inside a local Docker container (CPU). Uses file:// URIs for model/adapter/dataset.",
    )
    local_group.add_argument(
        "--local-gpu",
        dest="local_gpu",
        action="store_true",
        default=False,
        help="Run inside a local Docker container with GPU passthrough.",
    )
    args = parser.parse_args()

    config = ConfigLoader.load(args.config)

    if args.action == "list-adapters":
        _list_adapters(config)

    elif args.action == "evaluate":
        if not args.model_name:
            parser.error("--model-name is required for the evaluate action.")
        if not args.schema_name:
            parser.error("--schema-name is required for the evaluate action.")
        if not args.dataset_s3_uri:
            parser.error("--dataset-s3-uri is required for the evaluate action.")
        _evaluate(args, config)


if __name__ == "__main__":
    main()