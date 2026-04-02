"""QueryForge Fine-Tuning Launcher.

Lists base models and datasets available in S3, and launches a SageMaker
QLoRA Training Job for a chosen model/dataset pair.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict

from queryforge.utils.config import load_config
from queryforge.utils.s3 import S3Repository, generate_run_id, list_s3_objects


# ---------------------------------------------------------------------------
# Listing helpers
# ---------------------------------------------------------------------------

def _list_models(config) -> None:
    """Print a table of base model artifacts available in S3."""
    models_prefix = S3Repository.resolve_component_prefix(config, "models")
    all_keys = list_s3_objects(config.s3_bucket, models_prefix)

    # Keys follow: <models_prefix>/<model_name>/<version>/<filename>
    seen: dict[tuple[str, str], int] = defaultdict(int)
    for key in all_keys:
        relative = key[len(models_prefix):].lstrip("/")
        parts = relative.split("/")
        if len(parts) >= 3:  # model_name / version / filename
            seen[(parts[0], parts[1])] += 1

    if not seen:
        print("No base models found in S3.")
        return

    print(f"\n{'Model Name':<35} {'Version':<10} {'Files'}")
    print("-" * 60)
    for (name, version), count in sorted(seen.items()):
        print(f"{name:<35} {version:<10} {count}")
    print()


def _list_datasets(config, schema_name: str | None) -> None:
    """Print a table of dataset artifacts available in S3."""
    datasets_key = S3Repository.resolve_component_prefix(config, "datasets")
    all_keys = list_s3_objects(config.s3_bucket, datasets_key)

    # Keys follow: {datasets_key}/{schema}/{version}/{run_id}/{filename}
    seen: dict[tuple[str, str, str], int] = defaultdict(int)
    for key in all_keys:
        relative = key[len(datasets_key):].lstrip("/")
        parts = relative.split("/")
        if len(parts) >= 3:
            schema, version, run_id = parts[0], parts[1], parts[2]
            if schema_name and schema != schema_name:
                continue
            seen[(schema, version, run_id)] += 1

    if not seen:
        target = f" for schema '{schema_name}'" if schema_name else ""
        print(f"No datasets found in S3{target}.")
        return

    print(f"\n{'Schema':<20} {'Version':<10} {'Run ID':<20} {'Files'}")
    print("-" * 65)
    for (schema, version, run_id), count in sorted(seen.items()):
        print(f"{schema:<20} {version:<10} {run_id:<20} {count}")
    print()


def _resolve_dataset_run_id(config, schema_name: str, schema_version: str) -> str:
    """Return the most recent run_id for a given schema/version dataset.

    Args:
        config: Validated pipeline configuration.
        schema_name: Domain schema name (e.g., 'orders').
        schema_version: Schema version (e.g., 'v1').

    Returns:
        The lexicographically latest run_id found.

    Raises:
        SystemExit: If no matching dataset is found in S3.
    """
    datasets_key = S3Repository.resolve_component_prefix(config, "datasets")
    prefix = f"{datasets_key}/{schema_name}/{schema_version}/"
    keys = list_s3_objects(config.s3_bucket, prefix)
    run_ids = set()
    for key in keys:
        relative = key[len(prefix):].lstrip("/")
        parts = relative.split("/")
        if parts and parts[0]:
            run_ids.add(parts[0])

    if not run_ids:
        print(
            f"Error: No dataset found in S3 for schema '{schema_name}' version '{schema_version}'."
        )
        sys.exit(1)

    return sorted(run_ids)[-1]


# ---------------------------------------------------------------------------
# Train action
# ---------------------------------------------------------------------------

def _train(args, config) -> None:
    """Launch a SageMaker QLoRA Training Job."""
    # noqa: PLC0415 — heavy optional dependency, only needed for train action
    from queryforge.train.estimator import TrainingJobBuilder, build_training_inputs

    schema_name: str = args.schema_name
    schema_version: str = args.schema_version
    model_name: str = args.model_name
    model_version: str = args.model_version

    # URIs sourced from config artifact_uris (SSoT); fallback to conventional path.
    model_s3_uri = f"{S3Repository.component_uri(config, 'models')}/{model_name}/{model_version}"

    # --dataset-s3-uri takes priority over schema-based resolution.
    if args.dataset_s3_uri:
        dataset_s3_uri = args.dataset_s3_uri.rstrip("/")
        dataset_run_id = dataset_s3_uri.rstrip("/").split("/")[-1]
    else:
        dataset_run_id = args.dataset_run_id or _resolve_dataset_run_id(
            config, schema_name, schema_version
        )
        dataset_s3_uri = (
            f"{S3Repository.component_uri(config, 'datasets')}"
            f"/{schema_name}/{schema_version}/{dataset_run_id}"
        )

    run_id = generate_run_id()
    output_s3_uri = args.output_s3_uri or (
        f"{S3Repository.component_uri(config, 'adapters')}"
        f"/{schema_name}/{schema_version}/{run_id}"
    )

    print(f"Model   : {model_s3_uri}")
    print(f"Dataset : {dataset_s3_uri} (run_id={dataset_run_id})")
    print(f"Output  : {output_s3_uri}")
    print("\nLaunching SageMaker Training Job...")

    extra_hp = {}
    if args.epochs is not None:
        extra_hp["epochs"] = args.epochs

    trainer = (
        TrainingJobBuilder(config)
        .with_output_s3_uri(output_s3_uri)
        .with_input_data(build_training_inputs(model_s3_uri, dataset_s3_uri))
        .with_extra_hyperparameters(extra_hp)
        .build()
    )
    trainer.train(wait=True)

    job_name = trainer._latest_training_job.training_job_name
    artifact_uri = f"{output_s3_uri}/{job_name}/output/model.tar.gz"

    print(f"\nTraining complete.")
    print(f"Adapter saved at: {artifact_uri}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse arguments and dispatch to the requested action."""
    parser = argparse.ArgumentParser(
        description="Launch QLoRA fine-tuning on SageMaker or inspect available artifacts."
    )
    parser.add_argument(
        "--action",
        choices=["list-models", "list-datasets", "train"],
        required=True,
        help="Action to perform.",
    )
    parser.add_argument("--model-name", help="Base model name in S3 (required for train).")
    parser.add_argument("--model-version", default="v1", help="Base model version (default: v1).")
    parser.add_argument("--schema-name", help="Dataset schema name (required for train).")
    parser.add_argument("--schema-version", default="v1", help="Schema version (default: v1).")
    parser.add_argument(
        "--dataset-s3-uri",
        default=None,
        help="Direct S3 URI to the dataset folder (e.g. s3://bucket/prefix/run_id). "
             "When provided, --schema-name, --schema-version and --dataset-run-id are ignored.",
    )
    parser.add_argument(
        "--dataset-run-id",
        default=None,
        help="Dataset run ID. Defaults to the most recent run for the schema.",
    )
    parser.add_argument(
        "--output-s3-uri",
        default=None,
        help="S3 URI for the output adapter. Auto-generated if omitted.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs (default: value from pipeline.yaml).",
    )
    parser.add_argument("--config", default="config/pipeline.yaml", help="Path to pipeline.yaml.")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.action == "list-models":
        _list_models(config)

    elif args.action == "list-datasets":
        _list_datasets(config, args.schema_name)

    elif args.action == "train":
        if not args.model_name:
            parser.error("--model-name is required for the train action.")
        if not args.dataset_s3_uri and not args.schema_name:
            parser.error("--schema-name or --dataset-s3-uri is required for the train action.")
        _train(args, config)


if __name__ == "__main__":
    main()
