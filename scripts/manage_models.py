"""QueryForge Model Management Utility.

Lists local and S3 models, and provides mechanisms to upload local models
to the project's S3 bucket following the hierarchical naming convention.
"""

from __future__ import annotations
import argparse
import os

import boto3

from queryforge.utils.config import load_config
from queryforge.utils.s3 import upload_file


def list_local_models(models_dir: str = "models") -> list[str]:
    """List model directories in the local workspace."""
    if not os.path.exists(models_dir):
        return []
    return [
        d for d in os.listdir(models_dir)
        if os.path.isdir(os.path.join(models_dir, d))
    ]


def _models_s3_prefix(config) -> str:
    """Return the S3 key prefix for base models, derived from models_uri in config."""
    uri = config.artifact_uris.get(
        "models_uri", f"s3://{config.s3_bucket}/{config.s3_prefix}/models"
    )
    return uri.replace(f"s3://{config.s3_bucket}/", "")


def check_exists_in_s3(bucket: str, models_prefix: str, model_name: str, version: str = "v1") -> bool:
    """Check if the model version prefix exists in S3."""
    s3 = boto3.client("s3")
    prefix_key = f"{models_prefix}/{model_name}/{version}/"
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix_key, MaxKeys=1)
    return "Contents" in response


def main() -> None:
    """Execute model management actions based on CLI arguments."""
    parser = argparse.ArgumentParser(description="Manage QueryForge models.")
    parser.add_argument(
        "--action",
        choices=["list", "upload"],
        required=True,
        help="Action to perform: list local models or upload a local model.",
    )
    parser.add_argument("--model", help="Model name (required for upload).")
    parser.add_argument("--version", default="v1", help="Schema version (default: v1).")
    parser.add_argument("--run-id", default="base", help="Run identifier (default: base).")
    parser.add_argument("--config", default="config/pipeline.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.action == "list":
        print(f"{'Model Name':<30} | {'Status in S3 (base)':<20}")
        print("-" * 55)

        locals_found = list_local_models()
        if not locals_found:
            print("No local models found in models/ directory.")

        models_prefix = _models_s3_prefix(config)
        for m in locals_found:
            exists = check_exists_in_s3(config.s3_bucket, models_prefix, m, args.version)
            status = "[v] Uploaded" if exists else "[ ] Not in S3"
            print(f"{m:<30} | {status:<20}")

    elif args.action == "upload":
        if not args.model:
            print("Error: --model is required for upload.")
            return

        local_dir = os.path.join("models", args.model)
        if not os.path.exists(local_dir):
            print(f"Error: Local directory '{local_dir}' does not exist.")
            return

        models_prefix = _models_s3_prefix(config)
        dest = f"s3://{config.s3_bucket}/{models_prefix}/{args.model}/{args.version}"
        print(f"Uploading '{args.model}' to {dest}/...")
        uploaded = 0
        for root, _, files in os.walk(local_dir):
            for file in files:
                local_file_path = os.path.join(root, file)
                # Normalize path separators for S3 keys.
                relative_path = os.path.relpath(local_file_path, local_dir).replace("\\", "/")
                s3_uri = f"{dest}/{relative_path}"
                upload_file(local_file_path, s3_uri)
                print(f"  {relative_path}")
                uploaded += 1

        print(f"\nDone. {uploaded} file(s) uploaded.")


if __name__ == "__main__":
    main()
