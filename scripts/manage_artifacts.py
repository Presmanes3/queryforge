"""QueryForge Generic Artifact Management Utility.

Handles CRUD operations for schemas, datasets, and models on S3.
"""

from __future__ import annotations
import argparse
import os
import sys
from queryforge.utils.config import load_config
from queryforge.utils.s3 import (
    S3Repository,
    check_s3_uri_exists,
    upload_file,
    list_s3_objects,
    delete_s3_prefix,
    generate_run_id,
)


def main():
    """Execute the project's artifact lifecycle management."""
    parser = argparse.ArgumentParser(
        description="QueryForge Artifact Management (CRUD for schemas, datasets, models)."
    )
    parser.add_argument(
        "--component",
        required=True,
        choices=["schemas", "datasets", "models"],
        help="Type of artifact to manage.",
    )
    parser.add_argument(
        "--action",
        required=True,
        choices=["list", "upload", "delete"],
        help="Action to perform.",
    )
    parser.add_argument("--schema-name", help="Industry vertical name (e.g., orders).")
    parser.add_argument("--schema-version", default="v1", help="Semantic version.")
    parser.add_argument("--run-id", help="Unique ID (defaults to new for upload, wildcard for list/delete).")
    parser.add_argument("--local-path", help="Local file path for upload action.")
    parser.add_argument("--config", default="config/pipeline.yaml", help="Configuration path.")
    parser.add_argument("--force", action="store_true", help="Skip confirmation for delete.")

    args = parser.parse_args()
    config = load_config(args.config)

    if args.action == "upload":
        handle_upload(args, config)
    elif args.action == "list":
        handle_list(args, config)
    elif args.action == "delete":
        handle_delete(args, config)


def handle_upload(args, config):
    """Upload a local file to S3 with standardized paths."""
    if not args.local_path:
        print("Error: --local-path is required for upload.")
        sys.exit(1)
    if not args.schema_name:
        print("Error: --schema-name is required for upload.")
        sys.exit(1)

    run_id = args.run_id or generate_run_id()
    filename = os.path.basename(args.local_path)

    base_uri = S3Repository.component_uri(config, args.component)
    s3_uri = f"{base_uri}/{args.schema_name}/{args.schema_version}/{run_id}/{filename}"

    if check_s3_uri_exists(config.s3_bucket, s3_uri):
        print(f"Artifact already exists at {s3_uri}.")
        if not args.force:
            response = input("Overwrite? (y/N): ").lower().strip()
            if response != "y":
                print("Aborted.")
                sys.exit(0)

    print(f"Uploading {args.local_path} to {s3_uri}...")
    upload_file(args.local_path, s3_uri)
    print("Upload complete.")


def handle_list(args, config):
    """List artifacts in S3 based on component and schema filters."""
    prefix = S3Repository.resolve_component_prefix(config, args.component)
    if args.schema_name:
        prefix += f"/{args.schema_name}/{args.schema_version}"
        if args.run_id:
            prefix += f"/{args.run_id}"

    print(f"Listing artifacts under s3://{config.s3_bucket}/{prefix}...")

    keys = list_s3_objects(config.s3_bucket, prefix)
    if not keys:
        print("No artifacts found.")
        return

    for key in keys:
        print(f"s3://{config.s3_bucket}/{key}")


def handle_delete(args, config):
    """Delete artifacts or prefixes from S3."""
    if not args.schema_name:
        print("Error: --schema-name is required for delete to prevent accidental wipe.")
        sys.exit(1)

    # Derive delete target from artifact_uris in pipeline.yaml
    base_prefix = S3Repository.resolve_component_prefix(config, args.component)
    target_prefix = f"{base_prefix}/{args.schema_name}/{args.schema_version}"
    if args.run_id:
        target_prefix += f"/{args.run_id}"
    s3_uri = f"s3://{config.s3_bucket}/{target_prefix}"

    if not args.force:
        response = input(f"Are you sure you want to delete {s3_uri}? (y/N): ").lower().strip()
        if response != "y":
            print("Aborted.")
            sys.exit(0)

    print(f"Deleting {s3_uri}...")
    delete_s3_prefix(config.s3_bucket, target_prefix)
    print("Delete complete.")


if __name__ == "__main__":
    main()
