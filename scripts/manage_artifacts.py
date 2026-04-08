"""QueryForge Generic Artifact Management Utility.

Handles CRUD operations for schemas, datasets, and models on S3.
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

# Fix: Añadir el root del proyecto al sys.path para poder importar 'utils' y 'shared'
# cuando se ejecuta directamente desde la carpeta 'scripts' o la raíz.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

# Insertar raíz y src/ para que 'utils', 'shared' y 'src' sean alcanzables
for _p in [str(SRC_DIR), str(PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# sys.path ya incluye el root del proyecto, usamos las rutas directas
from utils.config import ConfigLoader
from utils.s3 import (
    S3Repository,
    check_s3_uri_exists,
    upload_file,
    list_s3_objects,
    delete_s3_prefix,
    generate_run_id,
)

def load_config(path=None):
    return ConfigLoader.load(path)


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
        choices=["list", "upload", "delete", "download"],
        help="Action to perform.",
    )
    parser.add_argument("--schema-name", help="Industry vertical name (e.g., orders).")
    parser.add_argument("--schema-version", default="v1", help="Semantic version.")
    parser.add_argument("--run-id", help="Unique ID (defaults to new for upload, wildcard for list/delete).")
    parser.add_argument("--local-path", help="Local file path for upload/download action.")
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
    elif args.action == "download":
        handle_download(args, config)


def handle_download(args, config):
    """Download artifacts from S3 to a local directory."""
    if not args.local_path:
        print("Error: --local-path is required for download.")
        sys.exit(1)
    if not args.schema_name or not args.run_id:
        print("Error: --schema-name and --run-id are required for download.")
        sys.exit(1)

    prefix = S3Repository.resolve_component_prefix(config, args.component)
    prefix += f"/{args.schema_name}/{args.schema_version}/{args.run_id}"

    import boto3
    repo = S3Repository(boto3.Session())
    
    keys = repo.list(config.s3_bucket, prefix)
    if not keys:
        print(f"No objects found under {prefix}")
        sys.exit(1)

    print(f"Downloading {len(keys)} objects to {args.local_path}...")
    for key in keys:
        rel_path = key[len(prefix):].lstrip("/")
        dest_path = os.path.join(args.local_path, rel_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        print(f"  s3://{config.s3_bucket}/{key} -> {dest_path}")
        repo.download(config.s3_bucket, key, dest_path)
    print("Download complete.")


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
