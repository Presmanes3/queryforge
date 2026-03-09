"""Core S3 path management and AWS SDK integration for QueryForge.

Guarantees adherence to the path naming contract across all project components.
"""

from __future__ import annotations
import boto3
from botocore.exceptions import ClientError
from datetime import datetime, timezone
import os

S3_PREFIX = "queryforge"


def generate_run_id() -> str:
    """Generate a sortable timestamp-based run identifier."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def build_s3_uri(
    bucket: str,
    schema_name: str,
    schema_version: str,
    component: str,
    run_id: str,
    filename: str | None = None,
    prefix: str = "queryforge",
) -> str:
    """Construct an S3 URI following the hierarchical naming convention.

    Args:
        bucket: Name of the project's artifact bucket.
        schema_name: Domain name (e.g., "orders").
        schema_version: Schema semver (e.g., "v1").
        component: Logical segment (dataset|model|adapter|gguf|metrics|schemas).
        run_id: Unique identifier for the experiment or run.
        filename: Optional leaf object name.
        prefix: Root prefix for the project.

    Returns:
        Formatted S3 URI string.
    """
    parts = [prefix, component, schema_name, schema_version, run_id]
    key = "/".join(parts)
    if filename:
        key = f"{key}/{filename}"
    return f"s3://{bucket}/{key}"


def check_s3_uri_exists(bucket: str, s3_uri: str) -> bool:
    """Determine if an object already exists at the specified URI."""
    key = s3_uri.replace(f"s3://{bucket}/", "")
    s3 = boto3.client("s3")
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        raise


def upload_file(local_path: str, s3_uri: str) -> None:
    """Copy a local file to S3 at the target location."""
    bucket, key = s3_uri.replace("s3://", "").split("/", 1)
    boto3.client("s3").upload_file(local_path, bucket, key)


def list_s3_objects(bucket: str, prefix: str) -> list[str]:
    """List all object keys for a given prefix in S3."""
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if "Contents" in page:
            keys.extend([obj["Key"] for obj in page["Contents"] if not obj["Key"].endswith("/")])
    return keys


def delete_s3_object(bucket: str, s3_uri: str) -> None:
    """Delete an object from S3."""
    key = s3_uri.replace(f"s3://{bucket}/", "")
    s3 = boto3.client("s3")
    s3.delete_object(Bucket=bucket, Key=key)


def delete_s3_prefix(bucket: str, prefix: str) -> None:
    """Delete all objects under a prefix in S3."""
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if "Contents" in page:
            objects = [{"Key": obj["Key"]} for obj in page["Contents"]]
            s3.delete_objects(Bucket=bucket, Delete={"Objects": objects})
