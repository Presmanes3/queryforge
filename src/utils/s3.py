"""S3 operations and path utilities for QueryForge.

Provides two interfaces:

- :class:`S3Repository` — typed, session-bound class for all S3 CRUD.
  Accepts an injected ``boto3.Session`` for testability.
- Module-level free functions — thin wrappers over ``S3Repository`` kept
  for backward compatibility with existing call sites.

Prefer ``S3Repository`` for new code.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import boto3
from botocore.exceptions import ClientError

if TYPE_CHECKING:
    from shared.schemas.config import PipelineConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_run_id() -> str:
    """Generate a sortable, UTC timestamp-based run identifier."""
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
    """Construct an S3 URI following the QueryForge hierarchical naming convention.

    Args:
        bucket: Name of the project's artifact bucket.
        schema_name: Domain name (e.g., ``"orders"``).
        schema_version: Schema semver (e.g., ``"v1"``).
        component: Logical segment (``dataset|model|adapter|gguf|metrics|schemas``).
        run_id: Unique identifier for the experiment or run.
        filename: Optional leaf object name appended to the path.
        prefix: Root prefix for the project.

    Returns:
        Formatted ``s3://`` URI string.
    """
    parts = [prefix, component, schema_name, schema_version, run_id]
    key = "/".join(parts)
    if filename:
        key = f"{key}/{filename}"
    return f"s3://{bucket}/{key}"


# ---------------------------------------------------------------------------
# S3Repository
# ---------------------------------------------------------------------------

class S3Repository:
    """Session-bound S3 CRUD operations for QueryForge artifacts.

    A single boto3 client is created at construction time and reused across
    all operations, avoiding per-call client instantiation overhead.

    Example::

        repo = S3Repository(boto3.Session(region_name="us-east-1"))
        if not repo.exists(config.s3_bucket, uri):
            repo.upload(local_path, uri)
    """

    def __init__(self, session: boto3.Session) -> None:
        self._client = session.client("s3")

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def resolve_component_prefix(config: "PipelineConfig", component: str) -> str:
        """Return the S3 key prefix for *component*, without the ``s3://bucket/`` scheme.

        Encapsulates the ``artifact_uris`` lookup and URI stripping pattern that
        was previously duplicated across ``manage_artifacts.py``,
        ``manage_models.py``, and ``run_finetuning.py``.

        Args:
            config: Validated pipeline configuration.
            component: Artifact folder name (e.g., ``"models"``, ``"datasets"``).

        Returns:
            S3 key prefix string (no leading ``s3://bucket/``).
        """
        uri = config.artifact_uris.get(
            f"{component}_uri",
            f"s3://{config.s3_bucket}/{config.s3_prefix}/{component}",
        )
        return uri.replace(f"s3://{config.s3_bucket}/", "")

    @staticmethod
    def component_uri(config: "PipelineConfig", component: str) -> str:
        """Return the full ``s3://`` URI for *component* from config.

        Args:
            config: Validated pipeline configuration.
            component: Artifact folder name (e.g., ``"models"``, ``"datasets"``).

        Returns:
            Full S3 URI string.
        """
        return config.artifact_uris.get(
            f"{component}_uri",
            f"s3://{config.s3_bucket}/{config.s3_prefix}/{component}",
        )

    # ------------------------------------------------------------------
    # Instance methods
    # ------------------------------------------------------------------

    def exists(self, bucket: str, s3_uri: str) -> bool:
        """Return ``True`` if an object exists at *s3_uri*."""
        key = s3_uri.replace(f"s3://{bucket}/", "")
        try:
            self._client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise

    def prefix_exists(self, bucket: str, prefix: str) -> bool:
        """Return ``True`` if any objects exist under *prefix*."""
        response = self._client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
        return "Contents" in response

    def upload(self, local_path: str, s3_uri: str) -> None:
        """Copy a local file to S3 at *s3_uri*."""
        bucket, key = s3_uri.replace("s3://", "").split("/", 1)
        self._client.upload_file(local_path, bucket, key)

    def download(self, bucket: str, key: str, local_path: str) -> None:
        """Copy an S3 object to *local_path*."""
        self._client.download_file(bucket, key, local_path)

    def put(self, bucket: str, key: str, body: str = "") -> None:
        """Create or overwrite an S3 object with *body* as content."""
        self._client.put_object(Bucket=bucket, Key=key, Body=body)

    def list(self, bucket: str, prefix: str) -> list[str]:
        """Return all object keys under *prefix* (directories excluded)."""
        paginator = self._client.get_paginator("list_objects_v2")
        keys: list[str] = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if "Contents" in page:
                keys.extend(
                    obj["Key"]
                    for obj in page["Contents"]
                    if not obj["Key"].endswith("/")
                )
        return keys

    def delete(self, bucket: str, s3_uri: str) -> None:
        """Delete a single object from S3."""
        key = s3_uri.replace(f"s3://{bucket}/", "")
        self._client.delete_object(Bucket=bucket, Key=key)

    def delete_prefix(self, bucket: str, prefix: str) -> None:
        """Delete all objects whose keys begin with *prefix*."""
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if "Contents" in page:
                objects = [{"Key": obj["Key"]} for obj in page["Contents"]]
                self._client.delete_objects(Bucket=bucket, Delete={"Objects": objects})


# ---------------------------------------------------------------------------
# Backward-compatible free functions
# ---------------------------------------------------------------------------
# These are thin wrappers over S3Repository that preserve the original call
# signatures.  Migrate call sites to S3Repository for new code.


def check_s3_uri_exists(bucket: str, s3_uri: str) -> bool:
    """Return ``True`` if an object exists at *s3_uri*.

    Deprecated: use :meth:`S3Repository.exists` with an injected session.
    """
    return S3Repository(boto3.Session()).exists(bucket, s3_uri)


def upload_file(local_path: str, s3_uri: str) -> None:
    """Copy a local file to S3.

    Deprecated: use :meth:`S3Repository.upload` with an injected session.
    """
    S3Repository(boto3.Session()).upload(local_path, s3_uri)


def list_s3_objects(bucket: str, prefix: str) -> list[str]:
    """List all object keys for a given prefix.

    Deprecated: use :meth:`S3Repository.list` with an injected session.
    """
    return S3Repository(boto3.Session()).list(bucket, prefix)


def delete_s3_object(bucket: str, s3_uri: str) -> None:
    """Delete a single object from S3.

    Deprecated: use :meth:`S3Repository.delete` with an injected session.
    """
    S3Repository(boto3.Session()).delete(bucket, s3_uri)


def delete_s3_prefix(bucket: str, prefix: str) -> None:
    """Delete all objects under a prefix.

    Deprecated: use :meth:`S3Repository.delete_prefix` with an injected session.
    """
    S3Repository(boto3.Session()).delete_prefix(bucket, prefix)
