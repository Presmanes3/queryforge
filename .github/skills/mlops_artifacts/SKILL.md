---
name: mlops_artifacts
description: >
  Rules for S3 path construction, artifact versioning, and configuration loading
  in QueryForge. Use this skill whenever writing code that reads from or writes to S3,
  constructs artifact paths, or loads configuration in src/queryforge/utils/ or config/.
---

# MLOps Artifacts Skill

## Principles

- No component constructs an S3 URI directly. All paths go through `src/queryforge/utils/s3.py`.
- All configuration is loaded at startup via Pydantic from `config/pipeline.yaml`
  and environment variables. No value is hardcoded in source code.
- Artifact paths embed `schema_name`, `schema_version`, `component`, and `run_id` to
  guarantee isolation between experiments and schema versions.

---

## 1. Pipeline configuration schema

```python
# shared/schemas/config.py
import os
from pydantic import BaseModel, Field, field_validator

class PipelineConfig(BaseModel):
    """Validated configuration for the QueryForge pipeline."""

    s3_bucket: str = Field(description="S3 bucket name for all QueryForge artifacts.")
    execution_role_arn: str = Field(description="IAM role ARN for SageMaker execution.")
    processing_image_uri: str = Field(description="ECR image URI for ProcessingStep containers.")
    train_instance_type: str = Field(description="SageMaker instance type for Training Jobs.")
    processing_instance_type: str = Field(description="SageMaker instance type for Processing Jobs.")
    accuracy_threshold: float = Field(
        default=0.75,
        description="Minimum Execution Accuracy required for model registration.",
    )
    aws_region: str = Field(description="AWS region for all SageMaker and S3 operations.")

    @field_validator("execution_role_arn")
    @classmethod
    def validate_arn(cls, v: str) -> str:
        if not v.startswith("arn:aws:iam::"):
            raise ValueError("execution_role_arn must be a valid IAM role ARN.")
        return v
```

### Loading convention

```python
# src/queryforge/utils/config.py
import os
import yaml
from shared.schemas.config import PipelineConfig

def load_config(path: str | None = None) -> PipelineConfig:
    """Load and validate pipeline configuration from a YAML file.

    Reads YAML from `path` (defaults to config/pipeline.yaml) and overlays any
    environment variables with matching uppercase keys (e.g., QF_S3_BUCKET).

    Args:
        path: Optional absolute path to a YAML config file.

    Returns:
        Validated PipelineConfig instance.

    Raises:
        ValidationError: If any required field is missing or invalid.
    """
    config_path = path or os.path.join(os.path.dirname(__file__), "../../../config/pipeline.yaml")
    with open(config_path) as f:
        data = yaml.safe_load(f)
    # Allow env vars to override: QF_S3_BUCKET overrides s3_bucket, etc.
    for key in PipelineConfig.model_fields:
        env_key = f"QF_{key.upper()}"
        if env_key in os.environ:
            data[key] = os.environ[env_key]
    return PipelineConfig(**data)
```

---

## 2. S3 path helper

```python
# src/queryforge/utils/s3.py
import boto3
from botocore.exceptions import ClientError

S3_PREFIX = "queryforge"

def build_s3_uri(
    bucket: str,
    schema_name: str,
    schema_version: str,
    component: str,
    run_id: str,
    filename: str | None = None,
) -> str:
    """Construct an S3 URI following the QueryForge path convention.

    Pattern: s3://<bucket>/queryforge/<schema_name>/<schema_version>/<component>/<run_id>/

    Args:
        bucket: S3 bucket name.
        schema_name: Logical name of the Pydantic schema (e.g., "orders").
        schema_version: Version string of the schema (e.g., "v1").
        component: Pipeline component segment: dataset | model | adapter | gguf | metrics.
        run_id: ISO timestamp or SageMaker execution ID.
        filename: Optional filename appended to the prefix.

    Returns:
        Full S3 URI string.
    """
    parts = [S3_PREFIX, schema_name, schema_version, component, run_id]
    key = "/".join(parts)
    if filename:
        key = f"{key}/{filename}"
    return f"s3://{bucket}/{key}"


def upload_file(local_path: str, s3_uri: str) -> None:
    """Upload a local file to S3.

    Args:
        local_path: Absolute path to the local file.
        s3_uri: Full S3 URI of the destination.

    Raises:
        ClientError: On S3 upload failure.
    """
    bucket, key = s3_uri.replace("s3://", "").split("/", 1)
    boto3.client("s3").upload_file(local_path, bucket, key)


def download_file(s3_uri: str, local_path: str) -> None:
    """Download a file from S3 to a local path.

    Args:
        s3_uri: Full S3 URI of the source.
        local_path: Absolute path to write the file.

    Raises:
        ClientError: On S3 download failure.
    """
    bucket, key = s3_uri.replace("s3://", "").split("/", 1)
    boto3.client("s3").download_file(bucket, key, local_path)
```

---

## 3. Config YAML template

```yaml
# config/pipeline.yaml
s3_bucket: ""                        # Set via QF_S3_BUCKET env var or fill in
execution_role_arn: ""               # Set via QF_EXECUTION_ROLE_ARN env var
processing_image_uri: ""             # versioned ECR URI
train_instance_type: "ml.g5.2xlarge"
processing_instance_type: "ml.m5.large"
accuracy_threshold: 0.75
aws_region: "us-east-1"
```

### Security note

The `config/pipeline.yaml` file **must** be listed in `.gitignore` if it contains real
values. Use the template above with empty strings and populate via environment variables
in CI/CD.

---

## 4. Run ID convention

```python
from datetime import datetime, timezone

def generate_run_id() -> str:
    """Generate a sortable run ID based on the current UTC timestamp.

    Returns:
        ISO 8601 string with colons replaced by hyphens (S3-safe): 20240315T143022Z.
    """
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
```

---

## 5. Dataset hash

```python
import hashlib

def compute_jsonl_hash(path: str) -> str:
    """Compute the SHA-256 hash of a JSONL file for dataset identity tracking.

    Args:
        path: Absolute path to the JSONL file.

    Returns:
        Hex-encoded SHA-256 digest string.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
```
