"""Configuration loader with environment variable overlay support.

Reads YAML from config/pipeline.yaml and overlays keys starting with QF_.
"""

from __future__ import annotations
import os
import yaml
from shared.schemas.config import PipelineConfig


def load_config(path: str | None = None) -> PipelineConfig:
    """Read and validate the QueryForge configuration.

    Args:
        path: Absolute path to the config file. Defaults to config/pipeline.yaml.

    Returns:
        Validated configuration instance.

    Raises:
        FileNotFoundError: If the YAML configuration file is missing.
        ValidationError: If the configuration fails strict Pydantic validation.
    """
    config_path = path or os.path.join(
        os.path.dirname(__file__), "../../../config/pipeline.yaml"
    )

    with open(config_path) as f:
        data = yaml.safe_load(f)

    # Environment overlay logic: QF_S3_BUCKET -> s3_bucket
    for key in PipelineConfig.model_fields:
        env_key = f"QF_{key.upper()}"
        if env_key in os.environ:
            data[key] = os.environ[env_key]

    return PipelineConfig(**data)
