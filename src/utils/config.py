"""Configuration loader with environment variable overlay support.

Reads a YAML file and validates it against :class:`~shared.schemas.config.PipelineConfig`.

Environment overrides apply at two depths:

- **Top-level scalar** — ``QF_S3_BUCKET=my-bucket`` sets ``config.s3_bucket``.
- **Nested scalar** — ``QF_TRAIN__EPOCHS=5`` (double underscore) sets
  ``config.train.epochs``.

The double-underscore separator follows the convention used by dynaconf and
pydantic-settings.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from shared.schemas.config import PipelineConfig


# Resolved once at import time; avoids the brittle os.path.join(dirname, "../../..") pattern.
_DEFAULT_CONFIG_PATH: Path = Path(__file__).resolve().parents[2] / "config" / "pipeline.yaml"


class ConfigLoader:
    """Load and validate a QueryForge pipeline configuration from YAML.

    Prefer the module-level :func:`load_config` alias for simple use-cases.
    """

    @staticmethod
    def load(path: str | None = None) -> PipelineConfig:
        """Read a YAML config file and return a validated :class:`PipelineConfig`.

        Args:
            path: Absolute or relative path to the YAML file. Defaults to
                ``config/pipeline.yaml`` at the workspace root.

        Returns:
            Validated pipeline configuration instance.

        Raises:
            FileNotFoundError: If the YAML file does not exist at ``path``.
            pydantic.ValidationError: If the configuration fails schema validation.
        """
        config_path = Path(path) if path else _DEFAULT_CONFIG_PATH

        with config_path.open() as fh:
            data: dict = yaml.safe_load(fh) or {}

        ConfigLoader._apply_env_overrides(data)
        return PipelineConfig(**data)

    @staticmethod
    def _apply_env_overrides(data: dict) -> None:
        """Mutate *data* in place using ``QF_``-prefixed environment variables.

        Single-underscore suffix maps to a top-level field::

            QF_S3_BUCKET=x  →  data["s3_bucket"] = "x"

        Double-underscore separator descends one level::

            QF_TRAIN__EPOCHS=5  →  data.setdefault("train", {})["epochs"] = "5"
        """
        for env_key, env_val in os.environ.items():
            if not env_key.startswith("QF_"):
                continue

            field_path = env_key[3:].lower()  # strip "QF_" prefix
            parts = field_path.split("__", maxsplit=1)

            if len(parts) == 1:
                # Top-level scalar field.
                if parts[0] in PipelineConfig.model_fields:
                    data[parts[0]] = env_val
            else:
                # Nested field: parent__child  →  data[parent][child].
                parent, child = parts
                if parent in PipelineConfig.model_fields:
                    if not isinstance(data.get(parent), dict):
                        data[parent] = {}
                    data[parent][child] = env_val