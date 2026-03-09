"""SageMaker ModelTrainer builder for QLoRA fine-tuning jobs.

Provides :class:`TrainingJobBuilder`, a fluent builder that is the single
source of truth for SageMaker training configuration: source directory,
entry script, instance type resolution, hyperparameter assembly, and job
naming conventions.

Both cloud (``SAGEMAKER_TRAINING_JOB``) and local Docker
(``LOCAL_CONTAINER``) modes are handled by the same builder class via
:meth:`TrainingJobBuilder.with_mode`.

Backward-compatible module-level functions :func:`build_estimator` and
:func:`build_training_inputs` are preserved for existing call sites.

Windows workaround
------------------
``ModelTrainer._prepare_train_script`` opens its output with ``mode="w"``,
which on Windows writes CRLF line endings.  The Linux SageMaker container
then fails with ``$'\\r': command not found``.  The monkey-patch below fixes
that single function in-process.  Delete lines 47-59 once AWS ships the fix.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from sagemaker.core.helper.session_helper import Session
from sagemaker.train.constants import TRAIN_SCRIPT
from sagemaker.train.model_trainer import ModelTrainer, Mode
from sagemaker.train.configs import (
    Compute,
    InputData,
    OutputDataConfig,
    S3DataSource,
    SourceCode,
    StoppingCondition,
    Tag,
)

from shared.schemas.config import PipelineConfig
from queryforge.train.hyperparameters import build_hyperparameters


# ---------------------------------------------------------------------------
# Windows CRLF monkey-patch
# ---------------------------------------------------------------------------

# Upstream bug: open("w") on Windows writes CRLF; bash in the Linux container
# rejects the \r bytes.  One-line fix applied in-process until AWS patches it.
_orig_prepare = ModelTrainer._prepare_train_script


def _prepare_train_script_lf(self, tmp_dir, source_code, distributed=None):
    _orig_prepare(self, tmp_dir, source_code, distributed)
    path = os.path.join(tmp_dir.name, TRAIN_SCRIPT)
    with open(path, "rb") as f:
        data = f.read()
    with open(path, "wb") as f:
        f.write(data.replace(b"\r\n", b"\n"))


ModelTrainer._prepare_train_script = _prepare_train_script_lf


# ---------------------------------------------------------------------------
# TrainingJobBuilder
# ---------------------------------------------------------------------------

class TrainingJobBuilder:
    """Fluent builder for SageMaker V3 :class:`ModelTrainer` instances.

    Single source of truth for training job constants: source directory,
    entry script, requirements file, job naming, and hyperparameter assembly.
    All required values are read from the injected :class:`PipelineConfig`;
    optional overrides are added via chained ``with_*`` methods.

    Example — cloud job::

        trainer = (
            TrainingJobBuilder(config)
            .with_output_s3_uri(output_uri)
            .with_input_data(build_training_inputs(model_uri, dataset_uri))
            .build()
        )

    Example — local Docker job::

        trainer = (
            TrainingJobBuilder(config)
            .with_mode(Mode.LOCAL_CONTAINER)
            .with_training_image(args.training_image)
            .with_instance_type("local_gpu")
            .with_input_data([
                InputData(channel_name="model", data_source=str(model_dir)),
                InputData(channel_name="training", data_source=str(dataset_dir)),
            ])
            .build()
        )
    """

    # Paths relative to the workspace root (used for cloud jobs).
    # For LOCAL_CONTAINER, _resolve_source_dir() returns the absolute path.
    _ENTRY_SCRIPT = "train.py"
    _REQUIREMENTS = "requirements-train.txt"
    # Absolute path to this file's directory == src/queryforge/train/
    _TRAIN_DIR: Path = Path(__file__).resolve().parent

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._mode: Mode = Mode.SAGEMAKER_TRAINING_JOB
        self._training_image: str | None = None
        self._instance_type: str | None = None
        self._output_s3_uri: str | None = None
        self._input_data: list[InputData] = []
        self._extra_hp: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Builder methods
    # ------------------------------------------------------------------

    def with_mode(self, mode: Mode) -> "TrainingJobBuilder":
        """Set the execution mode (cloud vs local container)."""
        self._mode = mode
        return self

    def with_training_image(self, uri: str) -> "TrainingJobBuilder":
        """Override the Docker image URI for the training container.

        Required for ``LOCAL_CONTAINER`` mode (no default exists for local).
        For cloud mode, defaults to ``config.processing_image_uri``.
        """
        self._training_image = uri
        return self

    def with_instance_type(self, override: str | None) -> "TrainingJobBuilder":
        """Override the compute instance type.

        Passing ``None`` is a no-op; the builder's default resolution runs.
        """
        if override is not None:
            self._instance_type = override
        return self

    def with_output_s3_uri(self, uri: str) -> "TrainingJobBuilder":
        """Set the S3 URI where SageMaker writes the adapter artifact (cloud only)."""
        self._output_s3_uri = uri
        return self

    def with_input_data(self, inputs: list[InputData]) -> "TrainingJobBuilder":
        """Set the data channel list for the training job."""
        self._input_data = inputs
        return self

    def with_extra_hyperparameters(self, extra: dict[str, Any]) -> "TrainingJobBuilder":
        """Merge additional hyperparameters on top of those derived from config."""
        self._extra_hp = extra
        return self

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> ModelTrainer:
        """Construct and return the configured :class:`ModelTrainer`.

        Returns:
            A ready-to-call ``ModelTrainer`` instance.

        Raises:
            ValueError: If a required field is missing (e.g., no training image
                for ``LOCAL_CONTAINER`` mode when no config default is available).
        """
        hyperparameters = {**build_hyperparameters(self._config.train), **self._extra_hp}

        boto_session = self._config.boto_session()
        sagemaker_session = Session(
            boto_session=boto_session,
            default_bucket=self._config.s3_bucket,
            default_bucket_prefix=self._config.s3_prefix,
        )

        is_local = self._mode == Mode.LOCAL_CONTAINER
        base_job_name = "queryforge-local-train" if is_local else "queryforge-train"

        trainer_kwargs: dict[str, Any] = dict(
            training_image=self._resolve_training_image(),
            sagemaker_session=sagemaker_session,
            source_code=SourceCode(
                source_dir=self._resolve_source_dir(),
                entry_script=self._ENTRY_SCRIPT,
                requirements=self._REQUIREMENTS,
            ),
            compute=Compute(
                instance_type=self._resolve_instance_type(),
                instance_count=1,
            ),
            hyperparameters=hyperparameters,
            input_data_config=self._input_data,
            base_job_name=base_job_name,
            training_mode=self._mode,
        )

        if not is_local:
            trainer_kwargs["role"] = self._config.execution_role_arn
            trainer_kwargs["tags"] = [Tag(key="project", value="queryforge")]
            trainer_kwargs["stopping_condition"] = StoppingCondition(
                max_runtime_in_seconds=self._config.train.max_runtime_seconds,
            )
            if self._output_s3_uri:
                trainer_kwargs["output_data_config"] = OutputDataConfig(
                    s3_output_path=self._output_s3_uri
                )

        return ModelTrainer(**trainer_kwargs)

    # ------------------------------------------------------------------
    # Private resolution helpers
    # ------------------------------------------------------------------

    def _resolve_source_dir(self) -> str:
        # LOCAL_CONTAINER requires an absolute path for Docker volume mounting.
        # Cloud jobs accept a workspace-relative path.
        if self._mode == Mode.LOCAL_CONTAINER:
            return str(self._TRAIN_DIR)
        return "src/queryforge/train"

    def _resolve_training_image(self) -> str:
        if self._training_image:
            return self._training_image
        return self._config.processing_image_uri

    def _resolve_instance_type(self) -> str:
        if self._instance_type:
            return self._instance_type
        if self._mode == Mode.LOCAL_CONTAINER:
            # noqa: PLC0415 — heavy optional dependency, only for GPU auto-detection
            try:
                import torch
                return "local_gpu" if torch.cuda.is_available() else "local_cpu"
            except ImportError:
                return "local_cpu"
        return self._config.train.instance_type


# ---------------------------------------------------------------------------
# Backward-compatible module-level functions
# ---------------------------------------------------------------------------

def build_estimator(
    config: PipelineConfig,
    hyperparameters: dict,
    output_s3_uri: str,
) -> ModelTrainer:
    """Build a cloud :class:`ModelTrainer` for QLoRA fine-tuning.

    Deprecated: use :class:`TrainingJobBuilder` directly for new call sites.

    Args:
        config: Validated pipeline configuration.
        hyperparameters: Hyperparameter dict (ignored; config values are used).
        output_s3_uri: S3 URI where SageMaker writes the adapter artifact.

    Returns:
        Configured :class:`ModelTrainer` ready to call ``.train()`` on.
    """
    return (
        TrainingJobBuilder(config)
        .with_output_s3_uri(output_s3_uri)
        .with_extra_hyperparameters(hyperparameters)
        .build()
    )


def build_training_inputs(model_s3_uri: str, dataset_s3_uri: str) -> list[InputData]:
    """Build the S3 data channel list for a cloud Training Job.

    Args:
        model_s3_uri: S3 URI of the base model directory (S3Prefix).
        dataset_s3_uri: S3 URI of the JSONL dataset directory (S3Prefix).

    Returns:
        List of :class:`InputData` channels: ``'model'`` and ``'training'``.
    """
    return [
        InputData(
            channel_name="model",
            data_source=S3DataSource(
                s3_uri=model_s3_uri,
                s3_data_type="S3Prefix",
            ),
        ),
        InputData(
            channel_name="training",
            data_source=S3DataSource(
                s3_uri=dataset_s3_uri,
                s3_data_type="S3Prefix",
            ),
        ),
    ]
