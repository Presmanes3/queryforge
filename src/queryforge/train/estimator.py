"""SageMaker ModelTrainer factory for QLoRA fine-tuning jobs.

Uses the sagemaker-train ModelTrainer, which is the correct high-level
abstraction in sagemaker v3 for launching Training Jobs.

Windows workaround: ModelTrainer._prepare_train_script uses open("w"), which on
Windows writes CRLF. The Linux SageMaker container then fails with
"$'\\r': command not found". The monkey-patch below fixes that single line
in-process. When AWS ships the fix upstream, delete lines 33-44.
"""

from __future__ import annotations

import os

import boto3
from sagemaker.core.helper.session_helper import Session
from sagemaker.train.constants import TRAIN_SCRIPT
from sagemaker.train.model_trainer import ModelTrainer
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


# Upstream bug: open("w") on Windows writes CRLF; bash in the Linux container
# rejects the \r bytes. One-line fix applied in-process until AWS patches it.
_orig_prepare = ModelTrainer._prepare_train_script


def _prepare_train_script_lf(self, tmp_dir, source_code, distributed=None):
    _orig_prepare(self, tmp_dir, source_code, distributed)
    path = os.path.join(tmp_dir.name, TRAIN_SCRIPT)
    with open(path, "rb") as f:
        data = f.read()
    with open(path, "wb") as f:
        f.write(data.replace(b"\r\n", b"\n"))


ModelTrainer._prepare_train_script = _prepare_train_script_lf


def build_estimator(
    config: PipelineConfig,
    hyperparameters: dict,
    output_s3_uri: str,
) -> ModelTrainer:
    """Build a ModelTrainer for QLoRA fine-tuning.

    Args:
        config: Validated pipeline configuration loaded from config/pipeline.yaml.
        hyperparameters: JSON-serializable dict of training hyperparameters.
        output_s3_uri: S3 URI where SageMaker will write the adapter artifact.

    Returns:
        Configured ModelTrainer ready to call .train() on.
    """
    boto_session = boto3.Session(region_name=config.aws_region)
    sagemaker_session = Session(
        boto_session=boto_session,
        default_bucket=config.s3_bucket,
        default_bucket_prefix=config.s3_prefix,
    )
    return ModelTrainer(
        sagemaker_session=sagemaker_session,
        training_image=config.processing_image_uri,
        source_code=SourceCode(
            source_dir="src/queryforge/train",
            entry_script="train.py",
            requirements="requirements-train.txt",
        ),
        compute=Compute(
            instance_type=config.train.instance_type,
            instance_count=1,
        ),
        stopping_condition=StoppingCondition(
            max_runtime_in_seconds=config.train.max_runtime_seconds,
        ),
        hyperparameters=hyperparameters,
        output_data_config=OutputDataConfig(s3_output_path=output_s3_uri),
        base_job_name="queryforge-train",
        role=config.execution_role_arn,
        tags=[Tag(key="project", value="queryforge")],
    )


def build_training_inputs(model_s3_uri: str, dataset_s3_uri: str) -> list[InputData]:
    """Build the data channel list for the SageMaker Training Job.

    Args:
        model_s3_uri: S3 URI of the base model directory (S3Prefix).
        dataset_s3_uri: S3 URI of the JSONL dataset directory (S3Prefix).

    Returns:
        List of InputData channels: 'model' and 'training'.
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
