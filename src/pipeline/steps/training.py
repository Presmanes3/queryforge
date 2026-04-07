import os
from pathlib import Path

from config import config
from pipeline.session import pipeline_session
from pipeline.parameters import training_dataset_uri
from pipeline.uris import model_s3_uri, output_s3_uri

from sagemaker.train.model_trainer import ModelTrainer, Mode
from sagemaker.train.configs import (
    Compute, InputData, OutputDataConfig, S3DataSource,
    SourceCode, StoppingCondition,
)
from sagemaker.train.constants import TRAIN_SCRIPT

from sagemaker.mlops.workflow.steps import TrainingStep

# ===== PATH TO ARTIFACTS =====
_TRAIN_DIR = Path(__file__).resolve().parents[2] / "train"

# Windows CRLF fix
_orig_prepare = ModelTrainer._prepare_train_script
def _prepare_lf(self, tmp_dir, source_code, distributed=None):
    _orig_prepare(self, tmp_dir, source_code, distributed)
    path = os.path.join(tmp_dir.name, TRAIN_SCRIPT)
    with open(path, "rb") as f: data = f.read()
    with open(path, "wb") as f: f.write(data.replace(b"\r\n", b"\n"))
ModelTrainer._prepare_train_script = _prepare_lf

hyperparameters = {
    "lora_r": config.train.lora_r,
    "lora_alpha": config.train.lora_alpha,
    "lora_dropout": config.train.lora_dropout,
    "epochs": config.train.epochs,
    "batch_size": config.train.batch_size,
    "grad_accum_steps": config.train.grad_accum_steps,
    "learning_rate": config.train.learning_rate,
    "max_seq_length": config.train.max_seq_length,
}


trainer = ModelTrainer(
    training_image      = config.effective_training_image_uri,  # Docker image URI for the training job
    sagemaker_session   = pipeline_session,             # Sagemaker Session for managing interactions with AWS services
    role                = config.execution_role_arn,    # IAM Role ARN with permissions for the training job
    source_code         = SourceCode(                   # Source code configuration for the training job
        source_dir      = str(_TRAIN_DIR),              # Directory containing the training code
        entry_script    = "train.py",
        requirements    = "requirements-train.txt",
    ),
    compute=Compute(
        instance_type   = config.train.instance_type,
        instance_count  = 1,
    ),
    hyperparameters     = hyperparameters,
    input_data_config=[
        InputData(channel_name="model",     data_source=S3DataSource(s3_uri=model_s3_uri,            s3_data_type="S3Prefix")),
        InputData(channel_name="training",  data_source=S3DataSource(s3_uri=training_dataset_uri,    s3_data_type="S3Prefix")),  
    ],
    output_data_config  = OutputDataConfig(s3_output_path=output_s3_uri),
    stopping_condition  = StoppingCondition(max_runtime_in_seconds=config.train.max_runtime_seconds),
    base_job_name       = "queryforge-train",
    training_mode       = Mode.SAGEMAKER_TRAINING_JOB,
)

training_step = TrainingStep(
    name="QLoraFineTune",
    step_args=trainer.train()
)