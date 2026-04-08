from __future__ import annotations

import boto3
from pydantic import BaseModel, Field, field_validator


class TrainConfig(BaseModel):
    """QLoRA training job configuration."""

    instance_type: str = Field(default="ml.g5.2xlarge", description="SageMaker instance type for the training job.")
    max_runtime_seconds: int = Field(default=3600, description="Wall-clock limit in seconds before SageMaker stops the job.")
    epochs: int = Field(default=1, description="Number of training epochs.")
    batch_size: int = Field(default=4, description="Per-device training batch size.")
    grad_accum_steps: int = Field(default=4, description="Gradient accumulation steps.")
    learning_rate: float = Field(default=2e-4, description="AdamW learning rate.")
    max_seq_length: int = Field(default=512, description="Maximum token sequence length.")
    lora_r: int = Field(default=16, description="LoRA rank.")
    lora_alpha: int = Field(default=32, description="LoRA scaling factor.")
    lora_dropout: float = Field(default=0.05, description="Dropout applied to LoRA layers.")


class InferenceConfig(BaseModel):
    """SageMaker Real-time Inference configuration."""

    instance_type: str = Field(
        default="ml.g4dn.xlarge",
        description="SageMaker instance type for the inference endpoint. Requires GPU for vLLM."
    )
    initial_instance_count: int = Field(
        default=1,
        description="Initial number of instances to launch for the endpoint."
    )
    image_uri: str | None = Field(
        default=None,
        description="ECR image URI for the vLLM inference container."
    )


class PipelineConfig(BaseModel):
    """Single Source of Truth (SSoT) for QueryForge environment and infrastructure."""

    # AWS Infrastructure
    s3_bucket: str = Field(
        description="Global S3 bucket for all project artifacts (datagen, models, metrics)."
    )
    s3_prefix: str = Field(
        default="queryforge",
        description="Root prefix for all S3 artifacts."
    )
    aws_region: str = Field(
        default="us-east-1",
        description="AWS region where resources are provisioned."
    )
    aws_profile: str | None = Field(
        default=None,
        description="Named AWS credentials profile from ~/.aws/config. Uses the default profile when omitted.",
    )
    execution_role_arn: str = Field(
        description="IAM role ARN used by SageMaker Training and Processing jobs."
    )

    # SageMaker Compute
    train: TrainConfig = Field(
        default_factory=TrainConfig,
        description="QLoRA training job configuration (instance, hyperparameters, LoRA).",
    )
    inference: InferenceConfig = Field(
        default_factory=InferenceConfig,
        description="SageMaker Real-time Inference configuration.",
    )
    processing_instance_type: str = Field(
        default="ml.m5.large",
        description="Default instance type for data generation Processing Jobs."
    )
    evaluation_instance_type: str = Field(
        default="ml.g5.2xlarge",
        description="Instance type for evaluation Processing Jobs. Requires GPU to run inference."
    )
    processing_image_uri: str = Field(
        description="ECR image URI for the evaluation Processing Job container."
    )
    training_image_uri: str | None = Field(
        default=None,
        description="ECR image URI for the training job container. When None, SageMaker selects the default PyTorch DLC automatically.",
    )

    @property
    def effective_training_image_uri(self) -> str | None:
        """Return the training image URI, or None to let SageMaker auto-select the DLC."""
        return self.training_image_uri or None

    # MLflow
    mlflow_tracking_uri: str | None = Field(
        default=None,
        description="SageMaker MLflow Tracking Server ARN (arn:aws:sagemaker:...:mlflow-tracking-server/name). When set, all training jobs report metrics to this server.",
    )
    mlflow_experiment_name: str = Field(
        default="queryforge-finetuning",
        description="MLflow experiment name used to group all QueryForge training runs.",
    )

    # Bedrock
    bedrock_model_id: str = Field(
        default="amazon.nova-pro-v1:0",
        description="AWS Bedrock model ID used for synthetic question-SQL pair generation."
    )

    # Pipeline Thresholds
    accuracy_threshold: float = Field(
        default=0.75,
        description="Minimum execution accuracy (0.0-1.0) required to register a model."
    )

    # Dynamic Infrastructure
    artifact_folders: list[str] = Field(
        default_factory=lambda: ["datasets", "models", "schemas", "metrics", "adapters", "gguf"],
        description="List of logical segments to initialize in S3.",
    )
    artifact_uris: dict[str, str] = Field(
        default_factory=dict,
        description="Populated S3 URIs for project folders (dataset_uri, model_uri, etc.).",
    )

    def boto_session(self) -> boto3.Session:
        """Build a boto3 Session bound to this config's region and optional profile.

        Returns:
            A ``boto3.Session`` configured with ``aws_region`` and, when set,
            ``aws_profile``.  All AWS SDK calls in the project should obtain
            their session from this method rather than calling
            ``boto3.Session()`` directly.
        """
        return boto3.Session(
            region_name=self.aws_region,
            profile_name=self.aws_profile,  # None → boto3 uses the default chain
        )

    @field_validator("execution_role_arn")
    @classmethod
    def validate_arn(cls, v: str) -> str:
        """Ensure the ARN follows a valid IAM role format."""
        if not v.startswith("arn:aws:iam::"):
            raise ValueError("execution_role_arn must be a valid AWS IAM role ARN.")
        return v
