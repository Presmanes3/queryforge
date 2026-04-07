from sagemaker.core.workflow.parameters import ParameterString, ParameterFloat

from config import config as _config

_bucket  = _config.s3_bucket
_prefix  = _config.s3_prefix
_default_schema  = "orders"
_default_version = "v1"
_default_dataset_base = (
    f"s3://{_bucket}/{_prefix}/datasets/{_default_schema}/{_default_version}"
)

model_name              = ParameterString("ModelName",              default_value="Llama-3.2-1B-Instruct")
schema_name             = ParameterString("SchemaName",             default_value=_default_schema)
schema_version          = ParameterString("SchemaVersion",          default_value=_default_version)
accuracy_threshold      = ParameterFloat("AccuracyThreshold",       default_value=0.75)
training_dataset_uri    = ParameterString("TrainingDatasetUri",     default_value=f"{_default_dataset_base}/{_default_schema}_{_default_version}_train.jsonl")
validation_dataset_uri  = ParameterString("ValidationDatasetUri",   default_value=f"{_default_dataset_base}/{_default_schema}_{_default_version}_test.jsonl")

