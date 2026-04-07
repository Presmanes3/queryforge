from sagemaker.mlops.workflow.pipeline import Pipeline

from pipeline.steps.training import training_step
from pipeline.steps.evaluate import validation_step
from pipeline.steps.register_model_check import condition_step

from pipeline.session import pipeline_session

from .parameters import model_name, schema_name, schema_version, accuracy_threshold, training_dataset_uri, validation_dataset_uri

pipeline = Pipeline(
    name                = "QueryForgeFinetuning",
    parameters          = [model_name, schema_name, schema_version, accuracy_threshold, training_dataset_uri, validation_dataset_uri],
    steps               = [training_step, validation_step, condition_step],
    sagemaker_session   = pipeline_session,
)