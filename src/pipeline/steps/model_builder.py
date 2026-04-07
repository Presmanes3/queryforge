from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.mlops.workflow.model_step import ModelStep
from sagemaker.core.workflow.functions import Join

from pipeline.session import pipeline_session
from config import config
from pipeline.uris import model_s3_uri
from pipeline.steps.evaluate import adapter_s3_uri
from pipeline.parameters import schema_name
from pipeline.steps.training import trainer

model_builder = ModelBuilder(
    s3_model_data_url   = adapter_s3_uri,
    image_uri           = trainer.training_image,
    role_arn            = config.execution_role_arn,
    sagemaker_session   = pipeline_session,
    content_type        = "application/json",
    accept_type         = "application/json",
)

register_model_step = ModelStep(
    name                            = "RegisterModel",
    step_args                       = model_builder.register(
        model_package_group_name    = Join(on="-", values=["queryforge", schema_name]),
        approval_status             = "PendingManualApproval",
        content_types               = ["application/json"],
        response_types              = ["application/json"],
    ),
)