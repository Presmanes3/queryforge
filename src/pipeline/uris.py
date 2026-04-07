from pipeline.parameters import model_name, schema_name, schema_version
from config import config

from sagemaker.core.workflow.functions import Join


bucket = config.s3_bucket
prefix = config.s3_prefix

model_s3_uri = Join(
    on="/",
    values=[f"s3://{bucket}/{prefix}/models", model_name, schema_version],
)

dataset_s3_uri = Join(
    on="/",
    values=[f"s3://{bucket}/{prefix}/datasets", schema_name, schema_version],
)

output_s3_uri = Join(
    on="/",
    values=[f"s3://{bucket}/{prefix}/adapters", schema_name, schema_version],
)

eval_output_uri = Join(
    on="/",
    values=[f"s3://{bucket}/{prefix}/evaluation", schema_name, schema_version],
)

