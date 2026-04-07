from pathlib import Path
from sagemaker.core.processing import ScriptProcessor, ProcessingInput, ProcessingOutput, ProcessingS3Input
from sagemaker.core.shapes.shapes import ProcessingS3Output
from sagemaker.mlops.workflow.steps import ProcessingStep
from sagemaker.core.workflow.properties import PropertyFile

from pipeline.session import pipeline_session
from config import config

from pipeline.uris import model_s3_uri, eval_output_uri
from pipeline.parameters import validation_dataset_uri

from pipeline.steps.training import training_step

# ===== PATH TO ARTIFACTS =====
_EVAL_DIR = Path(__file__).resolve().parents[2] / "evaluate" 

adapter_s3_uri = training_step.properties.ModelArtifacts.S3ModelArtifacts

validation_processor = ScriptProcessor(
    image_uri         = config.processing_image_uri,
    sagemaker_session = pipeline_session,
    command           = ["python3"],
    instance_type     = config.evaluation_instance_type,
    instance_count    = 1,
    role              = config.execution_role_arn,
    base_job_name     = "queryforge-evaluate",
)

metrics_output = PropertyFile(
    name        = "EvaluationMetrics",
    output_name = "metrics",
    path        = "metrics.json",
)

validation_step = ProcessingStep(
    name = "ModelEvaluation",
    step_args = validation_processor.run(
        code    = Path(_EVAL_DIR / "evaluate.py").as_uri(),
        inputs  = [
            ProcessingInput(
                input_name = "model",
                s3_input   = ProcessingS3Input(
                    s3_uri       = model_s3_uri,
                    local_path   = "/opt/ml/processing/input/model",
                    s3_data_type = "S3Prefix",
                ),
            ),
            ProcessingInput(
                input_name = "adapter",
                s3_input   = ProcessingS3Input(
                    s3_uri       = adapter_s3_uri,
                    local_path   = "/opt/ml/processing/input/adapter",
                    s3_data_type = "S3Prefix",
                ),
            ),
            ProcessingInput(
                input_name = "dataset",
                s3_input   = ProcessingS3Input(
                    s3_uri       = validation_dataset_uri,
                    local_path   = "/opt/ml/processing/input/dataset",
                    s3_data_type = "S3Prefix",
                ),
            ),
        ],
        outputs = [
            ProcessingOutput(
                output_name = "metrics",
                s3_output   = ProcessingS3Output(
                    s3_uri          = eval_output_uri,
                    local_path      = "/opt/ml/processing/output",
                    s3_upload_mode  = "EndOfJob",
                ),
            ),
        ],
    ),
    property_files = [metrics_output],
)