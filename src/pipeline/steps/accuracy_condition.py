from sagemaker.core.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.core.workflow.functions import JsonGet
from sagemaker.mlops.workflow.condition_step import ConditionStep
from sklearn.metrics import accuracy_score

from pipeline.steps.evaluate import validation_step, metrics_output
from pipeline.parameters import accuracy_threshold

accuracy_condition = ConditionGreaterThanOrEqualTo(
    left    = JsonGet(
        step_name       = validation_step.name,
        property_file   = metrics_output,
        json_path       = "execution_accuracy",
    ),
    right   = accuracy_threshold,
)