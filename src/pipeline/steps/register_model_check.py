from sagemaker.mlops.workflow.condition_step import ConditionStep

from pipeline.steps.model_builder import register_model_step
from pipeline.steps.accuracy_condition import accuracy_condition

condition_step = ConditionStep(
    name        = "AccuracyGate",
    conditions  = [accuracy_condition],
    if_steps    = [register_model_step],
    else_steps  = [],
)