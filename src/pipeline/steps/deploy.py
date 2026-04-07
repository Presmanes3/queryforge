from __future__ import annotations

from botocore.exceptions import ClientError

from shared.schemas.config import PipelineConfig


def deploy_endpoint(execution, schema_name: str, config: PipelineConfig) -> None:
    """Deploy the registered model package to a SageMaker real-time endpoint.

    Reads the model package ARN produced by the RegisterModel pipeline step,
    then creates or updates a SageMaker endpoint from that package.
    """
    sm = config.boto_session().client("sagemaker")

    steps = execution.list_steps()
    model_package_arn = None
    for step in steps.get("PipelineExecutionSteps", []):
        if step["StepName"] == "RegisterModel" and step["StepStatus"] == "Succeeded":
            model_package_arn = (
                step.get("Metadata", {})
                    .get("RegisteredModel", {})
                    .get("Arn", "")
            )
            break

    if not model_package_arn:
        raise ValueError("Step 'RegisterModel' not found or did not succeed.")

    endpoint_name = f"queryforge-{schema_name}"
    sm_model_name = f"{endpoint_name}-model"
    endpoint_config_name = f"{endpoint_name}-config"

    sm.create_model(
        ModelName        = sm_model_name,
        Containers       = [{"ModelPackageName": model_package_arn}],
        ExecutionRoleArn = config.execution_role_arn,
    )

    sm.create_endpoint_config(
        EndpointConfigName  = endpoint_config_name,
        ProductionVariants  = [{
            "VariantName":          "AllTraffic",
            "ModelName":            sm_model_name,
            "InstanceType":         config.evaluation_instance_type,
            "InitialInstanceCount": 1,
        }],
    )

    try:
        sm.create_endpoint(
            EndpointName        = endpoint_name,
            EndpointConfigName  = endpoint_config_name,
        )
        print(f"Creating endpoint '{endpoint_name}'...")
    except ClientError as e:
        if e.response["Error"]["Code"] == "ValidationException":
            sm.update_endpoint(
                EndpointName        = endpoint_name,
                EndpointConfigName  = endpoint_config_name,
            )
            print(f"Updating endpoint '{endpoint_name}'...")
        else:
            raise
