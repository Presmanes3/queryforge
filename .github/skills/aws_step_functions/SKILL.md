---
name: aws_step_functions
description: >
  Patterns for AWS Step Functions state machine definitions in QueryForge.
  Use this skill when writing or modifying code in src/queryforge/orchestration/
  or any ASL (Amazon States Language) state machine definition.
---

# AWS Step Functions Skill

## Principles

- Step Functions orchestrate product-level flows: retraining triggers, notifications,
  human-in-the-loop approval, and cross-schema coordination.
- The ML-specific workflow (datagen → train → evaluate → register) lives in SageMaker
  Pipelines. Step Functions invoke SageMaker Pipelines as a task.
- All state machine definitions are written in Python dicts then serialized to ASL JSON.
  Never write raw ASL JSON by hand.
- State machine ARNs and task resource ARNs are read from `config/pipeline.yaml`.

---

## 1. Invocation pattern — SageMaker Pipeline as Step Functions task

```python
# src/queryforge/orchestration/retraining_statemachine.py
import json

def build_start_pipeline_state(pipeline_arn: str, schema_name: str) -> dict:
    """Build an ASL state that starts a SageMaker Pipeline execution.

    Args:
        pipeline_arn: ARN of the SageMaker Pipeline to invoke.
        schema_name: Name of the schema; embedded in pipeline parameters.

    Returns:
        ASL Task state dict for use in the Step Functions state machine.
    """
    return {
        "Type": "Task",
        "Resource": "arn:aws:states:::sagemaker:startPipelineExecution.sync:2",
        "Parameters": {
            "PipelineName": pipeline_arn,
            "PipelineParameters": [
                {"Name": "SchemaName", "Value": schema_name},
                {"Name": "RunId.$", "Value": "$$.Execution.Name"},
            ],
        },
        "ResultPath": "$.PipelineResult",
        "Catch": [
            {
                "ErrorEquals": ["States.ALL"],
                "Next": "NotifyFailure",
                "ResultPath": "$.Error",
            }
        ],
    }
```

### Notes

- Use the `.sync:2` suffix to wait for pipeline completion before transitioning.
- `$$.Execution.Name` is a Step Functions context variable for the current execution ID;
  it is used as the `RunId` so all artifacts are traceable back to the execution.

---

## 2. Human-in-the-loop approval state

```python
def build_approval_state(callback_queue_url: str) -> dict:
    """Build an ASL state that pauses and waits for human approval via SQS.

    The approver sends a task token to the queue to resume or fail the execution.

    Args:
        callback_queue_url: SQS queue URL where the approver sends the token.

    Returns:
        ASL Task state dict using the waitForTaskToken integration pattern.
    """
    return {
        "Type": "Task",
        "Resource": "arn:aws:states:::sqs:sendMessage.waitForTaskToken",
        "Parameters": {
            "QueueUrl": callback_queue_url,
            "MessageBody": {
                "TaskToken.$": "$$.Task.Token",
                "Input.$": "$",
            },
        },
        "HeartbeatSeconds": 86400,   # fail if no response within 24 hours
        "ResultPath": "$.ApprovalResult",
    }
```

---

## 3. Full state machine structure

```python
def build_retraining_state_machine(config) -> dict:
    """Build the ASL definition for the retraining state machine.

    Args:
        config: Validated PipelineConfig with ARNs and resource identifiers.

    Returns:
        Dict representing the full ASL state machine definition.
    """
    return {
        "Comment": "QueryForge retraining orchestration — triggers SageMaker Pipeline "
                   "and gates promotion on human approval.",
        "StartAt": "StartPipeline",
        "States": {
            "StartPipeline": {
                **build_start_pipeline_state(
                    config.pipeline_arn, "$$.Execution.Input.schema_name"
                ),
                "Next": "WaitForApproval",
            },
            "WaitForApproval": {
                **build_approval_state(config.approval_queue_url),
                "Next": "CheckApproval",
            },
            "CheckApproval": {
                "Type": "Choice",
                "Choices": [
                    {
                        "Variable": "$.ApprovalResult.decision",
                        "StringEquals": "approved",
                        "Next": "PromoteModel",
                    }
                ],
                "Default": "RejectModel",
            },
            "PromoteModel": {"Type": "Succeed"},
            "RejectModel":  {"Type": "Fail", "Error": "ModelRejected"},
            "NotifyFailure": {"Type": "Fail", "Error": "PipelineFailed"},
        },
    }


def deploy_state_machine(definition: dict, name: str, role_arn: str) -> str:
    """Create or update a Step Functions state machine.

    Args:
        definition: ASL dict produced by a build_*_state_machine function.
        name: State machine name.
        role_arn: IAM role ARN that Step Functions will assume to run tasks.

    Returns:
        ARN of the created or updated state machine.
    """
    import boto3
    client = boto3.client("stepfunctions")
    asl = json.dumps(definition)
    try:
        response = client.create_state_machine(
            name=name, definition=asl, roleArn=role_arn, type="STANDARD"
        )
        return response["stateMachineArn"]
    except client.exceptions.StateMachineAlreadyExists:
        machines = client.list_state_machines()["stateMachines"]
        arn = next(m["stateMachineArn"] for m in machines if m["name"] == name)
        client.update_state_machine(stateMachineArn=arn, definition=asl, roleArn=role_arn)
        return arn
```

---

## 4. Naming conventions

| Resource | Convention | Example |
|---|---|---|
| State machine | `queryforge-<workflow>` | `queryforge-retraining` |
| State name | `PascalCase`, imperative | `StartPipeline`, `WaitForApproval` |
| Error state | always named `NotifyFailure` | — |
| Config key | `<resource>_arn` | `pipeline_arn`, `approval_queue_url` |

---

## 5. Useful commands

```bash
# Deploy or update a state machine
python scripts/run_orchestration.py --config config/pipeline.yaml --action deploy

# Start a retraining execution
python scripts/run_orchestration.py --action start --schema-name orders

# Describe the latest execution
python scripts/run_orchestration.py --action describe --state-machine queryforge-retraining
```
