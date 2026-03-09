---
name: sagemaker_patterns
description: >
  Canonical implementation patterns for QueryForge's SageMaker integration.
  Use this skill whenever creating or modifying Training Jobs, Processing Steps,
  SageMaker Pipelines, or Model Registry operations in src/queryforge/pipeline/,
  src/queryforge/train/, src/queryforge/registry/, or scripts/.
---

# SageMaker Patterns Skill

## Principles

- All AWS resource identifiers (IAM roles, S3 buckets, image URIs) are read from
  `config/pipeline.yaml` via the Pydantic config schema. Never hardcode them.
- Every SageMaker object is constructed from explicit, typed config values.
- Re-running a pipeline with the same inputs must produce the same outputs (idempotent).

---

## 1. Training Job — HuggingFace Estimator

```python
# src/queryforge/train/estimator.py
import sagemaker
from sagemaker.huggingface import HuggingFace
from shared.schemas.config import PipelineConfig

def build_estimator(config: PipelineConfig, hyperparameters: dict) -> HuggingFace:
    """Build a HuggingFace estimator for QLoRA fine-tuning.

    Args:
        config: Validated pipeline configuration loaded from config/pipeline.yaml.
        hyperparameters: JSON-serializable dict of training hyperparameters.

    Returns:
        Configured HuggingFace estimator ready for .fit() or Pipeline Step.
    """
    return HuggingFace(
        entry_point="train.py",
        source_dir="src/queryforge/train",
        role=config.execution_role_arn,
        instance_type=config.train_instance_type,
        instance_count=1,
        transformers_version="4.36",
        pytorch_version="2.1",
        py_version="py311",
        hyperparameters=hyperparameters,
        output_path=config.s3_model_output_uri,
        base_job_name="queryforge-train",
        tags=[{"Key": "project", "Value": "queryforge"}],
    )
```

### Rules

- `entry_point` must be a file inside `source_dir`; do not pass absolute paths.
- All hyperparameters must be JSON-serializable. No Python objects, no enums —
  serialize enums to their `.value` before passing.
- `base_job_name` follows the pattern `queryforge-<component>`.
- Always tag jobs with `{"Key": "project", "Value": "queryforge"}`.

---

## 2. Processing Step — ScriptProcessor

```python
# src/queryforge/pipeline/steps/datagen_step.py
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from shared.schemas.config import PipelineConfig

def build_datagen_processor(config: PipelineConfig) -> ScriptProcessor:
    """Build a processor for the dataset generation step."""
    return ScriptProcessor(
        role=config.execution_role_arn,
        image_uri=config.processing_image_uri,   # versioned URI from config
        instance_type=config.processing_instance_type,
        instance_count=1,
        base_job_name="queryforge-datagen",
        tags=[{"Key": "project", "Value": "queryforge"}],
    )
```

### ProcessingInput / ProcessingOutput conventions

| Parameter | Value |
|---|---|
| `source` | Full S3 URI constructed via `utils/s3.py` |
| `destination` | `/opt/ml/processing/<component>/input` or `output` |
| `s3_data_type` | `"S3Prefix"` for directories, `"S3Object"` for single files |

---

## 3. SageMaker Pipeline — Structure

```python
# src/queryforge/pipeline/finetune_pipeline.py
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterString, ParameterFloat

# --- Pipeline parameters are declared at the top, never inside step definitions ---
schema_name   = ParameterString(name="SchemaName",   default_value="orders")
schema_version = ParameterString(name="SchemaVersion", default_value="v1")
run_id        = ParameterString(name="RunId",         default_value="")
accuracy_threshold = ParameterFloat(name="AccuracyThreshold", default_value=0.75)

# ... build steps using parameters above ...

pipeline = Pipeline(
    name="QueryForgeFinetuning",
    parameters=[schema_name, schema_version, run_id, accuracy_threshold],
    steps=[datagen_step, train_step, eval_step, condition_step],
    sagemaker_session=sagemaker.Session(),
)
```

### Step ordering rules

1. `ProcessingStep` — datagen: generates JSONL dataset
2. `TrainingStep` — train: runs QLoRA fine-tuning
3. `ProcessingStep` — evaluate: measures Execution Accuracy
4. `ConditionStep` — gate: compares accuracy against `accuracy_threshold`
5. `RegisterModel` — register: only runs if condition is met

---

## 4. ConditionStep — Model Registration Gate

```python
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet

accuracy_condition = ConditionGreaterThanOrEqualTo(
    left=JsonGet(
        step_name=eval_step.name,
        property_file=eval_output,
        json_path="execution_accuracy",
    ),
    right=accuracy_threshold,
)

gate_step = ConditionStep(
    name="AccuracyGate",
    conditions=[accuracy_condition],
    if_steps=[register_step],
    else_steps=[],
)
```

---

## 5. Model Registry — Registration

```python
# src/queryforge/registry/register.py
from shared.schemas.registry import ModelRegistryMetadata

def register_model(metadata: ModelRegistryMetadata, model_data_url: str, config) -> None:
    """Register a model version with the SageMaker Model Registry.

    Args:
        metadata: Validated model metadata including schema name, version, and metrics.
        model_data_url: S3 URI of the model artifact (tar.gz).
        config: Validated pipeline configuration.
    """
    model_package_group_name = f"queryforge-{metadata.schema_name}"
    # ... registration code using boto3 or sagemaker SDK ...
```

### Metadata requirements

Every `create_model_package` call must include all fields from `ModelRegistryMetadata`
as `CustomerMetadataProperties`. See `shared/schemas/registry.py` for the full contract.

---

## Useful commands

```bash
# Upsert a pipeline definition (create or update)
python scripts/run_pipeline.py --config config/pipeline.yaml --action upsert

# Start a pipeline execution
python scripts/run_pipeline.py --config config/pipeline.yaml --action start \
    --schema-name orders --schema-version v1

# Describe a pipeline execution
python scripts/run_pipeline.py --action describe --execution-arn <arn>
```
