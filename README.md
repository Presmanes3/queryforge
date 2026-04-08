# QueryForge

QueryForge is an open-source framework for building schema-aware Text-to-SQL assistants. Starting from structured data models, it generates synthetic question-SQL pairs, fine-tunes lightweight LLMs on AWS SageMaker, evaluates model quality, and packages optimized versions for fast local inference with Ollama.

## Prerequisites

- Python 3.10+
- AWS Account with SageMaker and S3 access
- Docker (for image builds and local evaluation)
- [uv](https://github.com/astral-sh/uv) (recommended)

## Setup

1. Configure AWS environment:
   ```bash
   python scripts/setup/setup_aws.py
   ```
2. Prepare configuration:
   ```bash
   cp config/pipeline.yaml.example config/pipeline.yaml
   ```
   Edit `config/pipeline.yaml` with your AWS resource identifiers.
3. Install dependencies:
   ```bash
   uv sync --all-extras
   ```

## Component Overview

| Package | Responsibility |
|---|---|
| `src/datagen/` | Synthetic question-SQL pair generation via LangGraph. |
| `src/schemas/` | Pydantic source-of-truth for DDL and datasets. |
| `src/train/` | QLoRA fine-tuning logic for SageMaker Training Jobs. |
| `src/evaluate/` | Execution Accuracy evaluation in ephemeral SQLite environments. |
| `src/pipeline/` | SageMaker Pipeline DAG definitions and step orchestration. |
| `src/inference/` | Serving logic for SageMaker Endpoints and local testing. |
| `src/utils/` | S3 interactions, config loading, and common helpers. |

## Scripts Reference

| Script | Purpose |
|---|---|
| `scripts/setup/` | AWS role creation, ECR staging, and model asset downloads. |
| `scripts/run_pipeline.py` | Starts a full SageMaker Pipeline execution (Gen \u2192 Train \u2192 Eval). |
| `scripts/deploy_model.py` | Deploys a trained adapter to a SageMaker Real-Time Endpoint. |
| `scripts/inference/test_sagemaker_endpoint.py` | Smoke tests a running SageMaker endpoint with SQL prompts. |
| `scripts/run_local_inference.py` | Tests adapters locally without SageMaker infrastructure. |
| `scripts/datagen/run_datagen.py` | Standalone synthetic data generation for exploration. |

## Docker Container Images

- `docker/Dockerfile.evaluate`: Environment for execution-based SQL validation.
- `docker/Dockerfile.inference`: Optimized serving environment for SageMaker GPU instances.

## Pipeline Workflow

1. **Generation**: `src/datagen/` uses LLMs to produce `(question, sql, result)` triplets from `src/schemas/`.
2. **Fine-tuning**: `src/train/` applies QLoRA to a base model (e.g., Llama-3) on SageMaker.
3. **Evaluation**: `src/evaluate/` executes predicted SQL against test databases to calculate Execution Accuracy.
4. **Registration**: Validated models are pushed to the SageMaker Model Registry if they meet threshold metrics.
5. **Deployment**: `scripts/deploy_model.py` provisions production-ready endpoints.

## Project Status (TODO)

- **Pipeline**: QLoRA fine-tuning and S3 artifact storage are fully operational.
- **Evaluation**: Execution Accuracy is functional but requires expansion for complex SQL dialects.
- **Orchestration**: Automated SageMaker Endpoint creation based on evaluation thresholds is currently pending integration.

## Roadmap

- [ ] **Isolated Deployment Testing**: Validate `scripts/deploy_model.py` across different instance types.
- [ ] **Local Docker Inference**: Verify fine-tuned GGUF/adapter performance in local container environments.
- [ ] **Adaptive Retraining**: Implement automated triggers based on schema version increments.
- [ ] **Multi-dialect Support**: Support PostgreSQL and MySQL evaluation backends.


