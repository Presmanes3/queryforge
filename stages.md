# Maturity Roadmap: QueryForge

## Final Vision
Continuous retraining platform for SQL generation models (Text-to-SQL) with the following capabilities:
- Pydantic schema registration.
- Automatic DDL translation.
- Synthetic dataset generation.
- QLoRA fine-tuning on Amazon SageMaker.
- Functional evaluation (Execution Accuracy).
- Model registration and versioning in SageMaker Model Registry.
- GGUF quantization.
- Local deployment with Ollama.
- Promotion logic based on defined metric thresholds.

## Phase 0: Problem Design
Definition of scope and system contracts.
- Pydantic models for input structure.
- DDL derivation from schemas.
- JSONL format definition for training.
- Establishment of success metrics: Execution Accuracy over executed SQL rather than textual similarity.

## Phase 1: Manual Fine-tuning PoC
Implementation of the core training workflow.
- Offline synthetic generation for a single schema.
- Configuration of SageMaker Training Jobs.
- Hugging Face PEFT and QLoRA core implementation.
- Artifact storage in Amazon S3.

## Phase 2: Reproducible Pipeline
Workflow automation and orchestration.
- SageMaker Pipelines implementation for chaining processing, training, and evaluation.
- Integration with SageMaker Model Registry.
- Conditional logic for model registration and promotion decisions.

## Phase 3: Retraining Platform
Lifecycle management and continual learning.
- Input of new Pydantic models and dataset versioning.
- Implementation of Continual Learning strategies (accumulated dataset or replay).
- Automation of retraining triggers.
- Monitoring for model degradation.

## Phase 4: Local Serving and Quantization
Optimization for local inference.
- Fusion of LoRA adapters with the base model.
- Weight conversion to GGUF format.
- Variant publication for Ollama.
- High-performance local inference with quantized models.

