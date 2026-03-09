# queryforge
QueryForge is an open-source framework for building schema-aware Text-to-SQL assistants. Starting from structured data models, it generates synthetic question-SQL pairs, fine-tunes lightweight LLMs on AWS SageMaker, evaluates model quality, and packages optimized versions for fast local inference with Ollama.

## Setup

1. Copy the example configuration:
   ```bash
   cp config/pipeline.yaml.example config/pipeline.yaml
   ```
2. Edit `config/pipeline.yaml` with your actual AWS resource identifiers.

