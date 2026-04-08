"""Deploy the fine-tuned adapter and base model to a SageMaker Endpoint."""

import argparse
import time

from sagemaker.core.resources import Endpoint, EndpointConfig, Model
from src.config import config

def main() -> None:
    """Provision a SageMaker Endpoint for the specified model resources."""
    parser = argparse.ArgumentParser(description="Deploy a fine-tuned adapter to SageMaker.")
    parser.add_argument(
        "--adapter-uri",
        help="S3 URI of the model.tar.gz adapter (e.g., s3://bucket/path/to/model.tar.gz)",
    )
    parser.add_argument(
        "--base-model-uri",
        default=f"s3://{config.s3_bucket}/{config.s3_prefix}/models/Llama-3.2-1B-Instruct/v1/",
        help="S3 URI to the base model weights folder.",
    )
    args = parser.parse_args()

    if not args.adapter_uri:
        print("Error: --adapter-uri is required if not specified in config.")
        return

    timestamp = int(time.time())

    print("--- Deploying Model via SageMaker Core ---")

    print(f"1. Creating Model resource in AWS (using {config.inference.image_uri})...")
    # Map the model artifact to the container for SageMaker automatic extraction.
    model = Model.create(
        model_name=f"queryforge-model-vllm-{timestamp}",
        execution_role_arn=config.execution_role_arn,
        primary_container={
            "image": config.inference.image_uri,
            "model_data_url": args.adapter_uri,
            "environment": {
                "BASE_MODEL_S3_URI": args.base_model_uri,
                "HF_HUB_OFFLINE": "1",
            }
        }
    )
    print(f"Model created: {model.model_name}")

    print("2. Creating Endpoint Config...")
    # Require GPU instance class to support vLLM execution.
    endpoint_config = EndpointConfig.create(
        endpoint_config_name=f"queryforge-config-{timestamp}",
        production_variants=[
            {
                "variant_name": "AllTraffic",
                "model_name": model.model_name,
                "instance_type": config.inference.instance_type,
                "initial_instance_count": config.inference.initial_instance_count,
            }
        ]
    )
    print(f"Endpoint Config created: {endpoint_config.endpoint_config_name}")

    print(f"3. Launching Endpoint instance on {config.inference.instance_type} (ETA: ~8 mins)...")
    endpoint = Endpoint.create(
        endpoint_name=f"queryforge-endpoint-vllm-{timestamp}",
        endpoint_config_name=endpoint_config.endpoint_config_name
    )

    print("\n=============================")
    print("✓ Endpoint deployment initiated.")
    print(f"   Name: {endpoint.endpoint_name}")
    print("=============================")
    print("\nMonitor progress in the AWS SageMaker Console.")

if __name__ == "__main__":
    main()
