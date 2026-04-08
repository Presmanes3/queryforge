#!/usr/bin/env python3
"""Smoke-test the inference Docker image locally before pushing to ECR.

This script:
1. Builds the Docker image locally.
2. Starts the container with GPU support (if available) and local port mapping.
3. Mounts a local adapter directory to /opt/ml/model (simulating SageMaker).
4. Sets BASE_MODEL_S3_URI to allow the container to download the base model, 
   or uses a local mount if preferred.

Usage
-----
    python scripts/inference/test_inference_container.py \\
        --adapter-path D:\\Projects\\queryforge\\local_adapter \\
        --base-model-uri s3://your-bucket/path/to/base-model

    # After starting, test with:
    # curl -X POST http://localhost:8080/invocations -H "Content-Type: application/json" -d '{"inputs": "Tell me about orders"}'
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DOCKERFILE = _PROJECT_ROOT / "docker" / "Dockerfile.inference"
_IMAGE_NAME = "queryforge-vllm-inference:test"

def build_image():
    print(f"--- Building Docker image: {_IMAGE_NAME} ---")
    cmd = [
        "docker", "build",
        "-t", _IMAGE_NAME,
        "-f", str(_DOCKERFILE),
        "."
    ]
    # Run from project root
    result = subprocess.run(cmd, cwd=str(_PROJECT_ROOT))
    if result.returncode != 0:
        print("Error: Docker build failed.")
        sys.exit(1)

def run_container(adapter_path, base_model_uri, gpu=True):
    print(f"--- Starting container: {_IMAGE_NAME} ---")
    
    # Resolve absolute path for mount
    abs_adapter_path = os.path.abspath(adapter_path)
    
    cmd = [
        "docker", "run", "--rm",
        "-p", "8080:8080",
        "-e", f"BASE_MODEL_S3_URI={base_model_uri}",
        "-e", "SAGEMAKER_MODEL_DIR=/opt/ml/model",
        "-v", f"{abs_adapter_path}:/opt/ml/model",
        # Forward AWS credentials from environment for S3 download
        "-e", "AWS_ACCESS_KEY_ID",
        "-e", "AWS_SECRET_ACCESS_KEY",
        "-e", "AWS_SESSION_TOKEN",
        "-e", "AWS_REGION",
        "-e", "AWS_DEFAULT_REGION",
    ]

    if gpu:
        cmd.extend(["--gpus", "all"])
    
    cmd.append(_IMAGE_NAME)
    
    print(f"Running command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nStopping container...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test inference container locally.")
    parser.add_argument("--adapter-path", required=True, help="Local path to LoRA adapter files (adapter_config.json, etc.)")
    parser.add_argument("--base-model-uri", required=True, help="S3 URI for the base model weights")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU support (will likely fail vLLM but useful for testing imports)")
    parser.add_argument("--no-build", action="store_true", help="Skip docker build")

    args = parser.parse_args()

    if not args.no_build:
        build_image()
    
    # Note: vLLM absolutely requires a GPU. 
    # If the user doesn't have a local GPU with nvidia-container-toolkit, 
    # this will fail at runtime, but we can at least check the imports/startup.
    run_container(args.adapter_path, args.base_model_uri, gpu=not args.no_gpu)
