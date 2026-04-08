"""Build and push the local inference Docker image to Amazon ECR."""

import argparse
import subprocess
import sys
from pathlib import Path

import boto3

# Prepend project root to path for local execution.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Assuming config usage for AWS region and other parameters
from src.utils.config import ConfigLoader

def run_cmd(cmd: str) -> None:
    """Execute a shell command.
    
    Args:
        cmd: Shell command string.
    """
    print(f"Executing: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def main() -> None:
    """Build the Docker image and push it to the configured ECR repository."""
    try:
        config = ConfigLoader.load("config/pipeline.yaml")
        region = config.aws_region
    except Exception:
        region = "us-east-1"
        
    try:
        account_id = boto3.client("sts").get_caller_identity()["Account"]
    except Exception as e:
        print(f"Error fetching AWS credentials: {e}")
        sys.exit(1)

    repo_name = "queryforge-vllm-inference"
    image_tag = "latest"
    ecr_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{repo_name}:{image_tag}"

    # Authenticate local Docker daemon with the account ECR registry.
    login_cmd = f"aws ecr get-login-password --region {region} | docker login --username AWS --password-stdin {account_id}.dkr.ecr.{region}.amazonaws.com"
    run_cmd(login_cmd)

    # Authenticate with Public ECR to pull the base Deep Learning Container image.
    public_login_cmd = "aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com"
    run_cmd(public_login_cmd)

    ecr = boto3.client("ecr", region_name=region)
    try:
        ecr.create_repository(repositoryName=repo_name)
        print(f"Created ECR repository: {repo_name}")
    except ecr.exceptions.RepositoryAlreadyExistsException:
        print(f"ECR repository {repo_name} already exists.")

    print("\nBuilding local image...")
    run_cmd(f"docker build --provenance=false -t {repo_name} -f docker/Dockerfile.inference .")
    run_cmd(f"docker tag {repo_name}:latest {ecr_uri}")
    
    print("\nPushing to Amazon ECR...")
    run_cmd(f"docker push {ecr_uri}")
    print(f"\nImage successfully pushed to ECR: {ecr_uri}")

if __name__ == "__main__":
    main()
