"""Build and push the QueryForge evaluation Docker image to ECR.

Usage
-----
    # Build locally only (no push)
    python scripts/setup/build_ecr_image.py --action build

    # Push a locally built image to ECR (creates the repository if absent)
    python scripts/setup/build_ecr_image.py --action push

    # Build and push in one step (default action)
    python scripts/setup/build_ecr_image.py
    python scripts/setup/build_ecr_image.py --action build-and-push

After a successful push the script prints the full ECR URI. Copy that value
into config/pipeline.yaml under `processing_image_uri`.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.config import ConfigLoader

_PROJECT_ROOT  = Path(__file__).resolve().parents[2]
_DOCKERFILE    = _PROJECT_ROOT / "docker" / "Dockerfile.evaluate"
_ECR_REPO_NAME = "queryforge-processing"
_IMAGE_TAG     = "v1"


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    """Run a subprocess command, streaming output to stdout."""
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check)


def _get_account_id(config) -> str:
    """Return the AWS account ID associated with the configured credentials."""
    import boto3
    session = boto3.Session(
        region_name=config.aws_region,
        profile_name=config.aws_profile,
    )
    return session.client("sts").get_caller_identity()["Account"]


def _get_or_create_ecr_repo(config, account_id: str) -> str:
    """Return the ECR repository URI, creating the repository if it does not exist.

    Args:
        config: Loaded pipeline configuration.
        account_id: AWS account ID used to construct the ECR URI.

    Returns:
        Full ECR repository URI.
    """
    import boto3
    from botocore.exceptions import ClientError

    session = boto3.Session(
        region_name=config.aws_region,
        profile_name=config.aws_profile,
    )
    ecr = session.client("ecr", region_name=config.aws_region)

    try:
        resp = ecr.describe_repositories(repositoryNames=[_ECR_REPO_NAME])
        uri = resp["repositories"][0]["repositoryUri"]
        print(f"  ECR repository already exists: {uri}")
    except ClientError as e:
        if e.response["Error"]["Code"] != "RepositoryNotFoundException":
            raise
        print(f"  Creating ECR repository: {_ECR_REPO_NAME}")
        resp = ecr.create_repository(
            repositoryName=_ECR_REPO_NAME,
            imageScanningConfiguration={"scanOnPush": True},
            # Immutable tags prevent silent overwrites of a tagged image in production.
            imageTagMutability="MUTABLE",
        )
        uri = resp["repository"]["repositoryUri"]
        print(f"  Created: {uri}")

    return uri


def _ecr_login(config, account_id: str) -> None:
    """Authenticate the local Docker daemon with ECR."""
    import boto3

    session = boto3.Session(
        region_name=config.aws_region,
        profile_name=config.aws_profile,
    )
    ecr = session.client("ecr", region_name=config.aws_region)
    token = ecr.get_authorization_token()["authorizationData"][0]["authorizationToken"]

    import base64
    username, password = base64.b64decode(token).decode().split(":", 1)
    registry = f"{account_id}.dkr.ecr.{config.aws_region}.amazonaws.com"

    # Feed the password via stdin to avoid it appearing in process listings.
    subprocess.run(
        ["docker", "login", "--username", username, "--password-stdin", registry],
        input=password.encode(),
        check=True,
        capture_output=True,
    )
    print("  Docker authenticated with ECR.")


def _build(local_tag: str) -> None:
    """Build the evaluation image from Dockerfile.evaluate.

    Args:
        local_tag: Local image tag to assign (e.g. queryforge-processing:v1).
    """
    print(f"\n[1/1] Building image: {local_tag}")
    _run([
        "docker", "build",
        "--file", str(_DOCKERFILE),
        "--tag", local_tag,
        str(_PROJECT_ROOT),
    ])
    print(f"\nBuild complete: {local_tag}")


def _push(config, local_tag: str, ecr_uri: str) -> str:
    """Tag the local image and push it to ECR.

    Args:
        config: Loaded pipeline configuration.
        local_tag: Local image tag already built.
        ecr_uri: ECR repository URI (without tag).

    Returns:
        Full ECR image URI including tag.
    """
    full_uri = f"{ecr_uri}:{_IMAGE_TAG}"
    latest_uri = f"{ecr_uri}:latest"

    print(f"\n[1/3] Tagging {local_tag} → {full_uri}")
    _run(["docker", "tag", local_tag, full_uri])

    print(f"\n[2/3] Tagging {local_tag} → {latest_uri}")
    _run(["docker", "tag", local_tag, latest_uri])

    print(f"\n[3/3] Pushing {full_uri}")
    _run(["docker", "push", full_uri])
    _run(["docker", "push", latest_uri])

    return full_uri


def main() -> None:
    """Parse arguments and execute the requested build/push action."""
    parser = argparse.ArgumentParser(
        description="Build and push the QueryForge evaluation image to ECR."
    )
    parser.add_argument(
        "--action",
        choices=["build", "push", "build-and-push"],
        default="build-and-push",
        help="Action to perform (default: build-and-push).",
    )
    parser.add_argument(
        "--config",
        default="config/pipeline.yaml",
        help="Path to pipeline.yaml (default: config/pipeline.yaml).",
    )
    args = parser.parse_args()

    config     = ConfigLoader.load(args.config)
    account_id = _get_account_id(config)
    local_tag  = f"{_ECR_REPO_NAME}:{_IMAGE_TAG}"
    ecr_uri    = f"{account_id}.dkr.ecr.{config.aws_region}.amazonaws.com/{_ECR_REPO_NAME}"

    print("=== QueryForge ECR Image Builder ===")
    print(f"  account  : {account_id}")
    print(f"  region   : {config.aws_region}")
    print(f"  profile  : {config.aws_profile or 'default'}")
    print(f"  local tag: {local_tag}")
    print(f"  ecr uri  : {ecr_uri}:{_IMAGE_TAG}")
    print("=====================================\n")

    if args.action in ("build", "build-and-push"):
        _build(local_tag)

    if args.action in ("push", "build-and-push"):
        ecr_repo_uri = _get_or_create_ecr_repo(config, account_id)
        _ecr_login(config, account_id)
        full_uri = _push(config, local_tag, ecr_repo_uri)

        print("\n=== Done ===")
        print(f"Image pushed: {full_uri}")
        print("\nCopy the following value into config/pipeline.yaml:")
        print(f"  processing_image_uri: \"{full_uri}\"")
        print("============\n")


if __name__ == "__main__":
    main()
