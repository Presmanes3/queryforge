"""Run local vLLM inference using Docker and target GPU."""

import argparse
import subprocess
import sys
import tarfile
from pathlib import Path

import boto3

# Prepend project root to path for local execution.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Local base model initialization prevents redundant S3 downloads.
LOCAL_BASE_DIR = PROJECT_ROOT / "models" / "Llama-3.2-1B-Instruct"

# Adapter weights reside under the 'adapters/' prefix.
ADAPTER_S3_URI = (
    "s3://presmanes-queryforge-bucket"
    "/queryforge/adapters/orders/v1"
    "/pipelines-4g16u3ydjdhh-QLoraFineTune-Dxqd0jMPM5"
    "/output/model.tar.gz"
)

LOCAL_ADAPTER_DIR = PROJECT_ROOT / "local_output" / "cache" / "adapter"

def download_and_extract_adapter(s3_uri: str, dest_dir: Path) -> None:
    """Download and extract the adapter tarball from S3.
    
    Args:
        s3_uri: Full S3 URI of the adapter archive.
        dest_dir: Target local directory for extraction.
    """
    bucket, key = s3_uri.replace("s3://", "").split("/", 1)
    tarball = dest_dir / "model.tar.gz"

    print(f"Downloading adapter from {s3_uri}...")
    boto3.client("s3").download_file(bucket, key, str(tarball))

    print(f"Extracting {tarball} into {dest_dir}...")
    with tarfile.open(tarball, "r:gz") as tf:
        tf.extractall(path=dest_dir)

    tarball.unlink()
    print("Adapter ready.")

def run_command(cmd: list[str]) -> None:
    """Execute a system command.
    
    Args:
        cmd: List of command arguments.
    """
    print(f"Executing: {' '.join(str(c) for c in cmd)}")
    subprocess.check_call([str(c) for c in cmd])

def main() -> None:
    """Execute local inference container."""
    parser = argparse.ArgumentParser(description="Run local vLLM inference using Docker and GPU.")
    parser.add_argument("--skip-download", action="store_true", help="Skip adapter download if already cached.")
    args = parser.parse_args()

    if not (LOCAL_BASE_DIR / "config.json").exists():
        print(f"ERROR: Base model not found in {LOCAL_BASE_DIR}")
        print("Ensure weights are placed in models/Llama-3.2-1B-Instruct/")
        sys.exit(1)
    print(f"Base model found in {LOCAL_BASE_DIR}")

    LOCAL_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    if not args.skip_download:
        download_and_extract_adapter(ADAPTER_S3_URI, LOCAL_ADAPTER_DIR)
    else:
        print(f"--skip-download active, using cached adapter at {LOCAL_ADAPTER_DIR}")

    print("\n--- Launching Docker container (GPU enabled) ---")
    docker_cmd = [
        "docker", "run", "--gpus", "all",
        "-it", "--rm",
        "-p", "8080:8080",
        "-v", f"{LOCAL_BASE_DIR}:/tmp/base_model",
        "-v", f"{LOCAL_ADAPTER_DIR}:/opt/ml/model",
        "-e", "BASE_MODEL_S3_URI=",
        "-e", "HF_HUB_OFFLINE=1",
        "queryforge-vllm-inference:latest",
    ]

    try:
        run_command(docker_cmd)
    except KeyboardInterrupt:
        print("\nStopping container...")

if __name__ == "__main__":
    main()
