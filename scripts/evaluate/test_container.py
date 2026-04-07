"""Smoke-test the evaluation Docker image locally before pushing to ECR.

Mounts the local model, adapter, dataset, and output directories into the
container at the same paths SageMaker uses, then verifies that evaluate.py
runs to completion and writes metrics.json.

Usage
-----
    # Test the locally built image (default)
    python scripts/evaluate/test_container.py \\
        --model-name Llama-3.2-1B-Instruct \\
        --adapter-path D:\\Projects\\queryforge\\local_adapter \\
        --dataset-path D:\\Projects\\queryforge\\datasets\\orders_v1_test.jsonl

    # Test with GPU passthrough
    python scripts/evaluate/test_container.py --gpu \\
        --model-name Llama-3.2-1B-Instruct \\
        --adapter-path D:\\Projects\\queryforge\\local_adapter \\
        --dataset-path D:\\Projects\\queryforge\\datasets\\orders_v1_test.jsonl

    # Test a specific image (e.g. the ECR URI after push)
    python scripts/evaluate/test_container.py \\
        --image 619071308424.dkr.ecr.us-east-1.amazonaws.com/queryforge-processing:v1 \\
        --model-name Llama-3.2-1B-Instruct \\
        --adapter-path D:\\Projects\\queryforge\\local_adapter \\
        --dataset-path D:\\Projects\\queryforge\\datasets\\orders_v1_test.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_PROJECT_ROOT  = Path(__file__).resolve().parents[2]
_EVAL_SCRIPT   = _PROJECT_ROOT / "src" / "evaluate" / "evaluate.py"
_DEFAULT_IMAGE = "queryforge-processing:v1"

# SageMaker channel paths — must match the ProcessingInput local_path values.
_CONTAINER_MODEL   = "/opt/ml/processing/input/model"
_CONTAINER_ADAPTER = "/opt/ml/processing/input/adapter"
_CONTAINER_DATASET = "/opt/ml/processing/input/dataset"
_CONTAINER_OUTPUT  = "/opt/ml/processing/output"
_CONTAINER_CODE    = "/opt/ml/processing/input/code"


def _check_docker() -> None:
    """Verify that Docker is installed and the daemon is running."""
    if shutil.which("docker") is None:
        print("Error: 'docker' not found on PATH. Install Docker Desktop and try again.")
        sys.exit(1)
    result = subprocess.run(["docker", "info"], capture_output=True)
    if result.returncode != 0:
        print("Error: Docker daemon is not running. Start Docker Desktop and try again.")
        sys.exit(1)


def _check_gpu_support() -> bool:
    """Return True if the Docker runtime supports --gpus (nvidia-container-toolkit)."""
    result = subprocess.run(
        ["docker", "run", "--rm", "--gpus", "all", "ubuntu:22.04", "echo", "gpu-ok"],
        capture_output=True,
    )
    return result.returncode == 0


def _resolve_dataset_path(dataset_path: str) -> tuple[str, str]:
    """Return (host_dir, container_dataset_path) for a dataset file or directory.

    SageMaker always mounts a directory. When the user points to a single JSONL
    file the parent directory is mounted and the container path points to the file.

    Args:
        dataset_path: Local path to a JSONL file or a directory containing JSONL files.

    Returns:
        Tuple of (host_directory_to_mount, container_path_for_SM_CHANNEL_DATASET).
    """
    p = Path(dataset_path).resolve()
    if p.is_file():
        return str(p.parent), f"{_CONTAINER_DATASET}/{p.name}"
    return str(p), _CONTAINER_DATASET


def _build_docker_cmd(
    image: str,
    model_path: str,
    adapter_path: str,
    dataset_path: str,
    output_path: str,
    gpu: bool,
) -> list[str]:
    """Assemble the docker run command for the smoke test.

    Args:
        image: Docker image tag or URI to test.
        model_path: Local path to the base model directory.
        adapter_path: Local path to the PEFT adapter directory.
        dataset_path: Local path to the dataset file or directory.
        output_path: Local path where output files will be written.
        gpu: Whether to enable GPU passthrough via --gpus all.

    Returns:
        docker run command as a list of strings.
    """
    host_dataset_dir, container_dataset_path = _resolve_dataset_path(dataset_path)

    cmd = ["docker", "run", "--rm"]

    if gpu:
        cmd += ["--gpus", "all"]

    cmd += [
        # Volume mounts: host path → container path
        "-v", f"{Path(model_path).resolve()}:{_CONTAINER_MODEL}:ro",
        "-v", f"{Path(adapter_path).resolve()}:{_CONTAINER_ADAPTER}:ro",
        "-v", f"{host_dataset_dir}:{_CONTAINER_DATASET}:ro",
        "-v", f"{Path(output_path).resolve()}:{_CONTAINER_OUTPUT}",
        # evaluate.py injection — replicates what ScriptProcessor does via code=
        "-v", f"{_EVAL_SCRIPT.parent}:{_CONTAINER_CODE}:ro",
        # Environment variables
        "-e", f"SM_CHANNEL_MODEL={_CONTAINER_MODEL}",
        "-e", f"SM_CHANNEL_ADAPTER={_CONTAINER_ADAPTER}",
        "-e", f"SM_CHANNEL_DATASET={container_dataset_path}",
        "-e", f"SM_OUTPUT_DIR={_CONTAINER_OUTPUT}",
        # Skip NF4 by default; the caller can override via --quantize
        "-e", "SM_QUANTIZE=0",
        image,
        "python3", f"{_CONTAINER_CODE}/evaluate.py",
    ]

    return cmd


def main() -> None:
    """Parse arguments and run the container smoke test."""
    parser = argparse.ArgumentParser(
        description="Smoke-test the QueryForge evaluation Docker image locally."
    )
    parser.add_argument(
        "--image",
        default=_DEFAULT_IMAGE,
        help=f"Docker image to test (default: {_DEFAULT_IMAGE}).",
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Base model folder name under models/ (e.g. Llama-3.2-1B-Instruct).",
    )
    parser.add_argument(
        "--adapter-path",
        required=True,
        help="Local path to the PEFT adapter directory.",
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Local path to the test JSONL file or directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(_PROJECT_ROOT / "eval_output"),
        help="Local directory for output files (default: eval_output/).",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        default=False,
        help="Enable GPU passthrough via --gpus all (requires nvidia-container-toolkit).",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        default=False,
        help="Use NF4 4-bit quantization (SM_QUANTIZE=1). Default is bfloat16.",
    )
    args = parser.parse_args()

    model_path  = str(_PROJECT_ROOT / "models" / args.model_name)
    output_path = args.output_dir

    # --- Pre-flight checks ---
    _check_docker()

    if not Path(model_path).is_dir():
        print(f"Error: model directory not found: {model_path}")
        sys.exit(1)
    if not Path(args.adapter_path).exists():
        print(f"Error: adapter path not found: {args.adapter_path}")
        sys.exit(1)
    if not Path(args.dataset_path).exists():
        print(f"Error: dataset path not found: {args.dataset_path}")
        sys.exit(1)

    if args.gpu:
        if not _check_gpu_support():
            print(
                "Warning: --gpu requested but Docker GPU support is unavailable.\n"
                "  To enable GPU passthrough on Windows:\n"
                "    1. Install WSL2: wsl --install\n"
                "    2. Install nvidia-container-toolkit inside WSL2:\n"
                "       https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html\n"
                "  Falling back to CPU (SM_QUANTIZE=0)."
            )
            args.gpu = False

    Path(output_path).mkdir(parents=True, exist_ok=True)

    cmd = _build_docker_cmd(
        image        = args.image,
        model_path   = model_path,
        adapter_path = args.adapter_path,
        dataset_path = args.dataset_path,
        output_path  = output_path,
        gpu          = args.gpu,
    )

    # Override SM_QUANTIZE if the user explicitly wants quantization.
    if args.quantize:
        idx = cmd.index("SM_QUANTIZE=0")
        cmd[idx] = "SM_QUANTIZE=1"

    print("=== QueryForge Container Smoke Test ===")
    print(f"  image    : {args.image}")
    print(f"  model    : {model_path}")
    print(f"  adapter  : {args.adapter_path}")
    print(f"  dataset  : {args.dataset_path}")
    print(f"  output   : {output_path}")
    print(f"  gpu      : {args.gpu}")
    print(f"  quantize : {args.quantize}")
    print("========================================\n")

    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\nContainer exited with code {result.returncode}. Test FAILED.")
        sys.exit(result.returncode)

    # Verify that metrics.json was written — the primary success signal.
    metrics_path = Path(output_path) / "metrics.json"
    if not metrics_path.exists():
        print(f"\nmetrics.json not found at {metrics_path}. Test FAILED.")
        sys.exit(1)

    with open(metrics_path) as f:
        metrics = json.load(f)

    print("\n=== Smoke Test PASSED ===")
    print(f"  Accuracy : {metrics.get('execution_accuracy', 'n/a')}")
    print(f"  Samples  : {metrics.get('n_samples', 'n/a')}")
    print(f"  Output   : {output_path}")
    print("=========================\n")


if __name__ == "__main__":
    main()
