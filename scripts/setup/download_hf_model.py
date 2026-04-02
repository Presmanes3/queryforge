"""QueryForge Model Downloader.

Downloads a base model from Hugging Face Hub and uploads it to the project's
S3 bucket for use in SageMaker training jobs.
"""

from __future__ import annotations

import argparse
import sys

from huggingface_hub import snapshot_download


def main():
    """Execute the model download and S3 upload sequence."""
    parser = argparse.ArgumentParser(
        description="Download a model from Hugging Face and upload to S3."
    )
    parser.add_argument(
        "--model-id",
        default="unsloth/Llama-3.2-1B-Instruct",
        help="Hugging Face model ID (default: Llama-3.2-1B-Instruct)."
    )
    parser.add_argument(
        "--local-dir",
        default=None,
        help="Local directory to store the downloaded model. Defaults to models/<model_name>."
    )
    args = parser.parse_args()

    # Determine paths and names
    model_name = args.model_id.split("/")[-1]
    local_dir = args.local_dir or f"models/{model_name}"

    print(f"Downloading model '{args.model_id}' into '{local_dir}'...")
    try:
        local_model_path = snapshot_download(
            repo_id=args.model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            revision="main"
        )
        print(f"\nSuccessfully downloaded model to: {local_model_path}")
    except Exception as e:
        print(f"Failed to download model from Hugging Face: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
