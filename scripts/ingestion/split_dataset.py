"""Utility to split a synthetic dataset into training and test sets.

Takes a JSONL file from the datasets directory and produces two new files:
- <schema>_<version>_train.jsonl (90% - to be split further by SFTTrainer into train/val)
- <schema>_<version>_test.jsonl (10% - for final evaluation)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def split_dataset(input_file: str, train_ratio: float = 0.9) -> None:
    """Split a JSONL dataset into training and test files.

    Args:
        input_file: Path to the source .jsonl file.
        train_ratio: Fraction of samples to assign to the training set.
    """
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Error: File {input_file} not found.", file=sys.stderr)
        sys.exit(1)

    # Read all samples
    with open(input_path, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    if not samples:
        print(f"Error: {input_file} is empty.", file=sys.stderr)
        sys.exit(1)

    # Shuffle to ensure temporal/pattern diversity in both splits
    random.shuffle(samples)

    split_idx = int(len(samples) * train_ratio)
    train_samples = samples[:split_idx]
    test_samples = samples[split_idx:]

    # Define output paths
    base_name = input_path.stem
    output_dir = input_path.parent
    train_path = output_dir / f"{base_name}_train.jsonl"
    test_path = output_dir / f"{base_name}_test.jsonl"

    # Write training split
    with open(train_path, "w", encoding="utf-8") as f:
        for sample in train_samples:
            f.write(json.dumps(sample) + "\n")

    # Write test split
    with open(test_path, "w", encoding="utf-8") as f:
        for sample in test_samples:
            f.write(json.dumps(sample) + "\n")

    print(f"Split completed for {input_path.name}:")
    print(f"  - Training samples: {len(train_samples)} -> {train_path.name}")
    print(f"  - Test samples:     {len(test_samples)} -> {test_path.name}")


def main() -> None:
    """Run the split script from command line."""
    parser = argparse.ArgumentParser(
        description="Split a JSONL dataset into training and test sets."
    )
    parser.add_argument(
        "--input-schema",
        required=True,
        help="Name of the jsonl file in the datasets directory (e.g., orders_v1.jsonl).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Ratio of training samples (default: 0.9).",
    )
    args = parser.parse_args()

    # Resolve path relative to datasets/
    datasets_dir = Path("datasets")
    input_path = datasets_dir / args.input_schema

    split_dataset(str(input_path), args.train_ratio)


if __name__ == "__main__":
    main()
