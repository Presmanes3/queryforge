"""Entry point for synthetic dataset generation from a Pydantic schema file."""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import sys
from pathlib import Path

from pydantic import BaseModel

from queryforge.datagen.generator import DatasetGenerator


def _load_schema_class(schema_file: str, class_name: str | None) -> type[BaseModel]:
    """Import a Pydantic BaseModel subclass from an arbitrary .py file.

    Args:
        schema_file: Path to the Python file containing the schema class.
        class_name: Name of the class to load. If None, the first BaseModel
            subclass found in the file is used.

    Returns:
        The uninstantiated schema class.

    Raises:
        ValueError: When no matching class is found.
    """
    path = Path(schema_file).resolve()
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load module from {schema_file}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]

    if class_name:
        cls = getattr(module, class_name, None)
        if cls is None:
            raise ValueError(f"Class '{class_name}' not found in {schema_file}")
        return cls

    # Discover the first BaseModel subclass that is not BaseModel itself.
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, BaseModel) and obj is not BaseModel and obj.__module__ == module.__name__:
            return obj

    raise ValueError(f"No Pydantic BaseModel subclass found in {schema_file}")


def main() -> None:
    """Generate a JSONL training dataset from a Pydantic schema file."""
    parser = argparse.ArgumentParser(
        description="Generate a JSONL training dataset from a Pydantic schema file."
    )
    parser.add_argument(
        "--schema-file",
        required=True,
        help="Path to the .py file containing the Pydantic schema class.",
    )
    parser.add_argument(
        "--schema-class",
        default=None,
        help="Name of the schema class. Defaults to the first BaseModel subclass found.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=500,
        help="Number of question-SQL pairs to generate. Default: 500.",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets",
        help="Directory where the output JSONL file is written. Default: datasets/.",
    )
    args = parser.parse_args()

    try:
        schema_cls = _load_schema_class(args.schema_file, args.schema_class)
    except (ValueError, ImportError) as exc:
        print(f"Error loading schema: {exc}", file=sys.stderr)
        sys.exit(1)

    result = DatasetGenerator().run(
        schema_cls=schema_cls,
        n_samples=args.n_samples,
        output_dir=args.output_dir,
    )

    print(f"Written {result.n_written} samples to {result.output_path}")


if __name__ == "__main__":
    main()
