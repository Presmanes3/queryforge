"""I/O helpers: loading samples and writing evaluation outputs."""

from __future__ import annotations

import json
import os
from glob import glob

from _types import Sample, Metrics
from _sql import normalize_sql, is_valid_sql


def load_samples(dataset_channel: str) -> list[Sample]:
    """Load Sample records from all JSONL files found under *dataset_channel*.

    Accepts either a directory path (scanned recursively) or a direct path to
    a single JSONL file.
    """
    samples: list[Sample] = []
    if os.path.isfile(dataset_channel):
        paths = [dataset_channel]
    else:
        paths = sorted(glob(os.path.join(dataset_channel, "**", "*.jsonl"), recursive=True))
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(Sample.from_dict(json.loads(line)))
    return samples


def compute_metrics(samples: list[Sample], predictions: list[str]) -> Metrics:
    """Compute execution accuracy and exact match over all predictions."""
    if not samples:
        return Metrics(0.0, 0.0, 0.0, 0, "", "")
    exact = sum(
        1 for sample, pred in zip(samples, predictions)
        if normalize_sql(pred) == normalize_sql(sample.sql)
    )
    n = len(samples)
    acc = exact / n
    return Metrics(
        execution_accuracy=acc,
        exact_match=acc,
        invalid_query_rate=0.0,
        n_samples=n,
        schema_name=samples[0].schema_name,
        schema_version=samples[0].schema_version,
    )


def write_metrics(metrics: Metrics, output_dir: str) -> None:
    """Serialize *metrics* to metrics.json inside *output_dir*."""
    path = os.path.join(output_dir, "metrics.json")
    with open(path, "w") as f:
        json.dump(
            {
                "execution_accuracy": metrics.execution_accuracy,
                "exact_match": metrics.exact_match,
                "invalid_query_rate": metrics.invalid_query_rate,
                "n_samples": metrics.n_samples,
                "schema_name": metrics.schema_name,
                "schema_version": metrics.schema_version,
            },
            f,
            indent=2,
        )


def write_predictions(samples: list[Sample], predictions: list[str], output_dir: str) -> None:
    """Write per-sample predictions to predictions.jsonl inside *output_dir*."""
    path = os.path.join(output_dir, "predictions.jsonl")
    with open(path, "w") as f:
        for sample, pred in zip(samples, predictions):
            match = normalize_sql(pred) == normalize_sql(sample.sql)
            f.write(json.dumps({
                "expected": sample.sql,
                "predicted": pred,
                "match": match,
                "valid": is_valid_sql(sample.ddl, pred),
            }) + "\n")
