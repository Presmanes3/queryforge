# Changelog

## [2026-04-08]

### Added
- Created `scripts/inference/test_sagemaker_endpoint.py` for endpoint smoke testing.
- Created `scripts/inference/benchmark_sagemaker.py` for throughput and latency analysis.

### Fixed
- Fixed Triton core dump on T4 by switching to `nvidia/cuda:12.1.1-devel` base image.
- Resolved `libcudart.so.12` missing error by mounting correct CUDA 12.1 runtime.
- Stabilized vLLM on `ml.g4dn.xlarge` via `TRITON_CACHE_DIR` and `gpu_memory_utilization=0.80`.
- Eliminated invalid kernel images by disabling `custom_all_reduce` for Turing architecture.
- Replaced non-existent `/opt/conda/bin/serve` with standard `/usr/local/bin/serve`.

