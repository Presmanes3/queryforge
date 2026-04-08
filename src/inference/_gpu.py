"""GPU capability detection for vLLM dtype selection."""

import logging

logger = logging.getLogger(__name__)

# bfloat16 requires CUDA compute capability >= 8.0 (Ampere+).
_BFLOAT16_MIN_CAPABILITY = 8.0


def get_vllm_dtype() -> str:
    """Return the optimal vLLM dtype string for the current GPU.

    Returns ``"bfloat16"`` on Ampere-generation GPUs (compute capability ≥ 8.0)
    and ``"float16"`` on older architectures such as Turing (T4, compute
    capability 7.5) or Volta. Falls back to ``"float16"`` when CUDA is not
    available or the capability cannot be determined.
    """
    try:
        import torch  # noqa: PLC0415 — heavy optional dep, loaded only here

        if not torch.cuda.is_available():
            logger.warning("CUDA not available — defaulting to float16.")
            return "float16"

        major, minor = torch.cuda.get_device_capability()
        capability = major + minor / 10.0
        device_name = torch.cuda.get_device_name()

        if capability >= _BFLOAT16_MIN_CAPABILITY:
            logger.info(
                "GPU '%s' (capability %.1f) supports bfloat16.",
                device_name,
                capability,
            )
            return "bfloat16"

        logger.warning(
            "GPU '%s' (capability %.1f) does not support bfloat16 — using float16.",
            device_name,
            capability,
        )
        return "float16"

    except Exception as exc:  # pragma: no cover
        logger.warning("Could not determine GPU capability (%s) — defaulting to float16.", exc)
        return "float16"
