"""Single source of truth for QLoRA hyperparameter keys and in-container defaults.

This module is the only place where the hyperparameter dict contract is defined.
It is intentionally free of SageMaker SDK imports so it can be loaded inside the
training container (which packages only ``src/queryforge/train/``) just as easily
as by external builders that construct training jobs.

External orchestration code must use :func:`build_hyperparameters` with a
``TrainConfig`` instance to guarantee that Pydantic-validated values are passed
to the job — never build the dict by hand.

Train-time code (:mod:`train`) must use the ``DEFAULTS`` constant for fallback
values, which are kept in lockstep with the field defaults in
``shared.schemas.config.TrainConfig``.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# In-container defaults
# ---------------------------------------------------------------------------

# Mirrors the field defaults in shared.schemas.config.TrainConfig.
# When adding a new hyperparameter: update DEFAULTS here first, then add the
# corresponding Field to TrainConfig, then update build_hyperparameters.
DEFAULTS: dict[str, int | float] = {
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "epochs": 1,
    "batch_size": 4,
    "grad_accum_steps": 4,
    "learning_rate": 2e-4,
    "max_seq_length": 512,
}


# ---------------------------------------------------------------------------
# External builder helper
# ---------------------------------------------------------------------------

def build_hyperparameters(train_config) -> dict[str, int | float]:
    """Extract a JSON-serializable hyperparameter dict from a ``TrainConfig``.

    Used by :class:`~queryforge.train.estimator.TrainingJobBuilder` and
    ``scripts/run_finetuning.py`` to guarantee that keys and values precisely
    match the Pydantic SSoT rather than being hardcoded inline.

    Args:
        train_config: A ``shared.schemas.config.TrainConfig`` instance.

    Returns:
        Dict mapping each hyperparameter name to its configured value.
    """
    return {key: getattr(train_config, key) for key in DEFAULTS}
