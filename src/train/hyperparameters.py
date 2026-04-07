"""In-container fallback hyperparameters for QLoRA fine-tuning.

Used by ``train.py`` only when a key is absent from the SageMaker
hyperparameters JSON at ``/opt/ml/input/config/hyperparameters.json``.
Authoritative values live in ``config/pipeline.yaml`` and are injected
by ``pipeline/steps/training.py`` at pipeline definition time.
"""

hyperparameters = {
    "lora_r":           16,
    "lora_alpha":       32,
    "lora_dropout":     0.05,
    "epochs":           1,
    "batch_size":       4,
    "grad_accum_steps": 4,
    "learning_rate":    0.0002,
    "max_seq_length":   512,
}