"""SageMaker Training Job entry point for QLoRA fine-tuning.

Reads all configuration from the SageMaker hyperparameters JSON file.
Expects two data channels: 'model' (base model weights) and 'training' (JSONL dataset).
Writes the LoRA adapter to the SageMaker model output directory.

MLflow integration
------------------
When ``MLFLOW_TRACKING_URI`` is present in the environment this entry point
creates a single MLflow run and logs all hyperparameters, step-level metrics,
and job metadata.  The launcher injects ``MLFLOW_RUN_NAME``,
``MLFLOW_TRACKING_URI``, and ``MLFLOW_EXPERIMENT_NAME``; no run is created
outside the container.
"""

from __future__ import annotations
import glob
import json
import os

import mlflow
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

# hyperparameters.py is co-located in the same train/ directory and is packaged
# into the container alongside this entry point.  It has no SageMaker SDK
# dependency so it is safe to import here.
from hyperparameters import hyperparameters

_HP_PATH = "/opt/ml/input/config/hyperparameters.json"
_MODEL_CHANNEL = "/opt/ml/input/data/model"
_TRAINING_CHANNEL = "/opt/ml/input/data/training"
_OUTPUT_DIR = "/opt/ml/model"


def _get_bnb_config() -> BitsAndBytesConfig:
    """Return a BitsAndBytesConfig for NF4 double quantization."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def _get_lora_config(r: int, alpha: int, dropout: float) -> LoraConfig:
    """Return a LoraConfig targeting attention and feed-forward projections.

    Args:
        r: LoRA rank.
        alpha: LoRA scaling factor.
        dropout: Dropout applied to the LoRA layers.
    """
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )


def main() -> None:
    """Execute QLoRA fine-tuning inside a SageMaker Training Job container."""
    with open(_HP_PATH) as f:
        hp = json.load(f)

    print("\n=== Hyperparameters ===")
    for key, default in hyperparameters.items():
        if key in hp:
            print(f"  {key:<20}: {hp[key]}  (custom)")
        else:
            print(f"  {key:<20}: {default}  (default)")
    print("======================\n")

    # ------------------------------------------------------------------
    # MLflow setup — active only when the launcher injects the tracking URI.
    # The HF Trainer MLflow callback reads these same env vars automatically
    # and logs all train/eval metrics per step without extra code.
    # ------------------------------------------------------------------
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(
            os.environ.get("MLFLOW_EXPERIMENT_NAME", "queryforge-finetuning")
        )
        # Start the child run explicitly so the HF Trainer callback always
        # finds an active run and cannot fail silently without creating one.
        tags = {}
        mlflow.start_run(
            run_name=os.environ.get("MLFLOW_RUN_NAME") or os.environ.get("TRAINING_JOB_NAME", "queryforge-train"),
            tags=tags,
        )
        print(f"\n=== MLflow tracking ===")
        print(f"  Server  : {tracking_uri}")
        print(f"  Run     : {mlflow.active_run().info.run_id}")
        print(f"======================")

    # Disable the HuggingFace MLflowCallback: it uses log_batch internally,
    # which fails silently against sagemaker-mlflow<=0.2.0. All metrics are
    # logged manually from trainer.state.log_history after training completes.
    report_to = ["none"]

    model = AutoModelForCausalLM.from_pretrained(
        _MODEL_CHANNEL,
        quantization_config=_get_bnb_config(),
        device_map="auto",
        trust_remote_code=False,
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        _get_lora_config(
            r=int(hp.get("lora_r", hyperparameters["lora_r"])),
            alpha=int(hp.get("lora_alpha", hyperparameters["lora_alpha"])),
            dropout=float(hp.get("lora_dropout", hyperparameters["lora_dropout"])),
        ),
    )

    tokenizer = AutoTokenizer.from_pretrained(_MODEL_CHANNEL)
    tokenizer.pad_token = tokenizer.eos_token

    data_files = glob.glob(os.path.join(_TRAINING_CHANNEL, "**", "*.jsonl"), recursive=True)
    if not data_files:
        raise FileNotFoundError(
            f"No JSONL files found under {_TRAINING_CHANNEL}. "
            "Ensure the dataset was uploaded to the correct S3 prefix."
        )
    full_dataset = load_dataset(
        "json",
        data_files=data_files,
        split="train",
    )
    split = full_dataset.train_test_split(test_size=0.1, seed=42)
    print(
        f"\n=== Dataset split ==="
        f"\n  Total   : {len(full_dataset)} examples"
        f"\n  Train   : {len(split['train'])} examples"
        f"\n  Eval    : {len(split['test'])} examples"
        f"\n==================="
    )

    # Log all hyperparameters and job metadata now, while the run is active.
    # The HF Trainer callback will log step-level metrics during trainer.train().
    if tracking_uri and mlflow.active_run():
        sm_env = json.loads(os.environ.get("SM_TRAINING_ENV", "{}"))
        job_name = sm_env.get("job_name", os.environ.get("TRAINING_JOB_NAME", "unknown"))
        region = os.environ.get("AWS_REGION", "us-east-1")
        mlflow.log_params({
            "lora_r": hp.get("lora_r", hyperparameters["lora_r"]),
            "lora_alpha": hp.get("lora_alpha", hyperparameters["lora_alpha"]),
            "lora_dropout": hp.get("lora_dropout", hyperparameters["lora_dropout"]),
            "train_examples": len(split["train"]),
            "eval_examples": len(split["test"]),
            "sagemaker_job_name": job_name,
            "sagemaker_job_uri": (
                f"https://{region}.console.aws.amazon.com/sagemaker/home"
                f"?region={region}#/jobs/{job_name}"
            ),
        })

    training_args = SFTConfig(
        output_dir=_OUTPUT_DIR,
        num_train_epochs=int(hp.get("epochs", hyperparameters["epochs"])),
        per_device_train_batch_size=int(hp.get("batch_size", hyperparameters["batch_size"])),
        gradient_accumulation_steps=int(hp.get("grad_accum_steps", hyperparameters["grad_accum_steps"])),
        learning_rate=float(hp.get("learning_rate", hyperparameters["learning_rate"])),
        fp16=False,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        warmup_steps=100,
        lr_scheduler_type="cosine",
        dataset_text_field="text",
        max_seq_length=int(hp.get("max_seq_length", hyperparameters["max_seq_length"])),
        report_to=report_to,
        run_name=os.environ.get("TRAINING_JOB_NAME", "queryforge-train"),
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(_OUTPUT_DIR)

    if tracking_uri and mlflow.active_run():
        # Replay the full training history so every step-level metric (train
        # loss, eval loss, token accuracy, etc.) is persisted to the MLflow
        # tracking server via individual log_metrics calls, bypassing the
        # broken log_batch codepath in sagemaker-mlflow<=0.2.0.
        for entry in trainer.state.log_history:
            step = entry.get("step")
            if step is None:
                continue
            metrics = {
                k: v
                for k, v in entry.items()
                if k != "step" and isinstance(v, (int, float))
            }
            if metrics:
                mlflow.log_metrics(metrics, step=step)
        mlflow.end_run()


if __name__ == "__main__":
    main()
