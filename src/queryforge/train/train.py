"""SageMaker Training Job entry point for QLoRA fine-tuning.

Reads all configuration from the SageMaker hyperparameters JSON file.
Expects two data channels: 'model' (base model weights) and 'training' (JSONL dataset).
Writes the LoRA adapter to the SageMaker model output directory.
"""

from __future__ import annotations
import json
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

# hyperparameters.py is co-located in the same train/ directory and is packaged
# into the container alongside this entry point.  It has no SageMaker SDK
# dependency so it is safe to import here.
from hyperparameters import DEFAULTS

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
            r=int(hp.get("lora_r", DEFAULTS["lora_r"])),
            alpha=int(hp.get("lora_alpha", DEFAULTS["lora_alpha"])),
            dropout=float(hp.get("lora_dropout", DEFAULTS["lora_dropout"])),
        ),
    )

    tokenizer = AutoTokenizer.from_pretrained(_MODEL_CHANNEL)
    tokenizer.pad_token = tokenizer.eos_token

    full_dataset = load_dataset(
        "json",
        data_files=os.path.join(_TRAINING_CHANNEL, "*.jsonl"),
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

    training_args = SFTConfig(
        output_dir=_OUTPUT_DIR,
        num_train_epochs=int(hp.get("epochs", DEFAULTS["epochs"])),
        per_device_train_batch_size=int(hp.get("batch_size", DEFAULTS["batch_size"])),
        gradient_accumulation_steps=int(hp.get("grad_accum_steps", DEFAULTS["grad_accum_steps"])),
        learning_rate=float(hp.get("learning_rate", DEFAULTS["learning_rate"])),
        fp16=False,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        warmup_steps=100,
        lr_scheduler_type="cosine",
        dataset_text_field="text",
        max_seq_length=int(hp.get("max_seq_length", DEFAULTS["max_seq_length"])),
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


if __name__ == "__main__":
    main()
