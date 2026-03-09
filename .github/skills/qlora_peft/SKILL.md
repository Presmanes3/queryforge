---
name: qlora_peft
description: >
  Canonical QLoRA + HuggingFace PEFT configuration patterns for QueryForge.
  Use this skill whenever writing or modifying training scripts in
  src/queryforge/train/ or LoRA adapter merge code in src/queryforge/merge/.
---

# QLoRA / PEFT Skill

## Principles

- All training hyperparameters are declared in `shared/schemas/config.py` and passed
  through SageMaker's `HuggingFace` estimator. Never hardcode values in training scripts.
- The base model is loaded in 4-bit NF4 with double quantization (`BitsAndBytesConfig`).
- LoRA is applied only to attention projection layers (`q_proj`, `v_proj`, `k_proj`,
  `o_proj`) and feed-forward layers (`gate_proj`, `up_proj`, `down_proj`).
- The training entry point (`train.py`) reads all config from environment variables set
  by SageMaker from the `hyperparameters` dict.

---

## 1. BitsAndBytesConfig — 4-bit quantization

```python
from transformers import BitsAndBytesConfig
import torch

def get_bnb_config() -> BitsAndBytesConfig:
    """Return a BitsAndBytesConfig for NF4 double quantization."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
```

---

## 2. LoraConfig

```python
from peft import LoraConfig, TaskType

def get_lora_config(
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
) -> LoraConfig:
    """Return a LoraConfig targeting attention and feed-forward projections.

    Args:
        r: LoRA rank. Higher values increase expressivity but also memory use.
        alpha: LoRA scaling factor. Effective learning rate scales as alpha / r.
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
```

---

## 3. Training script structure

```python
# src/queryforge/train/train.py
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset

def main():
    """Entry point for SageMaker Training Job."""
    # SageMaker injects hyperparameters as /opt/ml/input/config/hyperparameters.json
    with open("/opt/ml/input/config/hyperparameters.json") as f:
        hp = json.load(f)

    model_id = hp["base_model_id"]
    output_dir = "/opt/ml/model"
    dataset_path = "/opt/ml/input/data/training"

    bnb_config = get_bnb_config()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=False,      # never enable without explicit review
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, get_lora_config(
        r=int(hp.get("lora_r", 16)),
        alpha=int(hp.get("lora_alpha", 32)),
        dropout=float(hp.get("lora_dropout", 0.05)),
    ))

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files=f"{dataset_path}/*.jsonl", split="train")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=int(hp.get("epochs", 3)),
        per_device_train_batch_size=int(hp.get("batch_size", 4)),
        gradient_accumulation_steps=int(hp.get("grad_accum_steps", 4)),
        learning_rate=float(hp.get("learning_rate", 2e-4)),
        fp16=False,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=int(hp.get("max_seq_length", 2048)),
    )
    trainer.train()
    trainer.save_model(output_dir)
```

---

## 4. Hyperparameter contract

All hyperparameters passed to the SageMaker estimator must map to keys in this table:

| Key | Type | Default | Description |
|---|---|---|---|
| `base_model_id` | `str` | required | HuggingFace model ID |
| `lora_r` | `int` | `16` | LoRA rank |
| `lora_alpha` | `int` | `32` | LoRA alpha |
| `lora_dropout` | `float` | `0.05` | LoRA dropout |
| `epochs` | `int` | `3` | Training epochs |
| `batch_size` | `int` | `4` | Per-device batch size |
| `grad_accum_steps` | `int` | `4` | Gradient accumulation steps |
| `learning_rate` | `float` | `2e-4` | AdamW learning rate |
| `max_seq_length` | `int` | `2048` | Maximum tokenized sequence length |

---

## 5. LoRA adapter merge

```python
# src/queryforge/merge/merge.py
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def merge_adapter(base_model_id: str, adapter_path: str, output_path: str) -> None:
    """Merge a LoRA adapter into the base model weights and save the result.

    Args:
        base_model_id: HuggingFace model ID of the original base model.
        adapter_path: Local path or S3 path to the trained PEFT adapter directory.
        output_path: Local path where the merged full-precision model will be saved.
    """
    model = AutoModelForCausalLM.from_pretrained(base_model_id, device_map="cpu")
    model = PeftModel.from_pretrained(model, adapter_path)
    merged = model.merge_and_unload()   # folds adapter weights into base weights
    merged.save_pretrained(output_path)
    AutoTokenizer.from_pretrained(base_model_id).save_pretrained(output_path)
```
