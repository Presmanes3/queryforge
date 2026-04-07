"""Model loading and inference helpers."""

from __future__ import annotations

import os
import tarfile
import tempfile
from glob import glob

import torch
import transformers.utils.logging as _hf_logging
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from _types import Sample
from _sql import normalize_sql, strip_code_fence

_RESPONSE_MARKER = "### Response:\n"
_MAX_NEW_TOKENS  = 256


def use_quantization() -> bool:
    """Return True when NF4 4-bit quantization should be used.

    Set SM_QUANTIZE=0 locally to load in bfloat16 directly on GPU.
    In SageMaker containers the variable is absent, so NF4 is always used.
    """
    return os.getenv("SM_QUANTIZE", "1") != "0"


def _get_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def resolve_adapter_dir(adapter_channel: str) -> str:
    """Return the directory containing the PEFT adapter files.

    SageMaker Training Job output is delivered as model.tar.gz; when found it
    is extracted to a temporary directory and the adapter root is located by
    searching for adapter_config.json.
    """
    tar_files = glob(os.path.join(adapter_channel, "*.tar.gz"))
    if not tar_files:
        return adapter_channel

    tmp = tempfile.mkdtemp()
    with tarfile.open(tar_files[0], "r:gz") as tar:
        tar.extractall(tmp)

    for root, _, files in os.walk(tmp):
        if "adapter_config.json" in files:
            return root

    return tmp


def _extract_prompt(text: str) -> str:
    idx = text.find(_RESPONSE_MARKER)
    if idx == -1:
        return text
    return text[: idx + len(_RESPONSE_MARKER)]


def load_model_and_tokenizer(model_channel: str, adapter_channel: str) -> tuple:
    """Load the base model, tokenizer, and PEFT adapter.

    Args:
        model_channel: Local path to the base model weights directory.
        adapter_channel: Local path to the adapter directory or .tar.gz archive.

    Returns:
        Tuple of (PeftModel, AutoTokenizer) ready for inference.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_channel)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    use_bnb    = use_quantization()
    device_map = "auto" if use_bnb else ("cuda:0" if torch.cuda.is_available() else "cpu")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_channel,
        quantization_config=_get_bnb_config() if use_bnb else None,
        dtype=None if use_bnb else torch.bfloat16,
        device_map=device_map,
        trust_remote_code=False,
    )
    print(f"Quantization: {'NF4 4-bit' if use_bnb else 'bfloat16 (no quantization)'}")

    adapter_dir = resolve_adapter_dir(adapter_channel)
    print(f"Loading PEFT adapter from: {adapter_dir}")
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()

    device = next(model.parameters()).device
    print(f"Model device: {device}")
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / 1024 ** 3
        reserved  = torch.cuda.memory_reserved(device) / 1024 ** 3
        print(f"VRAM allocated: {allocated:.2f} GB  |  reserved: {reserved:.2f} GB")

    return model, tokenizer


def generate_predictions(model, tokenizer, samples: list[Sample]) -> list[str]:
    """Run inference on all samples and return the predicted SQL strings.

    Args:
        model: Loaded PeftModel in eval mode.
        tokenizer: Tokenizer matching the base model.
        samples: Evaluation samples to predict.

    Returns:
        List of predicted SQL strings, one per sample, in the same order.
    """
    # Suppress repetitive generation warnings — not actionable during evaluation.
    _hf_logging.set_verbosity_error()

    n     = len(samples)
    width = len(str(n))
    predictions: list[str] = []

    for i, sample in enumerate(samples):
        prompt = _extract_prompt(sample.text)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=_MAX_NEW_TOKENS,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )

        # Decode only newly generated tokens, excluding the prompt.
        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        predicted_sql = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        predicted_sql = strip_code_fence(predicted_sql)
        predictions.append(predicted_sql)

        match  = normalize_sql(predicted_sql) == normalize_sql(sample.sql)
        status = "OK" if match else "MISS"
        print(f"  [{i + 1:>{width}}/{n}] {status}  pred: {predicted_sql[:72]!r}")
        if not match:
            print(f"  {' ' * (width * 2 + 7)}       exp:  {sample.sql[:72]!r}")

    return predictions
