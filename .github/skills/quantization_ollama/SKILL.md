---
name: quantization_ollama
description: >
  Patterns for GGUF quantization and Ollama publishing in QueryForge.
  Use this skill whenever writing or modifying code in src/queryforge/quantize/
  or src/queryforge/publish/.
---

# Quantization and Ollama Skill

## Principles

- GGUF conversion uses llama.cpp's `convert_hf_to_gguf.py`. Never convert using a
  different tool unless explicitly justified and documented.
- Quantization variants are declared in `config/pipeline.yaml` as a list. The
  `quantize` component iterates over all declared variants.
- The Ollama `Modelfile` is generated from a template — never written by hand.
- All intermediate files (`.gguf`, merged weights) are written to S3, not to local disk,
  except within the SageMaker processing container where `/tmp` is used.

---

## 1. Supported quantization formats

| Format | Description | Recommended use |
|---|---|---|
| `f16` | Full 16-bit float — no quantization | Reference, accuracy testing |
| `q8_0` | 8-bit integer | High accuracy, moderate size |
| `q4_k_m` | 4-bit, K-quants, medium | Default for production |
| `q5_k_m` | 5-bit, K-quants, medium | Balance accuracy / size |
| `q2_k` | 2-bit — aggressive compression | Edge devices only |

The default variant list in `config/pipeline.yaml`:
```yaml
gguf_variants:
  - q4_k_m
  - q5_k_m
```

---

## 2. GGUF conversion

```python
# src/queryforge/quantize/convert.py
import subprocess
import sys
import os

def convert_to_gguf(merged_model_path: str, output_path: str) -> str:
    """Convert a merged HuggingFace model to GGUF format using llama.cpp.

    Runs convert_hf_to_gguf.py from the llama.cpp submodule at third_party/llama.cpp/.

    Args:
        merged_model_path: Directory containing the merged HuggingFace weights and
                            tokenizer (output of merge_adapter).
        output_path: Destination path for the output .gguf file (f16, unquantized).

    Returns:
        Absolute path to the generated .gguf file.

    Raises:
        RuntimeError: If the conversion subprocess exits with a non-zero code.
    """
    converter = os.path.join(
        os.path.dirname(__file__), "../../../third_party/llama.cpp/convert_hf_to_gguf.py"
    )
    cmd = [sys.executable, converter, merged_model_path, "--outfile", output_path, "--outtype", "f16"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"GGUF conversion failed:\n{result.stderr}")
    return output_path
```

---

## 3. Weight quantization

```python
# src/queryforge/quantize/quantize.py
import subprocess
import os

def quantize_gguf(f16_path: str, output_path: str, variant: str) -> str:
    """Quantize a GGUF file to a specific quantization variant.

    Uses llama.cpp's llama-quantize binary.

    Args:
        f16_path: Path to the F16 GGUF file produced by convert_to_gguf.
        output_path: Path where the quantized GGUF file will be written.
        variant: Quantization variant string (e.g., "q4_k_m", "q5_k_m").

    Returns:
        Absolute path to the quantized .gguf file.

    Raises:
        RuntimeError: If quantization fails.
        ValueError: If the variant string is not in the supported list.
    """
    supported = {"f16", "q8_0", "q4_k_m", "q5_k_m", "q2_k"}
    if variant.lower() not in supported:
        raise ValueError(f"Unsupported quantization variant: {variant}. "
                         f"Choose from: {supported}")

    quantizer = os.path.join(
        os.path.dirname(__file__), "../../../third_party/llama.cpp/build/bin/llama-quantize"
    )
    cmd = [quantizer, f16_path, output_path, variant.upper()]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Quantization failed for variant {variant}:\n{result.stderr}")
    return output_path
```

---

## 4. Ollama Modelfile generation

```python
# src/queryforge/publish/modelfile.py
from string import Template

MODELFILE_TEMPLATE = Template("""\
FROM $gguf_path

TEMPLATE \"\"\"{{ if .System }}<|system|>
{{ .System }}<|end|>
{{ end }}{{ if .Prompt }}<|user|>
{{ .Prompt }}<|end|>
<|assistant|>
{{ end }}{{ .Response }}<|end|>\"\"\"

PARAMETER stop "<|end|>"
PARAMETER stop "<|user|>"
PARAMETER stop "<|assistant|>"

SYSTEM \"\"\"You are an expert SQL assistant. Given a database schema and a question,
write a single correct SQL query that answers the question.\"\"\"
""")

def generate_modelfile(gguf_path: str, output_path: str) -> None:
    """Generate an Ollama-compatible Modelfile pointing to a GGUF file.

    Args:
        gguf_path: Absolute or relative path to the .gguf file on the local machine.
        output_path: Path where the Modelfile will be written.
    """
    content = MODELFILE_TEMPLATE.substitute(gguf_path=gguf_path)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
```

---

## 5. Ollama publishing

```python
# src/queryforge/publish/publish.py
import subprocess

def publish_to_ollama(modelfile_path: str, model_name: str) -> None:
    """Build and register a model in the local Ollama instance.

    Args:
        modelfile_path: Path to the generated Modelfile.
        model_name: Name under which the model will be registered in Ollama
                    (e.g., "queryforge-orders-v1-q4").

    Raises:
        RuntimeError: If the `ollama create` command fails.
    """
    result = subprocess.run(
        ["ollama", "create", model_name, "-f", modelfile_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ollama create failed:\n{result.stderr}")
```

### Model naming convention

```
queryforge-<schema_name>-<schema_version>-<variant>
```

Examples: `queryforge-orders-v1-q4km`, `queryforge-inventory-v2-q5km`.
