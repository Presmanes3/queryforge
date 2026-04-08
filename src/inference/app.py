import logging
import os
import boto3

# Ensure Triton has a valid cache directory and doesn't conflict with system paths.
os.environ["TRITON_CACHE_DIR"] = "/tmp/triton"

from fastapi import FastAPI, Request
from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

# Importación absoluta compatible con la raíz del contenedor /app
try:
    from _gpu import get_vllm_dtype
except ImportError:
    # Fallback si se ejecuta desde src/ (local/tests)
    from src.inference._gpu import get_vllm_dtype

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
llm_engine = None
has_lora = False

def download_s3_prefix(s3_uri, local_dir):
    """Descarga el modelo base de S3 recursivamente."""
    bucket, prefix = s3_uri.replace("s3://", "").split("/", 1)
    prefix = prefix.rstrip("/") + "/"
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = key[len(prefix):]
            if not rel: continue
            dest = os.path.join(local_dir, rel)
            os.makedirs(os.path.dirname(dest) or local_dir, exist_ok=True)
            s3.download_file(bucket, key, dest)

@app.on_event("startup")
async def startup_event():
    global llm_engine, has_lora
    
    # 1. Descargar Modelo Base
    base_s3_uri = os.environ.get("BASE_MODEL_S3_URI")
    base_local_path = "/tmp/base_model"
    
    if base_s3_uri and not os.path.exists(os.path.join(base_local_path, "config.json")):
        logger.info(f"Descargando modelo base de {base_s3_uri}...")
        download_s3_prefix(base_s3_uri, base_local_path)
    elif not base_s3_uri and not os.path.exists(base_local_path):
        base_local_path = "/opt/ml/model/base_model" # Fallback si estuviéramos empaquetando todo
    
    # 2. Check for LoRA adapters
    # En AWS se montarán en /opt/ml/model (por el ModelDataUrl)
    # En local se puede usar variables de entorno o la ruta local de compilación
    adapter_path = os.environ.get("SAGEMAKER_MODEL_DIR", "/opt/ml/model")
    has_lora = os.path.exists(os.path.join(adapter_path, "adapter_config.json"))
    
    if has_lora:
        logger.info(f"Adaptador LoRA encontrado en {adapter_path}")
    else:
        logger.info("Iniciando modo base (sin adaptador LoRA explícito o empaquetado directo)")

    # 3. Inicializar vLLM con AsyncEngine para max throughput FastAPI
    logger.info("Inicializando motor Async vLLM...")
    engine_args = AsyncEngineArgs(
        model=base_local_path,
        enable_lora=has_lora,
        max_loras=1 if has_lora else 0,
        enforce_eager=True,           # Evita problemas con grafos CUDA en arquitecturas mixtas
        tensor_parallel_size=1,
        gpu_memory_utilization=0.80,  # Conservador para evitar OOM por fragmentación en T4 (16GB)
        max_model_len=2048,           # Reducido para mayor estabilidad en inferencia
        disable_custom_all_reduce=True, # Evita kernels personalizados que a veces fallan en T4
        dtype=get_vllm_dtype(),       # float16 en Turing (T4), bfloat16 en Ampere+
        disable_log_stats=False,
        trust_remote_code=True
    )
    llm_engine = AsyncLLMEngine.from_engine_args(engine_args)
    logger.info("vLLM Engine listo.")

@app.get("/ping")
def ping():
    """Obligatorio para SageMaker Health Checks."""
    return {"status": "ok"}

@app.post("/invocations")
async def invocations(request: Request):
    """Endpoint principal de inferencia (POST)."""
    global llm_engine, has_lora
    try:
        data = await request.json()
        prompt = data.get("inputs", "")
        
        # Parámetros por defecto para generación determinista de SQL
        sampling_params = SamplingParams(
            temperature=data.get("temperature", 0.0),
            max_tokens=data.get("max_new_tokens", 150)
        )
        
        request_id = random_uuid()
        
        # Inyectar LoRA si existe
        lora_request = None
        if has_lora:
            from vllm.lora.request import LoRARequest
            adapter_path = os.environ.get("SAGEMAKER_MODEL_DIR", "/opt/ml/model")
            lora_request = LoRARequest("sql_adapter", 1, adapter_path)

        # Generación eficiente asíncrona
        results_generator = llm_engine.generate(prompt, sampling_params, request_id, lora_request=lora_request)
        
        final_output = None
        async for request_output in results_generator:
            final_output = request_output
            
        text = final_output.outputs[0].text
        return {"text": text}
        
    except Exception as e:
        logger.error(f"Error inferencia: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    # Arranque local para pruebas fáciles (no usado por el DLC, que llamará un comando en el Dockerfile)
    uvicorn.run(app, host="0.0.0.0", port=8080)
