import argparse
import asyncio
import json
import time
import boto3
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def invoke_sagemaker(runtime_client, endpoint_name, payload):
    """Sincronous invocation to be run in a thread."""
    start = time.perf_counter()
    try:
        response = runtime_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Body=json.dumps(payload)
        )
        status = response["ResponseMetadata"]["HTTPStatusCode"]
        output = json.loads(response["Body"].read().decode())
        end = time.perf_counter()
        
        # Estimation of tokens (vLLM app.py returns {"text": "..."})
        generated_text = output.get("text", "")
        tokens = len(generated_text.split())
        
        return {
            "latency": end - start,
            "tokens": tokens,
            "success": status == 200
        }
    except Exception as e:
        return {"latency": 0, "tokens": 0, "success": False, "error": str(e)}

async def benchmark_sagemaker(endpoint_name, requests_count, concurrency):
    payload = {
        "inputs": "Table orders (id, customer_id, total). Question: Show me the total revenue by customer.",
        "temperature": 0.0,
        "max_new_tokens": 128
    }
    
    print(f"Starting SageMaker Benchmark: {requests_count} requests, concurrency {concurrency}...")
    print(f"Endpoint: {endpoint_name}\n")

    runtime_client = boto3.client("sagemaker-runtime")
    loop = asyncio.get_event_loop()
    
    # We use a ThreadPoolExecutor because boto3 is blocking
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        sem = asyncio.Semaphore(concurrency)
        
        async def wrapped_invoke():
            async with sem:
                return await loop.run_in_executor(
                    executor, 
                    invoke_sagemaker, 
                    runtime_client, 
                    endpoint_name, 
                    payload
                )

        start_time = time.perf_counter()
        tasks = [wrapped_invoke() for _ in range(requests_count)]
        results = await asyncio.gather(*tasks)
        end_time = time.perf_counter()

    duration = end_time - start_time
    success_results = [r for r in results if r["success"]]
    failed_results = [r for r in results if not r["success"]]
    
    total_tokens = sum(r["tokens"] for r in success_results)
    latencies = [r["latency"] for r in success_results]
    
    print("\n" + "="*40)
    print("      SAGEMAKER THROUGHPUT REPORT")
    print("="*40)
    print(f"Total Duration:      {duration:.2f}s")
    print(f"Requests/sec:        {len(success_results) / duration:.2f}")
    print(f"Tokens/sec (est.):   {total_tokens / duration:.2f}")
    print(f"Avg Latency:         {np.mean(latencies):.4f}s" if latencies else "Avg Latency: N/A")
    print(f"P95 Latency:         {np.percentile(latencies, 95):.4f}s" if latencies else "P95 Latency: N/A")
    print(f"Success Rate:        {len(success_results)}/{requests_count}")
    
    if failed_results:
        print(f"First Error:         {failed_results[0].get('error')}")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", type=str, required=True, help="SageMaker endpoint name")
    parser.add_argument("--requests", type=int, default=20, help="Total requests to send")
    parser.add_argument("--concurrency", type=int, default=5, help="Simultaneous requests")
    
    args = parser.parse_args()
    
    asyncio.run(benchmark_sagemaker(args.endpoint, args.requests, args.concurrency))
