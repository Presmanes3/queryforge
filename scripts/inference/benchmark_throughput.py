import argparse
import asyncio
import json
import time
import aiohttp
import numpy as np

async def send_request(session, url, payload):
    start = time.perf_counter()
    async with session.post(url, json=payload) as response:
        status = response.status
        output = await response.json()
        end = time.perf_counter()
        
        # vLLM usually returns text in 'text' or 'choices' depending on the API
        # Here we assume the structure of your app.py
        generated_text = output.get("text", [""])[0] if isinstance(output.get("text"), list) else ""
        tokens = len(generated_text.split()) # Rough token estimation
        
        return {
            "latency": end - start,
            "tokens": tokens,
            "success": status == 200
        }

async def benchmark(url, requests_count, concurrency):
    payload = {
        "inputs": "Table orders (id, customer_id, total). Show me the total revenue by customer.",
        "temperature": 0.0,
        "max_new_tokens": 128
    }
    
    print(f"Launching benchmark: {requests_count} requests with concurrency {concurrency}...")
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        # Concurrency control using a semaphore
        sem = asyncio.Semaphore(concurrency)
        
        async def sem_request():
            async with sem:
                return await send_request(session, url, payload)
        
        start_time = time.perf_counter()
        results = await asyncio.gather(*[sem_request() for _ in range(requests_count)])
        end_time = time.perf_counter()
        
    duration = end_time - start_time
    success_results = [r for r in results if r["success"]]
    total_tokens = sum(r["tokens"] for r in success_results)
    latencies = [r["latency"] for r in success_results]
    
    print("\n--- BENCHMARK RESULTS ---")
    print(f"Total Time: {duration:.2f}s")
    print(f"Request Throughput: {len(success_results) / duration:.2f} req/s")
    print(f"Token Throughput (est.): {total_tokens / duration:.2f} tokens/s")
    print(f"Average Latency: {np.mean(latencies):.4f}s")
    print(f"P95 Latency: {np.percentile(latencies, 95):.4f}s")
    print(f"Success: {len(success_results)}/{requests_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8080/invocations")
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=10)
    args = parser.parse_args()
    
    asyncio.run(benchmark(args.url, args.requests, args.concurrency))
