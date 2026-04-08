import boto3
import json
import argparse

def test_endpoint(endpoint_name, prompt):
    runtime = boto3.client("sagemaker-runtime")
    
    payload = {
        "inputs": prompt,
        "temperature": 0.0,
        "max_new_tokens": 150
    }
    
    print(f"Enviando petición a {endpoint_name}...")
    try:
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Body=json.dumps(payload)
        )
        
        result = json.loads(response["Body"].read().decode())
        print("\nRespuesta del modelo:")
        print("-" * 20)
        if "text" in result:
            print(result["text"])
        else:
            print(json.dumps(result, indent=2))
        print("-" * 20)
        
    except Exception as e:
        print(f"Error al invocar el endpoint: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", type=str, default="queryforge-endpoint-vllm-1775667572")
    parser.add_argument("--prompt", type=str, default="Table orders: id, customer_name, total_amount, created_at. Question: How many orders did we have yesterday?")
    
    args = parser.parse_args()
    test_endpoint(args.endpoint, args.prompt)
