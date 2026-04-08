
import os
import boto3
import sys
from pathlib import Path

# Add src to path for utils
sys.path.append(str(Path.cwd() / "src"))
sys.path.append(str(Path.cwd()))

from utils.config import ConfigLoader
from utils.s3 import S3Repository

def download():
    config = ConfigLoader.load('config/pipeline.yaml')
    session = boto3.Session(region_name=config.aws_region, profile_name=config.aws_profile)
    s3 = S3Repository(session)
    
    bucket = config.s3_bucket
    prefix = f"{S3Repository.resolve_component_prefix(config, 'adapters')}/orders/v1/pipelines-4g16u3ydjdhh-QLoraFineTune-Dxqd0jMPM5"
    
    local_dir = Path("local_output/adapter")
    local_dir.mkdir(parents=True, exist_ok=True)
    
    keys = s3.list(bucket, prefix)
    print(f"Found {len(keys)} objects in {prefix}")
    
    for key in keys:
        if key.endswith("/"):
            continue
        rel_path = key[len(prefix):].lstrip("/")
        dest = local_dir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {key} to {dest}")
        s3.download(bucket, key, str(dest))

if __name__ == "__main__":
    download()
