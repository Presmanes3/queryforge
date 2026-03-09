"""QueryForge AWS Infrastructure Setup Utility.

Configures the S3 artifact bucket and required resource paths.
"""

from __future__ import annotations
import argparse
import sys
import yaml
import boto3
from botocore.exceptions import ClientError
from queryforge.utils.config import load_config


def main():
    """Execute the project's AWS initialization sequence."""
    parser = argparse.ArgumentParser(
        description="Bootstrap the QueryForge project's S3 infrastructure."
    )
    parser.add_argument("--config", default="config/pipeline.yaml")
    args = parser.parse_args()

    config_path = args.config
    config = load_config(config_path)
    s3 = boto3.client("s3", region_name=config.aws_region)

    # 1. Bucket initialization
    try:
        s3.head_bucket(Bucket=config.s3_bucket)
        print(f"Bucket '{config.s3_bucket}' already exists.")
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            print(f"Creating bucket '{config.s3_bucket}' in region {config.aws_region}")
            # LocationConstraint required for regions other than us-east-1
            if config.aws_region == "us-east-1":
                s3.create_bucket(Bucket=config.s3_bucket)
            else:
                s3.create_bucket(
                    Bucket=config.s3_bucket,
                    CreateBucketConfiguration={"LocationConstraint": config.aws_region},
                )
        else:
            print(f"Failed to access bucket: {e}")
            sys.exit(1)

    # 2. Base artifact paths (Placeholder objects to establish prefix hierarchy)
    # Following the convention: s3://<bucket>/queryforge/<folder>/
    artifact_uris = {}
    for folder in config.artifact_folders:
        key_prefix = f"{config.s3_prefix}/{folder}"
        s3.put_object(Bucket=config.s3_bucket, Key=f"{key_prefix}/.keep", Body="")
        uri = f"s3://{config.s3_bucket}/{key_prefix}"
        artifact_uris[f"{folder}_uri"] = uri
        print(f"Initialized folder: {uri}")

    # 3. Update the YAML file with the generated URIs
    print(f"Updating {config_path} with artifact URIs...")
    with open(config_path, "r") as f:
        yaml_data = yaml.safe_load(f)

    yaml_data["artifact_uris"] = artifact_uris

    with open(config_path, "w") as f:
        yaml.dump(yaml_data, f, sort_keys=False, default_flow_style=False)

    print(f"Bucket bootstrap sequence completed successfully in {config.aws_region}.")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
