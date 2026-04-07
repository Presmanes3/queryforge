"""QueryForge IAM Role Provisioning Utility.

Creates or updates the SageMaker execution role with necessary S3 and SageMaker permissions.
Uses values from config/pipeline.yaml as the Source of Truth.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import boto3
from botocore.exceptions import ClientError
from utils.config import load_config


def main():
    """Execute the project's IAM role provisioning sequence."""
    parser = argparse.ArgumentParser(
        description="Provision the QueryForge SageMaker execution role."
    )
    parser.add_argument("--config", default="config/pipeline.yaml")
    args = parser.parse_args()

    # Load project SSoT
    config = load_config(args.config)
    session = boto3.Session(profile_name=config.aws_profile, region_name=config.aws_region)
    iam = session.client("iam")

    # Extract role name from ARN
    role_name = config.execution_role_arn.split("/")[-1]
    
    # Path to policies
    infra_dir = os.path.join(os.path.dirname(__file__), "../../infrastructure/iam")
    trust_policy_path = os.path.join(infra_dir, "trust-policy.json")

    with open(trust_policy_path) as f:
        trust_policy = f.read()

    # 1. Create role if it does not exist.  Never overwrite the trust policy of
    # an existing role — Studio service-roles carry additional trust statements
    # that must not be replaced.
    try:
        iam.get_role(RoleName=role_name)
        print(f"Role '{role_name}' already exists. Skipping trust policy update.")
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchEntity":
            print(f"Creating role '{role_name}'...")
            iam.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=trust_policy,
                Description="SageMaker execution role for QueryForge project."
            )
        else:
            print(f"Failed to check/create role: {e}")
            sys.exit(1)

    # 2. Define Inline Policy for S3 and SageMaker
    # We grant full access to the project bucket and standard SageMaker permissions
    project_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:ListBucket",
                    "s3:DeleteObject"
                ],
                "Resource": [
                    f"arn:aws:s3:::{config.s3_bucket}",
                    f"arn:aws:s3:::{config.s3_bucket}/*"
                ]
            },
            {
                "Effect": "Allow",
                "Action": [
                    "sagemaker:CreateTrainingJob",
                    "sagemaker:CreateProcessingJob",
                    "sagemaker:DescribeTrainingJob",
                    "sagemaker:DescribeProcessingJob",
                    "sagemaker:ListTrainingJobs",
                    "sagemaker:ListProcessingJobs",
                    "sagemaker:CreateModel",
                    "sagemaker:RegisterModel"
                ],
                "Resource": "*"
            },
            {
                "Effect": "Allow",
                "Action": [
                    "ecr:GetAuthorizationToken",
                    "ecr:BatchCheckLayerAvailability",
                    "ecr:GetDownloadUrlForLayer",
                    "ecr:BatchGetImage"
                ],
                "Resource": "*"
            },
            {
                "Effect": "Allow",
                "Action": [
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents"
                ],
                "Resource": "arn:aws:logs:*:*:log-group:/aws/sagemaker/*"
            },
            {
                "Effect": "Allow",
                "Action": "sagemaker-mlflow:*",
                "Resource": "*"
            }
        ]
    }

    # 3. Attach Inline Policy
    policy_name = "QueryForgeExecutionPolicy"
    print(f"Attaching/Updating inline policy '{policy_name}' to role '{role_name}'...")
    iam.put_role_policy(
        RoleName=role_name,
        PolicyName=policy_name,
        PolicyDocument=json.dumps(project_policy, indent=2)
    )

    print(f"\nSuccessfully provisioned IAM Role: {config.execution_role_arn}")
    print(f"Permissions granted for bucket: {config.s3_bucket}")


if __name__ == "__main__":
    main()
