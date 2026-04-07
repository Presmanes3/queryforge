import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.config import ConfigLoader as _ConfigLoader  # noqa: E402
os.environ.setdefault("AWS_DEFAULT_REGION", _ConfigLoader.load().aws_region)

from pipeline.definition import pipeline
from config import config

pipeline.upsert(    
    role_arn    = config.execution_role_arn,
    tags        = [{"Key": "project", "Value": "queryforge"}]
)