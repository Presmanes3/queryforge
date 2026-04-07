import os

from botocore.exceptions import ClientError

from config import config

if os.environ.get("QF_LOCAL_MODE") == "1":
    from sagemaker.core.workflow.pipeline_context import LocalPipelineSession as _BaseLocalSession
    from sagemaker.mlops.local.pipeline_entities import _LocalPipeline

    pipeline_session = _BaseLocalSession(
        boto_session            = config.boto_session(),
        default_bucket          = config.s3_bucket,
        default_bucket_prefix   = config.s3_prefix,
    )

    # _BaseLocalSession inherits from PipelineSession (required for pipeline variable
    # interception in ModelTrainer.train()), but LocalSagemakerClient lacks create_pipeline
    # and start_pipeline_execution.  Patch them onto the client instance.
    _client = pipeline_session.sagemaker_client
    _client._pipelines = {}

    def _create_pipeline(pipeline, pipeline_description=None, **kwargs):  # noqa: E306
        _client._pipelines[pipeline.name] = _LocalPipeline(
            pipeline=pipeline,
            pipeline_description=pipeline_description,
            local_session=pipeline_session,
        )
        return {"PipelineArn": pipeline.name}

    def _start_pipeline_execution(PipelineName, **kwargs):  # noqa: E306
        if PipelineName not in _client._pipelines:
            raise ClientError(
                {"Error": {"Code": "ResourceNotFound", "Message": f"Pipeline {PipelineName} does not exist"}},
                "start_pipeline_execution",
            )
        return _client._pipelines[PipelineName].start(**kwargs)

    _client.create_pipeline = _create_pipeline
    _client.start_pipeline_execution = _start_pipeline_execution

else:
    from sagemaker.core.workflow.pipeline_context import PipelineSession
    pipeline_session = PipelineSession(
        boto_session            = config.boto_session(),
        default_bucket          = config.s3_bucket,
        default_bucket_prefix   = config.s3_prefix,
    )