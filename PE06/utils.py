import os
from datetime import datetime


class MockS3Client:
    def upload_file(self, local_path, bucket, key):
        print(f"[SIMULATED S3] Uploaded {local_path} to s3://{bucket}/{key}")


class MockSageMakerClient:
    def create_model(self, ModelName):
        print(f"[SIMULATED SageMaker] Model '{ModelName}' deployed successfully")


class MockCloudWatchClient:
    def put_metric_data(self, Namespace, MetricData):
        print(f"[SIMULATED CloudWatch] Metrics logged to {Namespace}")


def get_s3_client():
    return MockS3Client()


def get_sagemaker_client():
    return MockSageMakerClient()


def get_cloudwatch_client():
    return MockCloudWatchClient()