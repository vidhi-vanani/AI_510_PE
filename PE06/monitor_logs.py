from utils import get_cloudwatch_client
import random

cloudwatch = get_cloudwatch_client()

latency = round(random.uniform(50, 200), 2)
error_rate = round(random.uniform(0, 5), 2)

metrics = [
    {"MetricName": "Latency", "Value": latency},
    {"MetricName": "ErrorRate", "Value": error_rate},
]

cloudwatch.put_metric_data(
    Namespace="MLOpsMonitoring",
    MetricData=metrics
)

print(f"Latency: {latency} ms")
print(f"Error Rate: {error_rate}%")