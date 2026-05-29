import random
from utils_gcp import log_message

latency = round(random.uniform(50, 150), 2)
cpu_usage = round(random.uniform(20, 80), 2)

print(f"Latency: {latency} ms")
print(f"CPU Usage: {cpu_usage}%")

log_message("Monitoring metrics captured.")