import random
import os
from datetime import datetime


os.makedirs("monitoring_logs", exist_ok=True)

latency = round(random.uniform(50, 300), 2)

error_rate = round(random.uniform(0, 0.05), 3)

log_message = (
    f"{datetime.now()} | "
    f"Latency={latency}ms | "
    f"Error Rate={error_rate}"
)

with open("monitoring_logs/log.txt", "a") as f:
    f.write(log_message + "\n")

print("[AZURE-SIMULATION] Azure Monitor Active")

print(f"Latency={latency}ms, Error Rate={error_rate}")