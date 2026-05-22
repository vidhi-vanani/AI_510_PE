import json
import os
from datetime import datetime

def create_workspace():
    workspace = {
        "workspace_name": "azure-mlops-workspace",
        "region": "westus",
        "created_time": str(datetime.now())
    }
    with open("workspace.json", "w") as f:
        json.dump(workspace, f, indent=4)

    print("[AZURE-SIMULATION] AZURE ML Workspace created.")

def log_model(model_name, accuracy):
    file_exists = os.path.isfile("model_registry.csv")

    with open("model_registry.csv", "a" if file_exists else "w") as f:
        if not file_exists:
            f.write("model_name,accuracy,timestamp\n")

        f.write(f"{model_name},{accuracy},{datetime.now()}\n")

    print(f"[AZURE-SIMULATION] Model registered: {model_name}")

    