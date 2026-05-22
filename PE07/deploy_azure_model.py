import os
import shutil
import pandas as pd


# Read registry
registry = pd.read_csv("model_registry.csv")

# Get latest model
latest_model = registry.iloc[-1]

model_name = latest_model["model_name"]

source_path = f"azure_storage/models/{model_name}"

# Create deployment folder
os.makedirs("deployed_model", exist_ok=True)

destination_path = f"deployed_model/{model_name}"

# Copy model
shutil.copy(source_path, destination_path)

print("[AZURE-SIMULATION] Azure ML Endpoint deployed.")

print(f"Model deployed successfully: {model_name}")