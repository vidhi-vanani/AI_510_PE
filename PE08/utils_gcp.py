import os
import shutil

os.makedirs("gcp_bucket/models", exist_ok=True)
os.makedirs("gcp_bucket/deployed_model", exist_ok=True)

def upload_to_gcs(souce_path, destination_path):
    """
    Uploads a file to Google Cloud Storage.
    """
    shutil.copy(souce_path, destination_path)
    print(f"Uploaded {souce_path} to {destination_path}")

def log_message(message):
    """
    Logs a message to the console.
    """
    print(f"LOG: {message}")