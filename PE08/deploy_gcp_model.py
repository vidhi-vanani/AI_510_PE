import shutil
from utils_gcp import log_message

source = "gcp_bucket/models/iris_model.pkl"
destination = "gcp_bucket/deployed_model/iris_model.pkl"

shutil.copy(source, destination)

print("Model deployed successfully!")

log_message("Deployment completed.")