from utils import get_sagemaker_client

model_name = "iris_model"

sagemaker = get_sagemaker_client()
sagemaker.create_model(ModelName=model_name)

print("Deployment simulation completed.")