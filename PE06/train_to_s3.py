import os
import joblib
from datetime import datetime
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from utils import get_s3_client

# Load dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# Timestamp for versioning
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create versioned folder
model_dir = f"model/{timestamp}"
os.makedirs(model_dir, exist_ok=True)

# Save model
model_path = f"{model_dir}/iris_model.pkl"
joblib.dump(model, model_path)

print(f"Model saved locally at {model_path}")
print(f"Accuracy: {accuracy:.4f}")

# Simulate upload to S3
s3 = get_s3_client()
bucket_name = "ml-model-bucket"
s3_key = f"{timestamp}/iris_model.pkl"

s3.upload_file(model_path, bucket_name, s3_key)

# Log metadata to registry
with open("model_registry.log", "a") as log:
    log.write(
        f"Model Name: iris_model.pkl | Timestamp: {timestamp} | Accuracy: {accuracy:.4f}\n"
    )

print("Model metadata logged in model_registry.log")