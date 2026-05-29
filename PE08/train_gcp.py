from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

from utils_gcp import upload_to_gcs, log_message

# Load dataset
iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(
    iris.data,
    iris.target,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Model Accuracy: {accuracy}")

# Save model
model_path = "iris_model.pkl"
joblib.dump(model, model_path)

# Simulate upload to GCS
upload_to_gcs(
    model_path,
    "gcp_bucket/models/iris_model.pkl"
)

log_message("Training completed successfully.")