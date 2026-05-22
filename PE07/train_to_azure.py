import os
import joblib

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from utils import create_workspace, log_model


# Create workspace
create_workspace()

# Load dataset
iris = load_iris()

X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestClassifier()

model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

# Create model storage folder
os.makedirs("azure_storage/models", exist_ok=True)

# Save model
model_name = "iris_model.pkl"

model_path = f"azure_storage/models/{model_name}"

joblib.dump(model, model_path)

print(f"[AZURE-SIMULATION] Model saved to {model_path}")

# Register model
log_model(model_name, accuracy)

print(f"[AZURE-SIMULATION] Accuracy: {accuracy:.2f}")