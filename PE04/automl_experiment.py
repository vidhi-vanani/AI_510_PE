"""
automl_experiment.py

Runs a simple AutoML experiment by:
- Training multiple models
- Selecting the best one
- Saving the best model
- Logging initial performance
"""
import os
import pandas as pd
import joblib
from datetime import datetime
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Create model directory
os.makedirs("model", exist_ok=True)

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Candidate models
models = {
    "RandomForest": RandomForestClassifier(),
    "SVM": SVC(),
    "LogisticRegression": LogisticRegression(max_iter=500)
}

best_score = 0
best_model = None
best_name = ""

print("Running AutoML Experiment...\n")

for name, model in models.items():
    score = cross_val_score(model, X, y, cv=5).mean()
    print(f"{name}: Accuracy = {score:.4f}")

    if score > best_score:
        best_score = score
        best_model = model
        best_name = name

# Train best model
best_model.fit(X, y)

# Save model
joblib.dump(best_model, "model/best_model.pkl")

# Create performance log
log = pd.DataFrame([{
    "timestamp": datetime.now(),
    "model": best_name,
    "accuracy": best_score,
    "status": "Initial Model"
}])

log.to_csv("model/performance_log.csv", index=False)

print(f"\nBest Model Selected: {best_name}")
print(f"Accuracy: {best_score:.4f}")
print("Saved to model/best_model.pkl")