"""
Simulates KaizenML continuous model improvement by:
- Comparing old and new model accuracy
- Replacing the model if improved
- Logging results with timestamp
"""
import pandas as pd
import joblib
from datetime import datetime
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load existing model
current_model = joblib.load("model/best_model.pkl")

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=None
)

# Evaluate current model
old_accuracy = accuracy_score(y_test, current_model.predict(X_test))

# Train new candidate model
new_model = RandomForestClassifier(n_estimators=200)
new_model.fit(X_train, y_train)

# Evaluate new model
new_accuracy = accuracy_score(y_test, new_model.predict(X_test))

# Print exact required format
print(f"Old Accuracy: {old_accuracy:.3f} | New Accuracy: {new_accuracy:.3f}")

# Load log
log = pd.read_csv("model/performance_log.csv")

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

if new_accuracy > old_accuracy:
    joblib.dump(new_model, "model/best_model.pkl")
    status = "Improved and Deployed"
    print("New model is better. Replacing the old model.")
else:
    status = "No Improvement"
    print("New model not better. Keeping the previous one.")

# Add log entry
new_entry = pd.DataFrame([{
    "timestamp": timestamp,
    "old_accuracy": round(old_accuracy, 3),
    "new_accuracy": round(new_accuracy, 3),
    "status": status
}])

log = pd.concat([log, new_entry], ignore_index=True)
log.to_csv("model/performance_log.csv", index=False)

print(f"Log updated at {timestamp}")
print("Model log saved to model/performance_log.csv")