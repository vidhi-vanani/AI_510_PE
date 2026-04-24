"""
Train a RandomForest model on the Iris dataset and save it to disk.

"""
import os
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

X, y = load_iris(return_X_y=True)

model = RandomForestClassifier()
model.fit(X, y)

os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/iris_model.pkl")

print("Model trained and saved!")
