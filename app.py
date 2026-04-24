"""
Iris Prediction API
A simple Flask API that loads a trained Iris classification model
and serves predictions via a REST endpoint.
"""
import socket
import platform
import sys
import flask
import joblib
import numpy as np

app = flask.Flask(__name__)

# Load trained model
model = joblib.load("model/iris_model.pkl")

@app.route("/")
def home():
    """
    Health check endpoint.

    Returns:
        str: Simple message confirming API is running.
    """
    return "Iris Prediction API is working !"

@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict the Iris flower class based on input features.

    Expected JSON input:
        {
            "features": [sepal_length, sepal_width, petal_length, petal_width]
        }

    Returns:
        flask.Response: JSON object containing predicted class.
    """
    data = flask.request.get_json()
    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features)

    return flask.jsonify({
        "prediction": int(prediction[0])
    })

@app.route("/runtime", methods=["GET"])
def runtime():
    """
    Returns runtime environment information for debugging and CI/CD verification.

    Returns:
        flask.Response: JSON object containing system and package details.
    """
    return flask.jsonify({
        "python_version": sys.version,
        "platform": platform.system(),
        "hostname": socket.gethostname(),
        "packages": {
            "flask": "installed",
            "scikit-learn": "installed",
            "joblib": "installed"
        }
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
