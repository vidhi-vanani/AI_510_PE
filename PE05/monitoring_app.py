from flask import Flask, request, jsonify
import logging
import os
import csv
from datetime import datetime
import numpy as np

app = Flask(__name__)

# Ensure logs folder exists
if not os.path.exists("logs"):
    os.makedirs("logs")

# Logging setup
logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

LOG_FILE = "logs/request_log.csv"

# Create CSV log file if not exists
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "input", "prediction", "status"])

# Dummy model (replace with trained model if needed)
def model_predict(features):
    """
    Dummy model prediction function that makes a simple prediction based on feature sum.

    Args:
        features (list of float): A list of 4 numerical features for prediction.

    Returns:
        int: Prediction result (0 or 1) based on whether the sum of features > 2.
    """
    return int(sum(features) > 2)  # simple logic

# PREDICT ROUTE (PE05 MODIFIED)
@app.route("/predict", methods=["POST"])
def predict():
    """
    Handle prediction requests via POST.

    Expects JSON payload with 'features' key containing a list of 4 floats.

    Validates input, makes prediction, logs the request, and returns JSON response.

    Returns:
        JSON: {"prediction": int} on success, or {"error": str} on failure.
    """
    data = request.get_json()

    # PE05: Invalid input handling
    if not data or "features" not in data:
        logging.error("Missing 'features' key in input.")
        return jsonify({"error": "Invalid input data."}), 400

    features = data["features"]

    if not isinstance(features, list) or len(features) != 4:
        logging.error(f"Malformed input: {features}")
        return jsonify({"error": "Invalid input data."}), 400

    try:
        features = list(map(float, features))
    except:
        logging.error(f"Non-numeric input: {features}")
        return jsonify({"error": "Invalid input data."}), 400

    prediction = model_predict(features)

    # Log success request
    with open(LOG_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now(), features, prediction, "OK"])

    logging.info(f"Prediction made: {features} -> {prediction}")

    return jsonify({"prediction": prediction})


# MONITOR ROUTE
@app.route("/monitor", methods=["GET"])
def monitor():
    """
    Provide monitoring statistics from the request logs.

    Reads the CSV log file and counts total requests and errors.

    Returns:
        JSON: {"total_requests": int, "errors": int}
    """
    total_requests = 0
    error_count = 0

    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                total_requests += 1
                if row["status"] != "OK":
                    error_count += 1

    return jsonify({
        "total_requests": total_requests,
        "errors": error_count
    })


# HEALTH ROUTE
@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint to verify API status.

    Returns:
        JSON: {"status": "API is running"}
    """
    return jsonify({"status": "API is running"})


# RUN APP
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)