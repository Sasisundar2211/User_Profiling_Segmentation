import os
import io
import joblib
import pandas as pd
from flask import Flask, request, jsonify

from src.feature_engineering import engineer_features
from src.data_preprocessing import preprocess_data

app = Flask(__name__)

MODEL_PATH = os.path.join("models", "clustering_model.pkl")
PREPROCESSOR_PATH = os.path.join("models", "preprocessor.pkl")

# Load model and preprocessor once at startup (lazy on first request)
_model = None
_preprocessor = None


def _load_artifacts():
    global _model, _preprocessor
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    if _preprocessor is None:
        _preprocessor = joblib.load(PREPROCESSOR_PATH)


@app.route("/")
def index():
    return jsonify({"status": "ok", "message": "User Profiling & Segmentation API"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept a CSV file upload and return cluster segment labels.
    Usage: POST /predict  with form-data field 'file' containing a CSV.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        df = pd.read_csv(io.StringIO(file.stream.read().decode("utf-8")))
    except Exception as e:
        return jsonify({"error": f"Failed to parse CSV: {str(e)}"}), 400

    try:
        _load_artifacts()
        df = engineer_features(df)
        X = _preprocessor.transform(df)
        labels = _model.predict(X)
        df["segment"] = labels
        return jsonify({"segments": df["segment"].tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
