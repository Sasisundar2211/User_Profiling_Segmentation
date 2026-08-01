"""HTTP interface for the user-profiling segmentation pipeline."""

from __future__ import annotations

from io import StringIO
from typing import Any

import pandas as pd
from flask import Flask, jsonify, request

from src.clustering import cluster_users
from src.data_preprocessing import preprocess_data
from src.feature_engineering import engineer_features
from src.profiling import profile_segments

app = Flask(__name__)

REQUIRED_COLUMNS = {
    "Age",
    "Gender",
    "Income Level",
    "Education Level",
    "Device Usage",
    "Time Spent Online (hrs/weekday)",
    "Time Spent Online (hrs/weekend)",
    "Click-Through Rates (CTR)",
    "Conversion Rates",
    "Ad Interaction Time (sec)",
}


def _segment(records: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(records)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if len(frame) < 2:
        raise ValueError("Provide at least two user records for segmentation.")

    frame = engineer_features(frame.copy())
    features, _ = preprocess_data(frame)
    clusters = min(4, len(frame))
    labels = cluster_users(features, k=clusters)
    profiles = profile_segments(frame.copy(), labels)
    return {
        "cluster_count": clusters,
        "records": [{"row": index, "segment": int(label)} for index, label in enumerate(labels)],
        "profiles": {str(segment): values for segment, values in profiles.items()},
    }


@app.get("/")
def home():
    return jsonify({
        "service": "User Profiling & Segmentation API",
        "endpoints": {"health": "/health", "segment": "/api/segment"},
    })


@app.get("/health")
def health():
    return jsonify({"status": "healthy"})


@app.post("/api/segment")
def segment_users():
    try:
        if "file" in request.files:
            records = pd.read_csv(StringIO(request.files["file"].read().decode("utf-8"))).to_dict("records")
        else:
            payload = request.get_json(silent=True) or {}
            records = payload if isinstance(payload, list) else payload.get("records", [])
        if not isinstance(records, list):
            raise ValueError("Send a JSON list or an object with a 'records' list.")
        return jsonify(_segment(records))
    except (UnicodeDecodeError, ValueError, pd.errors.ParserError) as error:
        return jsonify({"error": str(error)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
