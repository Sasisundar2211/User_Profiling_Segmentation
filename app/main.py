"""Production-grade Flask application for User Profiling & Segmentation API.

Provides endpoints for RESTful payload and CSV dataset processing using KMeans clustering.
"""

from __future__ import annotations

import logging
from io import StringIO
from typing import Any, Dict, List, Tuple

import pandas as pd
from flask import Flask, jsonify, request, Response

from src.clustering import cluster_users
from src.data_preprocessing import preprocess_data
from src.feature_engineering import engineer_features
from src.profiling import profile_segments

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

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


def segment_records(records: List[Dict[str, Any]], requested_k: int = 4) -> Dict[str, Any]:
    """Processes user records through the ML pipeline.

    Args:
        records: List of dictionaries representing user data.
        requested_k: Desired number of clusters.

    Returns:
        Dict containing cluster summaries, segment assignments, and segment profiles.
    """
    if not records:
        raise ValueError("Payload must contain at least one user record.")

    df = pd.DataFrame(records)
    missing_cols = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing_cols:
        raise ValueError(f"Missing required schema attributes: {', '.join(missing_cols)}")

    # Feature Engineering & Preprocessing
    df_engineered = engineer_features(df)
    X, _ = preprocess_data(df_engineered)

    # Dynamic cluster bound based on dataset size
    effective_k = min(max(1, requested_k), len(df_engineered))
    labels = cluster_users(X, k=effective_k)
    profiles = profile_segments(df_engineered, labels)

    assignments = [
        {"row": index, "segment": int(label)}
        for index, label in enumerate(labels)
    ]

    return {
        "status": "success",
        "cluster_count": effective_k,
        "record_count": len(df_engineered),
        "records": assignments,
        "profiles": profiles,
    }


@app.after_request
def add_cors_headers(response: Response) -> Response:
    """Adds CORS headers for cross-origin API integration."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/", methods=["GET"])
def index() -> Tuple[Response, int]:
    """API root detailing service metadata and available endpoints."""
    return (
        jsonify({
            "name": "User Profiling & Segmentation API",
            "version": "1.0.0",
            "status": "online",
            "description": "Enterprise ML segmentation engine built with Flask and scikit-learn.",
            "endpoints": {
                "health": "/health",
                "segmentation": "/api/segment",
            },
            "docs": "/docs",
        }),
        200,
    )


@app.route("/health", methods=["GET"])
def health() -> Tuple[Response, int]:
    """Health check endpoint for container probes and deployment monitoring."""
    return jsonify({"status": "healthy", "service": "user-segmentation-api"}), 200


@app.route("/api/segment", methods=["POST", "OPTIONS"])
def segment_endpoint() -> Tuple[Response, int]:
    """Handles POST requests containing JSON records or CSV uploads for user segmentation."""
    if request.method == "OPTIONS":
        return jsonify({}), 200

    try:
        requested_k = 4
        records: List[Dict[str, Any]] = []

        if "file" in request.files:
            file_obj = request.files["file"]
            if not file_obj.filename:
                raise ValueError("Uploaded file cannot be empty.")
            csv_content = file_obj.read().decode("utf-8")
            df_csv = pd.read_csv(StringIO(csv_content))
            records = df_csv.to_dict(orient="records")
        else:
            payload = request.get_json(silent=True) or {}
            if isinstance(payload, list):
                records = payload
            elif isinstance(payload, dict):
                records = payload.get("records", [])
                requested_k = payload.get("clusters", 4)
            else:
                raise ValueError("Invalid request format. Provide a JSON array or object with 'records'.")

        if not records or not isinstance(records, list):
            raise ValueError("No valid user records provided for segmentation.")

        result = segment_records(records, requested_k=requested_k)
        return jsonify(result), 200

    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as err:
        logger.warning("Validation error in /api/segment: %s", str(err))
        return jsonify({"status": "error", "error": str(err)}), 400
    except Exception as err:
        logger.error("Internal error in /api/segment: %s", str(err), exc_info=True)
        return jsonify({"status": "error", "error": "An internal error occurred."}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
