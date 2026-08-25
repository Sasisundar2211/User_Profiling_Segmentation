"""Integration tests for Flask web endpoints."""

from __future__ import annotations

from io import BytesIO
import pytest

from app.main import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "online"
    assert "endpoints" in data


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "healthy"


def test_segment_endpoint_valid_json(client):
    payload = {
        "records": [
            {
                "Age": "25-34", "Gender": "Female", "Income Level": "High",
                "Education Level": "Graduate", "Device Usage": "Mobile",
                "Time Spent Online (hrs/weekday)": 4.2,
                "Time Spent Online (hrs/weekend)": 5.1,
                "Click-Through Rates (CTR)": 0.11,
                "Conversion Rates": 0.04,
                "Ad Interaction Time (sec)": 18,
            },
            {
                "Age": "35-44", "Gender": "Male", "Income Level": "Medium",
                "Education Level": "Undergraduate", "Device Usage": "Desktop",
                "Time Spent Online (hrs/weekday)": 2.1,
                "Time Spent Online (hrs/weekend)": 3.4,
                "Click-Through Rates (CTR)": 0.06,
                "Conversion Rates": 0.02,
                "Ad Interaction Time (sec)": 9,
            },
        ]
    }
    response = client.post("/api/segment", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert len(data["records"]) == 2
    assert "profiles" in data


def test_segment_endpoint_invalid_schema(client):
    payload = {
        "records": [
            {"Age": "25-34"}
        ]
    }
    response = client.post("/api/segment", json=payload)
    assert response.status_code == 400
    data = response.get_json()
    assert data["status"] == "error"
    assert "Missing required schema attributes" in data["error"]


def test_segment_endpoint_csv_file(client):
    csv_data = (
        "Age,Gender,Income Level,Education Level,Device Usage,Time Spent Online (hrs/weekday),Time Spent Online (hrs/weekend),Click-Through Rates (CTR),Conversion Rates,Ad Interaction Time (sec)\n"
        "25-34,Female,High,Graduate,Mobile,4.2,5.1,0.11,0.04,18\n"
        "35-44,Male,Medium,Undergraduate,Desktop,2.1,3.4,0.06,0.02,9\n"
    )
    data = {
        "file": (BytesIO(csv_data.encode("utf-8")), "test.csv")
    }
    response = client.post("/api/segment", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    res_json = response.get_json()
    assert res_json["status"] == "success"
    assert res_json["record_count"] == 2
