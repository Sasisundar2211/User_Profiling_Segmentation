
# User Profiling & Segmentation Project

## Overview
This project uses machine learning to segment users based on their demographic and behavioral data for targeted advertising.

## Features
- Data preprocessing
- Feature engineering
- KMeans clustering
- Flask web app to upload user data and get segment predictions

## How to Run
1. Create a virtual environment and activate it
2. Install dependencies with `pip install -r requirements.txt`
3. Run the app: `python app/main.py`
4. Open browser at `http://127.0.0.1:5000`

## Deployment
You can deploy it to Google Cloud App Engine with:
```bash
gcloud app deploy
```
