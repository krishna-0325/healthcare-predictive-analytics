"""
api.py
======
Healthcare Predictive Analytics — Flask REST API

Endpoints
---------
  GET  /health                  → server status
  GET  /api/model-info          → metadata for both models
  POST /api/predict/diabetes    → diabetes risk prediction
  POST /api/predict/heart       → heart disease risk prediction

Usage
-----
  python api.py

The API runs on http://localhost:5000
CORS is enabled so the frontend (index.html) can call it directly.
"""

import json
import pickle
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)   # allow requests from the HTML frontend

# ─── Load models ──────────────────────────────────────────────────────────────

MODELS_DIR = Path("models")

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_json(path):
    with open(path) as f:
        return json.load(f)

try:
    diabetes_pipeline = load_pickle(MODELS_DIR / "diabetes_model.pkl")
    heart_pipeline    = load_pickle(MODELS_DIR / "heart_model.pkl")
    diabetes_meta     = load_json(MODELS_DIR / "diabetes_features.json")
    heart_meta        = load_json(MODELS_DIR / "heart_features.json")
    print("✓  Models loaded successfully.")
except FileNotFoundError:
    print("✗  Model files not found.")
    print("   Run  python train_models.py  first to generate them.")
    diabetes_pipeline = heart_pipeline = None
    diabetes_meta = heart_meta = {}


# ─── Health check ─────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({
        "status":  "ok",
        "models_loaded": diabetes_pipeline is not None,
    })


# ─── Model metadata ───────────────────────────────────────────────────────────

@app.route("/api/model-info")
def model_info():
    return jsonify({
        "diabetes": diabetes_meta,
        "heart":    heart_meta,
    })


# ─── Diabetes prediction ──────────────────────────────────────────────────────
#
# Expected POST body (JSON):
# {
#   "Pregnancies":              1,
#   "Glucose":                120,
#   "BloodPressure":           72,
#   "SkinThickness":           20,
#   "Insulin":                 80,
#   "BMI":                   28.0,
#   "DiabetesPedigreeFunction": 0.30,
#   "Age":                     35
# }

DIABETES_FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

@app.route("/api/predict/diabetes", methods=["POST"])
def predict_diabetes():
    if diabetes_pipeline is None:
        return jsonify({"error": "Model not loaded. Run train_models.py first."}), 503

    data = request.get_json(force=True)

    # Validate
    missing = [f for f in DIABETES_FEATURES if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        X = np.array([[float(data[f]) for f in DIABETES_FEATURES]])
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    # Predict
    prob  = diabetes_pipeline.predict_proba(X)[0][1]   # probability of diabetes
    score = round(prob * 100, 1)

    # Feature importances from trained model
    importances = diabetes_meta.get("importances", [])
    top_features = []
    for feat in importances[:5]:
        top_features.append({
            "name":       feat["name"],
            "value":      data[feat["name"]],
            "importance": feat["importance"],
            "pct":        feat["pct"],
        })

    risk_level = "Low" if score < 30 else "Moderate" if score < 60 else "High"

    return jsonify({
        "score":        score,
        "probability":  round(prob, 4),
        "risk_level":   risk_level,
        "top_features": top_features,
        "model":        "RandomForestClassifier",
        "dataset":      "Pima Indians Diabetes Dataset (UCI)",
        "normalized":   True,
    })


# ─── Heart disease prediction ─────────────────────────────────────────────────
#
# Expected POST body (JSON):
# {
#   "age":      52,
#   "sex":       1,      (1=male, 0=female)
#   "cp":        0,      (0=typical angina, 1=atypical, 2=non-anginal, 3=asymptomatic)
#   "trestbps": 130,
#   "chol":     240,
#   "fbs":        0,     (1 if fasting blood sugar > 120 mg/dL)
#   "restecg":    0,
#   "thalach":  150,
#   "exang":      0,     (1 if exercise-induced angina)
#   "oldpeak":  1.0,
#   "slope":      1,
#   "ca":         0,
#   "thal":       2
# }

HEART_FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

@app.route("/api/predict/heart", methods=["POST"])
def predict_heart():
    if heart_pipeline is None:
        return jsonify({"error": "Model not loaded. Run train_models.py first."}), 503

    data = request.get_json(force=True)

    missing = [f for f in HEART_FEATURES if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        X = np.array([[float(data[f]) for f in HEART_FEATURES]])
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    prob  = heart_pipeline.predict_proba(X)[0][1]
    score = round(prob * 100, 1)

    importances = heart_meta.get("importances", [])
    top_features = []
    for feat in importances[:5]:
        top_features.append({
            "name":       feat["name"],
            "value":      data.get(feat["name"]),
            "importance": feat["importance"],
            "pct":        feat["pct"],
        })

    risk_level = "Low" if score < 30 else "Moderate" if score < 60 else "High"

    return jsonify({
        "score":        score,
        "probability":  round(prob, 4),
        "risk_level":   risk_level,
        "top_features": top_features,
        "model":        "GradientBoostingClassifier",
        "dataset":      "Cleveland Heart Disease Dataset (UCI)",
        "normalized":   True,
    })


# ─── Comparison data ──────────────────────────────────────────────────────────

@app.route("/api/comparison")
def comparison():
    comparison_path = MODELS_DIR / "comparison.json"
    if not comparison_path.exists():
        return jsonify({"error": "Run train_models.py to generate comparison data"}), 404
    return jsonify(load_json(comparison_path))


# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  Healthcare Predictive Analytics — API")
    print("  Running on http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, port=5000)
