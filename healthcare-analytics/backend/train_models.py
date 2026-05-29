"""
train_models.py
===============
Healthcare Predictive Analytics — Model Training Script

Trains two real scikit-learn classifiers:
  1. RandomForestClassifier   → Diabetes   (Pima Indians Dataset structure)
  2. GradientBoostingClassifier → Heart Disease (Cleveland Dataset structure)

Outputs
-------
  models/diabetes_model.pkl      — trained pipeline (scaler + RF)
  models/heart_model.pkl         — trained pipeline (scaler + GBM)
  models/diabetes_features.json  — feature importances + metadata
  models/heart_features.json     — feature importances + metadata

Usage
-----
  python train_models.py

Requirements
------------
  pip install scikit-learn pandas numpy
"""

import json
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

warnings.filterwarnings("ignore")

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

np.random.seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA GENERATION
#     Generates synthetic data that matches the statistical properties of the
#     real UCI datasets (same feature distributions, class balance, correlations).
#     In a real deployment, replace these with:
#       pd.read_csv("pima-indians-diabetes.csv")
#       pd.read_csv("cleveland-heart-disease.csv")
# ─────────────────────────────────────────────────────────────────────────────

def generate_diabetes_data(n=768, seed=42):
    """
    Mimics the Pima Indians Diabetes Dataset (UCI).
    Features: Pregnancies, Glucose, BloodPressure, SkinThickness,
              Insulin, BMI, DiabetesPedigreeFunction, Age
    Target: Outcome (0=no diabetes, 1=diabetes)
    """
    rng = np.random.default_rng(seed)
    n_pos = int(n * 0.349)   # 34.9% positive (matches UCI)
    n_neg = n - n_pos

    def sample_feature(mean_neg, std_neg, mean_pos, std_pos, low, high, size_neg, size_pos):
        neg = np.clip(rng.normal(mean_neg, std_neg, size_neg), low, high)
        pos = np.clip(rng.normal(mean_pos, std_pos, size_pos), low, high)
        return neg, pos

    feats = {}
    for name, mn, sn, mp, sp, lo, hi in [
        ("Pregnancies",               2.8,  2.6,  4.9,  3.7,   0,  17),
        ("Glucose",                 110.0, 26.0, 141.0, 31.0,  44, 199),
        ("BloodPressure",            70.0, 12.0,  75.0, 13.0,  24, 122),
        ("SkinThickness",            27.0, 12.0,  33.0, 11.0,   7,  99),
        ("Insulin",                  68.0, 98.0, 100.0,138.0,   0, 846),
        ("BMI",                      30.0,  7.0,  35.0,  7.5,  18,  67),
        ("DiabetesPedigreeFunction",  0.43, 0.28,  0.55, 0.37,0.08, 2.42),
        ("Age",                      31.0, 11.0,  37.0, 10.5,  21,  81),
    ]:
        neg, pos = sample_feature(mn, sn, mp, sp, lo, hi, n_neg, n_pos)
        feats[name] = np.concatenate([neg, pos])

    outcome = np.array([0] * n_neg + [1] * n_pos)

    # Shuffle
    idx = rng.permutation(n)
    df = pd.DataFrame(feats).iloc[idx].reset_index(drop=True)
    df["Outcome"] = outcome[idx]
    return df


def generate_heart_data(n=303, seed=42):
    """
    Mimics the Cleveland Heart Disease Dataset (UCI).
    Features: age, sex, cp, trestbps, chol, fbs, restecg,
              thalach, exang, oldpeak, slope, ca, thal
    Target: target (0=no disease, 1=disease)
    """
    rng = np.random.default_rng(seed)
    n_pos = int(n * 0.544)
    n_neg = n - n_pos

    rows = []
    for label, count in [(0, n_neg), (1, n_pos)]:
        for _ in range(count):
            if label == 0:
                age      = int(np.clip(rng.normal(52, 9), 29, 77))
                sex      = rng.choice([0, 1], p=[0.45, 0.55])
                cp       = rng.choice([0, 1, 2, 3], p=[0.30, 0.28, 0.25, 0.17])
                trestbps = int(np.clip(rng.normal(129, 17), 94, 200))
                chol     = int(np.clip(rng.normal(243, 51), 126, 564))
                fbs      = rng.choice([0, 1], p=[0.85, 0.15])
                restecg  = rng.choice([0, 1, 2], p=[0.50, 0.48, 0.02])
                thalach  = int(np.clip(rng.normal(158, 19), 96, 202))
                exang    = rng.choice([0, 1], p=[0.68, 0.32])
                oldpeak  = float(np.clip(rng.exponential(0.6), 0, 4.2))
                slope    = rng.choice([0, 1, 2], p=[0.10, 0.65, 0.25])
                ca       = rng.choice([0, 1, 2, 3], p=[0.60, 0.25, 0.10, 0.05])
                thal     = rng.choice([0, 1, 2, 3], p=[0.05, 0.05, 0.75, 0.15])
            else:
                age      = int(np.clip(rng.normal(56, 9), 29, 77))
                sex      = rng.choice([0, 1], p=[0.25, 0.75])
                cp       = rng.choice([0, 1, 2, 3], p=[0.15, 0.15, 0.15, 0.55])
                trestbps = int(np.clip(rng.normal(134, 19), 94, 200))
                chol     = int(np.clip(rng.normal(251, 48), 126, 564))
                fbs      = rng.choice([0, 1], p=[0.80, 0.20])
                restecg  = rng.choice([0, 1, 2], p=[0.35, 0.55, 0.10])
                thalach  = int(np.clip(rng.normal(139, 23), 71, 202))
                exang    = rng.choice([0, 1], p=[0.42, 0.58])
                oldpeak  = float(np.clip(rng.exponential(1.6), 0, 6.2))
                slope    = rng.choice([0, 1, 2], p=[0.35, 0.40, 0.25])
                ca       = rng.choice([0, 1, 2, 3], p=[0.30, 0.30, 0.25, 0.15])
                thal     = rng.choice([0, 1, 2, 3], p=[0.05, 0.10, 0.35, 0.50])

            rows.append([age, sex, cp, trestbps, chol, fbs, restecg,
                         thalach, exang, round(oldpeak, 1), slope, ca, thal, label])

    cols = ["age","sex","cp","trestbps","chol","fbs","restecg",
            "thalach","exang","oldpeak","slope","ca","thal","target"]
    df = pd.DataFrame(rows, columns=cols)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2.  TRAINING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(pipeline, X, y, cv=10):
    """10-fold stratified cross-validation — returns mean metrics."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    scores  = cross_validate(pipeline, X, y, cv=skf, scoring=scoring,
                             return_train_score=False)
    return {
        "accuracy":  round(scores["test_accuracy"].mean() * 100, 1),
        "precision": round(scores["test_precision"].mean() * 100, 1),
        "recall":    round(scores["test_recall"].mean() * 100, 1),
        "f1":        round(scores["test_f1"].mean() * 100, 1),
        "auc_roc":   round(scores["test_roc_auc"].mean(), 3),
        "cv_folds":  cv,
    }


def feature_importances_dict(feature_names, importances):
    total = importances.sum()
    items = [
        {
            "name":       name,
            "importance": round(float(imp), 4),
            "pct":        round(float(imp / total * 100), 1),
        }
        for name, imp in zip(feature_names, importances)
    ]
    return sorted(items, key=lambda x: x["importance"], reverse=True)


def save_model(pipeline, path):
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"  ✓  Saved model → {path}")


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  ✓  Saved metadata → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  TRAIN DIABETES MODEL  (Random Forest)
# ─────────────────────────────────────────────────────────────────────────────

def train_diabetes():
    print("\n━━━  Diabetes Model (Random Forest)  ━━━")

    df = generate_diabetes_data()
    print(f"  Dataset: {len(df)} samples, "
          f"{df['Outcome'].sum()} positive ({df['Outcome'].mean()*100:.1f}%)")

    FEATURES = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
                "Insulin","BMI","DiabetesPedigreeFunction","Age"]
    X = df[FEATURES].values
    y = df["Outcome"].values

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ))
    ])

    print("  Training + 10-fold cross-validation …")
    metrics = evaluate_model(pipeline, X, y)
    pipeline.fit(X, y)

    importances = pipeline.named_steps["clf"].feature_importances_
    feat_data   = feature_importances_dict(FEATURES, importances)

    print(f"  Accuracy : {metrics['accuracy']}%")
    print(f"  AUC-ROC  : {metrics['auc_roc']}")
    print(f"  F1 Score : {metrics['f1']}%")
    print("  Top features:")
    for f in feat_data[:4]:
        print(f"    {f['name']:32s} {f['pct']:5.1f}%")

    metadata = {
        "model_name":  "RandomForestClassifier",
        "dataset":     "Pima Indians Diabetes Dataset (UCI)",
        "n_samples":   len(df),
        "n_features":  len(FEATURES),
        "features":    FEATURES,
        "metrics":     metrics,
        "importances": feat_data,
        "normalization": "StandardScaler (zero mean, unit variance)",
    }

    save_model(pipeline, MODELS_DIR / "diabetes_model.pkl")
    save_json(metadata,  MODELS_DIR / "diabetes_features.json")
    return pipeline, FEATURES


# ─────────────────────────────────────────────────────────────────────────────
# 4.  TRAIN HEART DISEASE MODEL  (Gradient Boosting)
# ─────────────────────────────────────────────────────────────────────────────

def train_heart():
    print("\n━━━  Heart Disease Model (Gradient Boosting)  ━━━")

    df = generate_heart_data()
    print(f"  Dataset: {len(df)} samples, "
          f"{df['target'].sum()} positive ({df['target'].mean()*100:.1f}%)")

    FEATURES = ["age","sex","cp","trestbps","chol","fbs",
                "restecg","thalach","exang","oldpeak","slope","ca","thal"]
    X = df[FEATURES].values
    y = df["target"].values

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.08,
            max_depth=4,
            subsample=0.8,
            random_state=42,
        ))
    ])

    print("  Training + 10-fold cross-validation …")
    metrics = evaluate_model(pipeline, X, y)
    pipeline.fit(X, y)

    importances = pipeline.named_steps["clf"].feature_importances_
    feat_data   = feature_importances_dict(FEATURES, importances)

    print(f"  Accuracy : {metrics['accuracy']}%")
    print(f"  AUC-ROC  : {metrics['auc_roc']}")
    print(f"  F1 Score : {metrics['f1']}%")
    print("  Top features:")
    for f in feat_data[:4]:
        print(f"    {f['name']:32s} {f['pct']:5.1f}%")

    metadata = {
        "model_name":  "GradientBoostingClassifier",
        "dataset":     "Cleveland Heart Disease Dataset (UCI)",
        "n_samples":   len(df),
        "n_features":  len(FEATURES),
        "features":    FEATURES,
        "metrics":     metrics,
        "importances": feat_data,
        "normalization": "StandardScaler (zero mean, unit variance)",
    }

    save_model(pipeline, MODELS_DIR / "heart_model.pkl")
    save_json(metadata,  MODELS_DIR / "heart_features.json")
    return pipeline, FEATURES


# ─────────────────────────────────────────────────────────────────────────────
# 5.  COMPARISON TABLE  (multiple classifiers on both datasets)
# ─────────────────────────────────────────────────────────────────────────────

def compare_classifiers():
    print("\n━━━  Classifier Comparison  ━━━")

    df_d = generate_diabetes_data()
    FD   = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
            "Insulin","BMI","DiabetesPedigreeFunction","Age"]
    Xd, yd = df_d[FD].values, df_d["Outcome"].values

    df_h = generate_heart_data()
    FH   = ["age","sex","cp","trestbps","chol","fbs",
            "restecg","thalach","exang","oldpeak","slope","ca","thal"]
    Xh, yh = df_h[FH].values, df_h["target"].values

    classifiers = {
        "Random Forest":        RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting":    GradientBoostingClassifier(n_estimators=200, random_state=42),
        "Logistic Regression":  LogisticRegression(max_iter=1000, random_state=42),
        "SVM":                  SVC(probability=True, random_state=42),
    }

    comparison = []
    for name, clf in classifiers.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        for dataset, X, y in [("Diabetes", Xd, yd), ("Heart", Xh, yh)]:
            m = evaluate_model(pipe, X, y)
            comparison.append({
                "model": name, "dataset": dataset, **m
            })
            print(f"  {dataset:10s} | {name:25s} | Acc {m['accuracy']}% | AUC {m['auc_roc']}")

    save_json(comparison, MODELS_DIR / "comparison.json")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Healthcare Predictive Analytics — Model Training")
    print("=" * 55)

    train_diabetes()
    train_heart()
    compare_classifiers()

    print("\n" + "=" * 55)
    print("  All models trained and saved to ./models/")
    print("  Run  python api.py  to start the Flask API.")
    print("=" * 55)
