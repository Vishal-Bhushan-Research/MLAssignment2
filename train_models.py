"""
Author: Vishal Bhushan
train_models.py
---------------
Trains five classifiers on the UCI "Predict Students' Dropout and Academic
Success" dataset and saves everything the Streamlit app needs.

Usage:   python train_models.py

Outputs
  model/<name>.joblib    one fitted sklearn Pipeline per classifier
  model/metadata.json    feature order, class labels, split info
  model/metrics.csv      hold-out metrics for every model (for the README)
  test_data.csv          the 20% hold-out split (features + Status column)
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
RAW_FILE = ROOT / "data" / "students_dropout.csv"
MODEL_DIR = ROOT / "model"
TARGET = "Status"          # Dropout / Enrolled / Graduate
SEED = 2026                # fixed so results are reproducible
TEST_SHARE = 0.20


def load_dataset():
    # The UCI file is semicolon separated.
    frame = pd.read_csv(RAW_FILE, sep=";")
    frame.columns = [c.strip() for c in frame.columns]
    return frame


def build_classifiers():
    """Each model is a Pipeline so scaling travels with the saved model.

    Scaling matters for LR, KNN and SVM (distance / gradient based).
    Trees and Gaussian NB do not need it, so they get the raw features.
    """
    return {
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, C=0.5)),
        ]),
        "Decision Tree": Pipeline([
            ("clf", DecisionTreeClassifier(max_depth=7, min_samples_leaf=10,
                                           random_state=SEED)),
        ]),
        "KNN": Pipeline([
            ("scale", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=17, weights="distance")),
        ]),
        "Naive Bayes": Pipeline([
            ("clf", GaussianNB()),
        ]),
        "Support Vector Machine": Pipeline([
            ("scale", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=1.0, probability=True,
                        random_state=SEED)),
        ]),
    }


def score_model(model, X, y):
    """The six metrics required by the assignment.

    This is a 3-class problem, so precision / recall / F1 use the
    'weighted' average (each class weighted by its support) and AUC is
    computed one-vs-rest on the predicted probabilities.
    """
    predicted = model.predict(X)
    proba = model.predict_proba(X)
    return {
        "Accuracy": accuracy_score(y, predicted),
        "AUC": roc_auc_score(y, proba, multi_class="ovr", average="weighted",
                             labels=model.classes_),
        "Precision": precision_score(y, predicted, average="weighted",
                                     zero_division=0),
        "Recall": recall_score(y, predicted, average="weighted"),
        "F1": f1_score(y, predicted, average="weighted"),
        "MCC": matthews_corrcoef(y, predicted),
    }


def file_name(model_name):
    return model_name.lower().replace(" ", "_") + ".joblib"


def main():
    students = load_dataset()
    features = [c for c in students.columns if c != TARGET]
    print(f"Dataset: {students.shape[0]} rows, {len(features)} features")
    print(students[TARGET].value_counts().to_string(), "\n")

    X_train, X_test, y_train, y_test = train_test_split(
        students[features], students[TARGET],
        test_size=TEST_SHARE, stratify=students[TARGET], random_state=SEED)

    MODEL_DIR.mkdir(exist_ok=True)
    results = []
    for name, model in build_classifiers().items():
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_DIR / file_name(name))
        row = {"Model": name, **score_model(model, X_test, y_test)}
        results.append(row)
        print(f"{name:<24} " + "  ".join(
            f"{k}={v:.4f}" for k, v in row.items() if k != "Model"))

    table = pd.DataFrame(results).round(4)
    table.to_csv(MODEL_DIR / "metrics.csv", index=False)

    # The hold-out rows become the file that is uploaded to the app.
    test_frame = X_test.copy()
    test_frame[TARGET] = y_test
    test_frame.to_csv(ROOT / "test_data.csv", index=False)

    meta = {
        "dataset": "UCI Predict Students' Dropout and Academic Success (id 697)",
        "target": TARGET,
        "features": features,
        "classes": sorted(students[TARGET].unique().tolist()),
        "models": {name: file_name(name) for name in build_classifiers()},
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "seed": SEED,
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))

    print("\nComparison table (markdown):\n")
    print("| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |")
    print("|---|---|---|---|---|---|---|")
    for _, r in table.iterrows():
        print(f"| {r.Model} | {r.Accuracy:.4f} | {r.AUC:.4f} | "
              f"{r.Precision:.4f} | {r.Recall:.4f} | {r.F1:.4f} | {r.MCC:.4f} |")


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
