"""
Author: Vishal Bhushan
Student Outcome Predictor - Streamlit front end for ML Assignment 2.

The app never trains anything. It loads the five pipelines saved by
train_models.py, scores the CSV you upload, and shows the metrics,
the confusion matrix and the classification report.
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "model"
SAMPLE_CSV = APP_DIR / "test_data.csv"

st.set_page_config(page_title="Student Outcome Predictor",
                   page_icon="🎓", layout="wide")


# ---------------------------------------------------------------- loading
@st.cache_resource
def load_artifacts():
    meta = json.loads((MODEL_DIR / "metadata.json").read_text())
    models = {name: joblib.load(MODEL_DIR / fname)
              for name, fname in meta["models"].items()}
    return meta, models


def read_upload(uploaded):
    """Accept comma or semicolon separated files (the raw UCI file uses ';')."""
    frame = pd.read_csv(uploaded, sep=None, engine="python")
    frame.columns = [c.strip() for c in frame.columns]
    return frame


def evaluate(model, X, y):
    pred = model.predict(X)
    proba = model.predict_proba(X)
    scores = {
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, average="weighted", zero_division=0),
        "Recall": recall_score(y, pred, average="weighted", zero_division=0),
        "F1": f1_score(y, pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }
    # AUC needs every class present in the uploaded labels.
    try:
        scores["AUC"] = roc_auc_score(y, proba, multi_class="ovr",
                                      average="weighted", labels=model.classes_)
    except ValueError:
        scores["AUC"] = float("nan")
    return pred, scores


def draw_confusion(y_true, y_pred, labels, title):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted outcome")
    ax.set_ylabel("Actual outcome")
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- layout
meta, models = load_artifacts()
target = meta["target"]
features = meta["features"]
class_labels = meta["classes"]

st.title("🎓 Student Outcome Predictor")
st.caption("Will a student drop out, stay enrolled, or graduate? "
           "Five pre-trained classifiers, evaluated on the data you upload. "
           f"Dataset: {meta['dataset']}.")

with st.sidebar:
    st.header("1. Upload test data")
    uploaded = st.file_uploader("CSV with the 36 features and the "
                                f"'{target}' column", type=["csv"])
    if SAMPLE_CSV.exists():
        st.download_button("Download sample test_data.csv",
                           SAMPLE_CSV.read_bytes(), "test_data.csv",
                           mime="text/csv")
    st.header("2. Choose a model")
    chosen = st.selectbox("Classifier", list(models.keys()))
    st.divider()
    st.markdown(f"Models were trained on **{meta['train_rows']}** rows "
                f"and are only *used* here, never retrained.")

if uploaded is None:
    st.info("Upload a CSV in the sidebar to start. You can grab the sample "
            "test file with the download button.")
    st.stop()

data = read_upload(uploaded)
missing = [c for c in features + [target] if c not in data.columns]
if missing:
    st.error(f"The file is missing {len(missing)} required column(s): "
             + ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else ""))
    st.stop()

X_new, y_new = data[features], data[target].astype(str).str.strip()
unknown = sorted(set(y_new) - set(class_labels))
if unknown:
    st.error(f"Unexpected values in '{target}': {unknown}. "
             f"Allowed: {class_labels}")
    st.stop()

with st.expander(f"Preview of uploaded data ({len(data)} rows)"):
    st.dataframe(data.head(20), width="stretch")
    st.bar_chart(y_new.value_counts())

tab_single, tab_compare = st.tabs([f"🔍 {chosen}", "📊 Compare all five"])

with tab_single:
    preds, scores = evaluate(models[chosen], X_new, y_new)
    cols = st.columns(6)
    for col, key in zip(cols, ["Accuracy", "AUC", "Precision",
                               "Recall", "F1", "MCC"]):
        col.metric(key, f"{scores[key]:.3f}")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Confusion matrix")
        st.pyplot(draw_confusion(y_new, preds, class_labels, chosen))
    with right:
        st.subheader("Classification report")
        report = classification_report(y_new, preds, labels=class_labels,
                                       output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).T.round(3), width="stretch")

    scored = data.copy()
    scored["Predicted_" + target] = preds
    st.download_button("Download predictions as CSV",
                       scored.to_csv(index=False).encode(),
                       f"predictions_{chosen.lower().replace(' ', '_')}.csv",
                       mime="text/csv")

with tab_compare:
    rows = []
    for name, mdl in models.items():
        _, sc = evaluate(mdl, X_new, y_new)
        rows.append({"Model": name, **sc})
    board = (pd.DataFrame(rows)
             .set_index("Model")[["Accuracy", "AUC", "Precision",
                                  "Recall", "F1", "MCC"]])
    st.dataframe(board.style.format("{:.4f}").highlight_max(color="#c6efce"),
                 width="stretch")
    best = board["F1"].idxmax()
    st.success(f"Best weighted F1 on this file: **{best}** "
               f"({board.loc[best, 'F1']:.4f})")
    st.bar_chart(board[["Accuracy", "F1", "MCC"]], stack=False)
