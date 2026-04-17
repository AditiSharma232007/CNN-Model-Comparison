from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt


st.set_page_config(page_title="CNN Model Comparison", layout="wide")

RESULTS_DIR = Path("artifacts")
SUMMARY_FILE = RESULTS_DIR / "summary.csv"


@st.cache_data
def load_summary() -> pd.DataFrame:
    if not SUMMARY_FILE.exists():
        return pd.DataFrame()
    return pd.read_csv(SUMMARY_FILE)


def render_metric_chart(df: pd.DataFrame, metric: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    chart_df = df.copy()
    chart_df["label"] = chart_df["model_name"] + " (" + chart_df["mode"] + ")"
    sns.barplot(data=chart_df, x=metric, y="label", hue="mode", dodge=False, ax=ax, palette="crest")
    ax.set_title(f"{metric.replace('_', ' ').title()} Comparison")
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_ylabel("")
    st.pyplot(fig, use_container_width=True)


def render_confusion_matrix(matrix_file: Path) -> None:
    payload = json.loads(matrix_file.read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(payload["matrix"], annot=True, fmt="d", cmap="Blues", xticklabels=payload["labels"], yticklabels=payload["labels"], ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig, use_container_width=True)


st.title("CNN Model Comparison Dashboard")
st.caption("Train multiple CNNs, compare outcomes, and publish this dashboard on the cloud.")

summary = load_summary()
if summary.empty:
    st.warning("No training results found yet. Run `python train.py --data-dir data/your_dataset` first.")
    st.stop()

best_row = summary.sort_values(by=["accuracy", "f1_score"], ascending=False).iloc[0]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Best Model", f"{best_row['model_name']} ({best_row['mode']})")
col2.metric("Accuracy", f"{best_row['accuracy']:.4f}")
col3.metric("F1 Score", f"{best_row['f1_score']:.4f}")
col4.metric("Train Time (sec)", f"{best_row['training_time_sec']:.1f}")

st.subheader("Experiment Summary")
st.dataframe(summary, use_container_width=True)

metric = st.selectbox("Select metric", ["accuracy", "precision", "recall", "f1_score", "training_time_sec"])
render_metric_chart(summary, metric)

st.subheader("Detailed View")
labels = [f"{row.model_name} ({row.mode})" for row in summary.itertuples()]
selected_label = st.selectbox("Select a trained model", labels)
selected = summary.iloc[labels.index(selected_label)]

st.json(
    {
        "model_name": selected["model_name"],
        "mode": selected["mode"],
        "accuracy": float(selected["accuracy"]),
        "precision": float(selected["precision"]),
        "recall": float(selected["recall"]),
        "f1_score": float(selected["f1_score"]),
        "training_time_sec": float(selected["training_time_sec"]),
        "checkpoint_path": selected["checkpoint_path"],
    }
)

confusion_file = Path(selected["confusion_matrix_path"])
if confusion_file.exists():
    render_confusion_matrix(confusion_file)

report_file = Path(selected["report_path"])
if report_file.exists():
    st.subheader("Classification Report")
    st.json(json.loads(report_file.read_text(encoding="utf-8")))
