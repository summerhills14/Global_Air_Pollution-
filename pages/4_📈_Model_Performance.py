import streamlit as st
import pandas as pd
import numpy as np
from utils.common import load_data

st.title("📈 Model Performance")
st.caption("Comparison structure follows the Kaggle notebook.")

df = load_data()
if df is None:
    st.error("Dataset missing.")
    st.stop()

st.subheader("Models in the Kaggle notebook")

regression = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest",
        "Gradient Boosting",
        "XGBoost",
        "Deep Learning (ANN)"
    ],
    "Task": ["Regression"] * 5,
    "Metrics": ["R², MAE, RMSE"] * 5
})

classification = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "XGBoost",
        "Deep Learning (ANN)"
    ],
    "Task": ["Classification"] * 4,
    "Metrics": ["Accuracy, Weighted F1"] * 4
})

st.dataframe(regression, use_container_width=True)
st.dataframe(classification, use_container_width=True)

st.info(
    "The numeric metric values are intentionally not hard-coded here. "
    "Run `train_models.py` in PyCharm to recreate the notebook's training/evaluation "
    "and write `model_metrics.json`; the page will then display the actual saved results."
)

from pathlib import Path
import json
metrics_path = Path(__file__).resolve().parents[1] / "models" / "model_metrics.json"

if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    st.subheader("Actual saved metrics")
    if "regression" in metrics:
        st.write("### Regression")
        st.dataframe(pd.DataFrame(metrics["regression"]).T, use_container_width=True)
    if "classification" in metrics:
        st.write("### Classification")
        st.dataframe(pd.DataFrame(metrics["classification"]).T, use_container_width=True)

st.subheader("Dataset")
c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Features", f"{df.shape[1]}")
c3.metric("AQI Categories", f"{df['AQI_Category'].nunique()}")
