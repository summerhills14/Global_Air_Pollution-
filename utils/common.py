from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "synthetic_global_air_pollution.csv"
MODEL_DIR = ROOT / "models"

NUMERIC_FEATURES = [
    "Temperature_C", "Humidity_%", "WindSpeed_mps", "Pressure_hPa",
    "Rainfall_mm", "Visibility_km", "UV_Index", "PM2_5", "PM10",
    "NO2", "SO2", "CO", "Ozone", "NH3", "Aerosol_Optical_Depth",
    "Traffic_Density", "Industrial_Activity", "Green_Cover_%",
    "Population_Density"
]

CATEGORICAL_FEATURES = ["Season", "Country", "City"]
ENGINEERED_FEATURES = ["Month_sin", "Month_cos", "DayOfYear"]
FEATURE_COLS = NUMERIC_FEATURES + [c + "_enc" for c in CATEGORICAL_FEATURES] + ENGINEERED_FEATURES

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        return None
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df

def apply_filters(df):
    st.sidebar.header("Filters")

    result = df.copy()

    countries = st.sidebar.multiselect(
        "Country",
        sorted(result["Country"].dropna().unique()),
        placeholder="All countries"
    )
    if countries:
        result = result[result["Country"].isin(countries)]

    cities = st.sidebar.multiselect(
        "City",
        sorted(result["City"].dropna().unique()),
        placeholder="All cities"
    )
    if cities:
        result = result[result["City"].isin(cities)]

    seasons = st.sidebar.multiselect(
        "Season",
        sorted(result["Season"].dropna().unique()),
        placeholder="All seasons"
    )
    if seasons:
        result = result[result["Season"].isin(seasons)]

    categories = st.sidebar.multiselect(
        "AQI Category",
        sorted(result["AQI_Category"].dropna().unique()),
        placeholder="All categories"
    )
    if categories:
        result = result[result["AQI_Category"].isin(categories)]

    if result["Date"].notna().any():
        min_date = result["Date"].min().date()
        max_date = result["Date"].max().date()
        date_range = st.sidebar.date_input(
            "Date range", value=(min_date, max_date),
            min_value=min_date, max_value=max_date
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            result = result[
                (result["Date"].dt.date >= date_range[0]) &
                (result["Date"].dt.date <= date_range[1])
            ]

    return result

def build_feature_row(values):
    """Replicates the Kaggle notebook feature engineering and label encoding."""
    row = pd.DataFrame([values])

    row["Date"] = pd.to_datetime(row["Date"], errors="coerce")
    row["Month_sin"] = np.sin(2 * np.pi * row["Month"] / 12)
    row["Month_cos"] = np.cos(2 * np.pi * row["Month"] / 12)
    row["DayOfYear"] = row["Date"].dt.dayofyear

    encoders = load_object("label_encoders.pkl")
    for col in CATEGORICAL_FEATURES:
        le = encoders[col]
        value = row.loc[0, col]
        if value not in le.classes_:
            raise ValueError(
                f"{col}='{value}' was not present in the training data."
            )
        row[col + "_enc"] = le.transform([value])[0]

    # Ensure exact notebook feature order.
    return row[FEATURE_COLS]

@st.cache_resource
def load_object(filename):
    path = MODEL_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {filename}. Run train_models.py first."
        )
    return joblib.load(path)

@st.cache_resource
def load_keras_model(filename):
    from tensorflow import keras
    path = MODEL_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {filename}. Run train_models.py first."
        )
    return keras.models.load_model(path)

def inject_css():
    st.markdown("""
    <style>
    .block-container {padding-top: 1.8rem; padding-bottom: 2rem;}
    [data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.25);
        padding: 12px;
        border-radius: 12px;
    }
    .app-card {
        padding: 1.2rem;
        border-radius: 16px;
        border: 1px solid rgba(128,128,128,.25);
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

def aqi_label(aqi):
    if aqi <= 50: return "Good"
    if aqi <= 100: return "Moderate"
    if aqi <= 150: return "Unhealthy for Sensitive Groups"
    if aqi <= 200: return "Unhealthy"
    if aqi <= 300: return "Very Unhealthy"
    return "Hazardous"
