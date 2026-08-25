import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.common import load_data, apply_filters

st.title("📊 Exploratory Data Analysis")
st.caption("EDA views based on the Kaggle notebook.")

df = load_data()
if df is None:
    st.error("Dataset missing. Add it to data/synthetic_global_air_pollution.csv.")
    st.stop()

df = apply_filters(df)

tab1, tab2, tab3, tab4 = st.tabs([
    "AQI Distribution", "Geography & Seasons", "Pollutants", "Correlation"
])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df["AQI"], bins=50, kde=True, ax=ax)
        ax.set_title("Distribution of AQI")
        st.pyplot(fig)
        plt.close(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(8, 4))
        order = df["AQI_Category"].value_counts().index
        sns.countplot(y="AQI_Category", data=df, order=order, ax=ax)
        ax.set_title("AQI Category Distribution")
        st.pyplot(fig)
        plt.close(fig)

with tab2:
    country_aqi = df.groupby("Country")["AQI"].mean().sort_values()
    season_aqi = df.groupby("Season")["AQI"].mean().sort_values()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Average AQI by Country")
        st.bar_chart(country_aqi)
    with c2:
        st.subheader("Average AQI by Season")
        st.bar_chart(season_aqi)

    st.subheader("Most Polluted Cities")
    city_aqi = df.groupby("City")["AQI"].agg(["mean", "count"]).sort_values("mean", ascending=False)
    city_aqi.columns = ["Average AQI", "Records"]
    st.dataframe(city_aqi.head(15), use_container_width=True)

with tab3:
    pollutants = ["PM2_5", "PM10", "NO2", "SO2", "CO", "Ozone"]
    available = [p for p in pollutants if p in df.columns]
    sample = df.sample(min(3000, len(df)), random_state=42)

    for start in range(0, len(available), 3):
        cols = st.columns(3)
        for ui_col, pollutant in zip(cols, available[start:start+3]):
            with ui_col:
                fig, ax = plt.subplots(figsize=(5, 3.5))
                sns.scatterplot(
                    x=pollutant, y="AQI", data=sample,
                    alpha=.3, ax=ax
                )
                ax.set_title(f"{pollutant} vs AQI")
                st.pyplot(fig)
                plt.close(fig)

with tab4:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Top correlations with AQI")
    st.dataframe(
        corr["AQI"].sort_values(ascending=False).head(10).to_frame("Correlation"),
        use_container_width=True
    )

st.divider()
st.subheader("Missing Values")
missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)
st.dataframe(missing.to_frame("Missing Count"), use_container_width=True)
