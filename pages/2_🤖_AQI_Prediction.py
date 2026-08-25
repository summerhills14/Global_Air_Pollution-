import streamlit as st
import numpy as np
from utils.common import (
    load_data, build_feature_row, load_object, load_keras_model,
    aqi_label, NUMERIC_FEATURES
)

st.title("🤖 AQI Prediction")
st.caption("Uses the trained Random Forest regressor saved by the Kaggle notebook.")

df = load_data()
if df is None:
    st.error("Dataset missing.")
    st.stop()

st.info(
    "Enter environmental, pollutant and urban-activity values. "
    "The app applies the notebook's Month sine/cosine features and categorical label encoders."
)

with st.form("aqi_prediction_form"):
    st.subheader("🌦️ Weather")
    c1, c2, c3 = st.columns(3)
    with c1:
        date = st.date_input("Date", value=df["Date"].max().date())
        temperature = st.number_input("Temperature (°C)", value=float(df["Temperature_C"].median()))
        humidity = st.number_input("Humidity (%)", value=float(df["Humidity_%"].median()), min_value=0.0, max_value=100.0)
    with c2:
        wind = st.number_input("Wind Speed (m/s)", value=float(df["WindSpeed_mps"].median()), min_value=0.0)
        pressure = st.number_input("Pressure (hPa)", value=float(df["Pressure_hPa"].median()))
        rainfall = st.number_input("Rainfall (mm)", value=float(df["Rainfall_mm"].median()), min_value=0.0)
    with c3:
        visibility = st.number_input("Visibility (km)", value=float(df["Visibility_km"].median()), min_value=0.0)
        uv = st.number_input("UV Index", value=float(df["UV_Index"].median()), min_value=0.0)
        season = st.selectbox("Season", sorted(df["Season"].unique()))

    st.subheader("🧪 Pollutants")
    c1, c2, c3 = st.columns(3)
    with c1:
        pm25 = st.number_input("PM2.5", value=float(df["PM2_5"].median()), min_value=0.0)
        pm10 = st.number_input("PM10", value=float(df["PM10"].median()), min_value=0.0)
        no2 = st.number_input("NO2", value=float(df["NO2"].median()), min_value=0.0)
    with c2:
        so2 = st.number_input("SO2", value=float(df["SO2"].median()), min_value=0.0)
        co = st.number_input("CO", value=float(df["CO"].median()), min_value=0.0)
        ozone = st.number_input("Ozone", value=float(df["Ozone"].median()), min_value=0.0)
    with c3:
        nh3 = st.number_input("NH3", value=float(df["NH3"].median()), min_value=0.0)
        aod = st.number_input("Aerosol Optical Depth", value=float(df["Aerosol_Optical_Depth"].median()), min_value=0.0)

    st.subheader("🏙️ Urban Activity")
    c1, c2 = st.columns(2)
    with c1:
        traffic = st.number_input("Traffic Density", value=float(df["Traffic_Density"].median()), min_value=0.0, max_value=100.0)
        industry = st.number_input("Industrial Activity", value=float(df["Industrial_Activity"].median()), min_value=0.0, max_value=100.0)
    with c2:
        green = st.number_input("Green Cover (%)", value=float(df["Green_Cover_%"].median()), min_value=0.0, max_value=100.0)
        population = st.number_input("Population Density", value=float(df["Population_Density"].median()), min_value=0.0)

    st.subheader("🌍 Location")
    c1, c2 = st.columns(2)
    with c1:
        country = st.selectbox("Country", sorted(df["Country"].unique()))
    with c2:
        city_options = sorted(df.loc[df["Country"] == country, "City"].unique())
        city = st.selectbox("City", city_options)

    submitted = st.form_submit_button("🔮 Predict AQI", type="primary", use_container_width=True)

if submitted:
    try:
        values = {
            "Date": str(date),
            "Month": date.month,
            "Temperature_C": temperature,
            "Humidity_%": humidity,
            "WindSpeed_mps": wind,
            "Pressure_hPa": pressure,
            "Rainfall_mm": rainfall,
            "Visibility_km": visibility,
            "UV_Index": uv,
            "PM2_5": pm25,
            "PM10": pm10,
            "NO2": no2,
            "SO2": so2,
            "CO": co,
            "Ozone": ozone,
            "NH3": nh3,
            "Aerosol_Optical_Depth": aod,
            "Traffic_Density": traffic,
            "Industrial_Activity": industry,
            "Green_Cover_%": green,
            "Population_Density": population,
            "Season": season,
            "Country": country,
            "City": city,
        }

        x = build_feature_row(values)
        scaler = load_object("scaler_reg.pkl")
        model = load_object("best_aqi_regressor_rf.pkl")

        prediction_value = float(model.predict(scaler.transform(x))[0])
        category = aqi_label(prediction_value)

        st.divider()
        c1, c2 = st.columns(2)
        c1.metric("Predicted AQI", f"{prediction_value:.2f}")
        c2.metric("Estimated Category", category)

        if prediction_value <= 100:
            st.success("Air quality is in the lower-risk range for this prediction.")
        elif prediction_value <= 200:
            st.warning("This prediction indicates elevated/unhealthy air quality.")
        else:
            st.error("This prediction indicates very unhealthy or hazardous air quality.")

    except Exception as e:
        st.error("Prediction could not be generated.")
        st.exception(e)
