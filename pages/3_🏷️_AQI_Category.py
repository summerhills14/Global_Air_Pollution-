import streamlit as st
from utils.common import load_data, build_feature_row, load_object

st.title("🏷️ AQI Category Prediction")
st.caption("Uses the trained Random Forest classifier saved by the Kaggle notebook.")

df = load_data()
if df is None:
    st.error("Dataset missing.")
    st.stop()

st.write("Use the same environmental inputs as the AQI predictor.")

with st.form("classification_form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        date = st.date_input("Date", value=df["Date"].max().date())
        season = st.selectbox("Season", sorted(df["Season"].unique()))
        country = st.selectbox("Country", sorted(df["Country"].unique()))
        city_options = sorted(df.loc[df["Country"] == country, "City"].unique())
        city = st.selectbox("City", city_options)

    with c2:
        temperature = st.number_input("Temperature (°C)", value=float(df["Temperature_C"].median()))
        humidity = st.number_input("Humidity (%)", value=float(df["Humidity_%"].median()))
        wind = st.number_input("Wind Speed (m/s)", value=float(df["WindSpeed_mps"].median()))
        pressure = st.number_input("Pressure (hPa)", value=float(df["Pressure_hPa"].median()))
        rainfall = st.number_input("Rainfall (mm)", value=float(df["Rainfall_mm"].median()))
        visibility = st.number_input("Visibility (km)", value=float(df["Visibility_km"].median()))
        uv = st.number_input("UV Index", value=float(df["UV_Index"].median()))

    with c3:
        pm25 = st.number_input("PM2.5", value=float(df["PM2_5"].median()))
        pm10 = st.number_input("PM10", value=float(df["PM10"].median()))
        no2 = st.number_input("NO2", value=float(df["NO2"].median()))
        so2 = st.number_input("SO2", value=float(df["SO2"].median()))
        co = st.number_input("CO", value=float(df["CO"].median()))
        ozone = st.number_input("Ozone", value=float(df["Ozone"].median()))
        nh3 = st.number_input("NH3", value=float(df["NH3"].median()))
        aod = st.number_input("Aerosol Optical Depth", value=float(df["Aerosol_Optical_Depth"].median()))

    st.subheader("Urban Activity")
    c1, c2, c3 = st.columns(3)
    with c1:
        traffic = st.number_input("Traffic Density", value=float(df["Traffic_Density"].median()))
    with c2:
        industry = st.number_input("Industrial Activity", value=float(df["Industrial_Activity"].median()))
    with c3:
        green = st.number_input("Green Cover (%)", value=float(df["Green_Cover_%"].median()))
    population = st.number_input("Population Density", value=float(df["Population_Density"].median()))

    submitted = st.form_submit_button("🏷️ Predict AQI Category", type="primary", use_container_width=True)

if submitted:
    try:
        values = {
            "Date": str(date), "Month": date.month,
            "Temperature_C": temperature, "Humidity_%": humidity,
            "WindSpeed_mps": wind, "Pressure_hPa": pressure,
            "Rainfall_mm": rainfall, "Visibility_km": visibility,
            "UV_Index": uv, "PM2_5": pm25, "PM10": pm10, "NO2": no2,
            "SO2": so2, "CO": co, "Ozone": ozone, "NH3": nh3,
            "Aerosol_Optical_Depth": aod, "Traffic_Density": traffic,
            "Industrial_Activity": industry, "Green_Cover_%": green,
            "Population_Density": population, "Season": season,
            "Country": country, "City": city,
        }

        x = build_feature_row(values)
        scaler = load_object("scaler_clf.pkl")
        model = load_object("best_aqi_classifier_rf.pkl")
        target_encoder = load_object("target_encoder.pkl")

        pred_encoded = model.predict(scaler.transform(x))
        category = target_encoder.inverse_transform(pred_encoded)[0]

        st.divider()
        st.success(f"### Predicted AQI Category: **{category}**")

    except Exception as e:
        st.error("Classification prediction could not be generated.")
        st.exception(e)
