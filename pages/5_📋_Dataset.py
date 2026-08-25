import streamlit as st
from utils.common import load_data, apply_filters

st.title("📋 Dataset Explorer")

df = load_data()
if df is None:
    st.error("Dataset missing.")
    st.stop()

filtered = apply_filters(df)

st.write(f"Showing **{len(filtered):,}** records.")
st.dataframe(filtered, use_container_width=True, height=650)

st.download_button(
    "⬇️ Download filtered CSV",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="filtered_global_air_pollution.csv",
    mime="text/csv",
)
