import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(page_title="Temperatuuranalyse 2023", layout="wide")

# ====== BESTANDSLOCATIES ======
CITY_FILES = {
    "Amsterdam": "preprocessed_data/data_daily_Amsterdam.csv",
    "Londen": "preprocessed_data/data_daily_Londen.csv",
    "Madrid": "preprocessed_data/data_daily_Madrid.csv"
}
MAP_FILE = "preprocessed_data/temperature_persistence_map.csv"

@st.cache_data
def load_city(file):
    df = pd.read_csv(file)
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data
def load_map(file):
    return pd.read_csv(file)

# ====== SIDEBAR ======
st.sidebar.title("🌡️ Analyse van Temperatuur 2023")
view = st.sidebar.radio(
    "Kies weergave:",
    [
        "Tijdreeks per stad",
        "Maandelijkse spreiding",
        "Persistentie per stad",
        "Persistentiekaart (grenzen)"
    ]
)

# ====== PAGINA'S ======
if view == "Tijdreeks per stad":
    st.title("📈 Dagelijkse gemiddelde temperatuur")
    city = st.selectbox("Kies stad:", list(CITY_FILES.keys()))
    df = load_city(CITY_FILES[city])
    fig = px.line(df, x="date", y="t2m_daily_mean_C", title=f"Temperatuurverloop in {city} (2023)", labels={"t2m_daily_mean_C": "°C"})
    st.plotly_chart(fig, use_container_width=True)

elif view == "Maandelijkse spreiding":
    st.title("📊 Maandelijkse temperatuurspreiding per stad")
    city = st.selectbox("Kies stad:", list(CITY_FILES.keys()))
    df = load_city(CITY_FILES[city])
    df["month"] = df["date"].dt.month
    fig = px.box(df, x="month", y="t2m_daily_mean_C", title=f"Temperatuurspreiding per maand ({city})", labels={"month": "Maand", "t2m_daily_mean_C": "°C"})
    st.plotly_chart(fig, use_container_width=True)

elif view == "Persistentie per stad":
    st.title("🔁 Temperatuurpersistentie (geheugen) per stad")
    city = st.selectbox("Kies stad:", list(CITY_FILES.keys()))
    df = load_city(CITY_FILES[city])
    df["month"] = df["date"].dt.month
    df["anomaly"] = df["t2m_daily_mean_C"] - df.groupby("month")["t2m_daily_mean_C"].transform("mean")

    max_lag = 30
    acf = [df["anomaly"].autocorr(lag=i) for i in range(1, max_lag + 1)]
    acf_df = pd.DataFrame({"lag_days": range(1, max_lag + 1), "acf": acf})
    memory_days = next((lag for lag, c in zip(acf_df["lag_days"], acf_df["acf"]) if c < 0.2), None)

    fig = px.bar(acf_df, x="lag_days", y="acf", title=f"Autocorrelatie van temperatuur-anomalie – {city}")
    fig.add_hline(y=0.2, line_dash="dot", line_color="red", annotation_text="grens 0.2")
    st.plotly_chart(fig, use_container_width=True)

    if memory_days:
        st.markdown(f"📈 **Geheugenlengte:** ± {memory_days} dagen")
    else:
        st.markdown("⚪ Geen duidelijke grens (correlatie blijft hoog of fluctueert).")

elif view == "Persistentiekaart (grenzen)":
    st.title("🗺️ Ruimtelijke grenzen van temperatuurpersistentie")
    df_map = load_map(MAP_FILE)
    fig = px.scatter_mapbox(
        df_map,
        lat="latitude",
        lon="longitude",
        color="acf_lag1",
        color_continuous_scale="RdBu_r",
        title="Dag-tot-dag persistentie (autocorrelatie lag=1)",
        mapbox_style="carto-positron",
        zoom=3,
        height=650
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    **Interpretatie:**
    - 🔴 Hogere waarden = hoge persistentie (temperatuur verandert traag, typisch maritiem).
    - 🔵 Lagere waarden = lage persistentie (temperatuur wisselt snel, vaak continentaal of bergachtig).
    """)

