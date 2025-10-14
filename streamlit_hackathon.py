import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os

st.set_page_config(page_title="Temperatuuranalyse 2023", layout="wide")

# ====== BESTANDSLOCATIES ======
# Aangepast: CSV's staan in de root van je repo
CITY_FILES = {
    "Amsterdam": "data_daily_Amsterdam.csv",
    "Londen": "data_daily_Londen.csv",
    "Madrid": "data_daily_Madrid.csv"
}
MAP_FILE = "temperature_persistence_map.csv"

# ====== FUNCTIES ======
@st.cache_data
def load_city(file):
    df = pd.read_csv(file)
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data
def load_map(file):
    df = pd.read_csv(file)
    return df

# ====== SIDEBAR ======
st.sidebar.title("Analyse van Temperatuur 2023")
view = st.sidebar.radio(
    "Kies weergave:",
    [
        "Tijdreeks per stad",
        "Persistentie per stad",
        "Persistentiekaart (grenzen)"
    ]
)

# ====== PAGINA'S ======
import plotly.express as px
import streamlit as st

if view == "Tijdreeks per stad":
    st.title("📈 Dagelijkse gemiddelde temperatuur per maand")

    # Stad selecteren
    city = st.selectbox("Kies stad:", list(CITY_FILES.keys()))
    df = load_city(CITY_FILES[city])

    # Maand toevoegen
    df["month"] = df["date"].dt.month
    df["month_name"] = df["date"].dt.strftime("%b")  # Jan, Feb, etc.

    # Slider voor periode-selectie
    min_month, max_month = st.slider(
        "Selecteer maanden om te tonen",
        min_value=1,
        max_value=12,
        value=(1, 12)
    )
    df_filtered = df[(df["month"] >= min_month) & (df["month"] <= max_month)]

    # Plot met kleurverloop op temperatuur
    fig = px.scatter(
        df_filtered,
        x="month_name",
        y="t2m_daily_mean_C",
        color="t2m_daily_mean_C",
        color_continuous_scale="RdBu_r",
        title=f"Temperatuurverloop in {city}",
        labels={"t2m_daily_mean_C": "Temperatuur (°C)", "month_name": "Maand"},
        hover_data={"date": True, "t2m_daily_mean_C": True, "month_name": False}
    )

    # Voeg lijn toe
    fig.update_traces(mode="lines+markers", line=dict(width=3), marker=dict(size=5))

    # X-as correct sorteren (maanden chronologisch)
    fig.update_xaxes(categoryorder="array",
                     categoryarray=["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])

    st.plotly_chart(fig, use_container_width=True)


elif view == "Persistentie per stad":
    st.title("🔁 Temperatuurpersistentie per stad")
    st.write('Betekenis: Hoe lang de temperatuur op een plek “hetzelfde gedrag” blijft vertonen voordat het terugkeert naar normaal.')
    st.write('Een voorbeeld hiervan is een hittegolf, waardoor het meerdere dagen warmer blijft dan gemiddeld.')
    st.write('Een hoge persistentie betekent dat de temperatuur traag verandert; een lage persistentie betekent dat de temperatuur snel fluctueert.')
    st.write('Dit wordt gemeten doormiddel van de autocorrelatie.')
    st.write('De rode lijn op y=0.2 geeft de grens van praktisch “geen persistentie aan')

    # Kies stad
    city = st.selectbox("Kies stad:", list(CITY_FILES.keys()))
    df = load_city(CITY_FILES[city])

    # Anomalie berekenen
    df["month"] = df["date"].dt.month
    df["anomaly"] = df["t2m_daily_mean_C"] - df.groupby("month")["t2m_daily_mean_C"].transform("mean")

    # Autocorrelatie berekenen
    max_lag = 30
    acf = [df["anomaly"].autocorr(lag=i) for i in range(1, max_lag + 1)]
    acf_df = pd.DataFrame({"lag_days": range(1, max_lag + 1), "acf": acf})

    # Bepaal geheugenlengte
    memory_days = next((lag for lag, c in zip(acf_df["lag_days"], acf_df["acf"]) if c < 0.2), None)

    # Plot: bar met kleurverloop (blauw = laag, rood = hoog)
    fig = px.bar(
        acf_df,
        x="lag_days",
        y="acf",
        title=f"Autocorrelatie van temperatuur-anomalie – {city}",
        color="acf",
        color_continuous_scale="RdBu_r",
        labels={"lag_days": "Dagen vertraging", "acf": "Autocorrelatie"},
        hover_data={"lag_days": True, "acf": True}
    )

    # Rode lijn voor grens 0.2
    fig.add_hline(
        y=0.2,
        line_dash="dot",
        line_color="red",
        annotation_text="grens 0.2",
        annotation_position="top left"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Toont geheugenlengte
    if memory_days:
        st.markdown(f"📈 **Geheugenlengte:** ± {memory_days} dagen")
    else:
        st.markdown("⚪ Geen duidelijke grens (correlatie blijft hoog of fluctueert).")


elif view == "Persistentiekaart (grenzen)":
    st.title("🗺️ Ruimtelijke grenzen van temperatuurpersistentie")
    df_map = load_map(MAP_FILE)

    # Heatmap plot
    fig = px.density_mapbox(
        df_map,
        lat='latitude',
        lon='longitude',
        z='acf_lag1',  # persistentie waarde
        radius=10,     # grootte van de 'blur' per punt; hoger = gladdere kaart
        center=dict(lat=52, lon=5),  # midden van Europa
        zoom=3,
        mapbox_style='carto-positron',
        color_continuous_scale='RdBu_r',
        title='Dag-tot-dag persistentie (autocorrelatie lag=1)',
        hover_data={'latitude': True, 'longitude': True, 'acf_lag1': True}
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Interpretatie:**
    - 🔴 Hogere waarden = hoge persistentie (temperatuur verandert traag, typisch maritiem).
    - 🔵 Lagere waarden = lage persistentie (temperatuur wisselt snel, vaak continentaal of bergachtig).
    """)

