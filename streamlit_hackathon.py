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
        "Seizoens- en dag/nacht patronen"
    ]
)

# ====== PAGINA'S ======
import plotly.express as px
import streamlit as st

if view == "Tijdreeks per stad":
    st.title("📈 Dagelijkse gemiddelde temperatuur per maand")

    # ✅ Checkbox voor vergelijking
    compare = st.checkbox("Vergelijk meerdere steden")

    # ✅ Maand selectie
    maanden = st.multiselect(
        "Kies maand(en) om te tonen:",
        options=list(range(1, 13)),
        default=list(range(1, 13)),
        format_func=lambda x: ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][x-1]
    )

    if compare:
        # Multi-city plot
        selected_cities = st.multiselect("Kies steden:", list(CITY_FILES.keys()), default=list(CITY_FILES.keys()))
        fig = px.line(title="Dagelijkse gemiddelde temperatuur 2023")
        for city in selected_cities:
            df = load_city(CITY_FILES[city])
            df_filtered = df[df["date"].dt.month.isin(maanden)]
            fig.add_scatter(x=df_filtered["date"], y=df_filtered["t2m_daily_mean_C"], mode="lines", name=city)
    else:
        # Single-city plot
        city = st.selectbox("Kies stad:", list(CITY_FILES.keys()))
        df = load_city(CITY_FILES[city])
        df_filtered = df[df["date"].dt.month.isin(maanden)]
        fig = px.line(
            df_filtered,
            x="date",
            y="t2m_daily_mean_C",
            title=f"Dagelijkse gemiddelde temperatuur in {city} (2023)",
            labels={"t2m_daily_mean_C": "°C"}
        )

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


elif view == "Seizoens- en dag/nacht patronen":
    st.title("🌡️ Dag/nacht en seizoenspatronen per stad")
    st.write("Heatmap van gemiddelde temperatuur per uur en per maand. Hiermee zie je dag/nachtverschillen en seizoenspatronen.")

    # Kies stad
    city = st.selectbox("Kies stad:", list(CITY_FILES.keys()))
    df = load_city(CITY_FILES[city])

    # Voorbereiden van data
    df['hour'] = df['date'].dt.hour
    df['month'] = df['date'].dt.month
    heatmap_data = df.groupby(['month', 'hour'])['t2m_daily_mean_C'].mean().reset_index()
    pivot_table = heatmap_data.pivot(index='month', columns='hour', values='t2m_daily_mean_C')

    # Plotten
    fig = px.imshow(
        pivot_table,
        labels=dict(x="Uur van de dag", y="Maand", color="°C"),
        x=list(range(24)),
        y=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
        aspect="auto",
        color_continuous_scale='RdBu_r'
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Interpretatie:**
    - 🔴 Rood = warmere periodes, 🔵 Blauw = koudere periodes.
    - Je ziet duidelijk dag/nachtverschillen (temperatuur daalt 's nachts) en seizoenspatronen (zomer = warm, winter = koud).
    - Verschillen tussen steden worden zichtbaar wanneer je van stad wisselt.
    """)


