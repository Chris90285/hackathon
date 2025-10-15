import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from PIL import Image


st.set_page_config(page_title="Temperatuuranalyse 2023", layout="wide")

# ====== BESTANDSLOCATIES ======
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
        "Data Visualisatie",
        "Tijdreeks per stad",
        "Persistentie per stad",
        "Seizoens- en dag/nacht patronen",
        "Simpel Voorspelmodel",
        "Voorspelmodel Berggebieden"
    ]
)

# ====== PAGINA'S ======
if view == "Data Visualisatie":
    st.title("Data visualisatie")

    image = Image.open("TEMP_MAP.png")
    st.image(image, caption="Temperatuurkaart (2023)", use_container_width=True)


if view == "Tijdreeks per stad":
    st.title("📈 Dagelijkse gemiddelde temperatuur per maand")

    compare = st.checkbox("Vergelijk meerdere steden")

    maanden = st.multiselect(
        "Kies maand(en) om te tonen:",
        options=list(range(1, 13)),
        default=list(range(1, 13)),
        format_func=lambda x: ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][x-1]
    )

    if compare:
        selected_cities = st.multiselect("Kies steden:", list(CITY_FILES.keys()), default=list(CITY_FILES.keys()))
        fig = px.line(title="Dagelijkse gemiddelde temperatuur 2023")
        for city in selected_cities:
            df = load_city(CITY_FILES[city])
            df_filtered = df[df["date"].dt.month.isin(maanden)]
            fig.add_scatter(x=df_filtered["date"], y=df_filtered["Dagelijks_Gemiddelde"], mode="lines", name=city)
    else:
        city = st.selectbox("Kies stad:", list(CITY_FILES.keys()))
        df = load_city(CITY_FILES[city])
        df_filtered = df[df["date"].dt.month.isin(maanden)]
        fig = px.line(
            df_filtered,
            x="date",
            y="Dagelijks_Gemiddelde",
            title=f"Dagelijkse gemiddelde temperatuur in {city} (2023)",
            labels={"Dagelijks_Gemiddelde": "°C"}
        )

    st.plotly_chart(fig, use_container_width=True)



elif view == "Persistentie per stad":
    st.title("Temperatuurpersistentie per stad")
    st.write('Betekenis: Hoe lang de temperatuur op een plek “hetzelfde gedrag” blijft vertonen voordat het terugkeert naar normaal.')
    st.write('Een voorbeeld hiervan is een hittegolf, waardoor het meerdere dagen warmer blijft dan gemiddeld.')
    st.write('Een hoge persistentie betekent dat de temperatuur traag verandert; een lage persistentie betekent dat de temperatuur snel fluctueert.')
    st.write('Dit wordt gemeten doormiddel van de autocorrelatie.')
    st.write('De rode lijn op y=0.2 geeft de grens van praktisch “geen persistentie aan')

    city = st.selectbox("Kies stad:", list(CITY_FILES.keys()))
    df = load_city(CITY_FILES[city])

    df["month"] = df["date"].dt.month
    df["anomaly"] = df["Dagelijks_Gemiddelde"] - df.groupby("month")["Dagelijks_Gemiddelde"].transform("mean")

    max_lag = 30
    acf = [df["anomaly"].autocorr(lag=i) for i in range(1, max_lag + 1)]
    acf_df = pd.DataFrame({"lag_days": range(1, max_lag + 1), "acf": acf})

    memory_days = next((lag for lag, c in zip(acf_df["lag_days"], acf_df["acf"]) if c < 0.2), None)

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

    fig.add_hline(
        y=0.2,
        line_dash="dot",
        line_color="red",
        annotation_text="grens 0.2",
        annotation_position="top left"
    )

    st.plotly_chart(fig, use_container_width=True)

    if memory_days:
        st.markdown(f"**Geheugenlengte:** ± {memory_days} dagen")
    else:
        st.markdown("⚪ Geen duidelijke grens (correlatie blijft hoog of fluctueert).")


elif view == "Seizoens- en dag/nacht patronen":
    st.title("Dag/nacht en seizoenspatronen per stad")
    st.write("""
    Heatmap van gemiddelde temperatuur per uur (om de 3 uur) en per maand. 
    Hiermee zie je dag/nachtverschillen en seizoenspatronen op een overzichtelijke manier.
    """)

    HOURLY_FILES = {
        "Amsterdam": "hourly_Amsterdam.csv",
        "Londen": "hourly_Londen.csv",
        "Madrid": "hourly_Madrid.csv"
    }

    city = st.selectbox("Kies stad:", list(HOURLY_FILES.keys()), key="hourly_city")

    @st.cache_data
    def load_hourly(file):
        df = pd.read_csv(file)
        df["valid_time"] = pd.to_datetime(df["valid_time"])
        df["hour"] = df["hour"].astype(int)
        df["month"] = df["month"].astype(int)
        df["t2m_C"] = df["t2m_C"].astype(float)
        return df

    try:
        df = load_hourly(HOURLY_FILES[city])
    except FileNotFoundError:
        st.error(f"❌ Het bestand '{HOURLY_FILES[city]}' is niet gevonden. Upload de juiste `hourly_*.csv` bestanden.")
        st.stop()

    if df["hour"].nunique() < 8:
        st.warning("⚠️ Let op: deze dataset bevat mogelijk geen volledige uurlijkse data (om de 3 uur).")

    heatmap_data = df.groupby(["month", "hour"])["t2m_C"].mean().reset_index()

    all_months = list(range(1, 13))
    all_hours = list(range(0, 24, 3))
    pivot_table = heatmap_data.pivot(index="month", columns="hour", values="t2m_C")
    pivot_table = pivot_table.reindex(index=all_months, columns=all_hours)
    pivot_table = pivot_table.fillna(np.nan)

    fig = px.imshow(
        pivot_table,
        labels=dict(x="Uur van de dag", y="Maand", color="Temperatuur (°C)"),
        x=all_hours,
        y=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title=f"Seizoens- en dag/nachtpatronen in {city} (2023)"
    )

    fig.update_traces(hovertemplate="Maand: %{y}<br>Uur: %{x}<br>Temp: %{z:.1f} °C")

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Interpretatie:**
    - 🔴 Rood = warmere periodes, 🔵 Blauw = koudere periodes.
    - Je ziet duidelijk dag/nachtverschillen (temperatuur daalt 's nachts) en seizoenspatronen (zomer = warm, winter = koud).
    - In zuidelijke steden zoals Madrid blijft het 's nachts relatief warm, terwijl noordelijke steden sterkere nachtelijke afkoeling tonen.
    """)



elif view == "Simpel Voorspelmodel":
    st.title("Simpel voorspellen van temperatuur")

    city = st.selectbox("Kies stad:", list(CITY_FILES.keys()))
    df = load_city(CITY_FILES[city])

    lag = st.slider("Aantal dagen vooruit voorspellen:", 1, 7, 1)

    df = df.sort_values("date").reset_index(drop=True)

    df['t2m_pred'] = df['Dagelijks_Gemiddelde'].shift(lag)

    df_eval = df.dropna(subset=['t2m_pred'])

    mae = np.mean(np.abs(df_eval['Dagelijks_Gemiddelde'] - df_eval['t2m_pred']))
    st.markdown(f"**MAE voor {lag} dagen vooruit:** {mae:.2f} °C")

    fig = px.line(
        df_eval,
        x="date",
        y=["Dagelijks_Gemiddelde", "t2m_pred"],
        labels={"value": "Temperatuur (°C)", "date": "Datum"},
        title=f"Voorspelling vs echte temperatuur in {city} ({lag}-dagen vooruit)"
    )

    fig.for_each_trace(lambda t: t.update(name="Echt" if t.name=="Dagelijks_Gemiddelde" else "Voorspeld"))

    st.plotly_chart(fig, use_container_width=True)

elif view == "Voorspelmodel Berggebieden":
    st.title("Temperatuurvoorspelling in berggebieden")
    st.write("""
    In berggebieden kan de temperatuur sterk variëren door hoogte, helling en lokale effecten.  
    Hier vergelijken we een **simpel persistentie model** met een **seizoensmodel (lineaire regressie)**.
    """)

    MOUNTAIN_FILES = {
        "Alpen": "data_daily_Alpen.csv",
        "Pyreneeën": "data_daily_Pyreneeen.csv",
        "Karpaten": "data_daily_Karpaten.csv"
    }

    region = st.selectbox("Kies berggebied:", list(MOUNTAIN_FILES.keys()))
    df = pd.read_csv(MOUNTAIN_FILES[region])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    lag = st.slider("Aantal dagen vooruit voorspellen:", 1, 7, 1)

    df["day_of_year"] = df["date"].dt.dayofyear
    df["sin_doy"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["cos_doy"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    df["Simpel"] = df["Dagelijks_Gemiddelde"].shift(lag)

    from sklearn.linear_model import LinearRegression

    df["lag_temp"] = df["Dagelijks_Gemiddelde"].shift(1)
    df = df.dropna()

    X = df[["lag_temp", "sin_doy", "cos_doy"]]
    y = df["Dagelijks_Gemiddelde"]

    model = LinearRegression()
    model.fit(X, y)

    df["Seizoensmodel"] = model.predict(X)

    mae_simple = np.mean(np.abs(df["Dagelijks_Gemiddelde"] - df["Simpel"]))
    mae_reg = np.mean(np.abs(df["Dagelijks_Gemiddelde"] - df["Seizoensmodel"]))

    st.markdown(f"""
    **MAE ({region}, {lag}-dagen horizon):**
    - Simpel model (persistence): {mae_simple:.2f} °C  
    - Seizoensmodel (lineaire regressie): {mae_reg:.2f} °C  
    """)

    fig = px.line(
        df,
        x="date",
        y=["Dagelijks_Gemiddelde", "Simpel", "Seizoensmodel"],
        labels={"value": "Temperatuur (°C)", "date": "Datum"},
        title=f"Voorspelling vs. observatie in {region}"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Interpretatie:**
    - Simpel model voorspelt temperatuur puur op basis van vorige dagen.  
    - Seizoensmodel houdt ook rekening met jaarlijkse cycli (koude winters, warme zomers).  
    - In berggebieden is het seizoenseffect meestal sterker, dus het regressiemodel zal vaak beter presteren.
    """)
