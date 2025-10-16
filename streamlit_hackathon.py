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

# ====== HELPERS ======
def _set_month_xaxis(fig, date_series):
    """
    Zet x-as ticks op maandlabels (Jan, Feb, ...) zonder jaartal.
    date_series: pd.Series met datetime-waarden
    """
    if date_series.isna().all():
        return
    min_date = pd.to_datetime(date_series.min())
    max_date = pd.to_datetime(date_series.max())
    # Start op de eerste van de maand van min_date
    start = pd.Timestamp(min_date.year, min_date.month, 1)
    # Maak ticks voor elke maand tussen start en max_date
    ticks = pd.date_range(start=start, end=max_date, freq='MS')
    if len(ticks) == 0:
        # fallback: gebruik min_date als enkele tick
        ticks = [min_date]
    ticktext = [d.strftime("%b") for d in ticks]
    fig.update_xaxes(tickvals=ticks, ticktext=ticktext, tickangle=0, tickformat=None)

# ====== FUNCTIES ======
@st.cache_data
def load_city(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.replace(" ", "_")  # normaliseer kolomnamen
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
        "Voorspelmodellen per gebied"
    ]
)

# ====== PAGINA'S ======
if view == "Data Visualisatie":
    st.title("Data visualisatie")
    image = Image.open("TEMP_MAP.png")
    st.image(image, caption="Temperatuurkaart (2023)", use_container_width=True)


# ====== TIJDREEKS ======
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
        fig = px.line(title="Dagelijkse gemiddelde temperatuur")
        # combineer voor tick-bereik
        combined = []
        for city in selected_cities:
            df = load_city(CITY_FILES[city])
            temp_col = [c for c in df.columns if "Gemiddelde" in c or "t2m" in c][0]
            df_filtered = df[df["date"].dt.month.isin(maanden)]
            combined.append(df_filtered[["date"]])
            fig.add_scatter(x=df_filtered["date"], y=df_filtered[temp_col], mode="lines", name=city)
        if combined:
            combined_dates = pd.concat(combined)["date"]
            _set_month_xaxis(fig, combined_dates)
    else:
        city = st.selectbox("Kies stad:", list(CITY_FILES.keys()))
        df = load_city(CITY_FILES[city])
        temp_col = [c for c in df.columns if "Gemiddelde" in c or "t2m" in c][0]
        df_filtered = df[df["date"].dt.month.isin(maanden)]
        fig = px.line(
            df_filtered,
            x="date",
            y=temp_col,
            title=f"Dagelijkse gemiddelde temperatuur in {city}",
            labels={temp_col: "°C"}
        )
        _set_month_xaxis(fig, df_filtered["date"])

    st.plotly_chart(fig, use_container_width=True)


# ====== PERSISTENTIE ======
elif view == "Persistentie per stad":
    st.title("Temperatuurpersistentie per stad")
    st.write('Betekenis: Hoe lang de temperatuur op een plek “hetzelfde gedrag” blijft vertonen voordat het terugkeert naar normaal.')
    st.write('Een hoge persistentie betekent dat de temperatuur traag verandert; een lage persistentie betekent dat de temperatuur snel fluctueert.')
    st.write('De rode lijn op y=0.2 geeft de grens van praktisch “geen persistentie” aan.')

    city = st.selectbox("Kies stad:", list(CITY_FILES.keys()))
    df = load_city(CITY_FILES[city])
    temp_col = [c for c in df.columns if "Gemiddelde" in c or "t2m" in c][0]

    df["month"] = df["date"].dt.month
    df["anomaly"] = df[temp_col] - df.groupby("month")[temp_col].transform("mean")

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


# ====== SEIZOENSPATRONEN ======
elif view == "Seizoens- en dag/nacht patronen":
    st.title("Dag/nacht en seizoenspatronen per stad")
    st.write("Heatmap van gemiddelde temperatuur per uur (om de 3 uur) en per maand.")

    HOURLY_FILES = {
        "Amsterdam": "hourly_Amsterdam.csv",
        "Londen": "hourly_Londen.csv",
        "Madrid": "hourly_Madrid.csv"
    }

    city = st.selectbox("Kies stad:", list(HOURLY_FILES.keys()), key="hourly_city")

    @st.cache_data
    def load_hourly(file):
        df = pd.read_csv(file)
        df.columns = df.columns.str.replace(" ", "_")
        df["valid_time"] = pd.to_datetime(df["valid_time"])
        df["hour"] = df["hour"].astype(int)
        df["month"] = df["month"].astype(int)
        df["t2m_C"] = df["t2m_C"].astype(float)
        return df

    try:
        df = load_hourly(HOURLY_FILES[city])
    except FileNotFoundError:
        st.error(f"❌ Bestand '{HOURLY_FILES[city]}' niet gevonden.")
        st.stop()

    if df["hour"].nunique() < 8:
        st.warning("⚠️ Mogelijk onvolledige uurlijkse data (om de 3 uur).")

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
        y=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title=f"Seizoens- en dag/nachtpatronen in {city}"
    )

    fig.update_traces(hovertemplate="Maand: %{y}<br>Uur: %{x}<br>Temp: %{z:.1f} °C")
    st.plotly_chart(fig, use_container_width=True)


# ====== SIMPEL VOORSPELMODEL ======
elif view == "Simpel Voorspelmodel":
    st.title("Simpel voorspellen van temperatuur")

    city = st.selectbox("Kies stad:", list(CITY_FILES.keys()))
    df = load_city(CITY_FILES[city])
    temp_col = [c for c in df.columns if "Gemiddelde" in c or "t2m" in c][0]

    lag = st.slider("Aantal dagen vooruit voorspellen:", 1, 7, 1)

    df = df.sort_values("date").reset_index(drop=True)
    df["t2m_pred"] = df[temp_col].shift(lag)
    df_eval = df.dropna(subset=["t2m_pred"])
    mae = np.mean(np.abs(df_eval[temp_col] - df_eval["t2m_pred"]))
    st.markdown(f"**MAE voor {lag} dagen vooruit:** {mae:.2f} °C")

    fig = px.line(
        df_eval,
        x="date",
        y=[temp_col, "t2m_pred"],
        labels={"value": "Temperatuur (°C)", "date": "Datum"},
        title=f"Voorspelling vs echte temperatuur in {city}"
    )
    # zorg dat x-as maandlabels toont zonder jaartal
    _set_month_xaxis(fig, df_eval["date"])
    fig.for_each_trace(lambda t: t.update(name="Echt" if t.name == temp_col else "Voorspeld"))
    st.plotly_chart(fig, use_container_width=True)


# ====== VOORSPELMODELLEN VOOR GEBIEDEN ======
elif view == "Voorspelmodellen per gebied":
    import streamlit as st
    import pandas as pd
    import numpy as np
    import plotly.express as px
    from sklearn.linear_model import LinearRegression

    st.title("🌍 Temperatuurvoorspelling per gebied")

    # === FILE DICTIONARIES ===
    MOUNTAIN_FILES = {
        "Alpen": "data_daily_Alpen.csv",
        "Pyreneeën": "data_daily_Pyreneeen.csv",
        "Karpaten": "data_daily_Karpaten.csv"
    }

    SEA_FILES = {
        "Noordzee": "data_daily_Noordzee.csv",
        "Middellandse Zee": "data_daily_MiddellandseZee.csv",
        "Atlantische Oceaan": "data_daily_AtlantischeOceaan.csv"
    }

    DESERT_FILES = {
        "Sahara_Spain": "data_daily_Sahara_Spain.csv",
        "Tabernas": "data_daily_Tabernas.csv",
        "Bardenas": "data_daily_Bardenas.csv"
    }

    # === COORDINATES FOR MAP ===
    COORDS = {
        "Alpen": (46.8, 9.8),
        "Pyreneeën": (42.6, 0.5),
        "Karpaten": (47.0, 24.0),
        "Noordzee": (55.0, 3.0),
        "Middellandse Zee": (42.5, 5.0),
        "Atlantische Oceaan": (41.0, -10.0),
        "Sahara_Spain": (37.0, -2.0),
        "Tabernas": (37.1, -2.3),
        "Bardenas": (42.3, -1.7)
    }

    gebied_type = st.selectbox("Kies type gebied:", ["Berggebieden", "Zeegebieden", "Woestijnen"])
    lag = st.slider("Aantal dagen vooruit voorspellen:", 1, 7, 1, key="main_lag")

    # === Select correct file set ===
    if gebied_type == "Berggebieden":
        files = MOUNTAIN_FILES
    elif gebied_type == "Zeegebieden":
        files = SEA_FILES
    else:
        files = DESERT_FILES

    region = st.selectbox(f"Kies {gebied_type.lower()[:-1]}:", list(files.keys()))
    df = pd.read_csv(files[region])
    df.columns = df.columns.str.replace(" ", "_")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    temp_col = [c for c in df.columns if "Gemiddelde" in c or "t2m" in c][0]

    # === MODEL DEFINITIONS ===
    if gebied_type == "Berggebieden":
        # Seasonal linear regression model
        df["day_of_year"] = df["date"].dt.dayofyear
        df["sin_doy"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
        df["cos_doy"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
        df["Simpel"] = df[temp_col].shift(lag)
        df["lag_temp"] = df[temp_col].shift(1)
        df = df.dropna()
        X = df[["lag_temp", "sin_doy", "cos_doy"]]
        y = df[temp_col]
        model = LinearRegression().fit(X, y)
        df["Model"] = model.predict(X)

    elif gebied_type == "Zeegebieden":
        # Moving average + harmonic model
        df["Simpel"] = df[temp_col].shift(lag)
        df["MA_7d"] = df[temp_col].rolling(window=7, center=True).mean()
        df["day_of_year"] = df["date"].dt.dayofyear
        df["sin1"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
        df["cos1"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
        df["sin2"] = np.sin(4 * np.pi * df["day_of_year"] / 365)
        df["cos2"] = np.cos(4 * np.pi * df["day_of_year"] / 365)
        df = df.dropna()
        X = df[["MA_7d", "sin1", "cos1", "sin2", "cos2"]]
        y = df[temp_col]
        model = LinearRegression().fit(X, y)
        df["Model"] = model.predict(X)

    else:  # Woestijnen
        # Daily temp variation + persistence + seasonality
        df["Simpel"] = df[temp_col].shift(lag)
        df["day_of_year"] = df["date"].dt.dayofyear
        df["sin1"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
        df["cos1"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
        df["day_night_diff"] = df[temp_col].rolling(window=2).apply(lambda x: x.max() - x.min())
        df = df.dropna()
        X = df[["Simpel", "sin1", "cos1", "day_night_diff"]]
        y = df[temp_col]
        model = LinearRegression().fit(X, y)
        df["Model"] = model.predict(X)

    # === ERROR METRICS ===
    mae_simple = np.mean(np.abs(df[temp_col] - df["Simpel"]))
    mae_model = np.mean(np.abs(df[temp_col] - df["Model"]))

    st.markdown(f"""
    **MAE ({region}, {lag}-dagen horizon):**
    - Simpel model: {mae_simple:.2f} °C  
    - Geavanceerd model: {mae_model:.2f} °C  
    """)

    # === PREDICTION GRAPH ===
    fig = px.line(
        df,
        x="date",
        y=[temp_col, "Simpel", "Model"],
        labels={"value": "Temperatuur (°C)", "date": "Datum"},
        title=f"📈 Voorspelling vs. Observatie – {region}"
    )
    fig.for_each_trace(lambda t: t.update(
        name="Werkelijke temperatuur" if t.name == temp_col
        else "Simpel model" if t.name == "Simpel"
        else "Voorspellend model"
    ))
    fig.update_layout(legend_title_text="Legenda")
    st.plotly_chart(fig, use_container_width=True)

    # === INTERACTIVE MAE MAP ===
    st.subheader("🌐 Vergelijk MAE per gebied op de kaart")

    # Compute MAE for all regions to visualize on one map
    all_regions = {**MOUNTAIN_FILES, **SEA_FILES, **DESERT_FILES}
    maes = []
    for r, path in all_regions.items():
        df_temp = pd.read_csv(path)
        df_temp.columns = df_temp.columns.str.replace(" ", "_")
        df_temp["date"] = pd.to_datetime(df_temp["date"])
        df_temp = df_temp.sort_values("date").reset_index(drop=True)
        tcol = [c for c in df_temp.columns if "Gemiddelde" in c or "t2m" in c][0]

        df_temp["Simpel"] = df_temp[tcol].shift(lag)
        df_temp = df_temp.dropna()
        mae = np.mean(np.abs(df_temp[tcol] - df_temp["Simpel"]))

        # categorize area type
        if r in MOUNTAIN_FILES:
            t = "Berg"
        elif r in SEA_FILES:
            t = "Zee"
        else:
            t = "Woestijn"

        maes.append({
            "region": r,
            "lat": COORDS[r][0],
            "lon": COORDS[r][1],
            "type": t,
            "MAE": mae
        })

    df_map = pd.DataFrame(maes)

    fig_map = px.scatter_geo(
        df_map,
        lat="lat",
        lon="lon",
        color="MAE",
        hover_name="region",
        hover_data=["type", "MAE"],
        color_continuous_scale="YlOrRd",
        size="MAE",
        projection="natural earth",
        title=f"MAE per regio (lag = {lag} dagen)",
        scope="europe"
    )

    # Allow zooming & movement
    fig_map.update_geos(
        resolution=50,
        showcountries=True,
        showcoastlines=True,
        showland=True,
        landcolor="rgb(240, 240, 240)",
        countrycolor="gray",
        projection_scale=4,  # zoom
        center={"lat": 46, "lon": 10},
    )

    fig_map.update_layout(
        coloraxis_colorbar=dict(title="MAE (°C)"),
        dragmode="pan"
    )

    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("""
    **Interpretatie:**
    - De kleurintensiteit toont de fout (MAE) per locatie.  
    - Je kunt de kaart inzoomen en rondbewegen om verschillen tussen gebieden te vergelijken.  
    - Lagere MAE (lichter) betekent nauwkeurigere voorspellingen.  
    """)
