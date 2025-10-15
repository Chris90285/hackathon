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
# ====== VOORSPELMODELLEN PER GEBIED (kaart + details) ======
elif view == "Voorspelmodellen per gebied":
    st.title("🌍 Temperatuurvoorspelling per gebied")

    # ------- region metadata (bestandsnaam + type + lat/lon) -------
    REGION_META = {
        # bergen
        "Alpen":           {"file": "data_daily_Alpen.csv",           "type": "Berg",   "lat": 46.8, "lon": 9.8},
        "Pyreneeën":       {"file": "data_daily_Pyreneeen.csv",      "type": "Berg",   "lat": 42.6, "lon": 0.5},
        "Karpaten":        {"file": "data_daily_Karpaten.csv",      "type": "Berg",   "lat": 47.0, "lon": 24.0},
        # zee
        "Noordzee":        {"file": "data_daily_Noordzee.csv",      "type": "Zee",    "lat": 55.0, "lon": 3.0},
        "MiddellandseZee":{"file": "data_daily_MiddellseZee.csv",  "type": "Zee",    "lat": 42.5, "lon": 5.0},  # let op bestandsnaam
        "AtlantischeOceaan":{"file":"data_daily_AtlantischeOceaan.csv","type":"Zee","lat":41.0, "lon": -10.0},
        # woestijn
        "Tabernas":        {"file": "data_daily_Tabernas.csv",      "type": "Woestijn", "lat": 37.0, "lon": -2.4},
        "Bardenas Reales": {"file": "data_daily_BardenasReales.csv","type": "Woestijn", "lat": 42.2, "lon": -1.5},
        "Oost-Kreta":      {"file": "data_daily_OostKreta.csv",     "type": "Woestijn", "lat": 35.1, "lon": 26.1},
    }

    # ------- kaart controls -------
    st.markdown("**Interactiekaart — MAE per locatie**")
    lag_for_map = st.slider("Voorspellingshorizon (dagen vooruit) voor kaart:", 1, 7, 1, key="map_lag")
    mae_metric = st.selectbox("Kleur op:", ["MAE model", "MAE simpel"], index=0, key="map_metric")
    show_types = st.multiselect("Toon gebiedstypes:", ["Berg", "Zee", "Woestijn"], default=["Berg", "Zee", "Woestijn"], key="map_types")

    # ------- helper: detecteer temperatuurkolom -------
    def _detect_temp_col(df):
        df.columns = df.columns.str.replace(" ", "_")
        candidates = [c for c in df.columns if ("Gemiddelde" in c) or ("t2m" in c.lower()) or ("t2m" in c)]
        if len(candidates) == 0:
            raise KeyError("Geen temperatuurkolom gevonden")
        return candidates[0]

    # ------- bereken MAE per regio voor gegeven lag -------
    rows = []
    for region, meta in REGION_META.items():
        # filter op type
        if meta["type"] not in show_types:
            continue
        file = meta["file"]
        if not os.path.exists(file):
            # bestand niet gevonden — sla over maar geef feedback
            rows.append({
                "region": region,
                "type": meta["type"],
                "lat": meta["lat"],
                "lon": meta["lon"],
                "mae_simple": np.nan,
                "mae_model": np.nan,
                "notes": f"file {file} missing"
            })
            continue
        try:
            df = pd.read_csv(file)
        except Exception as e:
            rows.append({
                "region": region,
                "type": meta["type"],
                "lat": meta["lat"],
                "lon": meta["lon"],
                "mae_simple": np.nan,
                "mae_model": np.nan,
                "notes": f"read error"
            })
            continue

        # prepare
        df.columns = df.columns.str.replace(" ", "_")
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        temp_col = None
        try:
            temp_col = _detect_temp_col(df)
        except KeyError:
            rows.append({
                "region": region,
                "type": meta["type"],
                "lat": meta["lat"],
                "lon": meta["lon"],
                "mae_simple": np.nan,
                "mae_model": np.nan,
                "notes": "no temp col"
            })
            continue

        # simple persistence MAE calculation (we will align indices when comparing)
        df_sorted = df.sort_values("date").reset_index(drop=True)
        df_sorted["Simpel"] = df_sorted[temp_col].shift(lag_for_map)

        # build model depending on type
        if meta["type"] == "Berg":
            # same as your bergmodel: lag1 + seasonal sin/cos
            df_sorted["day_of_year"] = df_sorted["date"].dt.dayofyear
            df_sorted["sin_doy"] = np.sin(2 * np.pi * df_sorted["day_of_year"] / 365)
            df_sorted["cos_doy"] = np.cos(2 * np.pi * df_sorted["day_of_year"] / 365)
            df_sorted["lag_temp"] = df_sorted[temp_col].shift(1)
            df_model = df_sorted.dropna(subset=[temp_col, "lag_temp", "sin_doy", "cos_doy", "Simpel"])
            if len(df_model) < 2:
                mae_model = np.nan
                mae_simple = np.nan
            else:
                X = df_model[["lag_temp", "sin_doy", "cos_doy"]]
                y = df_model[temp_col]
                reg = LinearRegression().fit(X, y)
                preds = reg.predict(X)
                mae_model = np.mean(np.abs(y - preds))
                # simple MAE on same index
                mae_simple = np.mean(np.abs(df_model[temp_col] - df_model["Simpel"]))
        elif meta["type"] == "Zee":
            # Zeemodel: MA_7d + harmonics
            df_sorted["MA_7d"] = df_sorted[temp_col].rolling(window=7, center=True).mean()
            df_sorted["day_of_year"] = df_sorted["date"].dt.dayofyear
            df_sorted["sin1"] = np.sin(2 * np.pi * df_sorted["day_of_year"] / 365)
            df_sorted["cos1"] = np.cos(2 * np.pi * df_sorted["day_of_year"] / 365)
            df_sorted["sin2"] = np.sin(4 * np.pi * df_sorted["day_of_year"] / 365)
            df_sorted["cos2"] = np.cos(4 * np.pi * df_sorted["day_of_year"] / 365)
            df_model = df_sorted.dropna(subset=[temp_col, "MA_7d", "sin1", "cos1", "Simpel"])
            if len(df_model) < 2:
                mae_model = np.nan
                mae_simple = np.nan
            else:
                X = df_model[["MA_7d", "sin1", "cos1", "sin2", "cos2"]]
                y = df_model[temp_col]
                reg = LinearRegression().fit(X, y)
                preds = reg.predict(X)
                mae_model = np.mean(np.abs(y - preds))
                mae_simple = np.mean(np.abs(df_model[temp_col] - df_model["Simpel"]))
        else:  # Woestijn
            # Woestijnmodel: AR(3) + korte dag-nacht sinus
            df_sorted["lag1"] = df_sorted[temp_col].shift(1)
            df_sorted["lag2"] = df_sorted[temp_col].shift(2)
            df_sorted["lag3"] = df_sorted[temp_col].shift(3)
            # kortere periode sinus to capture faster cycles (approx)
            df_sorted["day_of_year"] = df_sorted["date"].dt.dayofyear
            df_sorted["sin_day"] = np.sin(2 * np.pi * df_sorted["day_of_year"] / 30)
            df_model = df_sorted.dropna(subset=[temp_col, "lag1", "lag2", "lag3", "sin_day", "Simpel"])
            if len(df_model) < 2:
                mae_model = np.nan
                mae_simple = np.nan
            else:
                X = df_model[["lag1", "lag2", "lag3", "sin_day"]]
                y = df_model[temp_col]
                reg = LinearRegression().fit(X, y)
                preds = reg.predict(X)
                mae_model = np.mean(np.abs(y - preds))
                mae_simple = np.mean(np.abs(df_model[temp_col] - df_model["Simpel"]))

        rows.append({
            "region": region,
            "type": meta["type"],
            "lat": meta["lat"],
            "lon": meta["lon"],
            "mae_simple": float(mae_simple) if not pd.isna(mae_simple) else np.nan,
            "mae_model": float(mae_model) if not pd.isna(mae_model) else np.nan,
            "notes": ""
        })

    df_map = pd.DataFrame(rows)

    # ------- kaart plotten (Plotly) -------
    if df_map.shape[0] == 0:
        st.info("Geen regio's gevonden om te tonen op de kaart.")
    else:
        color_col = "mae_model" if mae_metric == "MAE model" else "mae_simple"
        fig_map = px.scatter_geo(
            df_map,
            lat="lat",
            lon="lon",
            hover_name="region",
            hover_data=["type", "mae_simple", "mae_model", "notes"],
            color=color_col,
            size=color_col,
            projection="natural earth",
            color_continuous_scale="YlOrRd",
            title=f"MAE per locatie — {mae_metric} (lag={lag_for_map}d)",
            symbol="type",
            scope="europe"
        )
        fig_map.update_layout(legend_title_text="Gebiedstype")
        st.plotly_chart(fig_map, use_container_width=True, theme="streamlit")

    st.markdown("---")
    st.write("### Details per gebied (zelfde UI als eerder)")
    # Nu de bestaande UI: gebruiker kan alsnog kiezen berg/zee/woestijn en zien details + grafieken
    gebied_type = st.selectbox("Kies type gebied:", ["Berggebieden", "Zeegebieden", "Woestijngebieden"], key="detail_area_choice")

    # ======================================================================
    # De rest: precieze gebied-specifieke UI en modellen (kopie van je vorige logica)
    # ======================================================================
    if gebied_type == "Berggebieden":
        MOUNTAIN_FILES = {
            "Alpen": "data_daily_Alpen.csv",
            "Pyreneeën": "data_daily_Pyreneeen.csv",
            "Karpaten": "data_daily_Karpaten.csv"
        }

        region = st.selectbox("Kies berggebied:", list(MOUNTAIN_FILES.keys()), key="berg_detail_region")
        df = pd.read_csv(MOUNTAIN_FILES[region])
        df.columns = df.columns.str.replace(" ", "_")
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        temp_col = [c for c in df.columns if "Gemiddelde" in c or "t2m" in c][0]
        lag = st.slider("Aantal dagen vooruit voorspellen:", 1, 7, 1, key="berg_lag_detail")

        df["day_of_year"] = df["date"].dt.dayofyear
        df["sin_doy"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
        df["cos_doy"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
        df["Simpel"] = df[temp_col].shift(lag)
        df["lag_temp"] = df[temp_col].shift(1)
        df = df.dropna()

        X = df[["lag_temp", "sin_doy", "cos_doy"]]
        y = df[temp_col]
        model = LinearRegression()
        model.fit(X, y)
        df["Seizoensmodel"] = model.predict(X)

        mae_simple = np.mean(np.abs(df[temp_col] - df["Simpel"]))
        mae_reg = np.mean(np.abs(df[temp_col] - df["Seizoensmodel"]))

        st.markdown(f"""
        **MAE ({region}, {lag}-dagen horizon):**
        - Simpel model (persistence): {mae_simple:.2f} °C  
        - Seizoensmodel (lineaire regressie): {mae_reg:.2f} °C  
        """)

        fig = px.line(
            df,
            x="date",
            y=[temp_col, "Simpel", "Seizoensmodel"],
            labels={"value": "Temperatuur (°C)", "date": "Datum"},
            title=f"Voorspelling vs. observatie in {region}"
        )
        fig.for_each_trace(lambda t: t.update(
            name=("Werkelijke temperatuur" if t.name == temp_col else
                  "Simpel model" if t.name == "Simpel" else
                  "Seizoensmodel")
        ))
        _set_month_xaxis(fig, df["date"])
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        **Interpretatie:**
        - Simpel model voorspelt temperatuur puur op basis van vorige dagen.  
        - Seizoensmodel houdt rekening met jaarlijkse cycli (koude winters, warme zomers).  
        - In berggebieden is het seizoenseffect meestal sterker, dus het regressiemodel presteert vaak beter.
        """)

    elif gebied_type == "Zeegebieden":
        SEA_FILES = {
            "Noordzee": "data_daily_Noordzee.csv",
            "Middellandse Zee": "data_daily_MiddellseZee.csv",
            "Atlantische Oceaan": "data_daily_AtlantischeOceaan.csv"
        }

        region = st.selectbox("Kies zeegebied:", list(SEA_FILES.keys()), key="zee_detail_region")
        df = pd.read_csv(SEA_FILES[region])
        df.columns = df.columns.str.replace(" ", "_")
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        temp_col = [c for c in df.columns if "Gemiddelde" in c or "t2m" in c][0]
        lag = st.slider("Aantal dagen vooruit voorspellen:", 1, 7, 1, key="zee_lag_detail")

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
        reg = LinearRegression()
        reg.fit(X, y)
        df["Zeemodel"] = reg.predict(X)

        mae_simple = np.mean(np.abs(df[temp_col] - df["Simpel"]))
        mae_reg = np.mean(np.abs(df[temp_col] - df["Zeemodel"]))

        st.markdown(f"""
        **MAE ({region}, {lag}-dagen horizon):**
        - Simpel model (persistence): {mae_simple:.2f} °C  
        - Zeemodel (harmonisch + moving average): {mae_reg:.2f} °C  
        """)

        fig = px.line(
            df,
            x="date",
            y=[temp_col, "Simpel", "Zeemodel"],
            labels={"value": "Temperatuur (°C)", "date": "Datum"},
            title=f"Voorspelling vs. observatie in {region}"
        )
        fig.for_each_trace(lambda t: t.update(
            name=("Werkelijke temperatuur" if t.name == temp_col else
                  "Simpel model" if t.name == "Simpel" else
                  "Zeemodel")
        ))
        _set_month_xaxis(fig, df["date"])
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        **Interpretatie:**
        - In zeegebieden veranderen temperaturen trager door de hoge warmtecapaciteit van water.  
        - Het model gebruikt een 7-daags gemiddelde om korte schommelingen uit te filteren.  
        - Harmonische functies (sin/cos) vangen de jaarlijkse en halfjaarlijkse golfbewegingen.  
        - Dit geeft vaak een stabielere voorspelling dan een simpel persistence-model.
        """)

    else:  # Woestijngebieden
        DESERT_FILES = {
            "Tabernas": "data_daily_Tabernas.csv",
            "Bardenas Reales": "data_daily_BardenasReales.csv",
            "Oost-Kreta": "data_daily_OostKreta.csv"
        }

        region = st.selectbox("Kies woestijngebied:", list(DESERT_FILES.keys()), key="woestijn_detail_region")
        df = pd.read_csv(DESERT_FILES[region])
        df.columns = df.columns.str.replace(" ", "_")
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        temp_col = [c for c in df.columns if "Gemiddelde" in c or "t2m" in c][0]
        lag = st.slider("Aantal dagen vooruit voorspellen:", 1, 5, 1, key="woestijn_lag_detail")

        df["lag1"] = df[temp_col].shift(1)
        df["lag2"] = df[temp_col].shift(2)
        df["lag3"] = df[temp_col].shift(3)
        df["day_of_year"] = df["date"].dt.dayofyear
        df["sin_day"] = np.sin(2 * np.pi * df["day_of_year"] / 30)
        df["Simpel"] = df[temp_col].shift(lag)

        df = df.dropna()
        X = df[["lag1", "lag2", "lag3", "sin_day"]]
        y = df[temp_col]
        model = LinearRegression()
        model.fit(X, y)
        df["Woestijnmodel"] = model.predict(X)

        mae_simple = np.mean(np.abs(df[temp_col] - df["Simpel"]))
        mae_reg = np.mean(np.abs(df[temp_col] - df["Woestijnmodel"]))

        st.markdown(f"""
        **MAE ({region}, {lag}-dagen horizon):**
        - Simpel model (persistence): {mae_simple:.2f} °C  
        - Woestijnmodel (AR(3) + dag-nachtcyclus): {mae_reg:.2f} °C  
        """)

        fig = px.line(
            df,
            x="date",
            y=[temp_col, "Simpel", "Woestijnmodel"],
            labels={"value": "Temperatuur (°C)", "date": "Datum"},
            title=f"Voorspelling vs. observatie in {region}"
        )
        fig.for_each_trace(lambda t: t.update(
            name=("Werkelijke temperatuur" if t.name == temp_col else
                  "Simpel model" if t.name == "Simpel" else
                  "Woestijnmodel")
        ))
        _set_month_xaxis(fig, df["date"])
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        **Interpretatie:**
        - Woestijngebieden hebben sterke dagelijkse temperatuurvariaties.  
        - Het AR(3)-model gebruikt de afgelopen drie dagen om korte-termijneffecten te vangen.  
        - De extra sinuscomponent simuleert de maandelijkse schommeling (dag/nacht-effect).  
        - Hierdoor is het model beter afgestemd op snelle temperatuurschommelingen dan de eerdere modellen.
        """)
