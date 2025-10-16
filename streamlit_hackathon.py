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
    st.title("🌍 Temperatuurvoorspelling per gebied")

    gebied_type = st.selectbox("Kies type gebied:", ["Berggebieden", "Zeegebieden", "Woestijngebieden"])

    # ========================
    # 🌄 B E R G G E B I E D E N
    # ========================
    if gebied_type == "Berggebieden":
        MOUNTAIN_FILES = {
            "Alpen": "data_daily_Alpen.csv",
            "Pyreneeën": "data_daily_Pyreneeen.csv",
            "Karpaten": "data_daily_Karpaten.csv"
        }

        region = st.selectbox("Kies berggebied:", list(MOUNTAIN_FILES.keys()))
        path = MOUNTAIN_FILES.get(region)
        if not os.path.exists(path):
            st.error(f"❌ Bestand '{path}' niet gevonden.")
            st.stop()

        df = pd.read_csv(path)
        df.columns = df.columns.str.replace(" ", "_")
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        temp_col = [c for c in df.columns if "Gemiddelde" in c or "t2m" in c][0]

        lag = st.slider("Aantal dagen vooruit voorspellen:", 1, 7, 1, key="berg_lag")

        # ---- Seizoenscomponent + persistence ----
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

        # ======= MAE KAART =======
        st.subheader("🌐 MAE per berggebied (interactieve kaart)")
        st.write("De kaart toont de nauwkeurigheid (MAE) van het seizoensmodel per gebied. Lagere MAE = betere voorspelling.")

        # DataFrame met MAE's voor alle berggebieden
        mae_data = []
        coords = {
            "Alpen": [46.8, 9.8],
            "Pyreneeën": [42.7, 1.8],
            "Karpaten": [47.0, 25.0]
        }

        for g, f in MOUNTAIN_FILES.items():
            if not os.path.exists(f):
                continue
            d = pd.read_csv(f)
            d.columns = d.columns.str.replace(" ", "_")
            d["date"] = pd.to_datetime(d["date"])
            temp = [c for c in d.columns if "Gemiddelde" in c or "t2m" in c][0]
            d["lag_temp"] = d[temp].shift(1)
            d["day_of_year"] = d["date"].dt.dayofyear
            d["sin_doy"] = np.sin(2 * np.pi * d["day_of_year"] / 365)
            d["cos_doy"] = np.cos(2 * np.pi * d["day_of_year"] / 365)
            d = d.dropna()
            X = d[["lag_temp", "sin_doy", "cos_doy"]]
            y = d[temp]
            m = LinearRegression().fit(X, y)
            y_pred = m.predict(X)
            mae = np.mean(np.abs(y - y_pred))
            mae_data.append({"Gebied": g, "lat": coords[g][0], "lon": coords[g][1], "MAE": mae})

        df_map = pd.DataFrame(mae_data)

        if df_map.empty:
            st.warning("Geen data om te tonen.")
        else:
            fig_map = px.scatter_mapbox(
                df_map,
                lat="lat",
                lon="lon",
                color="MAE",
                size="MAE",
                hover_name="Gebied",
                color_continuous_scale="RdYlBu_r",
                zoom=4,
                center={"lat": 46.5, "lon": 10.0},
                mapbox_style="carto-positron",
                title="MAE van seizoensmodel per berggebied"
            )
            fig_map.update_layout(
                coloraxis_colorbar=dict(
                    title="MAE (°C)",
                    ticksuffix=" °C",
                )
            )
            st.plotly_chart(fig_map, use_container_width=True)

    # ====================
    # 🌊 Z E E G E B I E D E N
    # ====================
    elif gebied_type == "Zeegebieden":
        # (ongewijzigde zee-code uit je originele script)
        SEA_FILES = {
            "Noordzee": "data_daily_Noordzee.csv",
            "Middellandse Zee": "data_daily_MiddellandseZee.csv",
            "Atlantische Oceaan": "data_daily_AtlantischeOceaan.csv"
        }
        region = st.selectbox("Kies zeegebied:", list(SEA_FILES.keys()))
        path = SEA_FILES.get(region)
        if not os.path.exists(path):
            st.error(f"❌ Bestand '{path}' niet gevonden.")
            st.stop()
        df = pd.read_csv(path)
        df.columns = df.columns.str.replace(" ", "_")
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        temp_col = [c for c in df.columns if "Gemiddelde" in c or "t2m" in c][0]
        lag = st.slider("Aantal dagen vooruit voorspellen:", 1, 7, 1, key="zee_lag")

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

    # ========================
    # 🏜️ W O E S T I J N G E B I E D E N
    # ========================
    else:
        # (ongewijzigde woestijn-code uit je originele script)
        DESERT_FILES = {
            "Tabernas": "data_daily_Tabernas.csv",
            "Bardenas Reales": "data_daily_BardenasReales.csv",
            "Oost-Kreta": "data_daily_OostKreta.csv"
        }
        region = st.selectbox("Kies woestijngebied:", list(DESERT_FILES.keys()))
        path = DESERT_FILES.get(region)
        if not os.path.exists(path):
            st.error(f"❌ Bestand '{path}' niet gevonden.")
            st.stop()
        df = pd.read_csv(path)
        df.columns = df.columns.str.replace(" ", "_")
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        temp_col = [c for c in df.columns if "Gemiddelde" in c or "t2m" in c][0]
        lag = st.slider("Aantal dagen vooruit voorspellen:", 1, 5, 1, key="woestijn_lag")

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


