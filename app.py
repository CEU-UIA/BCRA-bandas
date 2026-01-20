# ============================================================
# Macroeconomía
# Tipo de cambio (A3500) + Bandas 2025/2026
# Tasa de interés (BCRA Monetarias id 145)
# Precios (IPC INDEC FTP con selector)
# + HOME con navegación
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go


# ----------------------------
# Configuración general
# ----------------------------
st.set_page_config(page_title="Macroeconomía", layout="wide")
st.title("Macroeconomía")


# ----------------------------
# Navegación
# ----------------------------
if "section" not in st.session_state:
    st.session_state.section = "home"

params = st.query_params
if "section" in params:
    st.session_state.section = params["section"]


# ============================================================
# A3500 – API BCRA (idVariable = 84)
# ============================================================
@st.cache_data(ttl=60 * 60)
def get_a3500() -> pd.DataFrame:
    url = "https://api.bcra.gob.ar/estadisticas/v4.0/Monetarias/84"
    params = {"Limit": 1000, "Offset": 0}
    data = []

    for _ in range(3):
        try:
            while True:
                r = requests.get(url, params=params, timeout=10, verify=False)
                r.raise_for_status()
                payload = r.json()

                results = payload.get("results", [])
                if not results:
                    break

                detalle = results[0].get("detalle", [])
                if not detalle:
                    break

                data.extend(detalle)

                meta = payload["metadata"]["resultset"]
                params["Offset"] += params["Limit"]
                if params["Offset"] >= meta["count"]:
                    break
            break
        except requests.exceptions.RequestException:
            pass

    if not data:
        st.warning("⚠️ No se pudo conectar con la API del BCRA (A3500).")
        return pd.DataFrame(columns=["Date", "FX"])

    df = pd.DataFrame(data)
    df["Date"] = pd.to_datetime(df["fecha"], errors="coerce")
    df["FX"] = pd.to_numeric(df["valor"], errors="coerce")

    return (
        df[["Date", "FX"]]
        .dropna()
        .drop_duplicates(subset=["Date"])
        .sort_values("Date")
        .reset_index(drop=True)
    )


# ============================================================
# REM – última publicación
# ============================================================
@st.cache_data(ttl=60 * 60)
def get_rem_last():
    url = (
        "https://www.bcra.gob.ar/archivos/Pdfs/PublicacionesEstadisticas/"
        "historico-relevamiento-expectativas-mercado.xlsx"
    )
    df = pd.read_excel(url, sheet_name="Base de Datos Completa", skiprows=1)

    rem = df.loc[
        (df["Variable"] == "Precios minoristas (IPC nivel general; INDEC)")
        & (df["Referencia"] == "var. % mensual")
    ].copy()

    latest = rem["Fecha de pronóstico"].max()

    return (
        rem.loc[rem["Fecha de pronóstico"] == latest]
        .sort_values("Período")
        .tail(24)
        .rename(columns={"Período": "Date", "Mediana": "v_m_REM"})
        .assign(Date=lambda x: pd.to_datetime(x["Date"], errors="coerce"))
        .reset_index(drop=True)
    )


# ============================================================
# IPC – INDEC FTP (completo)
# ============================================================
@st.cache_data(ttl=12 * 60 * 60)
def get_ipc_indec_full() -> pd.DataFrame:
    url = "https://www.indec.gob.ar/ftp/cuadros/economia/serie_ipc_divisiones.csv"
    try:
        df = pd.read_csv(url, sep=";", decimal=",", encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(url, sep=";", decimal=",", encoding="latin1")

    df["Codigo"] = pd.to_numeric(df["Codigo"], errors="coerce")
    df["Periodo"] = pd.to_datetime(df["Periodo"].astype(str), format="%Y%m", errors="coerce")

    for c in ["Descripcion", "Clasificador", "Region"]:
        df[c] = df[c].astype(str).str.strip()

    for c in ["Indice_IPC", "v_m_IPC", "v_i_a_IPC"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return (
        df.dropna(subset=["Periodo"])
        .sort_values("Periodo")
        .reset_index(drop=True)
    )


@st.cache_data(ttl=12 * 60 * 60)
def get_ipc_nacional_nivel_general() -> pd.DataFrame:
    df = get_ipc_indec_full()

    tmp = (
        df[(df["Codigo"] == 0) & (df["Region"] == "Nacional")]
        .dropna(subset=["v_m_IPC"])
        .rename(columns={"Periodo": "Date"})
        .sort_values("Date")
    )
    tmp["Period"] = tmp["Date"].dt.to_period("M")
    tmp["v_m_CPI"] = tmp["v_m_IPC"] / 100.0

    return (
        tmp[["Date", "v_m_CPI", "Period"]]
        .drop_duplicates("Period")
        .sort_values("Period")
        .reset_index(drop=True)
    )


# ============================================================
# Bandas 2025 / 2026
# ============================================================
def build_bands_2025(start, end, lower0, upper0):
    g_up = (1 + 0.01) ** (1 / 30)
    g_dn = (1 - 0.01) ** (1 / 30)

    dates = pd.date_range(start, end, freq="D")
    t = np.arange(len(dates))

    return pd.DataFrame(
        {
            "Date": dates,
            "lower": lower0 * (g_dn**t),
            "upper": upper0 * (g_up**t),
        }
    )


def build_bands_2026(bands_2025, rem, ipc):
    rem_m = rem.assign(Period=rem["Date"].dt.to_period("M"))[["Period", "v_m_REM"]]
    m = ipc.merge(rem_m, on="Period", how="outer").sort_values("Period")

    m["v_m_dec"] = np.where(m["v_m_CPI"].notna(), m["v_m_CPI"], m["v_m_REM"] / 100)

    end_month = m.loc[m["v_m_REM"].notna(), "Period"].max() + 2
    b = pd.DataFrame({"Period": pd.period_range("2026-01", end_month, freq="M")})
    b["ref"] = b["Period"] - 2
    b = b.merge(
        m[["Period", "v_m_dec"]].rename(columns={"Period": "ref"}), on="ref", how="left"
    )

    lower0 = bands_2025.loc[bands_2025["Date"] == "2025-12-31", "lower"].iloc[0]
    upper0 = bands_2025.loc[bands_2025["Date"] == "2025-12-31", "upper"].iloc[0]

    cal = pd.DataFrame(
        {"Date": pd.date_range("2026-01-01", b["Period"].max().to_timestamp("M"), freq="D")}
    )
    cal["Period"] = cal["Date"].dt.to_period("M")
    cal = cal.merge(b[["Period", "v_m_dec"]], on="Period", how="left")

    r_d = (1 + cal["v_m_dec"]) ** (1 / 30) - 1
    cal["lower"] = lower0 * (1 - r_d).cumprod()
    cal["upper"] = upper0 * (1 + r_d).cumprod()

    return cal[["Date", "lower", "upper"]]


# ============================================================
# Tasas – BCRA Monetarias
# ============================================================
@st.cache_data(ttl=60 * 60)
def get_monetaria_serie(id_variable: int) -> pd.DataFrame:
    url = f"https://api.bcra.gob.ar/estadisticas/v4.0/Monetarias/{id_variable}"
    r = requests.get(url, timeout=10, verify=False)
    data = r.json()["results"][0]["detalle"]

    df = pd.DataFrame(data)
    df["Date"] = pd.to_datetime(df["fecha"], errors="coerce")
    df["value"] = pd.to_numeric(df["valor"], errors="coerce")

    return (
        df[["Date", "value"]]
        .dropna()
        .drop_duplicates(subset=["Date"])
        .sort_values("Date")
        .reset_index(drop=True)
    )


# ============================================================
# HOME
# ============================================================
if st.session_state.section == "home":

    st.markdown(
        """
        <style>
          /* Oculta SOLO el título/caption global en Home (sin matar textos internos) */
          div[data-testid="stAppViewContainer"] h1 { display:none; }
          div[data-testid="stAppViewContainer"] .stCaption { display:none; }

          /* Fondo gris claro */
          [data-testid="stAppViewContainer"] { background: #f2f4f7; }

          /* Contenedor centrado */
          .home-wrap{
            max-width: 980px;
            margin: 0 auto;
            padding-top: 10px;
            text-align: center;
          }

          .home-title{
            font-size: 44px;
            font-weight: 800;
            color: #0b2b4c;
            margin-bottom: 10px;
          }

          .home-subtitle{
            font-size: 18px;
            color: #243447;
            margin-bottom: 28px;
          }

          /* Cards SOLO para botones dentro de .home-cards */
          .home-cards div.stButton > button{
            width: 100% !important;
            background: #dbeafe !important;
            border: 1px solid rgba(11,43,76,0.18) !important;
            border-radius: 18px !important;
            padding: 18px 18px !important;
            height: 90px !important;
            box-shadow: 0 8px 22px rgba(0,0,0,0.06) !important;
            transition: all 0.15s ease-in-out !important;
          }

          .home-cards div.stButton > button:hover{
            transform: translateY(-2px);
            box-shadow: 0 12px 28px rgba(0,0,0,0.10) !important;
            border-color: rgba(11,43,76,0.30) !important;
          }

          /* Texto (emoji + nombre) */
          .home-cards div.stButton > button{
            color: #0b2b4c !important;
            font-weight: 800 !important;
            font-size: 20px !important;
          }

          @media (max-width: 900px){
            .home-title{ font-size: 36px; }
          }
        </style>

        <div class="home-wrap">
          <div class="home-title">Macroeconomía</div>
          <div class="home-subtitle">Seleccioná una variable</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Centrado de las cards
    left_pad, mid, right_pad = st.columns([1, 6, 1])
    with mid:
        st.markdown('<div class="home-cards">', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)

        with c1:
            if st.button("💱  Tipo de cambio", use_container_width=True):
                st.session_state.section = "fx"
                st.rerun()

        with c2:
            if st.button("📈  Tasa de interés", use_container_width=True):
                st.session_state.section = "tasa"
                st.rerun()

        with c3:
            if st.button("🛒  Precios", use_container_width=True):
                st.session_state.section = "precios"
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div style='height:100px'></div>", unsafe_allow_html=True)

        logo_col = st.columns([2, 1, 2])
        with logo_col[1]:
            st.image(
            "assets/logo_ceu.png",
    )

    


# ============================================================
# SECCIÓN: TIPO DE CAMBIO
# ============================================================
if st.session_state.section == "fx":

    st.divider()
    if st.button("← Volver al inicio"):
        st.session_state.section = "home"
        st.rerun()

    with st.spinner("Cargando datos..."):
        fx = get_a3500()
        rem = get_rem_last()
        ipc = get_ipc_nacional_nivel_general()

        bands_2025 = build_bands_2025("2025-04-14", "2025-12-31", 1000.0, 1400.0)
        bands_2026 = build_bands_2026(bands_2025, rem, ipc)
        bands = pd.concat([bands_2025, bands_2026]).sort_values("Date")

        df = bands.merge(fx, on="Date", how="left")

    st.subheader("Tipo de cambio")

    c_left, c_right = st.columns([1, 3])

    with c_left:
        last_fx = df["FX"].dropna().iloc[-1]
        st.markdown(
            f"<div style='font-size:46px; font-weight:700'>{last_fx:,.0f}</div>",
            unsafe_allow_html=True,
        )

    with c_right:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Date"], y=df["upper"], name="Banda superior", line=dict(dash="dash")))
        fig.add_trace(go.Scatter(x=df["Date"], y=df["lower"], name="Banda inferior", line=dict(dash="dash"), fill="tonexty"))
        fig.add_trace(go.Scatter(x=df["Date"], y=df["FX"], name="A3500"))
        fig.update_layout(hovermode="x unified", height=600)
        fig.update_yaxes(title_text="ARS / USD")
        fig.update_xaxes(title_text="")
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# SECCIÓN: TASA DE INTERÉS
# ============================================================
if st.session_state.section == "tasa":

    st.divider()
    if st.button("← Volver al inicio"):
        st.session_state.section = "home"
        st.rerun()

    tasa = get_monetaria_serie(145)

    c1, c2 = st.columns([1, 3])

    with c1:
        last_val = tasa["value"].iloc[-1]
        st.markdown(
            f"<div style='font-size:46px; font-weight:700'>{last_val:.1f}%</div>",
            unsafe_allow_html=True,
        )

    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tasa["Date"], y=tasa["value"], name="Tasa"))
        fig.update_layout(hovermode="x unified", height=450)
        fig.update_yaxes(title_text="% TNA", ticksuffix="%")
        fig.update_xaxes(title_text="")
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# SECCIÓN: PRECIOS
# ============================================================
if st.session_state.section == "precios":

    st.divider()
    if st.button("← Volver al inicio"):
        st.session_state.section = "home"
        st.rerun()

    ipc = get_ipc_indec_full()
    ipc = ipc[ipc["Region"] == "Nacional"]

    opciones = sorted(ipc["Descripcion"].unique())
    default = opciones.index("Nivel general") if "Nivel general" in opciones else 0
    desc = st.selectbox("Seleccioná una división", opciones, index=default)

    serie = ipc[ipc["Descripcion"] == desc].dropna(subset=["v_m_IPC"])

    c1, c2 = st.columns([1, 3])

    with c1:
        last_val = serie["v_m_IPC"].iloc[-1]
        st.markdown(
            f"<div style='font-size:46px; font-weight:700'>{last_val:.1f}%</div>",
            unsafe_allow_html=True,
        )

    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=serie["Periodo"], y=serie["v_m_IPC"], name="Variación mensual"))
        fig.update_layout(hovermode="x unified", height=450)
        fig.update_yaxes(title_text="Variación mensual (%)", ticksuffix="%")
        fig.update_xaxes(title_text="")
        st.plotly_chart(fig, use_container_width=True)

