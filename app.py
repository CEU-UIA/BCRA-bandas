# ============================================================
# Macroeconomía – CEU.UIA
# Tipo de cambio (A3500) + Bandas 2025/2026
# Tasa de interés (BCRA Monetarias id 145)
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
st.set_page_config(page_title="Macroeconomía - CEU.UIA", layout="wide")
st.title("Macroeconomía – CEU.UIA")
st.caption("Centro de Estudios de la Unión Industrial Argentina")


# ----------------------------
# A3500 – API BCRA (idVariable = 84)
# ----------------------------
@st.cache_data(ttl=60 * 60)
def get_a3500() -> pd.DataFrame:
    url = "https://api.bcra.gob.ar/estadisticas/v4.0/Monetarias/84"
    params = {"Limit": 1000, "Offset": 0}
    data = []

    for _ in range(3):  # hasta 3 intentos
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
        st.warning(
            "⚠️ No se pudo conectar con la API del BCRA (A3500). "
            "Se muestran solo las bandas."
        )
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


# ----------------------------
# REM (última publicación)
# ----------------------------
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


# ----------------------------
# IPC (datos.gob.ar)
# ----------------------------
@st.cache_data(ttl=24 * 60 * 60)
def get_ipc():
    url = (
        "https://infra.datos.gob.ar/catalog/sspm/dataset/145/distribution/145.3/download/"
        "indice-precios-al-consumidor-nivel-general-base-diciembre-2016-mensual.csv"
    )

    return (
        pd.read_csv(url)
        .rename(
            columns={
                "indice_tiempo": "Date",
                "ipc_ng_nacional_tasa_variacion_mensual": "v_m_CPI",
            }
        )[["Date", "v_m_CPI"]]
        .assign(
            Date=lambda x: pd.to_datetime(x["Date"], errors="coerce"),
            Period=lambda x: x["Date"].dt.to_period("M"),
        )
        .drop_duplicates("Period")
        .sort_values("Period")
    )


# ----------------------------
# Bandas 2025
# ----------------------------
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


# ----------------------------
# Bandas 2026 (inflación t−2)
# ----------------------------
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


# ----------------------------
# Tasas – BCRA Monetarias (id variable)
# ----------------------------
@st.cache_data(ttl=60 * 60)
def get_monetaria_serie(id_variable: int) -> pd.DataFrame:
    url = f"https://api.bcra.gob.ar/estadisticas/v4.0/Monetarias/{id_variable}"
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
        return pd.DataFrame(columns=["Date", "value"])

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


# ----------------------------
# Parámetros fijos
# ----------------------------
lower0 = 1000.0
upper0 = 1400.0


# ----------------------------
# Ejecución
# ----------------------------
with st.spinner("Cargando datos..."):
    fx = get_a3500()
    rem = get_rem_last()
    ipc = get_ipc()

    bands_2025 = build_bands_2025("2025-04-14", "2025-12-31", lower0, upper0)
    bands_2026 = build_bands_2026(bands_2025, rem, ipc)

    bands = (
        pd.concat([bands_2025, bands_2026], ignore_index=True)
        .sort_values("Date")
        .reset_index(drop=True)
    )

    fx["Date"] = pd.to_datetime(fx["Date"], errors="coerce")
    bands["Date"] = pd.to_datetime(bands["Date"], errors="coerce")

    fx = (
        fx.dropna(subset=["Date", "FX"])
        .drop_duplicates(subset=["Date"])
        .sort_values("Date")
        .reset_index(drop=True)
    )

    bands = (
        bands.dropna(subset=["Date", "lower", "upper"])
        .drop_duplicates(subset=["Date"])
        .sort_values("Date")
        .reset_index(drop=True)
    )

    df = bands.merge(fx, on="Date", how="left")


# ============================================================
# SECCIÓN: Tipo de cambio
# ============================================================
st.divider()
st.subheader("Tipo de cambio")

fx_obs = df.dropna(subset=["FX"]).sort_values("Date")

c_left, c_right = st.columns([1, 3], vertical_alignment="top")

with c_left:
    st.markdown("### Tipo de Cambio Mayorista")
    if fx_obs.empty:
        st.warning("Sin datos A3500")
    else:
        last_date = fx_obs["Date"].iloc[-1]
        last_fx = float(fx_obs["FX"].iloc[-1])

        st.markdown("**Último dato**")
        st.markdown(
            f"<div style='font-size:46px; font-weight:700; line-height:1.0'>{last_fx:,.0f}</div>",
            unsafe_allow_html=True,
        )
        st.caption(f"Fecha: {pd.to_datetime(last_date).date().isoformat()}")

with c_right:
    fig_fx = go.Figure()

    fig_fx.add_trace(
        go.Scatter(x=df["Date"], y=df["upper"], name="Banda superior", line=dict(dash="dash"))
    )
    fig_fx.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["lower"],
            name="Banda inferior",
            line=dict(dash="dash"),
            fill="tonexty",
            fillcolor="rgba(0,0,0,0.08)",
        )
    )
    fig_fx.add_trace(
        go.Scatter(x=df["Date"], y=df["FX"], name="A3500", mode="lines", connectgaps=True)
    )

    fig_fx.update_layout(
    title=None,
    hovermode="x unified",
    height=600,
    margin=dict(t=30),
    showlegend=True,
)
    fig_fx.update_xaxes(title_text="")           # 👈 mata el "undefined"
    fig_fx.update_yaxes(title_text="ARS / USD")


    

    st.plotly_chart(fig_fx, use_container_width=True)

st.download_button(
    "Descargar CSV (Tipo de cambio)",
    data=df[["Date", "FX", "lower", "upper"]].to_csv(index=False).encode("utf-8"),
    file_name="tipo_de_cambio_bandas.csv",
    mime="text/csv",
)

st.caption("Fuente: BCRA (A3500, API Monetarias id 84) | Bandas: REM + IPC (t−2)")


# ============================================================
# SECCIÓN: Tasa de interés (id 145)
# ============================================================
st.divider()
st.subheader("Tasa de interés")
st.caption("Fuente: BCRA (API Monetarias, id 145)")

tasa = get_monetaria_serie(145)
tasa = tasa[tasa["Date"] >= pd.Timestamp("2025-01-01")].copy()

c_left2, c_right2 = st.columns([1, 3], vertical_alignment="top")

with c_left2:
    st.markdown("### Tasa de Adelantos a CC de Empresa (% TNA)")
    if tasa.empty:
        st.warning("Sin datos")
    else:
        last_date_t = tasa["Date"].iloc[-1]
        last_val_t = float(tasa["value"].iloc[-1])

        st.markdown("**Último dato**")
        st.markdown(
            f"<div style='font-size:46px; font-weight:700; line-height:1.0'>{last_val_t:,.0f}%</div>",
            unsafe_allow_html=True,
        )
        st.caption(f"Fecha: {pd.to_datetime(last_date_t).date().isoformat()}")

with c_right2:
    if not tasa.empty:
        fig_tasa = go.Figure()
        fig_tasa.add_trace(
            go.Scatter(x=tasa["Date"], y=tasa["value"], mode="lines", name="Tasa")
        )

        fig_tasa.update_layout(
        title=None,
        hovermode="x unified",
        height=450,
        margin=dict(t=30),
        showlegend=True,
        )
        fig_tasa.update_xaxes(title_text="")          # 👈 mata el "undefined"
        fig_tasa.update_yaxes(title_text="% TNA", ticksuffix="%")


        st.plotly_chart(fig_tasa, use_container_width=True)

if not tasa.empty:
    st.download_button(
        "Descargar CSV (Tasa)",
        data=tasa.rename(columns={"value": "tasa"}).to_csv(index=False).encode("utf-8"),
        file_name="tasa_adelantos_id145.csv",
        mime="text/csv",
    )
