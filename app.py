# ============================================================
# BCRA – A3500 + Bandas Cambiarias 2025/2026
# Fuente A3500: API BCRA Monetarias (idVariable = 84)
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
st.set_page_config(page_title="BCRA BANDAS", layout="wide")
st.title("BCRA BANDAS")
st.caption("Fuente A3500: BCRA (API Monetarias, id 84) | Bandas: REM + IPC (t−2)")


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
                r = requests.get(
                    url,
                    params=params,
                    timeout=10,      # ⬅ timeout corto
                    verify=False     # ⬅ por certificados
                )
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

            break  # si salió bien, cortamos retry

        except requests.exceptions.RequestException as e:
            last_error = e

    if not data:
        # fallback limpio: no romper la app
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
        (df["Variable"] == "Precios minoristas (IPC nivel general; INDEC)") &
        (df["Referencia"] == "var. % mensual")
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
          .rename(columns={
              "indice_tiempo": "Date",
              "ipc_ng_nacional_tasa_variacion_mensual": "v_m_CPI"
          })[["Date", "v_m_CPI"]]
          .assign(
              Date=lambda x: pd.to_datetime(x["Date"], errors="coerce"),
              Period=lambda x: x["Date"].dt.to_period("M")
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

    return pd.DataFrame({
        "Date": dates,
        "lower": lower0 * (g_dn ** t),
        "upper": upper0 * (g_up ** t),
    })


# ----------------------------
# Bandas 2026 (inflación t−2)
# ----------------------------
def build_bands_2026(bands_2025, rem, ipc):
    rem_m = rem.assign(Period=rem["Date"].dt.to_period("M"))[["Period", "v_m_REM"]]
    m = ipc.merge(rem_m, on="Period", how="outer").sort_values("Period")

    m["v_m_dec"] = np.where(
        m["v_m_CPI"].notna(),
        m["v_m_CPI"],
        m["v_m_REM"] / 100
    )

    end_month = m.loc[m["v_m_REM"].notna(), "Period"].max() + 2
    b = pd.DataFrame({"Period": pd.period_range("2026-01", end_month, freq="M")})
    b["ref"] = b["Period"] - 2
    b = b.merge(
        m[["Period", "v_m_dec"]].rename(columns={"Period": "ref"}),
        on="ref", how="left"
    )

    lower0 = bands_2025.loc[bands_2025["Date"] == "2025-12-31", "lower"].iloc[0]
    upper0 = bands_2025.loc[bands_2025["Date"] == "2025-12-31", "upper"].iloc[0]

    cal = pd.DataFrame({
        "Date": pd.date_range("2026-01-01", b["Period"].max().to_timestamp("M"), freq="D")
    })
    cal["Period"] = cal["Date"].dt.to_period("M")
    cal = cal.merge(b[["Period", "v_m_dec"]], on="Period", how="left")

    r_d = (1 + cal["v_m_dec"]) ** (1 / 30) - 1
    cal["lower"] = lower0 * (1 - r_d).cumprod()
    cal["upper"] = upper0 * (1 + r_d).cumprod()

    return cal[["Date", "lower", "upper"]]



# ----------------------------
# Parámetros fijos (sin inputs)
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

    # asegurar tipos y limpiar duplicados
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

    # Base calendario = bandas (llega al último día)
    # A3500 queda solo donde existe (NaN en findes/feriados/futuro)
    df = bands.merge(fx, on="Date", how="left")

# ----------------------------
# Sidebar: Resumen (sin parámetros)
# ----------------------------
with st.sidebar:
    st.header("Resumen")

    fx_obs = df.dropna(subset=["FX"]).sort_values("Date")
    if fx_obs.empty:
        st.warning("No hay datos de A3500 disponibles ahora.")
    else:
        last_date = fx_obs["Date"].iloc[-1]
        last_fx = float(fx_obs["FX"].iloc[-1])

        # Bandas en la fecha del último FX
        row_last = df.loc[df["Date"] == last_date].tail(1)
        last_lower = float(row_last["lower"].iloc[0]) if not row_last.empty else np.nan
        last_upper = float(row_last["upper"].iloc[0]) if not row_last.empty else np.nan

        # % del TC respecto a banda superior (gap vs upper)
        # ejemplo: -2% significa que está 2% por debajo de la banda superior
        pct_vs_upper = (last_upper / last_fx - 1.0) if np.isfinite(last_upper) and last_upper != 0 else np.nan

        st.metric("Último A3500", f"{last_fx:,.2f}")
        st.caption(f"Fecha: {pd.to_datetime(last_date).date().isoformat()}")

        st.divider()

        c1, c2 = st.columns(2)
        c1.metric("Banda inferior (últ. fecha)", f"{last_lower:,.2f}")
        c2.metric("Banda superior (últ. fecha)", f"{last_upper:,.2f}")

        if np.isfinite(pct_vs_upper):
            st.metric("% vs banda superior", f"{pct_vs_upper*100:,.2f}%")
        else:
            st.metric("% vs banda superior", "—")

        st.divider()

        # Bandas al 31/01/2026
        target = pd.Timestamp("2026-01-31")
        row_3101 = df.loc[df["Date"] == target].tail(1)

        if row_3101.empty:
            st.warning("No hay dato de bandas para 2026-01-31.")
        else:
            b_lower_3101 = float(row_3101["lower"].iloc[0])
            b_upper_3101 = float(row_3101["upper"].iloc[0])

            st.subheader("Bandas al 31/01/2026")
            c3, c4 = st.columns(2)
            c3.metric("Lower 31/01/2026", f"{b_lower_3101:,.2f}")
            c4.metric("Upper 31/01/2026", f"{b_upper_3101:,.2f}")



# ----------------------------
# Gráfico
# ----------------------------
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df["Date"], y=df["upper"],
    name="Banda superior", line=dict(dash="dash")
))

fig.add_trace(go.Scatter(
    x=df["Date"], y=df["lower"],
    name="Banda inferior", line=dict(dash="dash"),
    fill="tonexty", fillcolor="rgba(0,0,0,0.08)"
))

fig.add_trace(go.Scatter(
    x=df["Date"], y=df["FX"],
    name="A3500",
    mode="lines",
    connectgaps=True
))

fig.update_layout(
    title="A3500 y Bandas Cambiarias",
    yaxis_title="ARS / USD",
    hovermode="x unified",
    height=650
)

st.plotly_chart(fig, use_container_width=True)

st.caption("A3500: BCRA API Monetarias (id 84) | Bandas: REM + IPC")
