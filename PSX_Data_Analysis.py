# PSX Streamlit Dashboard — polished, user‑friendly, and explanatory
# How to run (Anaconda Prompt):
#   1) cd to the folder containing this file
#   2) streamlit run psx_streamlit_app_v2.py
#
# Required packages: streamlit, pandas, plotly, matplotlib, seaborn, openpyxl
#   pip install streamlit pandas plotly matplotlib seaborn openpyxl
# or conda install -c conda-forge streamlit pandas plotly matplotlib seaborn openpyxl

from __future__ import annotations
import io
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =============================
# Page & global style
# =============================
st.set_page_config(
    page_title="PSX Dashboards",
    page_icon="📈",
    layout="wide",
)
st.markdown(
    """
    <style>
      /* tighten top padding */
      .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}

      /* st.metric smaller + bolder */
      [data-testid="stMetricValue"] > div {
        font-size: 1.05rem !important;   /* ↓ from ~2rem */
        line-height: 1.15 !important;
        font-weight: 700;
      }
      [data-testid="stMetricLabel"] > div {
        font-size: 0.85rem !important;
        line-height: 1.2 !important;
      }
      span[data-testid="stMetricDelta"] {
        font-size: 0.85rem !important;
        line-height: 1.2 !important;
      }

      /* wrap long sector labels on x-axis (matplotlib fig render) */
      .st-ax {white-space: pre-wrap;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Minimal CSS polish (keeps it light & safe)
st.markdown(
    """
    <style>
      /* tighten top padding */
      .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
      /* nicer metric numbers */
      [data-testid="stMetricValue"] {font-weight: 700;}
      /* wrap long sector labels on x-axis (matplotlib fig render) */
      .st-ax {white-space: pre-wrap;}
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================
# Helpers
# =============================
REQUIRED_COLS = {"Date", "Sector", "Symbol", "Open", "High", "Low", "Close", "Volume"}
DEFAULT_PATH = "Historical_PSX_Data_Featured.xlsx"
TODAY = datetime.today()

def _friendly_date(dt: pd.Timestamp | datetime) -> str:
    return pd.to_datetime(dt).strftime("%b %d, %Y")

@st.cache_data(show_spinner=True)
def load_excel(file_like_or_path: str | io.BytesIO) -> pd.DataFrame:
    df = pd.read_excel(file_like_or_path)
    # Basic cleaning
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date","High"]).sort_values("Date")
    # Normalize column names (strip)
    df.columns = [c.strip() for c in df.columns]
    return df

def validate_columns(df: pd.DataFrame) -> list[str]:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    return missing

def get_company_column(df: pd.DataFrame) -> str:
    return "Company Name" if "Company Name" in df.columns else "Symbol"

def aggregate_sector_daily(df: pd.DataFrame, sector: str, param: str) -> pd.DataFrame:
    df_sector = df.loc[df["Sector"] == sector, ["Date", param]].copy()
    if df_sector.empty:
        return pd.DataFrame(columns=["Date", f"Mean_{param}_{sector}"])
    out = (df_sector.groupby("Date", as_index=False)[param]
                   .mean()
                   .rename(columns={param: f"Mean_{param}_{sector}"}))
    return out

def aggregate_company_daily(df: pd.DataFrame, company_col: str, company: str, param: str, sector: str | None = None) -> pd.DataFrame:
    if sector:
        dfx = df[(df["Sector"] == sector) & (df[company_col] == company)][["Date", param]].copy()
    else:
        dfx = df[df[company_col] == company][["Date", param]].copy()
    if dfx.empty:
        return pd.DataFrame(columns=["Date", f"{company} — {param}"])
    out = (dfx.groupby("Date", as_index=False)[param]
              .mean()
              .rename(columns={param: f"{company} — {param}"}))
    return out

def sector_company_counts(df: pd.DataFrame) -> pd.DataFrame:
    return (df[["Symbol", "Sector"]]
              .drop_duplicates()
              .groupby("Sector")["Symbol"].nunique()
              .sort_values(ascending=False)
              .reset_index(name="Companies"))

# =============================
# Header
# =============================
st.title("Pakistan Stock Exchange Data Analysis")
st.caption(
    f"Interactive dashboard on PSX OHLCV data • Updated: {_friendly_date(TODAY)}"
)
st.caption("Data gets auto-updated daily up to the previous trading day by scraping the official PSX website.")
with st.expander("What you can do here (quick guide)", expanded=False):
    st.markdown(
        """
        - **Sector Trends:** See how many companies are in each sector and track the sector’s daily mean *Open/High/Low/Close/Volume*.\n
        - **Company Trends:** Select a company and compare it with the **sector average**. Optionally add **rolling means**.\n
        - **Downloads:** Export the current sector time series or the entire dataset as CSV.\n
        - **Tips:** Use the **range selector** below each chart to zoom to 1m/3m/6m/YTD/1y or view all.
        """
    )

# =============================
# Sidebar — data & global controls
# =============================
st.sidebar.header("Data & Controls")
df_update=pd.read_csv("https://raw.githubusercontent.com/moin-t/psx_action/refs/heads/main/PSX_Historical_update.csv")
if "Date" in df_update.columns:
    df_update["Date"] = pd.to_datetime(df_update["Date"], errors="coerce")
    df_update = df_update.dropna(subset=["Date","High"]).sort_values("Date")
    # Normalize column names (strip)
    df_update.columns = [c.strip() for c in df_update.columns]
# Load base dataset that your scraper updates on disk
try:
    df = load_excel(DEFAULT_PATH)
    df=pd.concat([df,df_update],ignore_index=True)
    st.sidebar.info(f"Using dataset: {DEFAULT_PATH}\n**Scraped from PSX Official** (Updates daily)")
    try:
        mtime = os.path.getmtime(DEFAULT_PATH)
        st.sidebar.caption(f"Last updated on disk: {datetime.fromtimestamp(mtime).strftime('%b %d, %Y %H:%M')}")
    except Exception:
        pass
except Exception as e:
    st.error(f"""Could not read base dataset at '{DEFAULT_PATH}'.

{e}""")
    st.stop()

# One-click refresh when your scraper has just written new data
if st.sidebar.button("Refresh data", help="Clear cache and reload the base dataset"):
    load_excel.clear()
    st.rerun()

# Basic facts
mind, maxd = df["Date"].min(), df["Date"].max()
num_rows = len(df)
num_companies = df[["Symbol"]].drop_duplicates().shape[0]
num_sectors = df[["Sector"]].drop_duplicates().shape[0]

# KPI row
k1, k2, k3, k4 = st.columns(4)
k1.metric("Rows", f"{num_rows:,}")
k2.metric("Companies", f"{num_companies:,}")
k3.metric("Sectors", f"{num_sectors:,}")
k4.metric("Date Range", f"{_friendly_date(mind)} → {_friendly_date(maxd)}")


# Sidebar filters
all_sectors = sorted(df["Sector"].dropna().unique().tolist())
all_params = ["High", "Low", "Open", "Close", "Volume"]

# Persisted selection (session state)
if "sector" not in st.session_state or st.session_state["sector"] not in all_sectors:
    st.session_state["sector"] = all_sectors[0]
if "param" not in st.session_state or st.session_state["param"] not in all_params:
    st.session_state["param"] = "High"

# Sidebar widgets
st.sidebar.subheader("Global filters")
sel_sector = st.sidebar.selectbox("Sector", all_sectors, index=all_sectors.index(st.session_state["sector"]))
sel_param  = st.sidebar.selectbox("Parameter", all_params, index=all_params.index(st.session_state["param"]))

if st.sidebar.button("Reset filters"):
    sel_sector = all_sectors[0]
    sel_param = "High"

# reflect sidebar into session_state
st.session_state["sector"] = sel_sector
st.session_state["param"] = sel_param

company_col = get_company_column(df)

# =============================
# Tabs layout
# =============================
tab1, tab2, tab3, tab4 = st.tabs(["📊 Sector Trends", "🏢 Company Trends", "🧾 Data & Downloads", "ℹ️ About"])

# =============================
# Tab 1 — Sector Trends
# =============================
with tab1:
    st.subheader("Sector‑Wise Trend Analysis")
    st.write(
        """
        Explore **company distribution** by sector and track **daily mean** values of your chosen parameter.
        Use the **selectors in the sidebar** to switch sector and parameter.
        """
    )

    # --- Company count by sector (interactive, Plotly) ---
    counts_df = sector_company_counts(df)
    c1, c2 = st.columns([2, 3], vertical_alignment="top")
    with c1:
        st.markdown("**Companies by Sector** (sorted)")
        bar = px.bar(
            counts_df,
            x="Companies",
            y="Sector",
            orientation="h",
            title=None,
        )
        bar.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=480)
        bar.update_traces(hovertemplate="%{y}: %{x}<extra></extra>")
        st.plotly_chart(bar, use_container_width=True)

    with c2:
        # Daily mean for selected sector/param
        sector_series = aggregate_sector_daily(df, sel_sector, sel_param)
        series_name = f"Mean_{sel_param}_{sel_sector}"
        if sector_series.empty:
            st.info("No data for this sector/parameter.")
        else:
            line = px.line(
                sector_series,
                x="Date",
                y=series_name,
                title=f"{sel_sector} Sector — Daily Mean of {sel_param}",
                labels={"Date": "Date", series_name: f"Daily Mean {sel_param}"},
            )
            line.update_traces(hovertemplate="Date=%{x|%Y-%m-%d}<br>Mean=%{y:.2f}<extra></extra>")
            line.update_layout(
                hovermode="x unified",
                xaxis=dict(
                    type="date",
                    rangeslider=dict(visible=True),
                    rangeselector=dict(buttons=[
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=3, label="3m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                margin=dict(l=40, r=20, t=60, b=40),
            )
            st.plotly_chart(line, use_container_width=True)

            # Quick stats
            last_date = sector_series["Date"].max()
            last_val = float(sector_series.loc[sector_series["Date"] == last_date, series_name].iloc[0])
            start_30 = last_date - timedelta(days=30)
            start_90 = last_date - timedelta(days=90)
            mask_30 = sector_series["Date"].between(start_30, last_date)
            mask_90 = sector_series["Date"].between(start_90, last_date)
            last_30_mean = sector_series.loc[mask_30, series_name].mean()
            last_90_high = sector_series.loc[mask_90, series_name].max()

            m1, m2, m3 = st.columns(3)
            m1.metric("Latest", f"{last_val:,.2f}", help=f"as of {_friendly_date(last_date)}")
            m2.metric("30‑Day Mean", "N/A" if pd.isna(last_30_mean) else f"{last_30_mean:,.2f}")
            m3.metric("90‑Day High", "N/A" if pd.isna(last_90_high) else f"{last_90_high:,.2f}")

    with st.expander("Explain this view"):
        st.markdown(
            f"""
            - **Left:** Number of listed companies in each sector (higher bars mean more companies).\n
            - **Right:** For **{sel_sector}** sector, we compute the **daily average {sel_param}** across all its companies, so a spike means many companies had high values that day. Use this to spot **seasonality**, **volatility clusters**, or **structural shifts**.
            """
        )

# =============================
# Tab 2 — Company Trends
# =============================
with tab2:
    st.subheader("Company‑Wise Trend Analysis")
    st.write(
        """
        Select a **company** and a **metric** (Open/High/Low/Close/Volume).\n
        Optionally overlay the **sector average** and add **rolling means** for context.
        """
    )

    # Sidebar-like controls inside tab for clarity
    _c1, _c2, _c3 = st.columns([2, 2, 3])
    with _c1:
        cw_param = st.selectbox("Parameter (Company view)", all_params, index=all_params.index(sel_param))
    with _c2:
        # Sector scope for company view (None => all)
        sector_opts = ["All"] + all_sectors
        cw_sector = st.selectbox("Scope sector", sector_opts, index=sector_opts.index(sel_sector) if sel_sector in sector_opts else 0)
    with _c3:
        # Company options depend on sector scope
        if cw_sector == "All":
            comp_list = df[get_company_column(df)].dropna().drop_duplicates().sort_values().tolist()
        else:
            comp_list = (
                df.loc[df["Sector"] == cw_sector, get_company_column(df)]
                  .dropna().drop_duplicates().sort_values().tolist()
            )
        if not comp_list:
            st.warning("No companies match the current filters.")
            st.stop()
        company = st.selectbox("Company", comp_list, index=0)

    opt1, opt2, opt3 = st.columns(3)
    with opt1:
        show_sector_overlay = st.checkbox("Overlay sector mean", value=(cw_sector != "All"))
    with opt2:
        show_markers = st.checkbox("Show markers", value=False)
    with opt3:
        roll_win = st.number_input("Rolling window (days)", min_value=0, max_value=120, value=0, step=5, help="0 = off")

    # Build series
    scope_sector = None if cw_sector == "All" else cw_sector
    comp_series = aggregate_company_daily(df, company_col, company, cw_param, sector=scope_sector)

    if comp_series.empty:
        st.info("No data for this company/parameter.")
    else:
        y_name = comp_series.columns[1]
        figc = px.line(
            comp_series, x="Date", y=y_name,
            title=f"{company} — Daily {cw_param}", labels={"Date": "Date", y_name: f"Daily {cw_param}"}
        )
        if show_markers:
            figc.update_traces(mode="lines+markers")

        # Rolling mean
        if roll_win and roll_win > 0:
            tmp = comp_series.copy()
            tmp["Roll"] = tmp[y_name].rolling(roll_win, min_periods=max(5, roll_win // 3)).mean()
            figc.add_scatter(x=tmp["Date"], y=tmp["Roll"], mode="lines", name=f"{roll_win}-day mean", line=dict(dash="dot"))

        # Sector overlay
        if show_sector_overlay and scope_sector is not None:
            sec_series = aggregate_sector_daily(df, scope_sector, cw_param)
            if not sec_series.empty:
                sname = sec_series.columns[1]
                figc.add_scatter(
                    x=sec_series["Date"], y=sec_series[sname], mode="lines",
                    name=f"{scope_sector} mean", line=dict(dash="dash")
                )

        figc.update_traces(hovertemplate="Date=%{x|%Y-%m-%d}<br>Value=%{y:.2f}<extra></extra>")
        figc.update_layout(
            hovermode="x unified",
            xaxis=dict(
                type="date",
                rangeslider=dict(visible=True),
                rangeselector=dict(buttons=[
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02, bgcolor="rgba(255,255,255,0.7)", bordercolor="rgba(0,0,0,0.1)", borderwidth=1),
            margin=dict(l=40, r=20, t=60, b=40),
        )
        st.plotly_chart(figc, use_container_width=True)

        # Company quick stats
        last_row = comp_series.iloc[-1]
        last_val = float(last_row[y_name])
        last_date = last_row["Date"]
        last_30 = comp_series.tail(30)[y_name].mean()
        last_90_high = comp_series.tail(90)[y_name].max()
        m1, m2, m3 = st.columns(3)
        m1.metric("Latest", f"{last_val:,.2f}", help=f"as of {_friendly_date(last_date)}")
        m2.metric("30‑Day Mean", "N/A" if pd.isna(last_30) else f"{last_30:,.2f}")
        m3.metric("90‑Day High", "N/A" if pd.isna(last_90_high) else f"{last_90_high:,.2f}")

    with st.expander("How to read this"):
        st.markdown(
            """
            - The **solid line** is the selected company's daily average for your chosen metric.\n
            - The **dashed line** (if enabled) shows the **sector's daily mean** as a benchmark.\n
            - A **rolling mean** (optional) smooths short‑term noise to reveal trend.
            """
        )

# =============================
# Tab 3 — Data & Downloads
# =============================
with tab3:
    st.subheader("Data Preview & Export")
    st.write("Quickly inspect the dataset and export what you need.")

    # Preview
    st.markdown("**Sample (first 200 rows):**")
    st.dataframe(df.head(200), use_container_width=True)

    # Full data download
    buf_all = io.StringIO()
    df.to_csv(buf_all, index=False)
    st.download_button(
        label="Download full dataset (CSV)",
        data=buf_all.getvalue().encode("utf-8"),
        file_name="psx_full_dataset.csv",
        mime="text/csv",
    )

    # Current sector time series download
    st.markdown("**Export current sector time series:**")
    sec_ts = aggregate_sector_daily(df, sel_sector, sel_param)
    if sec_ts.empty:
        st.info("No series for current selection.")
    else:
        buf_sec = io.StringIO()
        sec_ts.to_csv(buf_sec, index=False)
        st.download_button(
            label=f"Download {sel_sector} Sector — {sel_param} daily mean (CSV)",
            data=buf_sec.getvalue().encode("utf-8"),
            file_name=f"{sel_sector}_{sel_param}_daily_mean.csv",
            mime="text/csv",
        )

# =============================
# Tab 4 — About / Notes
# =============================
with tab4:
    st.subheader("About this app")
    st.markdown(
        f"""
        **PSX Dashboard (January 01, 2021– To Date)** — a clean, fast Streamlit app for interactive market exploration.
        
        **Data Window:** {_friendly_date(mind)} → {_friendly_date(maxd)} (Scraped from PSX Official website).
        
        **Pro tips**
        - Use the **Download** tab to export the exact time series behind the chart.
        - Customize Streamlit theme (☰ → Settings) for light/dark modes.
        - Press **R** (or click the re-run icon) to refresh the data file.

        **Developers**
        1. Moin Tariq - AI for Digital Earth Lab, Shandong University, Jinan, China.
        2. Muhammad Irfan Haider Khan - School of Artificial Intelligence, OPtics and ElectroNics (iOPEN), NWPU, Xi’an, Shaanxi, China. 
        3. Saif-Ur-Rehman - Turku Data Science Group - Univerity of Turku, Turku Finland.
        """
    )
