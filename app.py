import streamlit as st
import pandas as pd
import plotly.express as px
   
DATE_COL     = 'Date'
YEAR_COL     = 'Years'
CSV_PATH = "67_cleaned.csv"

st.set_page_config(page_title='Hospital AI Analysis', layout='wide')

st.title('Hospital AI Analysis')

HOSPITAL_META = {
    "AH":    {"name": "Alexandra Hospital (AH)",                     "region": "West",      "lat": 1.2879, "lon": 103.8021},
    "CGH":   {"name": "Changi General Hospital (CGH)",               "region": "East",      "lat": 1.3418, "lon": 103.9496},
    "KTPH":  {"name": "Khoo Teck Puat Hospital (KTPH)",              "region": "North",     "lat": 1.4246, "lon": 103.8388},
    "NTFGH": {"name": "Ng Teng Fong General Hospital (NTFGH)",       "region": "West",      "lat": 1.3347, "lon": 103.7450},
    "NUH(A)":{"name": "National University Hospital (NUH) (Adults)", "region": "West",      "lat": 1.2940, "lon": 103.7834},
    "SGH":   {"name": "Singapore General Hospital (SGH)",            "region": "Central",   "lat": 1.2794, "lon": 103.8340},
    "SKH":   {"name": "Sengkang General Hospital (SKH)",             "region": "Northeast", "lat": 1.3918, "lon": 103.8931},
    "TTSH":  {"name": "Tan Tock Seng Hospital (TTSH)",               "region": "Central",   "lat": 1.3210, "lon": 103.8450},
    "WH":    {"name": "Woodlands Health (WH)",                       "region": "North",     "lat": 1.4360, "lon": 103.7870},
}
@st.cache_data
def load_long_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip()
    # parse date
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

    # hospital columns are everything except Years/Date
    hosp_cols = [c for c in df.columns if c not in {DATE_COL, YEAR_COL}]
    # melt to long
    long_df = df.melt(
        id_vars=[col for col in [DATE_COL, YEAR_COL] if col in df.columns],
        value_vars=hosp_cols,
        var_name="HospitalAbbr",
        value_name="OccupancyPct"
    )
    # numeric occupancy
    long_df["OccupancyPct"] = pd.to_numeric(long_df["OccupancyPct"], errors="coerce")

    # attach meta
    long_df["Hospital"] = long_df["HospitalAbbr"].map(lambda a: HOSPITAL_META.get(a, {}).get("name", a))
    long_df["Region"]   = long_df["HospitalAbbr"].map(lambda a: HOSPITAL_META.get(a, {}).get("region"))
    long_df["Lat"]      = long_df["HospitalAbbr"].map(lambda a: HOSPITAL_META.get(a, {}).get("lat"))
    long_df["Lon"]      = long_df["HospitalAbbr"].map(lambda a: HOSPITAL_META.get(a, {}).get("lon"))
    return long_df

long_df = load_long_df(CSV_PATH)

if long_df.empty:
    st.error("CSV appears empty after loading.")
    st.stop()

# drop rows with no coords or occupancy
long_df = long_df.dropna(subset=["Lat", "Lon", "OccupancyPct", DATE_COL])

# ---------- Sidebar filters ----------
st.sidebar.header("🔎 Filters")

# Date range
min_d, max_d = long_df[DATE_COL].min(), long_df[DATE_COL].max()
start_d, end_d = st.sidebar.date_input(
    "Date range",
    value=(min_d.date(), max_d.date()),
    min_value=min_d.date(),
    max_value=max_d.date()
)
mask_date = (long_df[DATE_COL] >= pd.to_datetime(start_d)) & (long_df[DATE_COL] <= pd.to_datetime(end_d))
view = long_df.loc[mask_date].copy()

# Region
regions = ["All"] + sorted(view["Region"].dropna().unique().tolist())
sel_region = st.sidebar.selectbox("Region", regions)
if sel_region != "All":
    view = view[view["Region"] == sel_region]

# Hospitals (with “Select all”)
all_hosps = sorted(view["Hospital"].dropna().unique().tolist())
sel = st.sidebar.multiselect("Hospitals", ["Select all"] + all_hosps, default=["Select all"])
if "Select all" not in sel:
    view = view[view["Hospital"].isin(sel)]

# Threshold
thr = st.sidebar.slider("High occupancy threshold (%)", 70, 100, 85, 1)

# ---------- KPIs ----------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Hospitals selected", f"{view['Hospital'].nunique():,}")
k2.metric("Avg occupancy (%)", f"{view['OccupancyPct'].mean():.1f}" if not view.empty else "—")

# % of (hospital-days) over threshold in the period
if not view.empty:
    pct_over = (view["OccupancyPct"] >= thr).mean() * 100
    k3.metric(f"≥{thr}% (share of records)", f"{pct_over:.0f}%")
else:
    k3.metric(f"≥{thr}% (share of records)", "—")

# latest snapshot date in range
if not view.empty:
    latest_date = view[DATE_COL].max().date()
    k4.metric("Latest date shown", f"{latest_date}")
else:
    k4.metric("Latest date shown", "—")

st.divider()

# ---------- Map (latest snapshot in selected range) ----------
st.subheader("🗺️ Latest Snapshot Map")
if view.empty:
    st.info("No data for current filters.")
else:
    latest = view.loc[view[DATE_COL] == view[DATE_COL].max()].copy()
    # average across duplicates on same day (if any)
    latest = (latest.groupby(["Hospital","Region","Lat","Lon"], as_index=False)["OccupancyPct"]
                    .mean())
    fig_map = px.scatter_mapbox(
        latest,
        lat="Lat", lon="Lon",
        color="OccupancyPct",
        size="OccupancyPct",
        size_max=30,
        color_continuous_scale=[(0, "green"), (0.4, "yellow"), (0.8, "red")],
        range_color=[0, 100],
        hover_name="Hospital",
        hover_data={"Region": True, "OccupancyPct": True},
        zoom=11,
        mapbox_style="carto-positron",
    )
    st.plotly_chart(fig_map, width="stretch")

# ---------- Trend (time series) ----------
st.subheader("⏱️ Occupancy Trend")
if view.empty:
    st.info("No data for current filters.")
else:
    fig_line = px.line(
        view.sort_values(DATE_COL),
        x=DATE_COL, y="OccupancyPct",
        color="Hospital",
        markers=True,
        labels={"OccupancyPct": "Bed Occupancy (%)", DATE_COL: "Date"},
    )
    st.plotly_chart(fig_line, width="stretch")

# ---------- Ranking (mean in selected period) ----------
st.subheader("📊 Average Occupancy in Selected Period")
if view.empty:
    st.info("No data for current filters.")
else:
    rank = (view.groupby(["Hospital","Region"], as_index=False)["OccupancyPct"]
                 .mean()
                 .sort_values("OccupancyPct", ascending=True))
    fig_bar = px.bar(
        rank,
        x="Hospital", y="OccupancyPct",
        color="OccupancyPct",
        color_continuous_scale=[(0, "green"), (0.4, "yellow"), (0.8, "red")],
        range_color=[0, 100],
        text_auto=".1f",
        labels={"OccupancyPct":"Avg Bed Occupancy (%)"},
        category_orders={"Hospital": rank["Hospital"].tolist()}
    )
    st.plotly_chart(fig_bar, width="stretch")
