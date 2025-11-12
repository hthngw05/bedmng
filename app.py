import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------
st.set_page_config(page_title="MedTrack Analytics", layout="wide")
st.title("🏥 MedTrack — Hospital & LOS Analytics")

# Tabs
tab1, tab2 = st.tabs(["📊 Hospital Occupancy Dashboard", "🧩 LOS Analytics"])

# ----------------------------------------------------------
# TAB 1: Hospital Occupancy (your existing 67_cleaned.csv app)
# ----------------------------------------------------------
with tab1:
    DATE_COL = "Date"
    YEAR_COL = "Years"
    CSV_PATH = "67_cleaned.csv"

    HOSPITAL_META = {
        "AH": {"name": "Alexandra Hospital (AH)", "region": "West", "lat": 1.2879, "lon": 103.8021},
        "CGH": {"name": "Changi General Hospital (CGH)", "region": "East", "lat": 1.3418, "lon": 103.9496},
        "KTPH": {"name": "Khoo Teck Puat Hospital (KTPH)", "region": "North", "lat": 1.4246, "lon": 103.8388},
        "NTFGH": {"name": "Ng Teng Fong General Hospital (NTFGH)", "region": "West", "lat": 1.3347, "lon": 103.7450},
        "NUH(A)": {"name": "National University Hospital (Adults)", "region": "West", "lat": 1.2940, "lon": 103.7834},
        "SGH": {"name": "Singapore General Hospital (SGH)", "region": "Central", "lat": 1.2794, "lon": 103.8340},
        "SKH": {"name": "Sengkang General Hospital (SKH)", "region": "Northeast", "lat": 1.3918, "lon": 103.8931},
        "TTSH": {"name": "Tan Tock Seng Hospital (TTSH)", "region": "Central", "lat": 1.3210, "lon": 103.8450},
        "WH": {"name": "Woodlands Health (WH)", "region": "North", "lat": 1.4360, "lon": 103.7870},
    }

    @st.cache_data
    def load_long_df(path):
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
        hosp_cols = [c for c in df.columns if c not in {DATE_COL, YEAR_COL}]
        long_df = df.melt(id_vars=[DATE_COL, YEAR_COL], value_vars=hosp_cols,
                          var_name="HospitalAbbr", value_name="OccupancyPct")
        long_df["OccupancyPct"] = pd.to_numeric(long_df["OccupancyPct"], errors="coerce")
        long_df["Hospital"] = long_df["HospitalAbbr"].map(lambda a: HOSPITAL_META.get(a, {}).get("name", a))
        long_df["Region"] = long_df["HospitalAbbr"].map(lambda a: HOSPITAL_META.get(a, {}).get("region"))
        long_df["Lat"] = long_df["HospitalAbbr"].map(lambda a: HOSPITAL_META.get(a, {}).get("lat"))
        long_df["Lon"] = long_df["HospitalAbbr"].map(lambda a: HOSPITAL_META.get(a, {}).get("lon"))
        return long_df.dropna(subset=["OccupancyPct", DATE_COL])

    long_df = load_long_df(CSV_PATH)

    # Sidebar filters
    st.sidebar.header("🔎 Filters (Hospital Dashboard)")
    min_d, max_d = long_df[DATE_COL].min(), long_df[DATE_COL].max()
    start_d, end_d = st.sidebar.date_input("Date range", value=(min_d.date(), max_d.date()))
    mask = (long_df[DATE_COL] >= pd.to_datetime(start_d)) & (long_df[DATE_COL] <= pd.to_datetime(end_d))
    view = long_df[mask]

    thr = st.sidebar.slider("High occupancy threshold (%)", 70, 100, 85, 1)

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Hospitals selected", f"{view['Hospital'].nunique():,}")
    k2.metric("Avg occupancy (%)", f"{view['OccupancyPct'].mean():.1f}")
    k3.metric(f"≥{thr}% share", f"{(view['OccupancyPct']>=thr).mean()*100:.0f}%")
    k4.metric("Latest date shown", f"{view[DATE_COL].max().date()}")

    st.divider()

    # Snapshot map + table
    st.subheader("🗺️ Latest Snapshot Map")
    latest = view.loc[view[DATE_COL] == view[DATE_COL].max()].copy()
    fig_map = px.scatter_mapbox(latest, lat="Lat", lon="Lon",
                                color="OccupancyPct", size="OccupancyPct",
                                color_continuous_scale=[(0, "green"), (0.4, "yellow"), (0.8, "red")],
                                range_color=[0, 100], hover_name="Hospital", zoom=11,
                                mapbox_style="carto-positron")
    st.plotly_chart(fig_map, use_container_width=True)

    snap = latest.groupby(["Hospital", "Region"], as_index=False)["OccupancyPct"].mean()
    snap["Status"] = snap["OccupancyPct"].apply(lambda x: "≥ Thr" if x >= thr else "< Thr")
    snap.insert(0, " ", snap["OccupancyPct"].apply(lambda x: "🔴" if x >= thr else ("🟠" if x >= thr-10 else "🔵")))
    st.dataframe(snap.sort_values("OccupancyPct", ascending=False), use_container_width=True)

    st.divider()
    st.subheader("⏱️ Occupancy Trend")
    fig_line = px.line(view.sort_values(DATE_COL), x=DATE_COL, y="OccupancyPct",
                       color="Hospital", markers=True,
                       labels={"OccupancyPct": "Bed Occupancy (%)"})
    fig_line.add_hrect(y0=thr, y1=100, fillcolor="red", opacity=0.12)
    fig_line.add_hline(y=thr, line_dash="dot", line_color="red")
    st.plotly_chart(fig_line, use_container_width=True)
    st.subheader("📈 Occupancy Distribution by Hospital")

    # Multi-select hospital filter (optional if you already have one)
    hospitals = sorted(long_df["Hospital"].unique().tolist())
    sel_hosps = st.multiselect("Select hospitals to compare", hospitals, default=hospitals[:4])
    df_sel = long_df[long_df["Hospital"].isin(sel_hosps)]

    # Plotly density chart
    fig_dist = px.violin(
        df_sel, 
        x="Hospital", 
        y="OccupancyPct",
        color="Hospital",
        box=True, points="all",
        hover_data=["Region"],
        title="Occupancy Distribution for Major Hospitals (2018–2025)",
        labels={"OccupancyPct": "Occupancy Rate (%)"}
    )

    # Or, alternatively, for smoother density lines:
    fig_dist = px.histogram(df_sel, x="OccupancyPct", color="Hospital", marginal="box",
                             nbins=30, opacity=0.6, histnorm="probability density")

    st.plotly_chart(fig_dist, use_container_width=True)
# ----------------------------------------------------------
# TAB 2: LOS ANALYTICS (train_cleaned.csv)
# ----------------------------------------------------------
with tab2:
    from pathlib import Path
    PATH_TRAIN = "train_cleaned.csv"

    if not Path(PATH_TRAIN).exists():
        st.warning("⚠️ `train_cleaned.csv` not found. Place it in the same folder.")
    else:
        import numpy as np
        df = pd.read_csv(PATH_TRAIN)
        df.columns = df.columns.str.strip()

        STAY_ORDER = [
            "0-10", "11-20", "21-30", "31-40", "41-50", "51-60",
            "61-70", "71-80", "81-90", "91-100", "More than 100 Days"
        ]
        STAY_TO_MID = {s: i*10+5 for i, s in enumerate(STAY_ORDER[:-1])}
        STAY_TO_MID["More than 100 Days"] = 105
        df["Stay_midpoint_days"] = df["Stay"].map(STAY_TO_MID)

        # Sidebar
        st.sidebar.header("🔎 Filters (LOS Dashboard)")
        dept = st.sidebar.multiselect("Department", sorted(df["Department"].unique()))
        if dept:
            df = df[df["Department"].isin(dept)]

        st.metric("Rows", f"{len(df):,}")
        st.metric("Avg LOS midpoint", f"{df['Stay_midpoint_days'].mean():.1f}")

        # LOS Distribution
        st.subheader("🏨 Length of Stay Distribution")
        stay_ct = (df["Stay"].value_counts().reindex(STAY_ORDER).fillna(0).reset_index())
        stay_ct.columns = ["Stay", "Count"]
        fig_stay = px.bar(stay_ct, x="Stay", y="Count", text_auto=True)
        st.plotly_chart(fig_stay, use_container_width=True)

        # LOS by severity
        if {"Severity of Illness", "Stay"}.issubset(df.columns):
            tmp = df.groupby(["Severity of Illness", "Stay"]).size().reset_index(name="Count")
            tmp["Share"] = tmp.groupby("Severity of Illness")["Count"].transform(lambda s: s/s.sum())
            fig_stack = px.bar(tmp, x="Severity of Illness", y="Share", color="Stay", category_orders={"Stay": STAY_ORDER})
            st.subheader("🧪 LOS by Severity")
            st.plotly_chart(fig_stack, use_container_width=True)

        # Department ranking
        st.subheader("🏥 Top Departments by Patients")
        dept_ct = df["Department"].value_counts().head(10).reset_index()
        dept_ct.columns = ["Department", "Patients"]
        st.bar_chart(dept_ct.set_index("Department"))
