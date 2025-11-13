import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from pathlib import Path

# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------
st.set_page_config(page_title="MedTrack Analytics", layout="wide")
st.title("🏥 MedTrack — Hospital & LOS Analytics")

def kpi_card(title, value, caption="", color="#2D6CDF"):
    st.markdown(
        f"""
        <div style="
            background:{color};
            padding:18px 16px;
            border-radius:14px;
            color:white;
            box-shadow:0 2px 10px rgba(0,0,0,0.08);
            height:100%;
        ">
          <div style="font-size:38px; font-weight:700; line-height:1;">{value}</div>
          <div style="font-size:15px; opacity:0.95; margin-top:6px; font-weight:600;">{title}</div>
          <div style="font-size:12px; opacity:0.85; margin-top:4px;">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

tab1, tab2 = st.tabs(["📊 Hospital Occupancy Dashboard", "🧩 LOS Analytics"])

# ----------------------------------------------------------
# TAB 1: Hospital Occupancy (67_cleaned.csv)
# ----------------------------------------------------------
with tab1:
    DATE_COL = "Date"
    YEAR_COL = "Years"
    CSV_PATH = "67_cleaned.csv"

    HOSPITAL_META = {
        "AH":    {"name": "Alexandra Hospital (AH)",                     "region": "West",      "lat": 1.2879, "lon": 103.8021},
        "CGH":   {"name": "Changi General Hospital (CGH)",               "region": "East",      "lat": 1.3418, "lon": 103.9496},
        "KTPH":  {"name": "Khoo Teck Puat Hospital (KTPH)",              "region": "North",     "lat": 1.4246, "lon": 103.8388},
        "NTFGH": {"name": "Ng Teng Fong General Hospital (NTFGH)",       "region": "West",      "lat": 1.3347, "lon": 103.7450},
        "NUH(A)":{"name": "National University Hospital (Adults)",       "region": "West",      "lat": 1.2940, "lon": 103.7834},
        "SGH":   {"name": "Singapore General Hospital (SGH)",            "region": "Central",   "lat": 1.2794, "lon": 103.8340},
        "SKH":   {"name": "Sengkang General Hospital (SKH)",             "region": "Northeast", "lat": 1.3918, "lon": 103.8931},
        "TTSH":  {"name": "Tan Tock Seng Hospital (TTSH)",               "region": "Central",   "lat": 1.3210, "lon": 103.8450},
        "WH":    {"name": "Woodlands Health (WH)",                       "region": "North",     "lat": 1.4360, "lon": 103.7870},
    }

    @st.cache_data
    def load_long_df(path):
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
        id_vars = [c for c in [DATE_COL, YEAR_COL] if c in df.columns]  # guard if Years not present
        hosp_cols = [c for c in df.columns if c not in id_vars]
        long_df = df.melt(
            id_vars=id_vars,
            value_vars=hosp_cols,
            var_name="HospitalAbbr",
            value_name="OccupancyPct"
        )
        long_df["OccupancyPct"] = pd.to_numeric(long_df["OccupancyPct"], errors="coerce")
        long_df["Hospital"] = long_df["HospitalAbbr"].map(lambda a: HOSPITAL_META.get(a, {}).get("name", a))
        long_df["Region"]   = long_df["HospitalAbbr"].map(lambda a: HOSPITAL_META.get(a, {}).get("region"))
        long_df["Lat"]      = long_df["HospitalAbbr"].map(lambda a: HOSPITAL_META.get(a, {}).get("lat"))
        long_df["Lon"]      = long_df["HospitalAbbr"].map(lambda a: HOSPITAL_META.get(a, {}).get("lon"))
        long_df = long_df.dropna(subset=["OccupancyPct", DATE_COL])
        return long_df

    occ_df = load_long_df(CSV_PATH)

    # Sidebar filters
    st.sidebar.header("🔎 Filters (Hospital Dashboard)")
    min_d, max_d = occ_df[DATE_COL].min(), occ_df[DATE_COL].max()
    start_d, end_d = st.sidebar.date_input(
        "Date range",
        value=(min_d.date(), max_d.date())
    )
    mask = (occ_df[DATE_COL] >= pd.to_datetime(start_d)) & (occ_df[DATE_COL] <= pd.to_datetime(end_d))
    view_occ = occ_df[mask]

    thr = st.sidebar.slider("High occupancy threshold (%)", 50, 100, 85, 1)

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Hospitals selected", f"{view_occ['Hospital'].nunique():,}")
    k2.metric("Avg occupancy (%)", f"{view_occ['OccupancyPct'].mean():.1f}")
    k3.metric(f"≥{thr}% share", f"{(view_occ['OccupancyPct']>=thr).mean()*100:.0f}%")
    k4.metric("Latest date shown", f"{view_occ[DATE_COL].max().date()}")

    st.divider()

    # Snapshot map + table
    st.subheader("🗺️ Latest Snapshot Map")
    latest = view_occ.loc[view_occ[DATE_COL] == view_occ[DATE_COL].max()].copy()
    latest = latest.dropna(subset=["Lat", "Lon"])

    # Status buckets
    latest["Status"] = np.select(
        [
            latest["OccupancyPct"] >= thr,
            (latest["OccupancyPct"] >= (thr - 10)) & (latest["OccupancyPct"] < thr),
        ],
        ["High (≥ Thr)", "Near Thr (−10 to < Thr)"],
        default="Low (< Thr −10)"
    )

    status_colors = {
        "High (≥ Thr)": "#ef4444",            # red 🔴
        "Near Thr (−10 to < Thr)": "#f59e0b", # amber 🟠
        "Low (< Thr −10)": "#3b82f6",         # blue 🔵
    }

    fig_map = px.scatter_mapbox(
        latest, lat="Lat", lon="Lon",
        color="Status", size="OccupancyPct", size_max=28,
        hover_name="Hospital",
        hover_data={"Region": True, "OccupancyPct": True, "Lat": False, "Lon": False},
        zoom=11, mapbox_style="carto-positron",
        color_discrete_map=status_colors,
    )
    st.plotly_chart(fig_map, use_container_width=True)

    snap = (
        latest.groupby(["Hospital", "Region", "Status"], as_index=False)["OccupancyPct"]
        .mean()
    )
    snap["Δ from Thr (%)"] = (snap["OccupancyPct"] - thr).round(1)
    dot = {"High (≥ Thr)": "🔴", "Near Thr (−10 to < Thr)": "🟠", "Low (< Thr −10)": "🔵"}
    snap.insert(0, " ", snap["Status"].map(dot))
    snap_display = snap.style.background_gradient(subset=["OccupancyPct"], cmap="RdYlGn_r")
    st.dataframe(snap_display, use_container_width=True)

    st.divider()

    st.subheader("⏱️ Occupancy Trend")
    fig_line = px.line(
        view_occ.sort_values(DATE_COL),
        x=DATE_COL,
        y="OccupancyPct",
        color="Hospital",
        markers=True,
        labels={"OccupancyPct": "Bed Occupancy (%)"},
    )
    fig_line.add_hrect(y0=thr, y1=100, fillcolor="red", opacity=0.12)
    fig_line.add_hline(y=thr, line_dash="dot", line_color="red")
    st.plotly_chart(fig_line, use_container_width=True)

    st.subheader("📈 Occupancy Distribution by Hospital")
    hospitals = sorted(occ_df["Hospital"].unique().tolist())
    sel_hosps = st.multiselect("Select hospitals to compare", hospitals, default=hospitals[:4])
    df_sel = occ_df[occ_df["Hospital"].isin(sel_hosps)]
    fig_dist = px.histogram(
        df_sel,
        x="OccupancyPct",
        color="Hospital",
        marginal="box",
        nbins=30,
        opacity=0.6,
        histnorm="probability density",
        title="Occupancy Distribution for Major Hospitals (2018–2025)",
        labels={"OccupancyPct": "Occupancy Rate (%)"},
    )
    st.plotly_chart(fig_dist, use_container_width=True)

# ----------------------------------------------------------
# TAB 2: LOS ANALYTICS (train_cleaned.csv)
# ----------------------------------------------------------
with tab2:
    PATH_TRAIN = Path("train_cleaned.csv")
    if not PATH_TRAIN.exists():
        st.warning("⚠️ `train_cleaned.csv` not found. Place it in the same folder.")
    else:
        los_df = pd.read_csv(PATH_TRAIN)
        los_df.columns = los_df.columns.str.strip()

        # --- LOS mapping ---
        STAY_ORDER = [
            "0-10", "11-20", "21-30", "31-40", "41-50",
            "51-60", "61-70", "71-80", "81-90", "91-100",
            "More than 100 Days"
        ]
        STAY_TO_MID = {
            "0-10": 5,
            "11-20": 15,
            "21-30": 25,
            "31-40": 35,
            "41-50": 45,
            "51-60": 55,
            "61-70": 65,
            "71-80": 75,
            "81-90": 85,
            "91-100": 95,
            "More than 100 Days": 105,
        }

        if "Stay" in los_df.columns:
            los_df["Stay"] = pd.Categorical(los_df["Stay"], categories=STAY_ORDER, ordered=True)
            los_df["Stay_midpoint_days"] = los_df["Stay"].map(STAY_TO_MID).astype(float)
        else:
            st.error("Column 'Stay' is required in train_cleaned.csv")
            st.stop()

        # --- Sidebar filters (no readmission / visits) ---
        st.sidebar.header("🔎 Filters (LOS Dashboard)")

        def multisel(df, col, label=None):
            if col not in df.columns:
                return df
            opts = sorted(df[col].dropna().astype(str).unique().tolist())
            sel = st.sidebar.multiselect(label or col, opts, default=opts)
            if sel:
                df = df[df[col].astype(str).isin(sel)]
            return df

        view = los_df.copy()
        view = multisel(view, "Hospital_region_code", "Region")
        view = multisel(view, "Hospital_code", "Hospital code")
        view = multisel(view, "Department")
        view = multisel(view, "Ward_Type", "Ward type")
        view = multisel(view, "Ward_Facility_Code", "Ward facility")
        view = multisel(view, "Severity of Illness", "Severity")
        view = multisel(view, "Type of Admission", "Admission type")
        view = multisel(view, "Age")

        # Optional numeric filters
        if "Bed Grade" in view.columns and len(view) > 0:
            lo, hi = int(np.nanmin(view["Bed Grade"])), int(np.nanmax(view["Bed Grade"]))
            g0, g1 = st.sidebar.slider("Bed Grade", min_value=lo, max_value=hi, value=(lo, hi))
            view = view[(view["Bed Grade"] >= g0) & (view["Bed Grade"] <= g1)]

        if "Admission_Deposit" in view.columns and len(view) > 0:
            lo = int(np.nanpercentile(view["Admission_Deposit"], 1))
            hi = int(np.nanpercentile(view["Admission_Deposit"], 99))
            d0, d1 = st.sidebar.slider(
                "Admission deposit (trimmed)",
                min_value=lo,
                max_value=hi,
                value=(lo, hi),
            )
            view = view[(view["Admission_Deposit"] >= d0) & (view["Admission_Deposit"] <= d1)]

        st.sidebar.download_button(
            "⬇️ Download filtered rows (CSV)",
            data=view.to_csv(index=False).encode("utf-8"),
            file_name="train_filtered.csv",
            mime="text/csv",
        )

        # ===================== KPIs (4 cards) =====================
        total_patients = len(view)

        # Critical cases: Extreme severity OR Emergency admission
        crit_mask = pd.Series(False, index=view.index)
        if "Severity of Illness" in view.columns:
            crit_mask |= (view["Severity of Illness"] == "Extreme")
        if "Type of Admission" in view.columns:
            crit_mask |= (view["Type of Admission"] == "Emergency")
        critical_count = int(crit_mask.sum())

        # Admission type summary
        top_type, top_count, top_share, admit_summary_caption = "—", 0, 0.0, "—"
        admit_counts = pd.Series(dtype=int)
        if "Type of Admission" in view.columns and total_patients > 0:
            admit_counts = view["Type of Admission"].value_counts(dropna=True)
            admit_share = admit_counts / total_patients * 100
            if not admit_counts.empty:
                top_type  = str(admit_counts.index[0])
                top_count = int(admit_counts.iloc[0])
                top_share = float(admit_share.iloc[0])
                pieces = [
                    f"{t}: {int(c):,} ({admit_share[t]:.0f}%)"
                    for t, c in admit_counts.items()
                ]
                admit_summary_caption = " • ".join(pieces[:5]) + (
                    f" • +{len(pieces)-5} more" if len(pieces) > 5 else ""
                )

        # Average stay (days)
        avg_stay_days = float(view["Stay_midpoint_days"].mean()) if len(view) else float("nan")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi_card("Total Patients", f"{total_patients:,}", "Filtered selection", "#0ea5e9")
        with c2:
            kpi_card("Critical Cases", f"{critical_count:,}", "Extreme or Emergency", "#ef4444")
        with c3:
            kpi_card("Top Admission Type", top_type, f"{top_count:,} ", "#f59e0b")
        with c4:
            kpi_card("Average Stay (days)", f"{avg_stay_days:.1f}", "Midpoint estimate", "#10b981")

        # Optional full table under cards
        if "Type of Admission" in view.columns and total_patients > 0 and not admit_counts.empty:
            st.markdown("**Admission type breakdown**")
            st.dataframe(
                pd.DataFrame({
                    "Type of Admission": admit_counts.index,
                    "Patients": admit_counts.values,
                    "Share (%)": (admit_counts / total_patients * 100).round(1).values
                }),
                use_container_width=True,
                hide_index=True,
            )

        st.divider()

        # ===================== Visuals =====================
        st.subheader("🏨 Length of Stay distribution")
        if "Stay" in view.columns:
            stay_ct = (
                view["Stay"].value_counts(dropna=False)
                .reindex(STAY_ORDER)
                .fillna(0)
                .rename_axis("Stay")
                .reset_index(name="Patients")
            )
            fig_stay = px.bar(
                stay_ct,
                x="Stay",
                y="Patients",
                text_auto=True,
                labels={"Patients": "Patients"},
            )
            st.plotly_chart(fig_stay, use_container_width=True)
        else:
            st.info("Column 'Stay' not found — cannot plot LOS distribution.")

        st.subheader("👵 Age vs Average Length of Stay (LOS)")

        # Compute average LOS by age group
        if {"Age", "Stay_midpoint_days"}.issubset(view.columns):
            age_los = (
                view.groupby("Age", as_index=False)["Stay_midpoint_days"]
                .mean()
                .rename(columns={"Stay_midpoint_days": "Avg_LOS"})
                .sort_values("Age")
            )
        
            # Bar Chart
            fig_age_los = px.bar(
                age_los,
                x="Age",
                y="Avg_LOS",
                text_auto=".1f",
                color="Age",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                labels={"Avg_LOS": "Average Length of Stay (days)", "Age": "Age Group"},
                title="📈 Average Length of Stay (LOS) by Age Group",
            )
        
            fig_age_los.update_layout(showlegend=False)
            fig_age_los.update_traces(textposition="outside")
        
            st.plotly_chart(fig_age_los, use_container_width=True)
        else:
            st.info("Required columns 'Age' and 'Stay_midpoint_days' not found.")

        st.subheader("📊 Average LOS by Illness Severity and Admission Type")
        needed_cols = {"Severity of Illness", "Type of Admission", "Stay_midpoint_days"}
        if needed_cols.issubset(view.columns) and len(view) > 0:
            los_sev_adm = (
                view
                .groupby(["Severity of Illness", "Type of Admission"], as_index=False)["Stay_midpoint_days"]
                .mean()
            )

            fig_los = px.bar(
                los_sev_adm,
                x="Severity of Illness",
                y="Stay_midpoint_days",
                color="Type of Admission",
                barmode="group",
                labels={
                    "Severity of Illness": "Severity of Illness",
                    "Stay_midpoint_days": "Average Length of Stay (days)",
                    "Type of Admission": "Type of Admission",
                },
                title="Average Length of Stay by Illness Severity and Admission Type",
            )

            st.plotly_chart(fig_los, use_container_width=True)
        else:
            st.info("Columns 'Severity of Illness', 'Type of Admission' and 'Stay_midpoint_days' are required for this chart.")

        l1, r1 = st.columns(2)
        with l1:
            st.subheader("🚑 Admission type mix")
            if "Type of Admission" in view.columns:
                adm = (
                    view["Type of Admission"]
                    .value_counts()
                    .rename_axis("Type")
                    .reset_index(name="Count")
                )
                fig_adm = px.pie(adm, names="Type", values="Count", hole=0.35)
                st.plotly_chart(fig_adm, use_container_width=True)
            else:
                st.info("'Type of Admission' not found.")
        with r1:
            st.subheader("👥 Age band distribution")
            if "Age" in view.columns:
                age_ct = (
                    view["Age"].value_counts()
                    .rename_axis("Age")
                    .reset_index(name="Count")
                    .sort_values("Age")
                )
                fig_age = px.bar(age_ct, x="Age", y="Count", text_auto=True)
                st.plotly_chart(fig_age, use_container_width=True)
            else:
                st.info("'Age' not found.")

        st.subheader("🏥 Department volume (Top 15)")
        if "Department" in view.columns:
            dept = (
                view["Department"]
                .value_counts()
                .head(15)
                .rename_axis("Department")
                .reset_index(name="Patients")
            )
            fig_dept = px.bar(dept, x="Department", y="Patients", text_auto=True)
            st.plotly_chart(fig_dept, use_container_width=True)

        st.subheader("🛏️ Ward type volume")
        if "Ward_Type" in view.columns:
            ward = (
                view["Ward_Type"]
                .value_counts()
                .rename_axis("Ward_Type")
                .reset_index(name="Patients")
            )
            fig_ward = px.bar(ward, x="Ward_Type", y="Patients", text_auto=True)
            st.plotly_chart(fig_ward, use_container_width=True)

        with st.expander("🧾 View filtered rows"):
            st.dataframe(view.head(1000), use_container_width=True)
