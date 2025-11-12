import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------- paths --------
DATA_PATH = "train.csv"  # change if needed
OUT_DIR = "analytics_outputs"
CHART_DIR = os.path.join(OUT_DIR, "charts")
os.makedirs(CHART_DIR, exist_ok=True)

# -------- load --------
df = pd.read_csv(DATA_PATH)

# -------- helpers --------
STAY_MAP = {
    "0-10": 5, "11-20": 15, "21-30": 25, "31-40": 35, "41-50": 45,
    "51-60": 55, "61-70": 65, "71-80": 75, "81-90": 85, "91-100": 95,
    "More than 100 Days": 110
}

def ensure_numeric(series):
    return pd.to_numeric(series, errors="coerce")

def bar_chart(x, y, title, xlabel, ylabel, fname, rotation=45):
    plt.figure()
    plt.bar(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation, ha="right")
    plt.tight_layout()
    path = os.path.join(CHART_DIR, fname)
    plt.savefig(path, dpi=200)
    plt.close()
    return path

def line_chart(x, y, title, xlabel, ylabel, fname, rotation=0):
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation, ha="right" if rotation else "center")
    plt.tight_layout()
    path = os.path.join(CHART_DIR, fname)
    plt.savefig(path, dpi=200)
    plt.close()
    return path

def box_plot(groups, values, title, xlabel, ylabel, fname):
    plt.figure()
    uniq = groups.astype(str).unique()
    data = [values[groups.astype(str) == g].dropna().values for g in uniq]
    plt.boxplot(data, labels=uniq)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    path = os.path.join(CHART_DIR, fname)
    plt.savefig(path, dpi=200)
    plt.close()
    return path

# -------- feature engineering --------
df["LOS_days"] = df["Stay"].map(STAY_MAP)
df["Available Extra Rooms in Hospital"] = ensure_numeric(df["Available Extra Rooms in Hospital"])
df["Admission_Deposit"] = ensure_numeric(df["Admission_Deposit"])
if "Visitors with Patient" in df.columns:
    df["Visitors with Patient"] = ensure_numeric(df["Visitors with Patient"])

# order Age
if "Age" in df.columns:
    age_order = ["0-10","11-20","21-30","31-40","41-50","51-60","61-70","71-80","81-90","91-100"]
    df["Age"] = pd.Categorical(df["Age"], categories=age_order, ordered=True)

# -------- 1) Hospital efficiency --------
eff = (
    df.groupby(["Hospital_code","Hospital_region_code"], dropna=False)
      .agg(avg_los=("LOS_days","mean"),
           rooms=("Available Extra Rooms in Hospital","mean"),
           admissions=("LOS_days","size"))
      .reset_index()
)
eff["Efficiency_Index"] = eff["rooms"] / eff["avg_los"]
eff = eff.sort_values("Efficiency_Index", ascending=False)
eff.to_csv(os.path.join(OUT_DIR, "hospital_efficiency.csv"), index=False)

top_eff = eff.head(20)
bar_chart(
    x=top_eff["Hospital_code"].astype(str) + " (" + top_eff["Hospital_region_code"].astype(str) + ")",
    y=top_eff["Efficiency_Index"],
    title="Top 20 Hospitals by Efficiency Index (rooms / avg LOS)",
    xlabel="Hospital (Region)",
    ylabel="Efficiency Index",
    fname="top_efficiency_index.png",
    rotation=45
)

# -------- 2) Ward & department trends --------
dept = (
    df.groupby("Department", dropna=False)
      .agg(avg_los=("LOS_days","mean"), var_los=("LOS_days","var"), n=("LOS_days","size"))
      .reset_index()
      .sort_values("avg_los", ascending=False)
)
dept.to_csv(os.path.join(OUT_DIR, "department_los.csv"), index=False)

bar_chart(
    x=dept["Department"].astype(str),
    y=dept["avg_los"],
    title="Average LOS by Department",
    xlabel="Department",
    ylabel="Avg LOS (days)",
    fname="avg_los_by_department.png",
    rotation=45
)

if "Ward_Facility_Code" in df.columns:
    wfc = (
        df.groupby("Ward_Facility_Code", dropna=False)
          .agg(avg_los=("LOS_days","mean"), n=("LOS_days","size"))
          .reset_index()
          .sort_values("avg_los", ascending=False)
    )
    wfc.to_csv(os.path.join(OUT_DIR, "ward_facility_los.csv"), index=False)
    bar_chart(
        x=wfc["Ward_Facility_Code"].astype(str),
        y=wfc["avg_los"],
        title="Average LOS by Ward Facility Code",
        xlabel="Ward Facility Code",
        ylabel="Avg LOS (days)",
        fname="avg_los_by_ward_facility.png",
        rotation=0
    )

if "Bed Grade" in df.columns:
    bed = (
        df.groupby("Bed Grade", dropna=False)
          .agg(avg_los=("LOS_days","mean"), n=("LOS_days","size"))
          .reset_index()
          .sort_values("avg_los", ascending=False)
    )
    bed.to_csv(os.path.join(OUT_DIR, "bed_grade_los.csv"), index=False)
    bar_chart(
        x=bed["Bed Grade"].astype(str),
        y=bed["avg_los"],
        title="Average LOS by Bed Grade",
        xlabel="Bed Grade",
        ylabel="Avg LOS (days)",
        fname="avg_los_by_bed_grade.png",
        rotation=0
    )

# -------- 3) Patient demographics & admission --------
if "Age" in df.columns:
    age_los = (
        df.groupby("Age", dropna=False)
          .agg(avg_los=("LOS_days","mean"), n=("LOS_days","size"))
          .reset_index()
    )
    age_los.to_csv(os.path.join(OUT_DIR, "age_los.csv"), index=False)
    line_chart(
        x=age_los["Age"].astype(str),
        y=age_los["avg_los"],
        title="Average LOS by Age Group",
        xlabel="Age Group",
        ylabel="Avg LOS (days)",
        fname="avg_los_by_age.png",
        rotation=45
    )

if "Severity of Illness" in df.columns:
    severity_dist = pd.crosstab(df["Severity of Illness"], df["Stay"], normalize="index") * 100
    severity_dist.to_csv(os.path.join(OUT_DIR, "severity_stay_distribution.csv"))

df["is_long_stay"] = df["LOS_days"] >= 41
if "Type of Admission" in df.columns:
    adm = (
        df.groupby("Type of Admission", dropna=False)["is_long_stay"]
          .mean().reset_index(name="long_stay_rate")
          .sort_values("long_stay_rate", ascending=False)
    )
    adm.to_csv(os.path.join(OUT_DIR, "admission_long_stay_rate.csv"), index=False)
    bar_chart(
        x=adm["Type of Admission"].astype(str),
        y=adm["long_stay_rate"]*100,
        title="Long-Stay Rate (≥41 days) by Admission Type",
        xlabel="Admission Type",
        ylabel="Long-Stay Rate (%)",
        fname="long_stay_by_admission_type.png",
        rotation=0
    )

# -------- 4) Financial --------
corr = df[["Admission_Deposit","LOS_days"]].dropna().corr().iloc[0,1]
with open(os.path.join(OUT_DIR, "financial_summary.txt"), "w") as f:
    f.write(f"Pearson correlation between Admission_Deposit and LOS_days: {corr:.4f}\n")

box_plot(
    groups=df["Stay"],
    values=df["Admission_Deposit"],
    title="Admission Deposit by Stay Category",
    xlabel="Stay Category",
    ylabel="Admission Deposit",
    fname="deposit_by_stay.png"
)

if "Visitors with Patient" in df.columns:
    corr2 = df[["Visitors with Patient","LOS_days"]].dropna().corr().iloc[0,1]
    with open(os.path.join(OUT_DIR, "financial_summary.txt"), "a") as f:
        f.write(f"Pearson correlation between Visitors with Patient and LOS_days: {corr2:.4f}\n")

# -------- 5) Region --------
if "Hospital_region_code" in df.columns:
    region = (
        df.groupby("Hospital_region_code", dropna=False)
          .agg(avg_los=("LOS_days","mean"), n=("LOS_days","size"))
          .reset_index()
          .sort_values("avg_los", ascending=False)
    )
    region.to_csv(os.path.join(OUT_DIR, "region_los.csv"), index=False)
    bar_chart(
        x=region["Hospital_region_code"].astype(str),
        y=region["avg_los"],
        title="Average LOS by Hospital Region",
        xlabel="Region",
        ylabel="Avg LOS (days)",
        fname="avg_los_by_region.png",
        rotation=0
    )

# -------- narrative export --------
summary_lines = []
summary_lines.append(f"Admissions: {len(df):,}")
summary_lines.append(f"Hospitals: {df['Hospital_code'].nunique()}")
summary_lines.append(f"Avg LOS (mapped midpoint days): {df['LOS_days'].mean():.2f}")

top5_ineff = eff.nsmallest(5, "Efficiency_Index")[["Hospital_code","Hospital_region_code","Efficiency_Index","avg_los","rooms","admissions"]]
top5_eff = eff.nlargest(5, "Efficiency_Index")[["Hospital_code","Hospital_region_code","Efficiency_Index","avg_los","rooms","admissions"]]

summary_lines.append("\nTop 5 Efficient Hospitals (rooms/avg LOS):")
summary_lines.append(top5_eff.to_string(index=False))
summary_lines.append("\nBottom 5 Efficient Hospitals (rooms/avg LOS):")
summary_lines.append(top5_ineff.to_string(index=False))

summary_lines.append("\nDepartments with Longest Avg LOS:")
summary_lines.append(dept.head(5).to_string(index=False))

if "Ward_Facility_Code" in df.columns:
    summary_lines.append("\nWard Facilities with Longest Avg LOS:")
    summary_lines.append(wfc.head(5).to_string(index=False))

if "Bed Grade" in df.columns:
    summary_lines.append("\nBed Grades with Longest Avg LOS:")
    summary_lines.append(bed.head(5).to_string(index=False))

if "Age" in df.columns:
    summary_lines.append("\nAge Groups by Avg LOS:")
    summary_lines.append(age_los.to_string(index=False))

if "Severity of Illness" in df.columns:
    summary_lines.append("\nSeverity vs Stay distribution saved to CSV.")

if "Type of Admission" in df.columns:
    summary_lines.append("\nLong-Stay Rate by Admission Type:")
    summary_lines.append(adm.to_string(index=False))

summary_lines.append(f"\nCorrelation(Deposit, LOS_days): {corr:.4f}")
if "Visitors with Patient" in df.columns:
    summary_lines.append(f"Correlation(Visitors, LOS_days): {corr2:.4f}")

if "Hospital_region_code" in df.columns:
    summary_lines.append("\nAvg LOS by Region:")
    summary_lines.append(region.to_string(index=False))

with open(os.path.join(OUT_DIR, "data_story_summary.txt"), "w") as f:
    f.write("\n".join(summary_lines))

print("Done. Check the 'analytics_outputs' folder.")
