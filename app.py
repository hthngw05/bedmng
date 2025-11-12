import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import plotly.express as px
import plotly.graph_objects as go

# ────────────────────────────── CONFIG ──────────────────────────────
st.set_page_config(page_title="FlowCare AI – Discharge Analytics", layout="wide")

# ────────────────────────────── HELPERS ─────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(path_or_file):
    if path_or_file is None:
        return None
    try:
        if hasattr(path_or_file, "read"):  # Uploaded file-like object
            return pd.read_csv(path_or_file)
        return pd.read_csv(path_or_file)   # Local path string
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")
        return None

def infer_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    likely_cats = [
        "Hospital_code","Hospital_type_code","City_Code_Hospital","Hospital_region_code",
        "Department","Ward_Type","Ward_Facility_Code","Bed Grade","City_Code_Patient",
        "Type of Admission","Severity of Illness","Age"
    ]
    cats = [c for c in df.columns if c in likely_cats or df[c].dtype == "object"]
    if "Stay" in cats:
        cats.remove("Stay")
    nums = [c for c in df.columns if c not in cats and c != "Stay"]
    return cats, nums

def make_efficiency_index(df: pd.DataFrame) -> pd.DataFrame:
    stay_map = {
        "0-10": 5, "11-20": 15, "21-30": 25, "31-40": 35, "41-50": 45,
        "51-60": 55, "61-70": 65, "71-80": 75, "81-90": 85, "91-100": 95,
        "More than 100 Days": 110
    }
    tmp = df.copy()
    tmp["LOS_days"] = tmp["Stay"].map(stay_map) if "Stay" in tmp.columns else np.nan
    if "Available Extra Rooms in Hospital" not in tmp.columns:
        tmp["Available Extra Rooms in Hospital"] = np.nan

    by_hosp = (
        tmp.groupby(["Hospital_code","Hospital_region_code"], observed=False, dropna=False)
           .agg(avg_los=("LOS_days","mean"),
                rooms=("Available Extra Rooms in Hospital","mean"),
                count=("LOS_days","size"))
           .reset_index()
    )
    by_hosp["Efficiency_Index"] = by_hosp["rooms"] / by_hosp["avg_los"]
    return by_hosp

def tidy_age(df: pd.DataFrame) -> pd.DataFrame:
    if "Age" in df.columns:
        age_order = ["0-10","11-20","21-30","31-40","41-50","51-60","61-70","71-80","81-90","91-100"]
        if df["Age"].dtype != "category":
            df["Age"] = pd.Categorical(df["Age"], categories=age_order, ordered=True)
    return df

# ────────────────────────────── PAGES ───────────────────────────────
def page_load_data():
    st.subheader("Step 1 – Load your dataset")
    st.write("Upload `train.csv` from JanataHack Healthcare or paste a path in the sidebar.")
    default_path = st.sidebar.text_input("Optional: path to CSV (e.g., train.csv)", value="")
    uploaded = None
    if default_path.strip():
        df = load_data(default_path.strip())
    else:
        uploaded = st.sidebar.file_uploader("Upload your train.csv", type=["csv"])
        df = load_data(uploaded)

    if df is None:
        st.info("Waiting for CSV…")
        return None

    st.success(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
    st.dataframe(df.head(20), width="stretch")

    info = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "unique": df.nunique()
    }).sort_index()
    st.markdown("**Column summary**")
    st.dataframe(info, width="stretch")
    return df

def page_explore(df: pd.DataFrame):
    st.subheader("Step 2 – Explore Analytics (tell a story)")

    # KPIs
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Admissions", f"{len(df):,}")
    with c2:
        st.metric("Hospitals", f"{df['Hospital_code'].nunique() if 'Hospital_code' in df.columns else '—'}")
    with c3:
        val = df["Visitors with Patient"].mean() if "Visitors with Patient" in df.columns else np.nan
        st.metric("Avg Visitors", f"{val:.2f}" if pd.notnull(val) else "—")

    # Distributions
    target = "Stay" if "Stay" in df.columns else None

    if target:
        st.markdown("### Length of Stay Distribution")
        order = df[target].value_counts().sort_index().index
        fig = px.bar(df[target].value_counts().reindex(order),
                     labels={"index":"Stay (days)", "value":"Admissions"},
                     title="Count by Stay Category")
        st.plotly_chart(fig, width="stretch")

    if "Severity of Illness" in df.columns and target:
        st.markdown("### Severity of Illness vs Stay")
        pivot = pd.crosstab(df["Severity of Illness"], df[target], normalize="index")*100
        fig = px.imshow(pivot, aspect="auto", color_continuous_scale="Blues",
                        labels=dict(x="Stay", y="Severity", color="% within Severity"),
                        title="Within-Severity distribution across Stay")
        st.plotly_chart(fig, width="stretch")

    if "Admission_Deposit" in df.columns and target:
        st.markdown("### Deposit vs Stay")
        fig = px.box(df, x=target, y="Admission_Deposit", points="suspectedoutliers",
                     title="Admission_Deposit by Stay Category")
        st.plotly_chart(fig, width="stretch")

    if "Age" in df.columns and target:
        st.markdown("### Age vs Stay")
        df2 = tidy_age(df.copy())
        g = (df2.groupby(["Age", target], observed=False)
                 .size().reset_index(name="count"))
        fig = px.bar(g, x="Age", y="count", color=target,
                     title="Stay distribution by Age Group", barmode="stack")
        st.plotly_chart(fig, width="stretch")

    st.markdown("### Hospital Efficiency Index (rooms / avg LOS)")
    eff = make_efficiency_index(df)
    st.dataframe(eff.sort_values("Efficiency_Index", ascending=False), width="stretch")

def page_train(df: pd.DataFrame):
    st.subheader("Step 3 – Train a baseline classifier for Stay")
    target = "Stay"
    if target not in df.columns:
        st.error("No target column `Stay` found.")
        return

    drop_cols = ["case_id","patientid"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns] + [target], errors="ignore")
    y = df[target].astype(str)

    # Basic cleaning
    for c in X.columns:
        if X[c].dtype == "object":
            X[c] = X[c].fillna("Unknown")
        else:
            X[c] = X[c].fillna(X[c].median())

    cats, nums = infer_column_types(pd.concat([X, y], axis=1))
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"),
             [c for c in cats if c in X.columns]),
            ("num", "passthrough",
             [c for c in nums if c in X.columns])
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    from sklearn.pipeline import Pipeline
    clf = RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced"
    )
    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])

    with st.spinner("Training model…"):
        pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.success(f"Validation Accuracy: {acc:.3f}")

    st.markdown("#### Classification Report")
    st.text(classification_report(y_test, y_pred))

    cm = pd.DataFrame(confusion_matrix(y_test, y_pred),
                      index=pipe.classes_, columns=pipe.classes_)
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                    title="Confusion Matrix")
    st.plotly_chart(fig, width="stretch")

    st.markdown("#### Feature Importance (permutation on sample)")
    try:
        # Subsample for speed
        if len(X_test) > 2000:
            sample_idx = np.random.choice(len(X_test), size=2000, replace=False)
            X_eval = X_test.iloc[sample_idx]
            y_eval = y_test.iloc[sample_idx]
        else:
            X_eval, y_eval = X_test, y_test

        result = permutation_importance(
            pipe, X_eval, y_eval, n_repeats=5, random_state=42, n_jobs=-1
        )
        # Human-friendly names (OHE-expanded + numeric)
        ohe: OneHotEncoder = pipe.named_steps["pre"].named_transformers_["cat"]
        cat_cols = [c for c in cats if c in X.columns]
        ohe_names = ohe.get_feature_names_out(cat_cols).tolist() if hasattr(ohe, "get_feature_names_out") else []
        feature_names = ohe_names + [c for c in nums if c in X.columns]

        importances = pd.DataFrame({
            "feature": feature_names[:len(result.importances_mean)],
            "importance": result.importances_mean[:len(feature_names)]
        }).sort_values("importance", ascending=False).head(25)

        fig = px.bar(importances, x="importance", y="feature", orientation="h",
                     title="Top Features")
        st.plotly_chart(fig, width="stretch")
    except Exception as e:
        st.info(f"Permutation importance skipped: {e}")

    st.session_state["trained_model"] = pipe
    st.session_state["feature_cols"] = X.columns.tolist()

def page_simulator(df: pd.DataFrame):
    st.subheader("Step 4 – What-if Patient Simulator")
    if "trained_model" not in st.session_state:
        st.warning("Train a model in Step 3 first.")
        return

    pipe = st.session_state["trained_model"]
    feature_cols = st.session_state["feature_cols"]

    user_input = {}
    for col in feature_cols:
        if df[col].dtype == "object":
            opts = sorted(df[col].dropna().astype(str).unique().tolist())[:100]
            default = opts[0] if opts else "Unknown"
            user_input[col] = st.selectbox(col, opts or ["Unknown"], index=0)
        else:
            val = float(np.nanmedian(df[col]))
            user_input[col] = st.number_input(col, value=val)

    if st.button("Predict Stay Category"):
        X_new = pd.DataFrame([user_input])
        pred = pipe.predict(X_new)[0]
        proba = max(pipe.predict_proba(X_new)[0])
        readiness = (1 - proba) * 100  # heuristic for demo
        st.success(f"Predicted Stay: **{pred}**  |  Confidence: {proba*100:.1f}%")
        st.info(f"🟢 Discharge Readiness (heuristic): **{readiness:.1f} / 100** (higher = more ready)")

def page_explain():
    st.subheader("Step 5 – Explainability & Storytelling")
    st.markdown(
        """
- Use **Feature Importance** (Step 3) to explain *why* the model predicts longer stays (e.g., severity, age, admission type).
- Combine with analytics from **Step 2** to make **operations recommendations**:
  - Upgrade wards with consistently long stays.
  - Increase rooms where the **Efficiency Index** is low.
  - Prioritize discharge planning for elderly + severe cases early.
"""
    )

# ────────────────────────────── MAIN ────────────────────────────────
def main():
    st.title("🏥 FlowCare AI – Patient Discharge Analytics & Prediction")
    page = st.sidebar.radio(
        "Navigate",
        ["1) Load Data", "2) Explore Analytics", "3) Train Model", "4) Patient Simulator", "5) Model Explainability"],
        index=0
    )

    df = None
    if page == "1) Load Data":
        df = page_load_data()
        return

    # Persist data selection across pages
    default_path = st.sidebar.text_input("Optional: path to CSV (e.g., train.csv)", value="")
    uploaded = None
    if default_path.strip():
        df = load_data(default_path.strip())
    else:
        uploaded = st.sidebar.file_uploader("Upload your train.csv", type=["csv"])
        df = load_data(uploaded)

    if df is None:
        st.info("Load a CSV on the left to continue.")
        st.stop()

    # Normalize age categories for consistent grouping
    df = tidy_age(df)

    try:
        if page == "2) Explore Analytics":
            page_explore(df)
        elif page == "3) Train Model":
            page_train(df)
        elif page == "4) Patient Simulator":
            page_simulator(df)
        elif page == "5) Model Explainability":
            page_explain()
    except Exception as e:
        import traceback
        st.error("Unhandled error in the app:")
        st.code(traceback.format_exc())
        st.stop()

if __name__ == "__main__":
    main()
