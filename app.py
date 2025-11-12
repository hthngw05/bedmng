
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

st.set_page_config(page_title="FlowCare AI – Discharge Analytics", layout="wide")

st.title("🏥 FlowCare AI – Patient Discharge Analytics & Prediction")

st.markdown(
    """
This app helps you **explore, interpret, and predict patient length of stay** using your JanataHack Healthcare dataset.  
Use the sidebar to navigate steps. Upload your CSV (`train.csv`) or use the sample preloaded path.
"""
)

# Sidebar navigation
page = st.sidebar.radio(
    "Navigate",
    ["1) Load Data", "2) Explore Analytics", "3) Train Model", "4) Patient Simulator", "5) Model Explainability"],
)

@st.cache_data(show_spinner=False)
def load_data(path_or_file):
    try:
        if path_or_file is None:
            return None
        if hasattr(path_or_file, "read"):
            return pd.read_csv(path_or_file)
        else:
            return pd.read_csv(path_or_file)
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")
        return None

def infer_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    # Explicitly treat some columns as categorical based on known dataset
    likely_cats = [
        "Hospital_code","Hospital_type_code","City_Code_Hospital","Hospital_region_code",
        "Department","Ward_Type","Ward_Facility_Code","Bed Grade","City_Code_Patient",
        "Type of Admission","Severity of Illness","Age"
    ]
    cats = [c for c in df.columns if c in likely_cats or df[c].dtype == "object"]
    # Remove target from predictors
    if "Stay" in cats: cats.remove("Stay")
    nums = [c for c in df.columns if c not in cats and c != "Stay"]
    return cats, nums

def make_efficiency_index(df: pd.DataFrame) -> pd.DataFrame:
    # Map stay categories to midpoint days to approximate numeric LOS
    stay_map = {
        "0-10": 5, "11-20": 15, "21-30": 25, "31-40": 35, "41-50": 45,
        "51-60": 55, "61-70": 65, "71-80": 75, "81-90": 85, "91-100": 95,
        "More than 100 Days": 110
    }
    tmp = df.copy()
    if "Stay" in tmp.columns:
        tmp["LOS_days"] = tmp["Stay"].map(stay_map)
    else:
        tmp["LOS_days"] = np.nan
    if "Available Extra Rooms in Hospital" not in tmp.columns:
        tmp["Available Extra Rooms in Hospital"] = np.nan
    by_hosp = (
        tmp.groupby(["Hospital_code","Hospital_region_code"], dropna=False)
        .agg(avg_los=("LOS_days","mean"),
             rooms=("Available Extra Rooms in Hospital","mean"),
             count=("LOS_days","size"))
        .reset_index()
    )
    by_hosp["Efficiency_Index"] = by_hosp["rooms"] / by_hosp["avg_los"]
    return by_hosp

uploaded = None
default_path = st.sidebar.text_input("Optional: path to CSV (e.g., train.csv)", value="")
if default_path.strip():
    df = load_data(default_path)
else:
    uploaded = st.sidebar.file_uploader("Upload your train.csv", type=["csv"])
    df = load_data(uploaded)

if page == "1) Load Data":
    st.subheader("Step 1 – Load your dataset")
    st.write("Upload `train.csv` from JanataHack Healthcare or paste a path in the sidebar.")
    if df is not None:
        st.success(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
        st.dataframe(df.head(20), use_container_width=True)
        st.markdown("**Column summary**")
        info = pd.DataFrame({
            "dtype": df.dtypes.astype(str),
            "missing": df.isna().sum(),
            "unique": df.nunique()
        }).sort_index()
        st.dataframe(info, use_container_width=True)
    else:
        st.info("Waiting for CSV...")

if df is not None:
    cats, nums = infer_column_types(df)
    target = "Stay" if "Stay" in df.columns else None

    if page == "2) Explore Analytics":
        st.subheader("Step 2 – Explore Analytics (tell a story)")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total Admissions", f"{len(df):,}")
        with c2:
            st.metric("Hospitals", f"{df['Hospital_code'].nunique() if 'Hospital_code' in df.columns else '—'}")
        with c3:
            st.metric("Avg Visitors", f"{df['Visitors with Patient'].mean():.2f}" if "Visitors with Patient" in df.columns else "—")

        # Distribution of Stay
        if target:
            st.markdown("### Length of Stay Distribution")
            order = df[target].value_counts().sort_index().index
            fig = px.bar(df[target].value_counts().reindex(order),
                         labels={"index":"Stay (days)", "value":"Admissions"},
                         title="Count by Stay Category")
            st.plotly_chart(fig, use_container_width=True)

        # Severity vs Stay
        if "Severity of Illness" in df.columns and target:
            st.markdown("### Severity of Illness vs Stay")
            pivot = pd.crosstab(df["Severity of Illness"], df[target], normalize="index")*100
            fig = px.imshow(pivot, aspect="auto", color_continuous_scale="Blues",
                            labels=dict(x="Stay", y="Severity", color="% within Severity"),
                            title="Within-Severity distribution across Stay")
            st.plotly_chart(fig, use_container_width=True)

        # Deposit vs Stay
        if "Admission_Deposit" in df.columns and target:
            st.markdown("### Deposit vs Stay")
            fig = px.box(df, x=target, y="Admission_Deposit", points="suspectedoutliers",
                         title="Admission_Deposit by Stay Category")
            st.plotly_chart(fig, use_container_width=True)

        # Age vs Stay
        if "Age" in df.columns and target:
            st.markdown("### Age vs Stay")
            # Ensure age is ordered
            age_order = ["0-10","11-20","21-30","31-40","41-50","51-60","61-70","71-80","81-90","91-100"]
            tmp = df.copy()
            tmp["Age"] = pd.Categorical(tmp["Age"], categories=age_order, ordered=True)
            g = tmp.groupby(["Age", target]).size().reset_index(name="count")
            fig = px.bar(g, x="Age", y="count", color=target, title="Stay distribution by Age Group", barmode="stack")
            st.plotly_chart(fig, use_container_width=True)

        # Hospital Efficiency Index
        st.markdown("### Hospital Efficiency Index (rooms / avg LOS)")
        eff = make_efficiency_index(df)
        st.dataframe(eff.sort_values("Efficiency_Index", ascending=False), use_container_width=True)

    if page == "3) Train Model":
        st.subheader("Step 3 – Train a baseline classifier for Stay")
        if target is None:
            st.error("No target column `Stay` found.")
        else:
            # Drop columns that leak or are identifiers
            drop_cols = ["case_id","patientid"]
            X = df.drop(columns=[c for c in drop_cols if c in df.columns] + [target], errors="ignore")
            y = df[target].astype(str)

            # Basic cleaning: fill NA for categorical/numeric separately
            for c in X.columns:
                if X[c].dtype == "object":
                    X[c] = X[c].fillna("Unknown")
                else:
                    X[c] = X[c].fillna(X[c].median())

            cats, nums = infer_column_types(pd.concat([X, y], axis=1))
            pre = ColumnTransformer(
                transformers=[
                    ("cat", OneHotEncoder(handle_unknown="ignore"), [c for c in cats if c in X.columns]),
                    ("num", "passthrough", [c for c in nums if c in X.columns])
                ]
            )

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

            from sklearn.pipeline import Pipeline
            clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced")
            pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
            with st.spinner("Training model..."):
                pipe.fit(X_train, y_train)

            y_pred = pipe.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            st.success(f"Validation Accuracy: {acc:.3f}")

            st.markdown("#### Classification Report")
            st.text(classification_report(y_test, y_pred))

            # Confusion matrix
            cm = pd.DataFrame(confusion_matrix(y_test, y_pred), index=pipe.classes_, columns=pipe.classes_)
            fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", title="Confusion Matrix")
            st.plotly_chart(fig, use_container_width=True)

            # Permutation importance on a sample to save time
            st.markdown("#### Feature Importance (permutation on 2k sample)")
            try:
                # Sample to speed up
                sample_idx = np.random.choice(len(X_test), size=min(2000, len(X_test)), replace=False)
                result = permutation_importance(pipe, X_test.iloc[sample_idx], y_test.iloc[sample_idx], n_repeats=5, random_state=42, n_jobs=-1)
                # Permutation importance returns combined feature names after preprocessing; derive human-friendly names
                # We'll approximate by using original column names expanded for categories
                ohe: OneHotEncoder = pipe.named_steps["pre"].named_transformers_["cat"]
                cat_cols = [c for c in cats if c in X.columns]
                if hasattr(ohe, "get_feature_names_out"):
                    ohe_names = ohe.get_feature_names_out(cat_cols).tolist()
                else:
                    ohe_names = []
                feature_names = ohe_names + [c for c in nums if c in X.columns]
                importances = pd.DataFrame({
                    "feature": feature_names,
                    "importance": result.importances_mean[:len(feature_names)]
                }).sort_values("importance", ascending=False).head(25)
                fig = px.bar(importances, x="importance", y="feature", orientation="h", title="Top Features")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.info(f"Permutation importance skipped: {e}")

            st.session_state["trained_model"] = pipe
            st.session_state["feature_cols"] = X.columns.tolist()
            st.session_state["cats"] = cats
            st.session_state["nums"] = nums

    if page == "4) Patient Simulator":
        st.subheader("Step 4 – What‑if Patient Simulator")
        if "trained_model" not in st.session_state:
            st.warning("Train a model in Step 3 first.")
        else:
            pipe = st.session_state["trained_model"]
            feature_cols = st.session_state["feature_cols"]
            # Build simple UI dynamically based on column types
            user_input = {}
            for col in feature_cols:
                if df[col].dtype == "object":
                    options = sorted(df[col].dropna().astype(str).unique().tolist())[:100]
                    default = options[0] if options else "Unknown"
                    user_input[col] = st.selectbox(col, options, index=0)
                else:
                    val = float(np.nanmedian(df[col]))
                    user_input[col] = st.number_input(col, value=val)

            # Predict
            if st.button("Predict Stay Category"):
                X_new = pd.DataFrame([user_input])
                pred = pipe.predict(X_new)[0]
                proba = max(pipe.predict_proba(X_new)[0])
                readiness = (1 - proba) * 100  # heuristic
                st.success(f"Predicted Stay: **{pred}**  |  Confidence: {proba*100:.1f}%")
                st.info(f"🟢 Discharge Readiness (heuristic): **{readiness:.1f} / 100** (higher = more ready)")

    if page == "5) Model Explainability":
        st.subheader("Step 5 – Explainability & Storytelling")
        st.markdown(
            """
- Use the **Feature Importance** chart from Step 3 to explain _why_ the model predicts longer stays (e.g., severity, age, admission type).
- Combine with analytics from Step 2 (e.g., regional/department differences) to make **policy or operations recommendations**:
  - Upgrade wards with consistently long stays.
  - Increase rooms where the **Efficiency Index** is low.
  - Prioritize discharge planning for elderly + severe cases.
"""
        )
else:
    if page != "1) Load Data":
        st.warning("Please load a CSV on the left first.")
