# ============================================================
# STREAMLIT APP — CUSTOMER CHURN PREDICTION
# ============================================================
# This application provides a simple interface to run inference using the
# trained LightGBM pipeline. The pipeline performs preprocessing + feature
# engineering internally (via the project's Preprocessor).
#
# Output:
#  - churn_probability: risk score (0–1)
#  - churn_prediction: binary decision based on a calibrated threshold
#  - risk_band: human-readable risk interpretation
# ============================================================

# ---------- PROJECT SETUP ----------
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

# ---------- EXTERNAL LIBRARIES ----------
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import altair as alt

# ---------- INTERNAL IMPORTS ----------
from src.inference import predict_churn

# ---------- PATHS ----------
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
FEATURE_LIST_PATH = ARTIFACTS_DIR / "feature_list.pkl"
THRESHOLD_PATH = ARTIFACTS_DIR / "decision_threshold.pkl"

# ---------- PAGE CONFIGURATION ----------
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

# ---------- HELPERS ----------
def risk_band(p: float) -> str:
    """
    Simple risk interpretation bands for portfolio/demo purposes.
    Adjust cutoffs as needed for business policy.
    """
    if p < 0.20:
        return "Low"
    if p < 0.50:
        return "Medium"
    return "High"


# ---------- HEADER ----------
st.title("📉 Customer Churn Prediction")

st.markdown(
    """
This application predicts **customer churn probability** using a trained **LightGBM model**.

Upload a dataset containing the **original feature schema** used during training.  
All preprocessing and feature engineering steps are handled internally by the pipeline.
"""
)

# ---------- LOAD ARTIFACTS (schema + threshold) ----------
try:
    FEATURE_LIST = joblib.load(FEATURE_LIST_PATH)
except FileNotFoundError:
    FEATURE_LIST = None
    st.warning(
        "Feature schema file not found (feature_list.pkl). "
        "Predictions may fail if the input does not match the training schema."
    )

try:
    THRESHOLD = float(joblib.load(THRESHOLD_PATH))
except FileNotFoundError:
    THRESHOLD = None
    st.warning(
        "Threshold file not found (decision_threshold.pkl). "
        "Charts will be shown without a threshold reference line."
    )

# ============================================================
# CSV UPLOAD — PRIMARY INPUT METHOD
# ============================================================
uploaded_file = st.file_uploader("Upload a CSV file with customer data", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)

    st.subheader("Input Data Preview")
    st.dataframe(input_df.head())

    # Expected schema (for user/reference)
    if FEATURE_LIST is not None:
        with st.expander("Expected columns (training schema)"):
            st.code("\n".join(FEATURE_LIST))

    # ---------- RUN INFERENCE ----------
    if st.button("Run Prediction"):
        try:
            results = predict_churn(input_df).copy()

            # Add risk interpretation
            if "churn_probability" in results.columns:
                results["risk_band"] = results["churn_probability"].apply(risk_band)
                results = results.sort_values("churn_probability", ascending=False).reset_index(drop=True)

            st.subheader("Prediction Results (sorted by risk)")
            st.dataframe(results)

            # ---------- RISK SUMMARY ----------
            if "risk_band" in results.columns:
                st.subheader("Risk Summary")
                counts = (
                    results["risk_band"]
                    .value_counts()
                    .reindex(["High", "Medium", "Low"])
                    .fillna(0)
                    .astype(int)
                )

                c1, c2, c3 = st.columns(3)
                c1.metric("High risk", int(counts.get("High", 0)))
                c2.metric("Medium risk", int(counts.get("Medium", 0)))
                c3.metric("Low risk", int(counts.get("Low", 0)))

            # ---------- PROBABILITY DISTRIBUTION ----------
            if "churn_probability" in results.columns:
                st.subheader("Churn Probability Distribution")

                chart_df = results[["churn_probability"]].copy()

                hist = (
                    alt.Chart(chart_df)
                    .mark_bar()
                    .encode(
                        x=alt.X(
                            "churn_probability:Q",
                            bin=alt.Bin(maxbins=30),
                            title="Churn probability",
                        ),
                        y=alt.Y("count():Q", title="Customers"),
                        tooltip=[alt.Tooltip("count():Q", title="Customers")],
                    )
                )

                if THRESHOLD is not None:
                    rule = (
                        alt.Chart(pd.DataFrame({"threshold": [THRESHOLD]}))
                        .mark_rule(strokeDash=[6, 6])
                        .encode(x="threshold:Q")
                    )
                    st.altair_chart((hist + rule).interactive(), use_container_width=True)
                else:
                    st.altair_chart(hist.interactive(), use_container_width=True)

            # ---------- DOWNLOAD ----------
            st.download_button(
                label="Download results as CSV",
                data=results.to_csv(index=False),
                file_name="churn_predictions.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error("Prediction failed. Please verify the input schema and data types.")
            st.exception(e)

# ---------- FOOTER ----------
st.markdown("---")
st.caption(
    "Model: LightGBM | Threshold calibrated on validation set | Risk scoring output | SHAP-explainable"
)