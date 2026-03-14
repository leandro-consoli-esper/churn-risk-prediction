"""
Inference module — Customer Churn Prediction (LightGBM)

Loads trained artifacts (pipeline + decision threshold) and exposes a single
predict_churn() function for batch inference.
"""

import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "lgbm_pipeline.pkl"
THRESHOLD_PATH = ARTIFACTS_DIR / "decision_threshold.pkl"
FEATURE_LIST_PATH = ARTIFACTS_DIR / "feature_list.pkl"


def _load_artifact(path: Path, *, cast=None, optional: bool = False):
    try:
        obj = joblib.load(path)
        return cast(obj) if cast is not None else obj
    except FileNotFoundError as e:
        if optional:
            return None
        raise FileNotFoundError(f"Artifact not found at: {path}") from e


model_pipeline = _load_artifact(MODEL_PATH)
THRESHOLD = _load_artifact(THRESHOLD_PATH, cast=float)
FEATURE_LIST = _load_artifact(FEATURE_LIST_PATH, optional=True)

if FEATURE_LIST is not None:
    FEATURE_LIST = list(FEATURE_LIST)


def predict_churn(input_data: pd.DataFrame) -> pd.DataFrame:
    """
    Predict churn probability and binary churn flag using the trained pipeline.

    Parameters
    ----------
    input_data : pd.DataFrame
        Raw customer data containing the same base features used in training
        (engineered features are created internally by the pipeline).

    Returns
    -------
    pd.DataFrame
        Copy of input with:
        - churn_probability: float in [0, 1]
        - churn_prediction: int (0/1) after applying the persisted threshold
    """
    if not isinstance(input_data, pd.DataFrame):
        raise TypeError("input_data must be a pandas DataFrame")

    X = input_data.copy()

    # Normalize column names to match training schema
    X.columns = X.columns.str.strip().str.lower()

    # Schema enforcement: ensures the Preprocessor receives exactly the training columns
    if FEATURE_LIST is not None:
        missing = [c for c in FEATURE_LIST if c not in X.columns]
        if missing:
            raise ValueError(f"Missing required columns for inference: {missing}")
        X = X.reindex(columns=FEATURE_LIST)

    churn_proba = model_pipeline.predict_proba(X)[:, 1]
    churn_pred = (churn_proba >= THRESHOLD).astype(int)

    results = input_data.copy()
    results["churn_probability"] = churn_proba
    results["churn_prediction"] = churn_pred
    return results