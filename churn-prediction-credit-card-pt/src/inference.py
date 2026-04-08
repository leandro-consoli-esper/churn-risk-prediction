"""
Módulo de inferência — Previsão de Churn de Clientes (LightGBM)

Carrega os artefatos treinados (pipeline + threshold de decisão) e disponibiliza
uma função única predict_churn() para inferência em lote.
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
        raise FileNotFoundError(f"Artefato não encontrado em: {path}") from e


model_pipeline = _load_artifact(MODEL_PATH)
THRESHOLD = _load_artifact(THRESHOLD_PATH, cast=float)
FEATURE_LIST = _load_artifact(FEATURE_LIST_PATH, optional=True)

if FEATURE_LIST is not None:
    FEATURE_LIST = list(FEATURE_LIST)


def predict_churn(input_data: pd.DataFrame) -> pd.DataFrame:
    """
    Prevê a probabilidade de churn e a classificação binária de churn utilizando o pipeline treinado.

    Parâmetros
    ----------
    input_data : pd.DataFrame
        Dados brutos dos clientes contendo as mesmas variáveis base utilizadas no treinamento
        (as variáveis derivadas são criadas internamente pelo pipeline).

    Retorna
    -------
    pd.DataFrame
        Cópia dos dados de entrada contendo:
        - churn_probability: float no intervalo [0, 1]
        - churn_prediction: int (0/1) após aplicação do threshold persistido
    """
    if not isinstance(input_data, pd.DataFrame):
        raise TypeError("A data de entrada deve ser um DataFrame do pandas")

    X = input_data.copy()

    # Normaliza os nomes das colunas para corresponder ao formato de treinamento
    X.columns = X.columns.str.strip().str.lower()

    # Reforço de compliance: assegura que o preprocessador receba as colunas de treinamento corretamente
    if FEATURE_LIST is not None:
        missing = [c for c in FEATURE_LIST if c not in X.columns]
        if missing:
            raise ValueError(f"Faltam colunas necessárias para a inferência: {missing}")
        X = X.reindex(columns=FEATURE_LIST)

    churn_proba = model_pipeline.predict_proba(X)[:, 1]
    churn_pred = (churn_proba >= THRESHOLD).astype(int)

    results = input_data.copy()
    results["churn_probability"] = churn_proba
    results["churn_prediction"] = churn_pred
    return results