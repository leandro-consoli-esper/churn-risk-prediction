# Este notebook contém um transformer unificado de engenharia de atributos personalizado para previsão de churn de cartão de crédito.

# Importações
# ===============================
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Avisos
from warnings import filterwarnings
filterwarnings('ignore')


# --------- Preprocessador --------------------
 
class Preprocessor(BaseEstimator, TransformerMixin):
    """
    Transformer unificado de engenharia de atributos para previsão de churn de cartão de crédito,
    com "modos" específicos por modelo.

    Modos:
    - "catboost": conjunto COMPLETO de variáveis; colunas categóricas retornadas como object (strings) para categorias nativas do CatBoost.
    - "lgbm":     conjunto COMPLETO por padrão; colunas categóricas retornadas como 'category' do pandas para categorias nativas do LightGBM.
    - "xgb":      conjunto REDUZIDO por padrão; colunas categóricas retornadas como object (strings), destinadas a uma etapa posterior de codificação
                    (OneHotEncoder/OrdinalEncoder/TargetEncoder etc.).

    Observações:
    - Seguro contra vazamento de dados: o método fit não aprende estatísticas globais.
    - Este transformer NÃO codifica variáveis categóricas; apenas cria atributos e ajusta os tipos de dados.
    """

    SUPPORTED_MODES = ("catboost", "lgbm", "xgb")

    def __init__(
        self,
        mode: str = "xgb",
        feature_set: str | None = None,   # None => automático de acordo com o modo
        drop_cols: list[str] | None = None,
        strict_schema: bool = True,        # True => gera um erro explicativo se colunas obrigatórias estiverem ausentes
        keep_raw_categoricals: bool = True # mantém education/income/card_category (recomendado)
    ):
        self.mode = mode
        self.feature_set = feature_set
        self.drop_cols = drop_cols
        self.strict_schema = strict_schema
        self.keep_raw_categoricals = keep_raw_categoricals

    def fit(self, X, y=None):
        return self

    @staticmethod
    def _standardize_columns(X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X.columns = [c.lower() for c in X.columns]
        if "attrition_flag" in X.columns:
            X.rename(columns={"attrition_flag": "churn_flag"}, inplace=True)
        return X

    @staticmethod
    def _coerce_gender_to_numeric_if_needed(s: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(s):
            return s
        mapped = (
            s.astype(str)
             .str.strip()
             .str.lower()
             .map({"m": 1, "male": 1, "f": 0, "female": 0})
        )
        return mapped

    @staticmethod
    def _safe_div(a, b, eps=1e-3):
        return a / (b + eps)

    def _get_feature_set(self) -> str:
        if self.feature_set is not None:
            fs = self.feature_set.lower()
        else:
            fs = "full" if self.mode in ("catboost", "lgbm") else "lite"
        if fs not in ("full", "lite"):
            raise ValueError("feature_set must be 'full' or 'lite'")
        return fs

    def _required_columns(self) -> set[str]:
        # Necessário para os atributos combinados abaixo
        return {
            "avg_utilization_ratio",
            "total_trans_amt",
            "total_trans_ct",
            "months_on_book",
            "months_inactive_12_mon",
            "total_relationship_count",
            "contacts_count_12_mon",
            "total_ct_chng_q4_q1",
            "total_amt_chng_q4_q1",
            "total_revolving_bal",
            "credit_limit",
            "dependent_count",
        }

    def transform(self, X):
        if self.mode not in self.SUPPORTED_MODES:
            raise ValueError(f"mode must be one of {self.SUPPORTED_MODES}")

        X = self._standardize_columns(X)

        # -------------------------
        # Verificação de schema (opcional, mas recomendado)
        # -------------------------
        req = self._required_columns()
        missing = sorted([c for c in req if c not in X.columns])
        if missing and self.strict_schema:
            raise ValueError(
                "Preprocessor: missing required columns for feature engineering: "
                f"{missing}. If this is inference with partial schema, set strict_schema=False."
            )

        # -------------------------
        # REMOÇÕES BÁSICAS
        # -------------------------
        base_drop = [
            "clientnum",
            "naive_bayes_classifier_attrition_flag_card_category_contacts_count_12_mon_dependent_count_education_level_months_inactive_12_mon_2",
            "naive_bayes_classifier_attrition_flag_card_category_contacts_count_12_mon_dependent_count_education_level_months_inactive_12_mon_1",
        ]

        # Mantém variáveis categóricas originais por padrão (são importantes para CatBoost/LGBM e podem ser codificadas posteriormente para XGB)
        if not self.keep_raw_categoricals:
            base_drop += ["education_level", "income_category", "card_category"]

        # avg_open_to_buy geralmente é redundante com credit_limit e revolving/open to buy; mantém remoção como antes
        base_drop += ["avg_open_to_buy"]

        if self.drop_cols:
            base_drop += [c.lower() for c in self.drop_cols]

        X.drop(columns=base_drop, errors="ignore", inplace=True)

        # Se o schema não for estrito e colunas obrigatórias estiverem ausentes, ignora a engenharia de atributos com segurança
        req_now_missing = [c for c in req if c not in X.columns]
        if req_now_missing and not self.strict_schema:
            # Ainda aplica o ajuste de tipos (dtype) às colunas categóricas existentes e retorna
            return self._finalize_categoricals(X)

        # -------------------------
        # FEATURE ENGINEERING (FULL)
        # -------------------------
        # Limita a taxa de utilização para evitar valores fora dos intervalos definidos
        util = X["avg_utilization_ratio"].clip(lower=0.0, upper=1.0)

        X["risk_level"] = pd.cut(
            util,
            bins=[0, 0.3, 0.7, 0.9, 1.0],
            labels=["Low Risk", "Medium Risk", "High Risk", "Very High Risk"],
            include_lowest=True,
            right=True,
        )

        X["customer_value_score"] = (
            np.log1p(X["total_trans_amt"]) * 0.4
            + np.log1p(X["total_trans_ct"]) * 0.3
            + np.log1p(X["months_on_book"]) * 0.3
        )

        X["engagement_score"] = (
            ((12 - X["months_inactive_12_mon"]) / 12) * 0.3
            + np.log1p(X["total_trans_ct"]) * 0.4
            + np.log1p(X["total_relationship_count"]) * 0.3
        )

        X["inactivity_trans_ratio"] = X["months_inactive_12_mon"] / (X["total_trans_ct"] + 1)
        X["contacts_per_transaction"] = X["contacts_count_12_mon"] / (X["total_trans_ct"] + 1)
        X["trans_ct_per_contacts"] = X["total_trans_ct"] / (X["contacts_count_12_mon"] + 1)
        X["engagement_velocity"] = self._safe_div(X["total_trans_ct"], X["months_on_book"], eps=1e-3)
        X["activity_decay"] = 1 - X["total_ct_chng_q4_q1"]

        revolving_ratio = self._safe_div(X["total_revolving_bal"], X["credit_limit"], eps=1e-3)
        X["loyalty_index"] = (
            0.35 * np.log1p(X["months_on_book"])
            + 0.35 * np.log1p(X["total_trans_ct"])
            + 0.2 * (1 - revolving_ratio)
            + 0.1 * (1 - util)
        )

        X["engagement_stability"] = self._safe_div(X["total_ct_chng_q4_q1"], X["total_amt_chng_q4_q1"], eps=1e-3)
        X["inactivity_ratio"] = self._safe_div(X["months_inactive_12_mon"], X["months_on_book"], eps=1e-3)
        X["activity_decline_rate"] = X["total_ct_chng_q4_q1"] * X["total_amt_chng_q4_q1"]

        X["products_per_tenure"] = self._safe_div(X["total_relationship_count"], X["months_on_book"], eps=1e-3)
        X["relationship_density"] = self._safe_div(X["total_relationship_count"], (X["months_on_book"] / 12), eps=1e-3)
        X["active_product_ratio"] = X["total_relationship_count"] * (1 - (X["months_inactive_12_mon"] / 12))
        X["avg_product_value"] = self._safe_div(X["total_trans_amt"], X["total_relationship_count"], eps=1e-3)

        X["trans_amt_per_credit"] = self._safe_div(X["total_trans_amt"], X["credit_limit"], eps=1e-3)
        X["revolving_intensity"] = self._safe_div(X["total_revolving_bal"], X["total_trans_amt"], eps=1e-3)
        X["credit_usage_trend"] = util * X["total_ct_chng_q4_q1"]
        X["lifetime_value_proxy"] = (X["total_trans_amt"] + X["total_revolving_bal"]) * X["months_on_book"]
        X["revolving_rate_log"] = self._safe_div(np.log1p(X["total_revolving_bal"]), np.log1p(X["credit_limit"]), eps=1e-6)

        X["risk_behavior_score"] = (util + X["revolving_intensity"] + X["inactivity_ratio"]) / 3

        X["complaint_rate"] = X["contacts_count_12_mon"] / (X["months_inactive_12_mon"] + 1)
        X["frustration_index"] = (X["contacts_count_12_mon"] / (X["total_relationship_count"] + 1)) * X["inactivity_ratio"]
        X["spending_per_inactivity_month"] = X["total_trans_amt"] / (X["months_inactive_12_mon"] + 1)

        X["log_trans_amt_per_dependent"] = np.log1p(X["total_trans_amt"] / (X["dependent_count"] + 1))

        X["activity_momentum"] = X["total_ct_chng_q4_q1"] * X["engagement_velocity"]
        X["spending_momentum"] = X["total_amt_chng_q4_q1"] * X["total_trans_amt"]
        X["utilization_momentum"] = util * X["total_amt_chng_q4_q1"]
        X["inactivity_acceleration"] = X["months_inactive_12_mon"] * X["activity_decay"]

        X["high_value_ratio"] = self._safe_div(X["total_trans_amt"], (X["total_trans_amt"] + X["total_revolving_bal"]), eps=1e-3)
        X["credit_pressure_index"] = util * X["revolving_intensity"]
        X["relationship_efficiency"] = self._safe_div(X["total_trans_ct"], X["total_relationship_count"], eps=1e-3)
        X["complaint_intensity"] = self._safe_div(X["contacts_count_12_mon"], X["months_on_book"], eps=1.0)

        X["instability_index"] = (abs(X["total_ct_chng_q4_q1"]) + abs(X["total_amt_chng_q4_q1"]))
        X["stress_score"] = (util + X["revolving_intensity"] + X["complaint_intensity"])

        X["value_velocity"] = X["total_trans_amt"] / (X["months_on_book"] + 1)
        X["relationship_velocity"] = X["total_relationship_count"] / (X["months_on_book"] + 1)
        X["net_activity_flow"] = (X["total_trans_ct"] - X["months_inactive_12_mon"])

        X["retention_score"] = (
            0.4 * X["engagement_score"]
            + 0.3 * (1 - X["inactivity_ratio"])
            + 0.3 * X["relationship_velocity"]
        )
        X["dropout_signal"] = X["activity_decay"] + X["inactivity_ratio"] + X["complaint_intensity"]

        # -------------------------
        # CONJUNTO DE VARIÁVEIS: FILTRO LITE
        # -------------------------
        feature_set = self._get_feature_set()
        if feature_set == "lite":
            # Remove variáveis meta/compostas que tendem a ser redundantes e podem prejudicar a estabilidade do tuning
            lite_drop = [
                "stress_score",
                "risk_behavior_score",
                "dropout_signal",
                "instability_index",
                "relationship_density",  # mantém products_per_tenure em vez disso
            ]
            X.drop(columns=lite_drop, errors="ignore", inplace=True)

        # -------------------------
        # MANIPULAÇÃO DE CATEGÓRICAS
        # -------------------------
        return self._finalize_categoricals(X)

    def _finalize_categoricals(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Define os tipos categóricos (dtype) conforme o modo, garantindo que as colunas estejam no tipo correto
        ANTES de qualquer etapa de codificação (para XGB) ou para o tratamento categórico nativo (CatBoost/LGBM).
        """
        X = X.copy()

        # Candidatos à colunas categóricas (manter se presente)
        cat_candidates = ["education_level", "income_category", "card_category", "marital_status", "risk_level"]

        # Manipulação do gênero
        if "gender" in X.columns:
            if self.mode == "catboost":
                # Manter como string para permitir categóricas nativas no CatBoost
                if not pd.api.types.is_numeric_dtype(X["gender"]):
                    X["gender"] = X["gender"].astype(str).str.strip()
                else:
                    pass
            elif self.mode == "lgbm":
                g = self._coerce_gender_to_numeric_if_needed(X["gender"])
                if g.notna().any():
                    try:
                        X["gender"] = g.astype("Int64").astype("category")
                    except Exception:
                        X["gender"] = g.astype("category")
                else:
                    X["gender"] = X["gender"].astype("category")
            else:  # xgb
                # Preferir numérico 0/1 para XGB se possível; caso contrário manter como string para encoding posterior
                g = self._coerce_gender_to_numeric_if_needed(X["gender"])
                if g.notna().any():
                    X["gender"] = g.astype("float")  # allow NaNs
                else:
                    X["gender"] = X["gender"].astype(str).str.strip()

        # Outras categóricas
        for c in cat_candidates:
            if c not in X.columns:
                continue

            if self.mode == "catboost":
                X[c] = X[c].astype(str)
            elif self.mode == "lgbm":
                X[c] = X[c].astype("category")
            else:  # xgb
                # manter como string para o passo dos encoders externos
                X[c] = X[c].astype(str)

        return X