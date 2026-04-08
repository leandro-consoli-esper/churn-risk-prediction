# ============================================================
# APP STREAMLIT — PREVISÃO DE CHURN DE CLIENTES
# ============================================================
# Esta aplicação fornece uma interface simples para executar inferência utilizando o
# pipeline treinado em LightGBM. O pipeline realiza pré-processamento + engenharia
# de atributos internamente (por meio do Preprocessor do projeto).
#
# Saída:
#  - churn_probability: score de risco (0–1)
#  - churn_prediction: decisão binária baseada em um threshold calibrado
#  - risk_band: interpretação de risco em linguagem simples
# ============================================================

# ---------- CONFIGURAÇÃO DO PROJETO ----------
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

# ---------- BIBLIOTECAS EXTERNAS ----------
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import altair as alt

# ---------- IMPORTAÇÕES INTERNAS ----------
from src.inference import predict_churn

# ---------- CAMINHOS ----------
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
FEATURE_LIST_PATH = ARTIFACTS_DIR / "feature_list.pkl"
THRESHOLD_PATH = ARTIFACTS_DIR / "decision_threshold.pkl"

# ---------- CONFIGURAÇÃO DA PÁGINA ----------
st.set_page_config(page_title="Previsão de Churn de Clientes", layout="centered")

# ---------- FUNÇÕES AUXILIARES ----------
def risk_band(p: float) -> str:
    """
    Faixas simples de interpretação de risco para fins de portfólio/demonstração.
    Ajuste os limites conforme a política de negócio.
    """
    if p < 0.20:
        return "Low"
    if p < 0.50:
        return "Medium"
    return "High"


# ---------- CABEÇALHO ----------
st.title("📉 Previsão de Churn de Clientes")

st.markdown(
    """
Esta aplicação prevê a **probabilidade de churn de clientes** utilizando um modelo treinado de **LightGBM**.

Faça o upload de um dataset contendo o **mesmo esquema de variáveis utilizado no treinamento**.  
Todo o pré-processamento e a engenharia de atributos são realizados internamente pelo pipeline.
"""
)

# ---------- CARREGAMENTO DOS ARTEFATOS (schema + threshold) ----------
try:
    FEATURE_LIST = joblib.load(FEATURE_LIST_PATH)
except FileNotFoundError:
    FEATURE_LIST = None
    st.warning(
        "Arquivo de schema de variáveis não encontrado (feature_list.pkl). "
        "As previsões podem falhar se os dados de entrada não corresponderem ao esquema de treinamento."
    )

try:
    THRESHOLD = float(joblib.load(THRESHOLD_PATH))
except FileNotFoundError:
    THRESHOLD = None
    st.warning(
        "Arquivo de threshold não encontrado (decision_threshold.pkl). "
        "Os gráficos serão exibidos sem a linha de referência do threshold."
    )

# ============================================================
# UPLOAD DE CSV — MÉTODO PRINCIPAL DE ENTRADA
# ============================================================
uploaded_file = st.file_uploader("Envie um arquivo CSV com os dados dos clientes", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)

    st.subheader("Pré-visualização dos Dados de Entrada")
    st.dataframe(input_df.head())

    # Schema esperado (para usuário/referência)
    if FEATURE_LIST is not None:
        with st.expander("Colunas esperadas (schema de treinamento)"):
            st.code("\n".join(FEATURE_LIST))

    # ---------- EXECUTAR INFERÊNCIA ----------
    if st.button("Executar Previsão"):
        try:
            results = predict_churn(input_df).copy()

            # Adicionar interpretação de risco
            if "churn_probability" in results.columns:
                results["risk_band"] = results["churn_probability"].apply(risk_band)
                results = results.sort_values("churn_probability", ascending=False).reset_index(drop=True)

            st.subheader("Resultados da Previsão (ordenados por risco)")
            st.dataframe(results)

            # ---------- RESUMO DE RISCO ----------
            if "risk_band" in results.columns:
                st.subheader("Resumo de Risco")
                counts = (
                    results["risk_band"]
                    .value_counts()
                    .reindex(["High", "Medium", "Low"])
                    .fillna(0)
                    .astype(int)
                )

                c1, c2, c3 = st.columns(3)
                c1.metric("Alto risco", int(counts.get("High", 0)))
                c2.metric("Risco médio", int(counts.get("Medium", 0)))
                c3.metric("Baixo risco", int(counts.get("Low", 0)))

            # ---------- DISTRIBUIÇÃO DE PROBABILIDADE ----------
            if "churn_probability" in results.columns:
                st.subheader("Distribuição da Probabilidade de Churn")

                chart_df = results[["churn_probability"]].copy()

                hist = (
                    alt.Chart(chart_df)
                    .mark_bar()
                    .encode(
                        x=alt.X(
                            "churn_probability:Q",
                            bin=alt.Bin(maxbins=30),
                            title="Probabilidade de churn",
                        ),
                        y=alt.Y("count():Q", title="Clientes"),
                        tooltip=[alt.Tooltip("count():Q", title="Clientes")],
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
                label="Baixar resultados em CSV",
                data=results.to_csv(index=False),
                file_name="churn_predictions.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error("A previsão falhou. Verifique o schema de entrada e os tipos de dados.")
            st.exception(e)

# ---------- RODAPÉ ----------
st.markdown("---")
st.caption(
    "Modelo: LightGBM | Threshold calibrado no conjunto de validação | Saída com score de risco | Explicável com SHAP"
)