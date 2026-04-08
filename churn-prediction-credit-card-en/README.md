# 📉 Customer Churn Risk Prediction — Credit Card Portfolio

![Churn](images/churn.jpg)

A machine learning project focused on predicting customer churn using behavioral and financial data, with an emphasis on business impact, risk scoring, and model interpretability.

This project demonstrates an end-to-end data science workflow — from exploratory analysis and feature engineering to model optimization, explainability, and deployment through a Streamlit application.

---

# 🧠 Business Context

Customer churn represents a significant risk for financial institutions, as losing customers directly impacts long-term revenue and customer lifetime value.

Traditional rule-based approaches often fail to capture complex behavioral patterns that precede churn. This project aims to provide a data-driven solution capable of identifying high-risk customers early, enabling proactive retention strategies.

---

# 🎯 Objective

The primary objective of this project is to:

- Predict the probability of customer churn using historical transactional data  
- Maximize recall for churners while maintaining controlled false positives
- Provide a risk scoring framework to support targeted retention actions

---

# 🧭 Modeling Strategy

## Business Objective

Reduce churn by identifying as many at-risk customers as possible while maintaining operational efficiency.

## Modeling Positioning

Risk scoring model — outputs calibrated churn probabilities that can be used for prioritization and segmentation.

## Optimization Metrics

- **F2 Score** → prioritizes recall (higher cost for missed churners)  
- **PR-AUC** → robust evaluation for imbalanced datasets  

---

# 📊 Dataset Overview

- ~10,000 customers  
- Churn rate ≈ 16% (unbalanced dataset)
- Rich behavioral and financial attributes  
- Multiple engagement and transaction features  

![Target Distribution](images/targetdist.png)

---

# ⚙️ Project Workflow
This project development is built over the CRISP-DM framework.

## 1️⃣ Exploratory Data Analysis

- Distribution analysis
- Univariate analysis
- Outlier validation
- Multivariate analysys  
- Behavioral segmentation    
- Correlation assessment  

## 2️⃣ Feature Engineering

Domain-driven features capturing customer behavior dynamics, including:

- Activity flow metrics  
- Engagement ratios  
- Spending momentum  
- Utilization patterns  
- Interaction effects  

## 3️⃣ Modeling

Models evaluated:

- LightGBM  
- CatBoost  
- XGBoost  
- Random Forest (baseline)  

![Models](images/models.png)

Hyperparameter tuning performed using **Optuna** with cross-validation on training data only to avoid data leakage.

## 4️⃣ Threshold Calibration

Decision threshold selected on the validation set using constraint-aware optimization to balance recall and false positives.

## 5️⃣ Final Training

The final model was retrained on **Train + Validation datasets** to maximize the learning signal before the final test evaluation.

---

# 🏆 Final Model — LightGBM

LightGBM was selected based on overall performance, stability, and interpretability.

## 📈 Test Performance

| Metric | Value |
|------|------|
| ROC-AUC | 0.9935 |
| PR-AUC | 0.9748 |
| Recall (Churn) | 0.9568 |
| Precision (Churn) | 0.8659 |
| F2 Score | 0.9371 |

The model demonstrates strong discriminative power while maintaining high recall, aligning with the business objective of minimizing missed churners. Put simply, the model accurately classified 155 of 162 churners and 827 of 851 non-churners.
![LightGBM Consusion Matrix](images/light.png)

---

# 🔍 Model Explainability — SHAP

SHAP analysis confirms that predictions are driven by meaningful behavioral signals.

Top drivers include:

- Transaction amount  
- Activity flow  
- Transaction counts  
- Revolving balance  
- Inactivity patterns  
- Spending dynamics  

These insights reinforce the model’s alignment with real customer behavior and support trust in predictions.

![Shap Analysis](images/shap.png)

---

# 💰 Estimated Business Impact: Conservative Retention Scenario

To estimate financial impact conservatively, we assume:

- Only **15% of identified churners** are successfully retained  
- Customer value is estimated based on realistic revenue drivers in a credit card portfolio  
- No additional revenue from cross-sell, engagement uplift, or cost reduction is considered  

---

## 📊 Customer Value Estimation

Given the dataset represents a credit card portfolio, customer value was estimated considering two primary revenue streams:

### 1. Transaction-based revenue (interchange)
- Average annual transaction volume: **~4,400 monetary units**
- Assumed interchange rate: **~2%**
- Estimated annual revenue: **~88**

### 2. Interest on revolving balances
- Average revolving balance (proxy): **~1,500 monetary units**
- Assumed annual interest rate: **~25%**
- Estimated annual revenue: **~375**

###  Total estimated annual revenue per customer: > **~460 monetary units**
###  Observed churn rate: > **~16%**

---

### 📊 Estimated Customer Lifetime Value (LTV):
LTV ≈ Annual Revenue / Churn Rate
> **~2,800 monetary units per customer**

---

### Retention Impact

**Identified churners:**
1,551 customers  

**Successfully retained (15%):**
1,551 × 15% ≈ **233 customers**

## 💰 Estimated Retained Value

233 × 2,800 ≈ **~650,000 monetary units**

---

### ⚠️ Conservative Assumptions

This estimate intentionally:

- Uses simplified assumptions for interchange and interest rates  
- Approximates customer value without full revenue decomposition  
- Excludes cross-sell, upsell, and behavioral uplift  
- Does not account for operational cost reductions  

Therefore, this should be interpreted as a **conservative and lower-bound estimate** of financial impact.

---

## 🚀 Strategic Implications

Given:

- ~10,127 customers  
- Churn rate ≈ 16%  

And a model recall above **95%**, the solution is capable of identifying the majority of at-risk customers.

Even under conservative assumptions, the model enables:

- **Hundreds of thousands in preserved value**
- Risk-based customer prioritization  
- More efficient allocation of retention resources  
- Integration with CRM-driven retention workflows  

Beyond direct financial impact, this approach strengthens decision-making by aligning predictive modeling with real business value drivers.

---

# 🚀 Application — Streamlit Interface

The project includes an interactive application for running predictions.

Features:

- Upload customer dataset  
- Generate churn probability  
- Risk categorization  
- Download prediction results  
- Probability distribution visualization  

Run the app locally with:

```bash
streamlit run app.py
