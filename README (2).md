# Telecom Customer Churn Prediction & Retention Recommendation System

> An end-to-end machine learning pipeline that predicts customer churn, explains predictions using SHAP, segments customers into risk profiles, and delivers personalized retention recommendations — deployed as a Streamlit web application.

---

## Table of Contents

- [Overview](#overview)
- [Live Demo](#live-demo)
- [Project Structure](#project-structure)
- [Pipeline Architecture](#pipeline-architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Notebooks](#notebooks)
- [Results](#results)
- [Key Findings](#key-findings)
- [Tech Stack](#tech-stack)
- [Research Paper](#research-paper)
- [License](#license)

---

## Overview

Telecom operators face annual churn rates of 15–25% in competitive markets. This project builds a complete analytical system that moves beyond standalone churn prediction to deliver an integrated pipeline covering:

- **Predictive Analytics** — Binary churn classification using four ML models
- **Explainable AI** — SHAP-based global and instance-level explanations
- **Customer Segmentation** — K-Means clustering into four behavioral risk profiles
- **Prescriptive Analytics** — Rule-based, priority-ranked retention recommendations
- **Deployment** — Four-page Streamlit web application with single and batch inference

---

## Live Demo

```
https://your-app-name.streamlit.app
```

> Replace with your Streamlit Cloud deployment URL.

---

## Project Structure

```
telecom_churn_project/
│
├── data/
│   └── TelcoCustomerChurn.csv          # IBM Telco dataset (7,043 records)
│
├── notebooks/
│   ├── 01_data_understanding.ipynb     # Dataset exploration and profiling
│   ├── 02_data_cleaning.ipynb          # Missing values, types, duplicates
│   ├── 03_eda.ipynb                    # Exploratory data analysis
│   ├── 04_feature_engineering.ipynb    # Derived features and encoding
│   ├── 05_modeling.ipynb               # Model training and comparison
│   ├── 06_model_selection.ipynb        # Threshold optimization and evaluation
│   ├── 07_shap_explainability.ipynb    # SHAP global and local explanations
│   ├── 08_customer_segmentation.ipynb  # K-Means clustering and profiling
│   └── 09_recommendation_system.ipynb # Rule-based recommendation engine
│
├── models/
│   ├── final_model.pkl                 # Serialized Random Forest model
│   ├── scaler.pkl                      # Fitted StandardScaler
│   ├── optimal_threshold.pkl           # Optimized classification threshold (0.51)
│   └── kmeans_model.pkl                # Fitted K-Means segmentation model
│
├── pages/
│   ├── 01_predict.py                   # Single customer prediction page
│   ├── 02_batch.py                     # Batch CSV upload and analysis page
│   └── 03_insights.py                  # Model insights and SHAP visualizations
│
├── app.py                              # Streamlit app entry point (Overview page)
├── styles.css                          # Custom Streamlit styling
├── requirements.txt                    # Python dependencies
├── research_paper.docx                 # Full academic research paper
└── README.md
```

---

## Pipeline Architecture

```
Raw Data (CSV)
     │
     ▼
Data Cleaning & Preprocessing
     │
     ▼
Exploratory Data Analysis
     │
     ▼
Feature Engineering (6 derived features)
     │
     ▼
SMOTE Resampling (training set only)
     │
     ▼
Model Training & Comparison
(Logistic Regression | Decision Tree | Random Forest | XGBoost)
     │
     ▼
Threshold Optimization → Final Model Selection (Random Forest)
     │
     ├──────────────────────────┐
     ▼                          ▼
SHAP Explainability       K-Means Segmentation
(Global + Local)          (4 Customer Segments)
     │                          │
     └──────────┬───────────────┘
                ▼
     Retention Recommendation Engine
                │
                ▼
     Streamlit Web Application
     (Overview | Predict | Batch | Insights)
```

---

## Dataset

**Source:** [IBM Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

| Property | Value |
|---|---|
| Records | 7,032 (after cleaning) |
| Features | 21 original + 6 engineered |
| Target | Churn (binary: Yes / No) |
| Churn Rate | 26.5% (class imbalance) |
| Numerical Features | tenure, MonthlyCharges, TotalCharges |
| Categorical Features | Contract, InternetService, PaymentMethod, and 15 others |

**Engineered Features:**

| Feature | Description |
|---|---|
| `ChargesPerTenure` | MonthlyCharges / (tenure + 1) — value density proxy |
| `TenureGroup` | Lifecycle stage bins: Early / Developing / Established / Loyal |
| `NumAddons` | Count of active add-on service subscriptions (0–6) |
| `HasStreaming` | 1 if StreamingTV or StreamingMovies is active |
| `IsLongTermContract` | 1 if contract is one-year or two-year |
| `IsAutoPayment` | 1 if payment method is automated (credit card / bank transfer) |

---

## Installation

**Prerequisites:** Python 3.9+

```bash
# 1. Clone the repository
git clone https://github.com/your-username/telecom-churn-prediction.git
cd telecom-churn-prediction

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

**requirements.txt includes:**
```
pandas
numpy
scikit-learn
imbalanced-learn
xgboost
shap
matplotlib
seaborn
plotly
streamlit
joblib
```

---

## Usage

### Run the Streamlit App Locally

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501` with four pages:

| Page | Description |
|---|---|
| **Overview** | KPIs, segment summary, pipeline introduction |
| **Customer Predict** | Manual input form → churn probability + recommendations |
| **Batch Analysis** | CSV upload → downloadable predictions for all customers |
| **Model Insights** | SHAP plots, segment profiles, model comparison |

### Run Individual Notebooks

Execute notebooks sequentially in the order numbered `01` through `09`. Each notebook saves outputs (cleaned data, models, plots) consumed by the next.

```bash
jupyter notebook notebooks/
```

### Batch Inference via CSV

Upload a CSV to the Batch Analysis page with the same column structure as the source dataset. The app returns a table with:
- Predicted churn probability
- Binary churn prediction
- Risk tier (Critical / High / Medium / Low)
- Assigned customer segment
- Top retention recommendations

---

## Notebooks

| Notebook | Purpose | Key Output |
|---|---|---|
| `01_data_understanding` | Schema review, data types, value distributions | Data profile report |
| `02_data_cleaning` | TotalCharges conversion, null removal, encoding | `cleaned_data.csv` |
| `03_eda` | Churn distributions, feature-target relationships, correlation analysis | 15+ visualizations |
| `04_feature_engineering` | 6 derived features, one-hot encoding, feature selection | `engineered_data.csv` |
| `05_modeling` | SMOTE resampling, training all 4 classifiers, cross-validation | Model comparison table |
| `06_model_selection` | Threshold optimization, final model selection, confusion matrix | `final_model.pkl` |
| `07_shap_explainability` | TreeExplainer, global importance, waterfall plots | SHAP visualizations |
| `08_customer_segmentation` | Elbow + Silhouette analysis, K=4 clustering, segment profiling | `kmeans_model.pkl` |
| `09_recommendation_system` | Rule engine logic, priority assignment, sample outputs | Recommendation outputs |

---

## Results

### Model Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.741 | 0.508 | 0.757 | 0.608 | 0.824 |
| Decision Tree | 0.735 | 0.501 | 0.733 | 0.595 | 0.808 |
| **Random Forest** ✓ | **0.750** | **0.520** | **0.765** | **0.619** | **0.831** |
| XGBoost | 0.746 | 0.517 | 0.706 | 0.597 | 0.821 |

> Evaluated on held-out test set (n = 1,407) at optimized threshold of 0.51.

### Customer Segments

| Segment | Size | Churn Rate | Avg. Tenure | Monthly Charge | Priority |
|---|---|---|---|---|---|
| Critical Churn-Risk | 2,126 (30.2%) | 54.7% | 21 months | $87 | Critical |
| Vulnerable High-Risk | 1,744 (24.8%) | 28.5% | 13 months | $41 | High |
| Stable Mid-Risk | 2,029 (28.9%) | 9.2% | 56 months | $84 | Medium |
| Loyal Low-Risk | 1,133 (16.1%) | 2.0% | 42 months | $25 | Low |

### Top SHAP Features

| Rank | Feature | SHAP Score | Direction |
|---|---|---|---|
| 1 | IsLongTermContract | 0.0721 | Protective |
| 2 | ChargesPerTenure | 0.0552 | Risk |
| 3 | Contract_Month-to-month | 0.0527 | Risk |
| 4 | InternetService_Fiber optic | 0.0446 | Risk |
| 5 | Contract_Two year | 0.0318 | Protective |

---

## Key Findings

- **Contract type is the primary churn lever.** Month-to-month customers churn at 43.0% versus 2.8% for two-year contract holders — a 15× differential and the top SHAP predictor.
- **Early tenure is the critical risk window.** Customers in months 0–12 churn at 47.4%, falling to 6.1% beyond four years. Over half of all churned customers leave within the first two years.
- **Value perception drives churn more than billing amount.** The engineered `ChargesPerTenure` feature ranks 2nd in SHAP importance, above raw `MonthlyCharges` (rank 7).
- **Add-on adoption functions as a retention anchor.** Churn rate drops from 31.7% (zero add-ons) to 7.8% (four or more add-ons).
- **SMOTE favors Random Forest over XGBoost.** Gradient boosting's residual-weighting mechanism conflicts with SMOTE-generated synthetic samples, degrading test-set generalization.
- **55% of the customer base requires active intervention.** The Critical and High segments combined represent 3,870 customers needing targeted retention action.

---

## Tech Stack

| Category | Tools |
|---|---|
| Data Processing | pandas, numpy |
| Machine Learning | scikit-learn, xgboost, imbalanced-learn |
| Explainability | shap |
| Visualization | matplotlib, seaborn, plotly |
| Deployment | Streamlit, Streamlit Cloud |
| Serialization | joblib |
| Environment | Python 3.9+, Jupyter Notebook |

---

## Research Paper

A full academic research paper documenting the methodology, results, and findings is included in the repository:

```
research_paper.docx
```

The paper covers:
- Problem Statement and Research Objectives
- Literature Review (churn prediction, SMOTE, SHAP, K-Means)
- Complete Methodology with Data Dictionary
- Results across all pipeline phases
- Key Findings Summary with business implications
- Evidence-based recommendations for telecom operators

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## Citation

If you use this project or reference its findings:

```
[Author Name] (2025). Telecom Customer Churn Prediction and Retention Recommendation System
Using Machine Learning and Explainable Artificial Intelligence.
[University Name], [Department], [Course].
```

---

*Built as part of a course project in Applied Machine Learning / Data Science.*
