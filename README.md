# Explainable Credit Risk Modeling with XGBoost and SHAP
This project implements an end-to-end **credit risk (default prediction) pipeline** using XGBoost, careful target leakage prevention, and a strong focus on **explainability and real-world modeling practices**.

Data was taken from here: https://www.kaggle.com/datasets/wordsforthewise/lending-club/data

The workflow emphasizes:
- **Target leakage prevention** through careful temporal feature selection  
- **Domain-aware missing value handling** for sparse credit bureau variables  
- **Robust feature engineering** aligned with lending use cases  
- **Model validation via a regularized logistic regression baseline**  
- **Global and local explainability using SHAP**, including loan-level risk explanations
- **Bayesian Optimization** to tune hyperparameters

The final model produces both **accurate default risk predictions** and **transparent explanations** suitable for credit analysts, auditors, and regulatory review.

## Project Pipeline
-### 1. **Data Processing & Feature Engineering
      - Missing value handling (zero vs median logic based on financial meaning)
      - Target encoding for categorical variables
      - Standard scaling for numerical features
    2. **Modeling
      - Baseline Logistic regression model
      - Default XGBoost model
      - Tune XGBoost using Bayesian optimization and Stratified 5-Fold Cross Validation
      - Calibrated probability using isotonic regression

## Model Comparison
| Model | AUC | Log loss |
--------------------------
| Logistic Regression | 0.720985 | 0.616411

### Key Highlights
- Scales to large tabular datasets (1M+ rows)
- Handles highly sparse bureau features without information loss
- Captures non-linear risk patterns and feature interactions
- Generates borrower-level explanations for individual loan decisions


This project is designed to reflect **industry-style credit risk modeling**, rather than a purely academic approach.

