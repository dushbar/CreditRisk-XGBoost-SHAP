# Explainable Credit Risk Modeling with XGBoost and SHAP
This project implements an end-to-end **credit risk (default prediction) pipeline** using XGBoost, careful target leakage prevention, and a strong focus on **explainability and real-world modeling practices**.
Data was taken from here: https://www.kaggle.com/datasets/wordsforthewise/lending-club/data

The workflow emphasizes:
- **Target leakage prevention** through careful temporal feature selection  
- **Domain-aware missing value handling** for sparse credit bureau variables  
- **Robust feature engineering** aligned with lending use cases  
- **Model validation via a regularized logistic regression baseline**  
- **Global and local explainability using SHAP**, including loan-level risk explanations  

The final model produces both **accurate default risk predictions** and **transparent explanations** suitable for credit analysts, auditors, and regulatory review.

### Key Highlights
- Scales to large tabular datasets (1M+ rows)
- Handles highly sparse bureau features without information loss
- Captures non-linear risk patterns and feature interactions
- Generates borrower-level explanations for individual loan decisions

This project is designed to reflect **industry-style credit risk modeling**, rather than a purely academic approach.

