from skopt import gp_minimize
from skopt.space import Real, Integer
from xgboost import XGBClassifier
from sklearn.metrics import log_loss, roc_auc_score
import numpy as np


def tune_xgb_credit_risk(
    X_train, y_train,
    X_valid, y_valid,
    n_calls=40,
    random_state=42,
):
    """
    Bayesian optimization for credit risk:
    - Optimize log loss
    - Report AUC for best model
    """


    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    #define the parameters
    param_space = {
        "learning_rate": Real(0.01, 0.15, prior="log-uniform"),
        "max_depth": Integer(3, 6),
        "min_child_weight": Integer(20, 200),
        "subsample": Real(0.7, 1.0),
        "colsample_bytree": Real(0.7, 1.0),
        "scale_pos_weight": Real(
        0.5 * pos_weight, 2.0 * pos_weight, prior="log-uniform"
        ),
    }


    dimensions = list(param_space.values())
    param_names = list(param_space.keys())

    
    def objective(params_list):
        params = dict(zip(param_names, params_list))

        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=2000,
            early_stopping_rounds=50,
            random_state=random_state,
            verbosity=0,
            n_jobs=-1,
            **params,
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False,
        )

        y_proba = model.predict_proba(X_valid)[:, 1]
        return log_loss(y_valid, y_proba)


    # Bayesian optimization
    res = gp_minimize(
        func=objective,
        dimensions=dimensions,
        n_calls=n_calls,
        random_state=random_state,
        verbose=True,
    )

    #Train the best model
    best_params = dict(zip(param_names, res.x))

    best_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=2000,
        early_stopping_rounds=50,
        random_state=random_state,
        verbosity=0,
        n_jobs=-1,
        **best_params,
    )

    best_model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )


    y_proba = best_model.predict_proba(X_valid)[:, 1]

    best_logloss = log_loss(y_valid, y_proba)
    best_auc = roc_auc_score(y_valid, y_proba)

    print(f"Best Log Loss : {best_logloss:.5f}")
    print(f"Best AUC      : {best_auc:.5f}")

    return best_model, best_params, best_logloss, best_auc
