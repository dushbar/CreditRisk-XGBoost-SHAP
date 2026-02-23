from skopt import gp_minimize
from skopt.space import Real, Integer
from xgboost import XGBClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import numpy as np


def tune_xgb_credit_risk_cv(
    X, y,
    n_calls=25,
    random_state=42,
):
    """
    Bayesian optimization with Stratified 5-Fold CV.
    Optimizes mean log loss across folds.
    Reports mean + std AUC.
    """

    pos_weight = (y == 0).sum() / (y == 1).sum()

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

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    def objective(params_list):

        params = dict(zip(param_names, params_list))

        fold_logloss = []
        fold_auc = []

        for train_idx, valid_idx in cv.split(X, y):

            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

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

            fold_logloss.append(log_loss(y_valid, y_proba))
            fold_auc.append(roc_auc_score(y_valid, y_proba))

        return np.mean(fold_logloss)

    res = gp_minimize(
        func=objective,
        dimensions=dimensions,
        n_calls=n_calls,
        random_state=random_state,
        verbose=True,
    )

    best_params = dict(zip(param_names, res.x))

    print("Best Parameters:", best_params)

    return best_params