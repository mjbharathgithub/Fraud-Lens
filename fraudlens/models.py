"""Model definitions: one baseline and three ways of handling the imbalance.

Each candidate is a single Pipeline, so every fitted step (scaler, SMOTE,
classifier) only ever sees the rows it is trained on.
"""

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from . import config
from .features import TransactionFeatures

XGB_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "eval_metric": "aucpr",
    "n_jobs": 6,
    "random_state": config.SEED,
}

# How each candidate deals with a 0.17% positive class.
IMBALANCE_HANDLING = {
    "logistic_balanced": "class_weight=balanced",
    "xgb_plain": "none (ranking only)",
    "xgb_weighted": "scale_pos_weight",
    "xgb_smote_in_fold": "SMOTE inside training folds",
}


def make_candidates(pos_weight: float) -> dict:
    """Fresh, unfitted candidates. `pos_weight` is the legit-to-fraud ratio of the training data."""
    return {
        "logistic_balanced": Pipeline(
            [
                ("features", TransactionFeatures()),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(class_weight="balanced", max_iter=2000)),
            ]
        ),
        "xgb_plain": Pipeline(
            [
                ("features", TransactionFeatures()),
                ("clf", XGBClassifier(**XGB_PARAMS)),
            ]
        ),
        "xgb_weighted": Pipeline(
            [
                ("features", TransactionFeatures()),
                ("clf", XGBClassifier(**XGB_PARAMS, scale_pos_weight=pos_weight)),
            ]
        ),
        # imblearn's Pipeline applies SMOTE during fit only, so inside
        # cross-validation it resamples the training folds and never the held-out one.
        "xgb_smote_in_fold": ImbPipeline(
            [
                ("features", TransactionFeatures()),
                ("smote", SMOTE(random_state=config.SEED)),
                ("clf", XGBClassifier(**XGB_PARAMS)),
            ]
        ),
    }
