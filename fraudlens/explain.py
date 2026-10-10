"""Per-alert reasons: which inputs pushed this transaction's score up, in plain words.

The V-features are the issuer's anonymised PCA components, so a reason cannot
say "unusual merchant". It can honestly say "V14 is lower than 99.9% of genuine
transactions", and in production the issuer maps V14 back to its raw fields.
"""

import numpy as np
import pandas as pd
import xgboost as xgb

from .features import hour_of_day

CARD_TEST_MAX_AMOUNT = 1.00  # EDA: a third of all frauds are EUR 1.00 or less


def model_inputs(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """The feature table exactly as the final estimator sees it."""
    feats = pipeline.named_steps["features"].transform(X)
    if "scale" in pipeline.named_steps:
        feats = pd.DataFrame(pipeline.named_steps["scale"].transform(feats), columns=feats.columns, index=feats.index)
    return feats


def contributions(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """How much each input moved each score, in log-odds. Positive = towards fraud.

    Tree models: exact SHAP values from XGBoost itself (`pred_contribs`).
    Linear model: coefficient x standardised value.
    The two hour columns are merged into one "hour" contribution.
    """
    feats = model_inputs(pipeline, X)
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "get_booster"):
        booster = clf.get_booster()
        matrix = xgb.DMatrix(feats.to_numpy(), feature_names=booster.feature_names)
        values = booster.predict(matrix, pred_contribs=True)[:, :-1]  # last column is the base value
    else:
        values = feats.to_numpy() * clf.coef_[0]
    contrib = pd.DataFrame(values, columns=feats.columns, index=X.index)
    contrib["hour"] = contrib.pop("hour_sin") + contrib.pop("hour_cos")
    return contrib.rename(columns={"amount_log": "Amount"})


def _percentile(ref_quantiles: pd.DataFrame, feature: str, value: float) -> float:
    """Share of genuine training transactions with a smaller value of this feature."""
    return float(np.searchsorted(ref_quantiles[feature].to_numpy(), value) / (len(ref_quantiles) - 1))


def _position(pct: float) -> tuple[str, str]:
    """(short label, full phrase) for where a value sits among genuine transactions."""
    low = pct < 0.5
    tail = (1 - pct) if low else pct  # share of genuine transactions this value is beyond
    share = f"{min(100 * tail, 99.9):.1f}%"
    side, word = ("lower", "low") if low else ("higher", "high")
    if tail >= 0.999:
        degree = "extremely "
    elif tail >= 0.99:
        degree = "very "
    elif tail >= 0.95:
        degree = ""
    else:
        return "in normal range", "within the normal range; it raises risk only in combination with the other signals"
    return f"{degree}{word}", f"{side} than {share} of genuine transactions"


def describe(feature: str, row: pd.Series, bundle: dict) -> dict:
    """One reason as a short label (for the one-line note) and a full sentence."""
    if feature == "hour":
        hour = hour_of_day(row["Time"])
        lift = float(bundle["hour_lift"].iloc[int(hour)])
        clock = f"{int(hour):02d}:{int(hour % 1 * 60):02d}"
        return {
            "feature": "hour",
            "short": f"time {clock} ({lift:.1f}x fraud rate)",
            "text": f"Time of day {clock}: the fraud rate at this hour was {lift:.1f}x the daily average on the training day",
        }
    if feature == "Amount":
        amount = float(row["Amount"])
        if amount <= CARD_TEST_MAX_AMOUNT:
            return {
                "feature": "Amount",
                "short": f"micro-amount EUR {amount:.2f}",
                "text": f"Amount EUR {amount:.2f}: a micro-amount, the typical card-testing pattern",
            }
        short, phrase = _position(_percentile(bundle["ref_quantiles"], "Amount", amount))
        return {"feature": "Amount", "short": f"amount {short}", "text": f"Amount EUR {amount:.2f}: {phrase}"}
    value = float(row[feature])
    short, phrase = _position(_percentile(bundle["ref_quantiles"], feature, value))
    return {"feature": feature, "short": f"{feature} {short}", "text": f"{feature} = {value:.2f}: {phrase}"}


def top_reasons(bundle: dict, X: pd.DataFrame, n: int = 3) -> list[list[dict]]:
    """For each row, the n inputs that pushed the score up the most."""
    contrib = contributions(bundle["pipeline"], X)
    reasons = []
    for idx, row in X.iterrows():
        strongest = contrib.loc[idx].sort_values(ascending=False).head(n)
        reasons.append([{**describe(feature, row, bundle), "contribution": float(value)} for feature, value in strongest.items()])
    return reasons
