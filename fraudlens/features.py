"""Feature engineering as a scikit-learn transformer, so it lives inside the Pipeline."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from . import config

FEATURES_OUT = [*config.V_COLS, "amount_log", "hour_sin", "hour_cos"]


def hour_of_day(time_seconds):
    """Hours since midnight, assuming the dataset's clock starts at midnight.

    `Time` is seconds since the first transaction. The overnight dip in volume
    lines up with hours 1-5 under this assumption (see the EDA notebook).
    """
    return (np.asarray(time_seconds) % config.SECONDS_PER_DAY) / 3600


class TransactionFeatures(TransformerMixin, BaseEstimator):
    """Turn raw columns into model inputs.

    - V1..V28 pass through unchanged (already PCA components).
    - Amount is log-transformed: it is heavily right-skewed.
    - Raw `Time` is dropped. It is a running counter, so day-2 values lie
      outside anything seen in training. Hour of day replaces it, encoded on a
      circle so 23:00 and 01:00 are neighbours.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        hour = hour_of_day(X["Time"])
        out = X[config.V_COLS].copy()
        out["amount_log"] = np.log1p(X["Amount"].to_numpy())
        out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array(FEATURES_OUT)
