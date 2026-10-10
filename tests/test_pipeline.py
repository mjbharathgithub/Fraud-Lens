"""Leakage guards and data-handling contracts."""

import numpy as np
import pandas as pd

from fraudlens import config, data
from fraudlens.features import FEATURES_OUT, TransactionFeatures
from fraudlens.metrics import band_profile, capacity_curve, queue_metrics
from fraudlens.models import make_candidates


def test_clean_removes_exact_duplicates(transactions):
    doubled = pd.concat([transactions, transactions.head(50).assign(txn_id=lambda d: d["txn_id"] + 10_000)])
    cleaned, report = data.clean(doubled)
    assert report["duplicates_removed"] == 50
    assert len(cleaned) == len(transactions)
    assert not cleaned.duplicated(subset=[*config.RAW_FEATURES, config.TARGET]).any()


def test_day_split_never_looks_ahead(transactions):
    cleaned, _ = data.clean(transactions)
    train, test = data.split_by_day(cleaned)
    assert set(train["txn_id"]).isdisjoint(test["txn_id"])
    assert train["Time"].max() < test["Time"].min()
    assert len(train) + len(test) == len(cleaned)


def test_features_drop_raw_time_and_need_no_fitting(transactions):
    feats = TransactionFeatures().transform(transactions)
    assert list(feats.columns) == FEATURES_OUT
    assert "Time" not in feats.columns
    # Stateless: transforming one row alone gives the same values as within the full table.
    alone = TransactionFeatures().transform(transactions.iloc[[7]])
    pd.testing.assert_frame_equal(alone, feats.iloc[[7]])


def test_scaler_is_fitted_on_training_rows_only(transactions):
    cleaned, _ = data.clean(transactions)
    train, _ = data.split_by_day(cleaned)
    pipe = make_candidates(pos_weight=30.0)["logistic_balanced"]
    pipe.fit(train[config.RAW_FEATURES], train[config.TARGET])
    train_mean = TransactionFeatures().transform(train).mean().to_numpy()
    all_mean = TransactionFeatures().transform(cleaned).mean().to_numpy()
    np.testing.assert_allclose(pipe.named_steps["scale"].mean_, train_mean)
    assert not np.allclose(pipe.named_steps["scale"].mean_, all_mean)


def test_smote_pipeline_resamples_only_while_fitting(transactions):
    cleaned, _ = data.clean(transactions)
    train, test = data.split_by_day(cleaned)
    pipe = make_candidates(pos_weight=30.0)["xgb_smote_in_fold"]
    pipe.set_params(clf__n_estimators=20)
    pipe.fit(train[config.RAW_FEATURES], train[config.TARGET])
    scores = pipe.predict_proba(test[config.RAW_FEATURES])[:, 1]
    assert len(scores) == len(test)  # no synthetic rows are added at prediction time


def test_queue_metrics_on_a_known_ranking():
    y = np.array([1, 0, 1, 0, 0, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
    amounts = np.array([10, 5, 30, 5, 5, 60, 5, 5])
    m = queue_metrics(y, scores, amounts, k=4)
    assert m["frauds_in_queue"] == 2
    assert m["precision"] == 0.5
    assert m["recall"] == 2 / 3
    assert m["value_recall"] == 40 / 100
    assert m["cutoff_score"] == 0.6


def test_capacity_curve_matches_queue_metrics():
    rng = np.random.default_rng(1)
    y = (rng.random(500) < 0.1).astype(int)
    scores, amounts = rng.random(500), rng.random(500) * 100
    curve = capacity_curve(y, scores, amounts, [10, 50, 200])
    for _, row in curve.iterrows():
        m = queue_metrics(y, scores, amounts, int(row["alerts_per_day"]))
        assert row["frauds_caught"] == m["frauds_in_queue"]
        assert np.isclose(row["recall"], m["recall"])


def test_band_profile_counts_only_the_queue():
    y = np.array([1, 1, 1, 0, 1, 0, 0, 0])
    scores = np.array([0.999, 0.995, 0.95, 0.93, 0.80, 0.70, 0.10, 0.05])
    profile = band_profile(y, scores, {"CRITICAL": 0.99, "HIGH": 0.90}, k=6)
    assert profile["CRITICAL"] == {"alerts": 2, "frauds": 2, "precision": 1.0}
    assert profile["HIGH"] == {"alerts": 2, "frauds": 1, "precision": 0.5}
    assert profile["ELEVATED"] == {"alerts": 2, "frauds": 1, "precision": 0.5}
