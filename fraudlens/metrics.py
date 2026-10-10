"""Metrics that survive a 0.17% fraud rate, all tied to a review capacity of k alerts."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from . import config


def ranking_metrics(y, scores) -> dict:
    """PR-AUC is the headline. ROC-AUC is kept only to show how much it flatters."""
    return {
        "pr_auc": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
    }


def queue_metrics(y, scores, amounts, k: int = config.DAILY_CAPACITY) -> dict:
    """What the analysts get if they review the k highest-scored transactions."""
    y, scores, amounts = np.asarray(y), np.asarray(scores), np.asarray(amounts)
    top = np.argsort(-scores, kind="stable")[:k]
    caught = int(y[top].sum())
    total = int(y.sum())
    fraud_value = float(amounts[y == 1].sum())
    caught_value = float(amounts[top][y[top] == 1].sum())
    return {
        "k": int(k),
        "frauds_in_queue": caught,
        "frauds_total": total,
        "precision": caught / k,
        "recall": caught / total,
        "max_recall_at_capacity": min(k, total) / total,
        "fraud_value_caught": caught_value,
        "fraud_value_total": fraud_value,
        "value_recall": caught_value / fraud_value,
        "cutoff_score": float(scores[top[-1]]),
    }


def bootstrap_queue_ci(y, scores, k: int = config.DAILY_CAPACITY, n_boot: int = 1000) -> dict:
    """95% range for precision and recall at k, by resampling the day's transactions.

    One test day holds about 200 frauds, so a single precision number hides
    real uncertainty. This says how far it could plausibly move on another day.
    """
    y, scores = np.asarray(y), np.asarray(scores)
    rng = np.random.default_rng(config.SEED)
    n = len(y)
    precision, recall = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        y_b, s_b = y[idx], scores[idx]
        top = np.argpartition(-s_b, k)[:k]
        caught = y_b[top].sum()
        precision.append(caught / k)
        recall.append(caught / max(y_b.sum(), 1))
    lo_p, hi_p = np.percentile(precision, [2.5, 97.5])
    lo_r, hi_r = np.percentile(recall, [2.5, 97.5])
    return {"precision_ci": [float(lo_p), float(hi_p)], "recall_ci": [float(lo_r), float(hi_r)]}


def capacity_curve(y, scores, amounts, ks) -> pd.DataFrame:
    """Precision, recall and fraud value caught for a range of daily review capacities."""
    y, scores, amounts = np.asarray(y), np.asarray(scores), np.asarray(amounts)
    order = np.argsort(-scores, kind="stable")
    hits = np.cumsum(y[order])
    value = np.cumsum(amounts[order] * y[order])
    ks = np.asarray(ks)
    return pd.DataFrame(
        {
            "alerts_per_day": ks,
            "frauds_caught": hits[ks - 1],
            "precision": hits[ks - 1] / ks,
            "recall": hits[ks - 1] / y.sum(),
            "fraud_value_caught": value[ks - 1],
        }
    )


def band_profile(y, scores, thresholds: dict[str, float], k: int = config.DAILY_CAPACITY) -> dict:
    """Within the k-alert queue: how many alerts fall in each risk band and how often they are fraud."""
    y, scores = np.asarray(y), np.asarray(scores)
    top = np.argsort(-scores, kind="stable")[:k]
    y_top, s_top = y[top], scores[top]
    edges = {
        "CRITICAL": (thresholds["CRITICAL"], np.inf),
        "HIGH": (thresholds["HIGH"], thresholds["CRITICAL"]),
        "ELEVATED": (-np.inf, thresholds["HIGH"]),
    }
    profile = {}
    for band, (low, high) in edges.items():
        in_band = (s_top >= low) & (s_top < high)
        alerts = int(in_band.sum())
        frauds = int(y_top[in_band].sum())
        profile[band] = {"alerts": alerts, "frauds": frauds, "precision": frauds / alerts if alerts else None}
    return profile
