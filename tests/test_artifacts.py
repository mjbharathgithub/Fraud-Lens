"""The artefacts committed to the repo must agree with each other and with the model."""

import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fraudlens import config, triage
from fraudlens.api import app

pytestmark = pytest.mark.skipif(not config.MODEL_BUNDLE.exists(), reason="run `python -m fraudlens all` first")


@pytest.fixture(scope="module")
def metrics() -> dict:
    return json.loads(config.METRICS_JSON.read_text())


def test_committed_model_reproduces_the_committed_scores():
    bundle = triage.load_bundle()
    sample = pd.read_parquet(config.DAY2_SAMPLE)
    stored = pd.read_parquet(config.DAY2_EVAL).set_index("txn_id").loc[sample["txn_id"], "score"].to_numpy()
    np.testing.assert_allclose(triage.score(bundle, sample), stored, atol=1e-6)


def test_queue_file_matches_the_reported_metrics(metrics):
    queue = pd.read_csv(config.QUEUE_EVALUATED_CSV)
    day2 = metrics["champion"]["day2"]
    assert len(queue) == metrics["capacity"] == 200
    assert list(queue["rank"]) == list(range(1, 201))
    assert queue["risk_score"].is_monotonic_decreasing
    assert (queue["status"] == "PENDING_APPROVAL").all()
    assert int(queue["actual_fraud"].sum()) == day2["frauds_in_queue"]
    assert "actual_fraud" not in pd.read_csv(config.QUEUE_CSV).columns  # analysts never see the label


def test_champion_was_chosen_on_cross_validation(metrics):
    best_cv = max(metrics["models"], key=lambda name: metrics["models"][name]["cv_pr_auc_mean"])
    assert metrics["champion"]["name"] == best_cv


def test_scoring_api_flags_a_queue_transaction_and_passes_a_quiet_one():
    sample = pd.read_parquet(config.DAY2_SAMPLE)
    scores = pd.read_parquet(config.DAY2_EVAL).set_index("txn_id")["score"]
    ranked = sample.assign(score=sample["txn_id"].map(scores)).sort_values("score")
    payload = {"transactions": [row[["txn_id", *config.RAW_FEATURES]].to_dict() for _, row in ranked.iloc[[0, -1]].iterrows()]}

    response = TestClient(app).post("/score", json=payload)
    assert response.status_code == 200
    quiet, risky = response.json()
    assert quiet["alert"] is False and "action" not in quiet
    assert risky["alert"] is True and risky["status"] == "PENDING_APPROVAL" and len(risky["reasons"]) == 3
