"""The alert queue contract and the agent's guardrails."""

import re

import numpy as np
import pandas as pd
import pytest

from fraudlens import agent, config, data, store, triage
from fraudlens.models import make_candidates

LINE_FORMAT = re.compile(
    r"^#\d{3} \| (CRITICAL|HIGH|ELEVATED) \d\.\d{3} \| TXN \d+ \| EUR \d+\.\d{2} at D[12] \d{2}:\d{2} \| WHY .+ \| DO [A-Z_]+ \| PENDING_APPROVAL$"
)


@pytest.fixture(scope="module")
def bundle_and_day(transactions):
    cleaned, _ = data.clean(transactions)
    train, test = data.split_by_day(cleaned)
    pipe = make_candidates(pos_weight=30.0)["xgb_plain"].set_params(clf__n_estimators=30)
    pipe.fit(train[config.RAW_FEATURES], train[config.TARGET])
    legit = train[train[config.TARGET] == 0]
    bundle = {
        "pipeline": pipe,
        "band_thresholds": config.RISK_BANDS,
        "ref_quantiles": legit[[*config.V_COLS, "Amount"]].quantile(np.linspace(0, 1, 1001)),
        "hour_lift": pd.Series(1.0, index=range(24)),
    }
    return bundle, test


def test_queue_is_ranked_capped_and_pending(bundle_and_day):
    bundle, day = bundle_and_day
    alerts = triage.build_queue(bundle, day, k=25)
    assert len(alerts) == 25
    assert [a["rank"] for a in alerts] == list(range(1, 26))
    scores = [a["risk_score"] for a in alerts]
    assert scores == sorted(scores, reverse=True)
    assert all(a["status"] == "PENDING_APPROVAL" for a in alerts)
    assert all(len(a["reasons"]) == 3 for a in alerts)
    assert all(LINE_FORMAT.match(a["analyst_note"]) for a in alerts), alerts[0]["analyst_note"]


def test_queue_puts_the_obvious_frauds_first(bundle_and_day):
    bundle, day = bundle_and_day
    alerts = triage.build_queue(bundle, day, k=25)
    truth = day.set_index("txn_id")[config.TARGET]
    assert truth.loc[[a["txn_id"] for a in alerts]].mean() > 0.8
    assert alerts[0]["reasons"][0]["feature"] == "V14"  # the feature the synthetic frauds differ on


def test_bands_follow_the_score():
    thresholds = {"CRITICAL": 0.99, "HIGH": 0.90}
    assert triage.risk_band(0.995, thresholds) == "CRITICAL"
    assert triage.risk_band(0.95, thresholds) == "HIGH"
    assert triage.risk_band(0.10, thresholds) == "ELEVATED"
    assert "card testing" in triage.recommend("CRITICAL", 1.00)[1]
    assert "card testing" not in triage.recommend("CRITICAL", 250.0)[1]


@pytest.fixture()
def alert(bundle_and_day):
    bundle, day = bundle_and_day
    return triage.build_queue(bundle, day, k=5)[0]


def test_template_note_passes_its_own_guardrail(alert):
    evidence = agent.evidence_for(alert, queue_size=5)
    assert agent.check_grounding(agent.template_note(evidence), evidence) is None


def test_guardrail_rejects_invented_numbers_and_overclaims(alert):
    evidence = agent.evidence_for(alert, queue_size=5)
    txn = evidence["transaction_id"]
    assert "number not in evidence" in agent.check_grounding(f"Transaction {txn} matches 37 earlier fraud cases.", evidence)
    assert "overclaims" in agent.check_grounding(f"Transaction {txn} is confirmed fraud.", evidence)
    assert agent.check_grounding("A transaction looks risky.", evidence) == "transaction id missing"


def test_draft_falls_back_to_template_when_llm_is_unreachable(alert, monkeypatch):
    monkeypatch.setattr(agent, "llm_note", lambda evidence: None)
    result = agent.draft(alert, queue_size=5, use_llm=True)
    assert result["status"] == "PENDING_APPROVAL"
    assert result["source"].startswith("template")
    assert str(alert["txn_id"]) in result["note"]


def test_draft_rejects_an_ungrounded_llm_note(alert, monkeypatch):
    monkeypatch.setattr(agent, "llm_note", lambda evidence: f"Transaction {alert['txn_id']} was seen at 4242 merchants.")
    result = agent.draft(alert, queue_size=5, use_llm=True)
    assert "rejected" in result["source"]
    assert "4242" not in result["note"]


def test_case_store_records_decisions(alert, tmp_path):
    db = tmp_path / "cases.db"
    alert = {**alert, "review_note": "draft", "note_source": "template"}
    store.seed([alert], db)
    store.seed([alert], db)  # seeding twice must not duplicate the case
    assert len(store.cases(db)) == 1
    assert store.cases(db).loc[0, "status"] == "PENDING_APPROVAL"

    assert store.decide(alert["txn_id"], "approve", "analyst-1", "verified by phone", db) == "APPROVED"
    case = store.cases(db).loc[0]
    assert case["status"] == "APPROVED" and case["decided_by"] == "analyst-1"
    assert list(store.audit_log(db)["event"]) == ["APPROVED", "DRAFTED"]
