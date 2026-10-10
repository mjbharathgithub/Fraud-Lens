"""The triage brain: score a day of transactions and build the analysts' alert queue."""

import joblib
import numpy as np
import pandas as pd

from . import config
from .explain import CARD_TEST_MAX_AMOUNT, top_reasons
from .features import hour_of_day

STATUS_PENDING = "PENDING_APPROVAL"

# Rule-based action policy. Nothing here executes: every action is a draft
# that an analyst must approve.
ACTION_POLICY = {
    "CRITICAL": (
        "HOLD_AND_CALL",
        "Hold the transaction, place a temporary block on the card and call the cardholder to verify.",
    ),
    "HIGH": (
        "STEP_UP_VERIFY",
        "Send a step-up verification (OTP or in-app confirmation); block the card if it fails or is not answered within 30 minutes.",
    ),
    "ELEVATED": (
        "ANALYST_REVIEW",
        "Review the card's recent activity within 4 hours; no customer contact unless something else looks wrong.",
    ),
}
CARD_TEST_ADVICE = " Micro-amount: likely card testing, so watch this card for larger follow-on charges."


def load_bundle(path=config.MODEL_BUNDLE) -> dict:
    return joblib.load(path)


def score(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    """Fraud risk score between 0 and 1 for each transaction."""
    return bundle["pipeline"].predict_proba(X[config.RAW_FEATURES])[:, 1]


def risk_band(score_value: float, thresholds: dict) -> str:
    if score_value >= thresholds["CRITICAL"]:
        return "CRITICAL"
    if score_value >= thresholds["HIGH"]:
        return "HIGH"
    return "ELEVATED"


def recommend(band: str, amount: float) -> tuple[str, str]:
    code, text = ACTION_POLICY[band]
    if amount <= CARD_TEST_MAX_AMOUNT and band != "ELEVATED":
        text += CARD_TEST_ADVICE
    return code, text


def clock(time_seconds: float) -> str:
    hour = float(hour_of_day(time_seconds))
    return f"{int(hour):02d}:{int(hour % 1 * 60):02d}"


def analyst_line(alert: dict) -> str:
    """The one-line analyst note format:

    #RANK | BAND SCORE | TXN id | EUR amount at Dd HH:MM | WHY r1; r2; r3 | DO ACTION | STATUS
    """
    why = "; ".join(r["short"] for r in alert["reasons"])
    return (
        f"#{alert['rank']:03d} | {alert['risk_band']} {alert['risk_score']:.3f} | TXN {alert['txn_id']} | "
        f"EUR {alert['amount_eur']:.2f} at D{alert['day']} {alert['time']} | WHY {why} | "
        f"DO {alert['action_code']} | {alert['status']}"
    )


def build_alert(row: pd.Series, score_value: float, rank: int, reasons: list[dict], bundle: dict) -> dict:
    """Everything an analyst needs for one transaction, with the action still pending."""
    band = risk_band(score_value, bundle["band_thresholds"])
    code, action = recommend(band, float(row["Amount"]))
    alert = {
        "rank": int(rank),
        "txn_id": int(row["txn_id"]),
        "day": int(row["Time"] // config.SECONDS_PER_DAY) + 1,
        "time": clock(row["Time"]),
        "amount_eur": round(float(row["Amount"]), 2),
        "risk_score": round(float(score_value), 4),
        "risk_band": band,
        "reasons": reasons,
        "action_code": code,
        "action": action,
        "status": STATUS_PENDING,
    }
    alert["analyst_note"] = analyst_line(alert)
    return alert


def build_queue(bundle: dict, day: pd.DataFrame, k: int = config.DAILY_CAPACITY) -> list[dict]:
    """Score every transaction of the day and return the k riskiest as alerts, ranked."""
    scores = score(bundle, day)
    top = np.argsort(-scores, kind="stable")[:k]
    rows = day.iloc[top]
    reasons = top_reasons(bundle, rows[config.RAW_FEATURES])
    return [
        build_alert(row, scores[pos], rank, why, bundle)
        for rank, (pos, (_, row), why) in enumerate(zip(top, rows.iterrows(), reasons, strict=True), start=1)
    ]


def queue_frame(alerts: list[dict]) -> pd.DataFrame:
    """Flat table for the CSV artefact: one row per alert, reasons spread over three columns."""
    records = []
    for a in alerts:
        flat = {key: value for key, value in a.items() if key != "reasons"}
        for i, reason in enumerate(a["reasons"], start=1):
            flat[f"reason_{i}"] = reason["text"]
        records.append(flat)
    columns = [
        "rank", "txn_id", "day", "time", "amount_eur", "risk_score", "risk_band",
        "reason_1", "reason_2", "reason_3", "action_code", "action", "analyst_note", "status",
    ]  # fmt: skip
    return pd.DataFrame(records)[columns]


def queue_statement(queue: dict, ci: dict) -> str:
    """The precision/recall statement at the capacity cutoff, in plain words."""
    k, hit, total = queue["k"], queue["frauds_in_queue"], queue["frauds_total"]
    p_lo, p_hi = (round(100 * v) for v in ci["precision_ci"])
    r_lo, r_hi = (round(100 * v) for v in ci["recall_ci"])
    return (
        f"Of {k} alerts, {hit} are real fraud (precision {queue['precision']:.0%}, 95% range {p_lo}-{p_hi}%). "
        f"The queue catches {queue['recall']:.0%} of the day's {total} frauds (95% range {r_lo}-{r_hi}%) "
        f"and {queue['value_recall']:.0%} of the fraud value "
        f"(EUR {queue['fraud_value_caught']:,.0f} of EUR {queue['fraud_value_total']:,.0f})."
    )
