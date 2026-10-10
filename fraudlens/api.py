"""Scoring API: how FraudLens plugs into a payment flow.

    uvicorn fraudlens.api:app --port 8000
    POST /score  {"transactions": [{"Time": 100000, "V1": 0.1, ..., "V28": 0.0, "Amount": 12.5}]}

In a live flow there is no end-of-day ranking, so a transaction becomes an
alert when its score reaches the cutoff of the last completed day's queue.
"""

from functools import lru_cache

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field, create_model

from . import __version__, config, triage
from .explain import top_reasons

app = FastAPI(title="FraudLens scoring API", version=__version__)

Transaction = create_model(
    "Transaction",
    txn_id=(int | None, None),
    **{name: (float, ...) for name in config.RAW_FEATURES},
)


class ScoreRequest(BaseModel):
    transactions: list[Transaction] = Field(min_length=1, max_length=1000)


@lru_cache
def bundle() -> dict:
    return triage.load_bundle()


@app.get("/health")
def health() -> dict:
    model = bundle()
    return {"status": "ok", "model": model["name"], "built": model["created"], "alert_cutoff": model["queue_cutoff_score"]}


@app.post("/score")
def score(request: ScoreRequest) -> list[dict]:
    model = bundle()
    frame = pd.DataFrame([t.model_dump() for t in request.transactions])
    frame["txn_id"] = frame["txn_id"].fillna(pd.Series(range(len(frame)))).astype(int)
    scores = triage.score(model, frame)
    reasons = top_reasons(model, frame[config.RAW_FEATURES])

    results = []
    for (_, row), value, why in zip(frame.iterrows(), scores, reasons, strict=True):
        result = {"txn_id": int(row["txn_id"]), "risk_score": round(float(value), 4), "alert": bool(value >= model["queue_cutoff_score"])}
        if result["alert"]:
            alert = triage.build_alert(row, value, rank=0, reasons=why, bundle=model)
            result |= {
                "risk_band": alert["risk_band"],
                "reasons": [r["text"] for r in why],
                "action_code": alert["action_code"],
                "action": alert["action"],
                "status": alert["status"],
            }
        results.append(result)
    return results
