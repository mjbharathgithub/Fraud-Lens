"""The fraud-review agent.

For each alert it (1) gathers the evidence, (2) drafts a review note, and
(3) leaves it PENDING_APPROVAL for a human. Two drafters share one evidence
object:

- a deterministic template, which always works and is the default;
- an optional local LLM (Ollama) that writes the same facts as a short
  narrative. Its draft is accepted only if every number in it appears in the
  evidence; otherwise the template is used.
"""

import hashlib
import json
import re

import requests

from . import config
from .triage import STATUS_PENDING

PROMPT_VERSION = "review_note_v1"
MAX_WORDS = 120
BANNED_PHRASES = ("confirmed fraud", "is fraudulent", "is a fraud", "definitely", "guaranteed")
NUMBER = re.compile(r"\d+(?:\.\d+)?")

ANALYST_CHECKS = {
    "micro": "look for a run of small test charges on the same card",
    "hour": "confirm whether this cardholder normally transacts at this hour",
    "behaviour": "compare merchant, device and location with the card's usual pattern",
}


def analyst_checks(alert: dict) -> list[str]:
    """What the analyst should verify first, chosen from the reasons behind the alert."""
    checks = []
    for reason in alert["reasons"]:
        if reason["feature"] == "Amount" and "micro-amount" in reason["short"]:
            checks.append(ANALYST_CHECKS["micro"])
        elif reason["feature"] == "hour":
            checks.append(ANALYST_CHECKS["hour"])
    checks.append(ANALYST_CHECKS["behaviour"])
    return checks


def evidence_for(alert: dict, queue_size: int = config.DAILY_CAPACITY) -> dict:
    """The complete set of facts a drafter is allowed to use."""
    return {
        "transaction_id": alert["txn_id"],
        "queue_rank": alert["rank"],
        "queue_size": queue_size,
        "risk_score": alert["risk_score"],
        "risk_band": alert["risk_band"],
        "amount_eur": alert["amount_eur"],
        "day": alert["day"],
        "time": alert["time"],
        "signals": [reason["text"] for reason in alert["reasons"]],
        "recommended_action": alert["action"],
        "analyst_checks": analyst_checks(alert),
    }


def template_note(evidence: dict) -> str:
    return (
        f"Transaction {evidence['transaction_id']}: EUR {evidence['amount_eur']:.2f} on day {evidence['day']} "
        f"at {evidence['time']}. Alert #{evidence['queue_rank']} of {evidence['queue_size']}, risk score "
        f"{evidence['risk_score']:.3f} ({evidence['risk_band']}). Signals: {'; '.join(evidence['signals'])}. "
        f"Recommended: {evidence['recommended_action']} Before approving: {'; '.join(evidence['analyst_checks'])}."
    )


def load_prompt(version: str = PROMPT_VERSION) -> tuple[str, str]:
    """Return (system, user_template) from the versioned prompt file."""
    text = (config.PROMPTS / f"{version}.md").read_text(encoding="utf-8")
    system, user = text.split("## System", 1)[1].split("## User", 1)
    return system.strip(), user.strip()


def llm_note(evidence: dict, model: str = config.OLLAMA_MODEL, timeout: int = 90) -> str | None:
    """Ask the local model for a narrative note. Returns None when no model is reachable."""
    system, user = load_prompt()
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user.replace("{evidence}", json.dumps(evidence))},
        ],
        # A 2048-token context is plenty for one alert and lets the model fit on a 4 GB GPU.
        "options": {"temperature": 0.2, "seed": config.SEED, "num_predict": 220, "num_ctx": 2048},
    }
    try:
        response = requests.post(f"{config.OLLAMA_HOST}/api/chat", json=payload, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException:
        return None
    return " ".join(response.json()["message"]["content"].split())


def check_grounding(note: str, evidence: dict) -> str | None:
    """Return None if the draft is acceptable, otherwise the reason it is rejected.

    The rule is deliberately simple to explain: every number in the note must
    be a number in the evidence. A model that invents or recomputes a figure
    fails it.
    """
    if len(note.split()) > MAX_WORDS:
        return "too long"
    if str(evidence["transaction_id"]) not in note:
        return "transaction id missing"
    lowered = note.lower()
    for phrase in BANNED_PHRASES:
        if phrase in lowered:
            return f"overclaims ('{phrase}')"
    allowed = [float(n) for n in NUMBER.findall(json.dumps(evidence))]
    for token in NUMBER.findall(note):
        decimals = len(token.split(".")[1]) if "." in token else 0
        if not any(round(value, decimals) == float(token) for value in allowed):
            return f"number not in evidence ({token})"
    return None


def cache_key(evidence: dict) -> str:
    """Identifies one LLM draft: same prompt, same model, same evidence gives the same key."""
    content = json.dumps([PROMPT_VERSION, config.OLLAMA_MODEL, evidence], sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def draft(alert: dict, queue_size: int = config.DAILY_CAPACITY, use_llm: bool = False, cache: dict | None = None) -> dict:
    """Draft the review note for one alert. The result is always PENDING_APPROVAL.

    `cache` maps cache_key(evidence) to an LLM draft from an earlier run. A
    cached draft is re-checked against the evidence like a fresh one.
    """
    evidence = evidence_for(alert, queue_size)
    note, source = template_note(evidence), "template"
    if use_llm:
        key = cache_key(evidence)
        candidate = cache.get(key) if cache is not None else None
        if candidate is None:
            candidate = llm_note(evidence)
            if cache is not None and candidate is not None:
                cache[key] = candidate
        if candidate is None:
            source = "template (local LLM not reachable)"
        else:
            problem = check_grounding(candidate, evidence)
            if problem is None:
                note, source = candidate, f"llm:{config.OLLAMA_MODEL}"
            else:
                source = f"template (LLM draft rejected: {problem})"
    return {
        "txn_id": alert["txn_id"],
        "note": note,
        "source": source,
        "prompt_version": PROMPT_VERSION,
        "status": STATUS_PENDING,
    }
