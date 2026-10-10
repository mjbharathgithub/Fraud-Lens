"""Case store: every drafted action waits here for a human decision, and every decision is logged."""

import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from . import config
from .triage import STATUS_PENDING

DECISIONS = {
    "approve": "APPROVED",
    "reject": "REJECTED_FALSE_POSITIVE",
    "escalate": "ESCALATED",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    txn_id      INTEGER PRIMARY KEY,
    rank        INTEGER NOT NULL,
    action_code TEXT NOT NULL,
    note        TEXT NOT NULL,
    note_source TEXT NOT NULL,
    status      TEXT NOT NULL,
    decided_by  TEXT,
    decided_at  TEXT,
    comment     TEXT
);
CREATE TABLE IF NOT EXISTS audit_log (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    at      TEXT NOT NULL,
    txn_id  INTEGER NOT NULL,
    event   TEXT NOT NULL,
    actor   TEXT NOT NULL,
    detail  TEXT
);
"""


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


def connect(path: Path = config.CASES_DB) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    return conn


def seed(alerts: list[dict], path: Path = config.CASES_DB) -> None:
    """Open a pending case for every alert that does not have one yet."""
    with closing(connect(path)) as conn, conn:
        for alert in alerts:
            inserted = conn.execute(
                "INSERT OR IGNORE INTO cases (txn_id, rank, action_code, note, note_source, status) VALUES (?, ?, ?, ?, ?, ?)",
                (alert["txn_id"], alert["rank"], alert["action_code"], alert["review_note"], alert["note_source"], STATUS_PENDING),
            ).rowcount
            if inserted:
                conn.execute(
                    "INSERT INTO audit_log (at, txn_id, event, actor, detail) VALUES (?, ?, ?, ?, ?)",
                    (_now(), alert["txn_id"], "DRAFTED", "fraudlens-agent", f"{alert['action_code']} via {alert['note_source']}"),
                )


def update_note(txn_id: int, note: str, source: str, path: Path = config.CASES_DB) -> None:
    """Replace a pending case's draft note (for example with an LLM rewrite)."""
    with closing(connect(path)) as conn, conn:
        conn.execute(
            "UPDATE cases SET note = ?, note_source = ? WHERE txn_id = ? AND status = ?",
            (note, source, txn_id, STATUS_PENDING),
        )
        conn.execute(
            "INSERT INTO audit_log (at, txn_id, event, actor, detail) VALUES (?, ?, ?, ?, ?)",
            (_now(), txn_id, "REDRAFTED", "fraudlens-agent", source),
        )


def decide(txn_id: int, decision: str, actor: str, comment: str = "", path: Path = config.CASES_DB) -> str:
    """Record a human decision on a pending case and return the new status."""
    status = DECISIONS[decision]
    with closing(connect(path)) as conn, conn:
        conn.execute(
            "UPDATE cases SET status = ?, decided_by = ?, decided_at = ?, comment = ? WHERE txn_id = ?",
            (status, actor, _now(), comment, txn_id),
        )
        conn.execute(
            "INSERT INTO audit_log (at, txn_id, event, actor, detail) VALUES (?, ?, ?, ?, ?)",
            (_now(), txn_id, status, actor, comment),
        )
    return status


def cases(path: Path = config.CASES_DB) -> pd.DataFrame:
    with closing(connect(path)) as conn:
        return pd.read_sql_query("SELECT * FROM cases ORDER BY rank", conn)


def audit_log(path: Path = config.CASES_DB) -> pd.DataFrame:
    with closing(connect(path)) as conn:
        return pd.read_sql_query("SELECT at, txn_id, event, actor, detail FROM audit_log ORDER BY id DESC", conn)
