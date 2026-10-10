"""FraudLens analyst console.

    streamlit run streamlit_app.py

Reads only the small artefacts committed to the repo (model bundle, day-2
sample, metrics), so it runs the same on a laptop and on Streamlit Community
Cloud. Nothing here retrains a model.
"""

import json
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from fraudlens import agent, config, store, triage
from fraudlens.explain import top_reasons

st.set_page_config(page_title="FraudLens", page_icon=":mag:", layout="wide")

SURFACE, INK, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#b9b8b1"
BAND_ICON = {"CRITICAL": "🔴", "HIGH": "🟠", "ELEVATED": "🟡"}
STATUS_ICON = {"PENDING_APPROVAL": "⏳", "APPROVED": "✅", "REJECTED_FALSE_POSITIVE": "❌", "ESCALATED": "⬆️"}


# --- data ---------------------------------------------------------------------
@st.cache_resource
def load_bundle() -> dict:
    return triage.load_bundle()


@st.cache_data
def load_metrics() -> dict:
    return json.loads(config.METRICS_JSON.read_text())


@st.cache_data
def load_alerts() -> list[dict]:
    return json.loads((config.OUTPUTS / "alert_queue_top200.json").read_text(encoding="utf-8"))


@st.cache_data
def load_day2_eval() -> pd.DataFrame:
    """Score, label and amount for every day-2 transaction, highest score first."""
    return pd.read_parquet(config.DAY2_EVAL).sort_values("score", ascending=False, kind="stable").reset_index(drop=True)


@st.cache_data
def load_day2_sample() -> pd.DataFrame:
    return pd.read_parquet(config.DAY2_SAMPLE)


@st.cache_data(ttl=30)
def local_llm_available() -> bool:
    try:
        return requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=1).ok
    except requests.RequestException:
        return False


# --- small helpers --------------------------------------------------------------
def tidy(fig: go.Figure, height: int = 320, **layout) -> go.Figure:
    fig.update_layout(
        height=height,
        margin={"l": 10, "r": 10, "t": 30 if len(fig.data) > 1 else 8, "b": 10},
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font={"family": "Segoe UI, system-ui, sans-serif", "color": INK, "size": 13},
        legend={"orientation": "h", "y": 1.02, "yanchor": "bottom", "x": 0},
        hoverlabel={"bgcolor": "white"},
        **layout,
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False)
    return fig


def show(fig: go.Figure, key: str, title: str | None = None) -> None:
    if title:
        st.markdown(f"**{title}**")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=key)


def reasons_chart(reasons: list[dict]) -> go.Figure:
    """How strongly each of the top reasons pushed the score towards fraud."""
    ordered = reasons[::-1]
    fig = go.Figure(
        go.Bar(
            x=[r["contribution"] for r in ordered],
            y=[r["short"] for r in ordered],
            orientation="h",
            marker={"color": BLUE, "cornerradius": 4},
            text=[f"+{r['contribution']:.2f}" for r in ordered],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br>push towards fraud: %{x:.2f} log-odds<extra></extra>",
        )
    )
    return tidy(fig, height=60 + 38 * len(reasons), xaxis_title="Push towards fraud (log-odds)", bargap=0.45)


def show_alert_body(alert: dict, key: str) -> None:
    """Band, reasons and recommended action for one alert."""
    st.markdown(
        f"{BAND_ICON[alert['risk_band']]} **{alert['risk_band']}** · risk score **{alert['risk_score']:.3f}** · "
        f"EUR **{alert['amount_eur']:,.2f}** · day {alert['day']} at {alert['time']}"
    )
    st.caption("Why it was flagged")
    show(reasons_chart(alert["reasons"]), key=f"{key}_reasons")
    for reason in alert["reasons"]:
        st.markdown(f"- {reason['text']}")
    st.caption("Recommended action (drafted, not executed)")
    st.info(f"**{alert['action_code']}**: {alert['action']}")


# --- tab 1: the queue -----------------------------------------------------------
def tab_queue(metrics: dict, analyst: str, reveal: bool) -> None:
    alerts = load_alerts()
    by_txn = {a["txn_id"]: a for a in alerts}
    store.seed(alerts)  # idempotent: opens a pending case for any alert that has none
    cases = store.cases().set_index("txn_id")
    day2 = metrics["champion"]["day2"]
    ci = metrics["champion"]["day2_ci"]

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Alerts in today's queue", day2["k"])
    k2.metric(
        "Real fraud among them",
        f"{day2['frauds_in_queue']} ({day2['precision']:.0%})",
        help=f"95% range for precision: {ci['precision_ci'][0]:.0%} to {ci['precision_ci'][1]:.0%}",
    )
    k3.metric(
        "Share of the day's fraud caught",
        f"{day2['recall']:.0%}",
        help=f"{day2['frauds_in_queue']} of {day2['frauds_total']} frauds. 95% range: {ci['recall_ci'][0]:.0%} to {ci['recall_ci'][1]:.0%}",
    )
    k4.metric("Fraud value caught", f"EUR {day2['fraud_value_caught']:,.0f}", help=f"of EUR {day2['fraud_value_total']:,.0f} on the day")
    k5.metric("Awaiting approval", int((cases["status"] == triage.STATUS_PENDING).sum()))
    st.info(triage.queue_statement(day2, ci))

    table = pd.DataFrame(
        [
            {
                "rank": a["rank"],
                "txn_id": a["txn_id"],
                "time": a["time"],
                "amount_eur": a["amount_eur"],
                "risk_score": a["risk_score"],
                "band": f"{BAND_ICON[a['risk_band']]} {a['risk_band']}",
                "status": f"{STATUS_ICON[cases.loc[a['txn_id'], 'status']]} {cases.loc[a['txn_id'], 'status']}",
            }
            for a in alerts
        ]
    )
    if reveal:
        truth = load_day2_eval().set_index("txn_id")[config.TARGET]
        table["actual"] = table["txn_id"].map(truth).map({1: "FRAUD", 0: "genuine"})

    f1, f2, _ = st.columns([1, 1, 2])
    bands = f1.multiselect("Risk band", list(BAND_ICON), placeholder="All bands")
    statuses = f2.multiselect("Status", list(STATUS_ICON), placeholder="All statuses")
    if bands:
        table = table[table["band"].str.split().str[1].isin(bands)]
    if statuses:
        table = table[table["status"].str.split().str[1].isin(statuses)]

    left, right = st.columns([1.25, 1], gap="large")
    with left:
        st.subheader("Today's alert queue")
        event = st.dataframe(
            table,
            hide_index=True,
            width="stretch",
            height=560,
            on_select="rerun",
            selection_mode="single-row",
            key="queue_table",
            column_config={
                "rank": st.column_config.NumberColumn("#", width="small"),
                "txn_id": st.column_config.NumberColumn("Transaction", format="%d"),
                "time": "Time",
                "amount_eur": st.column_config.NumberColumn("Amount (EUR)", format="%.2f"),
                "risk_score": st.column_config.ProgressColumn("Risk score", min_value=0, max_value=1, format="%.3f"),
                "band": "Band",
                "status": "Status",
                "actual": "Actual (hidden from analysts)",
            },
        )
        st.caption("Select a row to open the case. Actions are drafts until an analyst approves them.")
        st.download_button("Download the queue (CSV)", config.QUEUE_CSV.read_bytes(), file_name=config.QUEUE_CSV.name, mime="text/csv")

    if event.selection.rows and event.selection.rows[0] < len(table):
        st.session_state["open_txn"] = int(table.iloc[event.selection.rows[0]]["txn_id"])
    txn = st.session_state.get("open_txn", alerts[0]["txn_id"])
    alert, case = by_txn[txn], cases.loc[txn]

    with right:
        st.subheader(f"Case #{alert['rank']:03d} · transaction {txn}")
        show_alert_body(alert, key="case")
        if reveal:
            actual = int(load_day2_eval().set_index("txn_id").loc[txn, config.TARGET])
            st.markdown(f"Ground truth (evaluation mode only): **{'FRAUD' if actual else 'genuine'}**")

        st.caption(f"Review note · drafted by {case['note_source']} · prompt {agent.PROMPT_VERSION}")
        st.write(case["note"])
        st.markdown(
            f"Status: {STATUS_ICON[case['status']]} **{case['status']}**"
            + (f" by {case['decided_by']} at {case['decided_at']}" if case["decided_by"] else "")
        )

        pending = case["status"] == triage.STATUS_PENDING
        comment = st.text_input("Comment for the audit log", key=f"comment_{txn}", disabled=not pending)
        b1, b2, b3 = st.columns(3)
        if b1.button("Approve action", type="primary", disabled=not pending, key=f"approve_{txn}", width="stretch"):
            store.decide(txn, "approve", analyst, comment)
            st.rerun()
        if b2.button("Reject: false positive", disabled=not pending, key=f"reject_{txn}", width="stretch"):
            store.decide(txn, "reject", analyst, comment)
            st.rerun()
        if b3.button("Escalate", disabled=not pending, key=f"escalate_{txn}", width="stretch"):
            store.decide(txn, "escalate", analyst, comment)
            st.rerun()

        if pending and local_llm_available():
            if st.button(f"Redraft the note with the local LLM ({config.OLLAMA_MODEL})", key=f"redraft_{txn}"):
                with st.spinner("Drafting from the evidence, then checking every number against it..."):
                    result = agent.draft(alert, use_llm=True)
                if result["source"].startswith("llm"):
                    store.update_note(txn, result["note"], result["source"])
                    st.rerun()
                st.warning(f"Kept the template note: {result['source']}.")
        st.caption("One-line analyst note format")
        st.code(alert["analyst_note"], language=None, wrap_lines=True)

    with st.expander("Audit log: every draft and every decision"):
        st.dataframe(store.audit_log(), hide_index=True, width="stretch")


# --- tab 2: live scoring --------------------------------------------------------
def _use_chosen_txn() -> None:
    if st.session_state["chosen_txn"] is not None:
        st.session_state["live_txn"] = int(st.session_state["chosen_txn"])


def tab_live(bundle: dict) -> None:
    sample = load_day2_sample()
    day = load_day2_eval()
    day_scores = np.sort(day["score"].to_numpy())
    capacity = bundle["capacity"]

    st.subheader("Score one transaction, live")
    st.caption(
        "Transactions come from day 2, which the model never saw in training. "
        "The score is computed now, by the same model that built the queue."
    )
    queue_ids = day.head(capacity)["txn_id"]
    c1, c2, c3 = st.columns([1, 1, 2])
    if c1.button("Pick one from today's queue", width="stretch"):
        st.session_state["live_txn"] = int(queue_ids.sample(1).iloc[0])
    if c2.button("Pick a random transaction", width="stretch"):
        st.session_state["live_txn"] = int(sample.loc[~sample["txn_id"].isin(day.head(1000)["txn_id"]), "txn_id"].sample(1).iloc[0])
    c3.selectbox(
        "...or choose a transaction id from the day-2 sample",
        sorted(sample["txn_id"]),
        index=None,
        key="chosen_txn",
        on_change=_use_chosen_txn,
        placeholder="Type an id",
    )
    if "live_txn" not in st.session_state:
        st.session_state["live_txn"] = int(queue_ids.iloc[0])

    txn = st.session_state["live_txn"]
    row = sample[sample["txn_id"] == txn].iloc[[0]].copy()
    original_amount = float(row["Amount"].iloc[0])
    default_hour = round(float(row["Time"].iloc[0] % config.SECONDS_PER_DAY) / 3600 * 4) / 4 % 24

    with st.expander("What if? Change the amount or the time and watch the score move"):
        w1, w2 = st.columns(2)
        amount = w1.number_input("Amount (EUR)", min_value=0.0, value=original_amount, step=1.0, key=f"amount_{txn}")
        hour = w2.slider("Hour of day", 0.0, 23.75, value=default_hour, step=0.25, key=f"hour_{txn}")
        st.caption("V1 to V28 are the issuer's anonymised behaviour features and stay as recorded.")
    edited = abs(amount - original_amount) > 1e-9 or hour != default_hour
    if edited:
        row["Amount"] = amount
        if hour != default_hour:
            row["Time"] = config.SECONDS_PER_DAY + hour * 3600

    started = time.perf_counter()
    score = float(triage.score(bundle, row)[0])
    reasons = top_reasons(bundle, row[config.RAW_FEATURES])[0]
    elapsed_ms = 1000 * (time.perf_counter() - started)
    rank = int(len(day_scores) - np.searchsorted(day_scores, score, side="right")) + 1
    in_queue = rank <= capacity
    alert = triage.build_alert(row.iloc[0], score, rank, reasons, bundle)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Transaction", f"{txn}" + (" (edited)" if edited else ""))
    m2.metric("Risk score", f"{score:.4f}")
    m3.metric("Rank among today's transactions", f"#{rank:,} of {len(day_scores):,}")
    m4.metric("Scored and explained in", f"{elapsed_ms:.0f} ms")

    left, right = st.columns(2, gap="large")
    with left:
        if in_queue:
            st.error(f"ALERT: enters today's queue at #{rank} of {capacity}.", icon="🚨")
            show_alert_body(alert, key="live")
        else:
            st.success(f"No alert: below today's cutoff score of {bundle['queue_cutoff_score']:.3f}. Nothing is drafted.", icon="✅")
            st.caption("What moved the score most")
            show(reasons_chart(reasons), key="live_quiet_reasons")
    with right:
        if in_queue:
            note = agent.draft(alert, queue_size=capacity)
            draft_key = (txn, amount, hour)
            if local_llm_available() and st.button(f"Redraft with the local LLM ({config.OLLAMA_MODEL})"):
                with st.spinner("Drafting from the evidence, then checking every number against it..."):
                    st.session_state["live_llm_note"] = (draft_key, agent.draft(alert, queue_size=capacity, use_llm=True))
            if st.session_state.get("live_llm_note", (None, None))[0] == draft_key:
                note = st.session_state["live_llm_note"][1]
            st.caption(f"Review note · drafted by {note['source']} · {STATUS_ICON[note['status']]} {note['status']}")
            st.write(note["note"])
            st.caption("One-line analyst note format")
            st.code(alert["analyst_note"], language=None, wrap_lines=True)
        with st.expander("The raw transaction as the model receives it"):
            st.dataframe(row[config.RAW_FEATURES].T.rename(columns=lambda _: "value"), width="stretch", height=300)

    st.divider()
    st.subheader("Score a whole file")
    st.caption(
        f"Upload a CSV with the columns {', '.join(config.RAW_FEATURES[:3])}, ..., V28, Amount. The {capacity} riskiest rows come back ranked."
    )
    example = sample.drop(columns=[config.TARGET]).sample(500, random_state=config.SEED)
    st.download_button(
        "Download an example file (500 day-2 transactions)",
        example.to_csv(index=False),
        file_name="example_transactions.csv",
        mime="text/csv",
    )
    upload = st.file_uploader("Transactions CSV", type="csv")
    if upload is not None:
        batch = pd.read_csv(upload)
        missing = [c for c in config.RAW_FEATURES if c not in batch.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        else:
            if "txn_id" not in batch.columns:
                batch.insert(0, "txn_id", range(len(batch)))
            ranked = pd.DataFrame(triage.build_queue(bundle, batch, k=min(capacity, len(batch))))
            ranked = ranked[["rank", "txn_id", "time", "amount_eur", "risk_score", "risk_band", "action_code", "analyst_note", "status"]]
            st.dataframe(ranked, hide_index=True, width="stretch")
            st.download_button("Download this queue (CSV)", ranked.to_csv(index=False), file_name="alert_queue.csv", mime="text/csv")


# --- tab 3: model evidence ------------------------------------------------------
def tab_evidence(metrics: dict) -> None:
    champion = metrics["champion"]["name"]
    day2 = metrics["champion"]["day2"]
    never = metrics["baselines"]["all_legit"]

    st.subheader("Why accuracy is the wrong yardstick")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Accuracy of answering 'never fraud'", f"{never['accuracy']:.2%}")
    a2.metric("Frauds that 'model' catches", f"0 of {day2['frauds_total']}")
    a3.metric("Champion ROC-AUC (flattering)", f"{day2['roc_auc']:.3f}")
    a4.metric("Champion PR-AUC (honest)", f"{day2['pr_auc']:.3f}")
    st.caption(
        "With 99.86% genuine transactions, accuracy and ROC-AUC are dominated by the easy negatives. "
        "PR-AUC and precision/recall at 200 alerts only move when the frauds themselves are ranked better."
    )

    st.subheader("Model comparison")
    random_hits = metrics["baselines"]["random_review"]["frauds_in_queue"]
    largest_hits = metrics["baselines"]["largest_amount_first"]["frauds_in_queue"]
    st.markdown(
        f"Without a model, 200 reviews chosen at random catch **{random_hits:.1f}** frauds on day 2, "
        f"and reviewing the 200 largest amounts catches **{largest_hits}**. The models:"
    )
    rows = []
    for name, m in metrics["models"].items():
        rows.append(
            {
                "model": name + (" ★ champion" if name == champion else ""),
                "handles imbalance by": m["imbalance_handling"],
                "CV PR-AUC (day 1)": m["cv_pr_auc_mean"],
                "± std": m["cv_pr_auc_std"],
                "PR-AUC (day 2)": m["day2"]["pr_auc"],
                "ROC-AUC (day 2)": m["day2"]["roc_auc"],
                "frauds in top 200 (day 2)": m["day2"]["frauds_in_queue"],
                "precision at 200": m["day2"]["precision"],
                "recall at 200": m["day2"]["recall"],
            }
        )
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        column_config={
            c: st.column_config.NumberColumn(format="%.3f")
            for c in ("CV PR-AUC (day 1)", "± std", "PR-AUC (day 2)", "ROC-AUC (day 2)", "precision at 200", "recall at 200")
        },
    )
    st.caption(
        f"The champion ({champion}) was chosen by cross-validation on day 1 only. Day-2 numbers are shown for every model "
        "for transparency; they were not used to choose. Every run is tracked in MLflow (`mlflow ui`)."
    )

    curves = pd.read_csv(config.CURVES_CSV)
    left, right = st.columns(2, gap="large")
    for column, kind, title, xlabel, ylabel in (
        (left, "pr", "Precision-recall: the gap is visible", "Recall (share of frauds caught)", "Precision (alerts that are fraud)"),
        (right, "roc", "ROC: both models look near-perfect", "False-positive rate", "True-positive rate"),
    ):
        fig = go.Figure()
        for name, colour in (("logistic_balanced", GREY), (champion, BLUE)):
            part = curves[(curves["curve"] == kind) & (curves["model"] == name)].sort_values(["x", "y"])
            fig.add_trace(
                go.Scatter(
                    x=part["x"],
                    y=part["y"],
                    name=name,
                    mode="lines",
                    line={"color": colour, "width": 2},
                    hovertemplate=f"{name}<br>{xlabel}: %{{x:.2f}}<br>{ylabel}: %{{y:.2f}}<extra></extra>",
                )
            )
        if kind == "pr":
            fig.add_trace(
                go.Scatter(
                    x=[day2["recall"]],
                    y=[day2["precision"]],
                    name="200 alerts a day",
                    mode="markers",
                    marker={"color": BLUE, "size": 12, "line": {"color": SURFACE, "width": 2}},
                    hovertemplate="200 alerts a day<br>recall %{x:.0%}, precision %{y:.0%}<extra></extra>",
                )
            )
        with column:
            show(
                tidy(fig, xaxis_title=xlabel, yaxis_title=ylabel, xaxis_range=[0, 1.02], yaxis_range=[0, 1.04]),
                title=title,
                key=f"curve_{kind}",
            )

    st.subheader("The resampling trap, measured")
    exhibit = metrics["leakage_exhibit"]
    e1, e2, e3 = st.columns([1, 1, 2])
    e1.metric("SMOTE before the split (wrong)", f"PR-AUC {exhibit['smote_before_split']['pr_auc']:.3f}")
    e2.metric("SMOTE after the split (right)", f"PR-AUC {exhibit['smote_after_split']['pr_auc']:.3f}")
    e3.caption(
        "Same data, same model, same random split. Oversampling first puts synthetic copies of test frauds into training "
        f"and makes the test set {exhibit['smote_before_split']['test_fraud_share']:.0%} fraud, so the score says almost nothing "
        "about real transactions. In FraudLens, SMOTE only ever runs inside the training fold of a pipeline."
    )

    st.subheader("Does the way we split matter?")
    bench = metrics["stratified_benchmark"][champion]
    split_rows = [
        {"evaluation": "Day replay: train day 1, test day 2 (headline)", "test transactions": metrics["data"]["day2"]["rows"], "PR-AUC": day2["pr_auc"], "frauds in top 200": f"{day2['frauds_in_queue']} of {day2['frauds_total']}"},
        {"evaluation": "Random stratified 75/25 split (textbook)", "test transactions": bench["test_rows"], "PR-AUC": bench["pr_auc"], "frauds in top 200": f"{bench['frauds_in_queue']} of {bench['frauds_total']}"},
    ]  # fmt: skip
    st.dataframe(
        pd.DataFrame(split_rows), hide_index=True, width="stretch", column_config={"PR-AUC": st.column_config.NumberColumn(format="%.3f")}
    )
    st.caption(
        "The random split mixes both days, so the model is tested on the same hours it trained on. The day replay is the harder, "
        "more honest test: it is what the client would see the morning after deployment. A random 25% test set is also only about "
        "half a day of transactions, so its top 200 is not a daily queue."
    )

    st.subheader("How often each risk band is right")
    profile = metrics["champion"]["band_profile"]
    thresholds = metrics["champion"]["band_thresholds"]
    band_rows = []
    for band, rule in (
        ("CRITICAL", f"score ≥ {thresholds['CRITICAL']}"),
        ("HIGH", f"score ≥ {thresholds['HIGH']}"),
        ("ELEVATED", "rest of the queue"),
    ):
        d1, d2 = profile["day1_out_of_fold"][band], profile["day2"][band]
        band_rows.append(
            {
                "band": f"{BAND_ICON[band]} {band}",
                "rule": rule,
                "day 1 alerts (out-of-fold)": d1["alerts"],
                "day 1 fraud share": d1["precision"],
                "day 2 alerts": d2["alerts"],
                "day 2 fraud share": d2["precision"],
            }
        )
    st.dataframe(
        pd.DataFrame(band_rows),
        hide_index=True,
        width="stretch",
        column_config={c: st.column_config.NumberColumn(format="percent") for c in ("day 1 fraud share", "day 2 fraud share")},
    )


# --- tab 4: business impact -----------------------------------------------------
def tab_business(metrics: dict) -> None:
    day = load_day2_eval()
    y, amounts = day[config.TARGET].to_numpy(), day["Amount"].to_numpy()
    total_frauds, total_value = int(y.sum()), float(amounts[y == 1].sum())

    st.subheader("Capacity planner")
    st.caption("Day 2 replayed at any staffing level. The cost figures are assumptions for the client to replace with their own.")
    s1, s2, s3, s4 = st.columns(4)
    k = s1.slider("Alerts reviewed per day", 20, 600, metrics["capacity"], step=10)
    review_cost = s2.number_input("Cost of one review (EUR)", 0.0, 100.0, 5.0, step=0.5)
    follow_on = s3.number_input(
        "Follow-on loss avoided per fraud caught (EUR)",
        0.0,
        5000.0,
        150.0,
        step=10.0,
        help="Blocking a compromised card stops later charges. The dataset cannot link transactions to cards, so this is an assumption.",
    )
    friction = s4.number_input("Cost of a false alarm to the customer (EUR)", 0.0, 100.0, 2.0, step=0.5)

    ks = np.arange(10, 601, 10)
    caught = np.cumsum(y)[ks - 1]
    value = np.cumsum(amounts * y)[ks - 1]
    net = value + caught * follow_on - ks * review_cost - (ks - caught) * friction
    at = int(np.flatnonzero(ks == k)[0])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Frauds caught", f"{caught[at]} of {total_frauds}", help=f"Most that {k} reviews could catch: {min(k, total_frauds)}")
    m2.metric("Alerts that are real fraud", f"{caught[at] / k:.0%}")
    m3.metric("Fraud value intercepted", f"EUR {value[at]:,.0f}", help=f"of EUR {total_value:,.0f} on the day")
    m4.metric(
        "Net value per day", f"EUR {net[at]:,.0f}", help="Fraud value intercepted + follow-on loss avoided - review cost - false-alarm cost"
    )

    order_amount = np.argsort(-amounts, kind="stable")[:k]
    largest_caught = int(y[order_amount].sum())
    how = f"How the {k} reviews are chosen"
    compare = pd.DataFrame(
        [
            {how: "FraudLens ranking", "frauds caught": int(caught[at]), "fraud value intercepted (EUR)": float(value[at])},
            {how: "Largest amounts first", "frauds caught": largest_caught, "fraud value intercepted (EUR)": float(amounts[order_amount][y[order_amount] == 1].sum())},
            {how: "At random (expected)", "frauds caught": round(k * y.mean(), 1), "fraud value intercepted (EUR)": round(k * y.mean() * total_value / total_frauds, 0)},
        ]
    )  # fmt: skip
    st.dataframe(
        compare,
        hide_index=True,
        width="stretch",
        column_config={"fraud value intercepted (EUR)": st.column_config.NumberColumn(format="%.0f")},
    )

    left, right = st.columns(2, gap="large")
    with left:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=ks,
                y=caught / total_frauds,
                name="Recall (frauds caught)",
                mode="lines",
                line={"color": BLUE, "width": 2},
                hovertemplate="%{x} alerts/day<br>recall %{y:.0%}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=ks,
                y=caught / ks,
                name="Precision (alerts that are fraud)",
                mode="lines",
                line={"color": ORANGE, "width": 2},
                hovertemplate="%{x} alerts/day<br>precision %{y:.0%}<extra></extra>",
            )
        )
        fig.add_vline(x=k, line={"color": MUTED, "width": 1})
        show(
            tidy(fig, xaxis_title="Alerts reviewed per day", yaxis_tickformat=".0%", yaxis_range=[0, 1.04]),
            title="What more reviews buy",
            key="capacity",
        )
    with right:
        best = int(np.argmax(net))
        fig = go.Figure(
            go.Scatter(
                x=ks,
                y=net,
                mode="lines",
                line={"color": BLUE, "width": 2},
                hovertemplate="%{x} alerts/day<br>net EUR %{y:,.0f}<extra></extra>",
            )
        )
        fig.add_vline(x=k, line={"color": MUTED, "width": 1})
        fig.add_trace(
            go.Scatter(
                x=[ks[best]],
                y=[net[best]],
                mode="markers+text",
                text=[f"best: {ks[best]} alerts/day"],
                textposition="bottom center",
                marker={"color": BLUE, "size": 11, "line": {"color": SURFACE, "width": 2}},
                hoverinfo="skip",
            )
        )
        show(
            tidy(
                fig,
                xaxis_title="Alerts reviewed per day",
                yaxis_title="EUR per day",
                showlegend=False,
            ),
            title="Net value per day under these assumptions",
            key="net_value",
        )
    st.caption(
        f"Under these assumptions the value peaks near {ks[best]} alerts a day: beyond that, each extra review costs more than the fraud it finds. "
        "One day of data is thin evidence for a staffing decision, so treat this as the shape of the trade-off, not a forecast."
    )


# --- page -----------------------------------------------------------------------
def main() -> None:
    bundle, metrics = load_bundle(), load_metrics()

    with st.sidebar:
        st.header("FraudLens")
        st.caption("The triage brain for card-fraud analysts")
        analyst = st.text_input("Signed in as", "analyst-1")
        reveal = st.toggle("Evaluation mode: reveal ground truth", value=False, help="Analysts never see the true label. Judges should.")
        st.divider()
        if local_llm_available():
            st.markdown(f"🟢 Local LLM available ({config.OLLAMA_MODEL})")
        else:
            st.markdown("⚪ No local LLM on this host: notes use the deterministic template and the pre-generated LLM drafts")
        if st.button("Reset demo decisions"):
            config.CASES_DB.unlink(missing_ok=True)
            st.session_state.pop("open_txn", None)
            st.rerun()
        st.divider()
        st.caption(
            f"Model: {bundle['name']}  \nTrained on: {bundle['trained_on']}  \nBuilt: {bundle['created']}  \n"
            f"scikit-learn {bundle['versions']['scikit-learn']} · xgboost {bundle['versions']['xgboost']}"
        )

    st.title("FraudLens")
    st.caption(
        f"Which {bundle['capacity']} card transactions should the fraud team review today? "
        "Trained on day 1, shown here scoring day 2, which the model has never seen."
    )
    tabs = {"queue": "Alert queue", "live": "Live scoring", "evidence": "Model evidence", "business": "Business impact"}
    opened = tabs.get(st.query_params.get("tab"), "Alert queue")  # ?tab=live opens straight on live scoring
    queue, live, evidence, business = st.tabs(list(tabs.values()), default=opened)
    with queue:
        tab_queue(metrics, analyst, reveal)
    with live:
        tab_live(bundle)
    with evidence:
        tab_evidence(metrics)
    with business:
        tab_business(metrics)


main()
