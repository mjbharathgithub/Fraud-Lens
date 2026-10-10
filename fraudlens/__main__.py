"""Command line entry point.

python -m fraudlens download        fetch the public dataset
python -m fraudlens train           compare models, pick the champion, write metrics
python -m fraudlens queue           build the top-200 alert queue with drafted notes
python -m fraudlens queue --llm 20  ...and let the local LLM write the top 20 notes
python -m fraudlens figures         render the report figures
python -m fraudlens all             everything above, in order
"""

import argparse
import json

from . import agent, config, data, store, triage


def build_queue(llm_top: int = 0) -> None:
    metrics = json.loads(config.METRICS_JSON.read_text())
    bundle = triage.load_bundle()
    df, _ = data.load_clean()
    _, day2 = data.split_by_day(df)

    alerts = triage.build_queue(bundle, day2)
    cache = json.loads(config.LLM_NOTES_JSON.read_text(encoding="utf-8")) if config.LLM_NOTES_JSON.exists() else {}
    for alert in alerts:
        note = agent.draft(alert, use_llm=alert["rank"] <= llm_top, cache=cache)
        alert["review_note"], alert["note_source"] = note["note"], note["source"]
        if alert["rank"] <= llm_top:
            print(f"[queue] note {alert['rank']:>3}/{llm_top}: {note['source']}", flush=True)

    frame = triage.queue_frame(alerts)
    frame.insert(frame.columns.get_loc("status"), "review_note", [a["review_note"] for a in alerts])
    frame.to_csv(config.QUEUE_CSV, index=False, encoding="utf-8-sig")

    # Same queue with the answer revealed. Analysts never see this column; judges should.
    truth = day2.set_index("txn_id")[config.TARGET]
    frame.assign(actual_fraud=frame["txn_id"].map(truth)).to_csv(config.QUEUE_EVALUATED_CSV, index=False, encoding="utf-8-sig")

    statement = triage.queue_statement(metrics["champion"]["day2"], metrics["champion"]["day2_ci"])
    config.QUEUE_STATEMENT_TXT.write_text(statement + "\n", encoding="utf-8")
    (config.OUTPUTS / "alert_queue_top200.json").write_text(json.dumps(alerts, indent=1), encoding="utf-8")

    config.LLM_NOTES_JSON.write_text(json.dumps(cache, indent=1), encoding="utf-8")

    config.CASES_DB.unlink(missing_ok=True)
    store.seed(alerts)
    print(f"[queue] {len(alerts)} alerts written to {config.QUEUE_CSV.name}")
    print(f"[queue] {statement}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="fraudlens", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["download", "train", "queue", "figures", "all"])
    parser.add_argument("--llm", type=int, default=0, metavar="N", help="draft the top N review notes with the local LLM")
    args = parser.parse_args()

    if args.command in ("download", "all"):
        print(f"[data] {data.download()}")
    if args.command in ("train", "all"):
        from . import train  # imported here so the app and the queue never need MLflow

        metrics = train.run()
        print(json.dumps(metrics["champion"], indent=2))
    if args.command in ("queue", "all"):
        build_queue(args.llm)
    if args.command in ("figures", "all"):
        from . import figures

        figures.render_all()


if __name__ == "__main__":
    main()
