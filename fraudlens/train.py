"""Train, compare and select models, then write every artefact the app and docs read.

Evaluation design
-----------------
1. Model selection: 5-fold stratified cross-validation on day 1 only.
2. Final test: the winner is refitted on all of day 1 and scores day 2 once.
   The 200 highest scores are the day's alert queue, which is exactly the
   client's "200 alerts per day".
3. A random stratified 75/25 split is also reported, for comparability.
4. One deliberately wrong run (SMOTE before the split) shows how much that
   mistake inflates the score.
"""

import json
from datetime import UTC, datetime

import joblib
import mlflow
import numpy as np
import pandas as pd
import sklearn
import xgboost
from imblearn.over_sampling import SMOTE
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from xgboost import XGBClassifier

from . import config, data
from .features import TransactionFeatures, hour_of_day
from .metrics import band_profile, bootstrap_queue_ci, capacity_curve, queue_metrics, ranking_metrics
from .models import IMBALANCE_HANDLING, XGB_PARAMS, make_candidates

TRUSTED_MODEL_TYPES = [
    "fraudlens.features.TransactionFeatures",
    "xgboost.core.Booster",
    "xgboost.sklearn.XGBClassifier",
]
LOGGED_PARAMS = {*XGB_PARAMS, "scale_pos_weight", "class_weight", "C", "max_iter"}


def ranking_scores(pipe, X) -> np.ndarray:
    """Scores used to rank transactions.

    Logistic regression is ranked on its log-odds: its probabilities round to
    exactly 1.0 for the most extreme transactions, which would tie hundreds of
    them at the top of the queue and make the baseline look worse than it is.
    """
    if hasattr(pipe, "decision_function"):
        return pipe.decision_function(X)
    return pipe.predict_proba(X)[:, 1]


def cross_validate(pipe, X, y) -> tuple[np.ndarray, list[float]]:
    """Out-of-fold scores for every training row, plus PR-AUC per fold."""
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.SEED)
    if hasattr(pipe, "decision_function"):
        oof = cross_val_predict(pipe, X, y, cv=cv, method="decision_function")
    else:
        oof = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    fold_pr_auc = [float(average_precision_score(y.iloc[test], oof[test])) for _, test in cv.split(X, y)]
    return oof, fold_pr_auc


def leakage_exhibit(df: pd.DataFrame) -> dict:
    """Same data, same model, SMOTE in two different places.

    WRONG: oversample the whole dataset, then split. Synthetic frauds built
    from test-set frauds land in training, and the test set itself is half
    synthetic. RIGHT: split first, oversample the training part only, test on
    untouched real transactions.
    """
    X = TransactionFeatures().transform(df[config.RAW_FEATURES])
    y = df[config.TARGET]

    X_res, y_res = SMOTE(random_state=config.SEED).fit_resample(X, y)
    X_tr, X_te, y_tr, y_te = train_test_split(X_res, y_res, test_size=0.25, stratify=y_res, random_state=config.SEED)
    leaky = XGBClassifier(**XGB_PARAMS).fit(X_tr, y_tr)
    wrong = ranking_metrics(y_te, leaky.predict_proba(X_te)[:, 1])
    wrong["test_fraud_share"] = float(y_te.mean())

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=config.SEED)
    X_bal, y_bal = SMOTE(random_state=config.SEED).fit_resample(X_tr, y_tr)
    honest = XGBClassifier(**XGB_PARAMS).fit(X_bal, y_bal)
    right = ranking_metrics(y_te, honest.predict_proba(X_te)[:, 1])
    right["test_fraud_share"] = float(y_te.mean())

    return {"smote_before_split": wrong, "smote_after_split": right}


def _curve_points(name: str, y, scores) -> pd.DataFrame:
    """PR and ROC curve coordinates, one point per fraud reached.

    The raw curves have a point per transaction, almost all of them in the
    low-precision tail. Keeping the best point at each distinct recall level
    preserves the shape exactly where it matters and leaves about 200 points.
    """
    precision, recall, _ = precision_recall_curve(y, scores)
    pr = pd.DataFrame({"x": recall, "y": precision}).groupby("x", as_index=False)["y"].max()
    fpr, tpr, _ = roc_curve(y, scores)
    roc = pd.DataFrame({"x": fpr, "y": tpr}).groupby("y", as_index=False)["x"].min()
    roc = pd.concat([roc, pd.DataFrame({"x": [1.0], "y": [1.0]})])  # close the curve at the top-right corner
    return pd.concat([pr.assign(curve="pr"), roc.assign(curve="roc")]).assign(model=name)[["model", "curve", "x", "y"]]


def _setup_mlflow() -> None:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(config.MLFLOW_EXPERIMENT) is None:
        mlflow.create_experiment(config.MLFLOW_EXPERIMENT, artifact_location=config.MLFLOW_ARTIFACTS.as_uri())
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)


def run() -> dict:
    df, data_report = data.load_clean()
    train, test = data.split_by_day(df)
    X_train, y_train = train[config.RAW_FEATURES], train[config.TARGET]
    X_test, y_test = test[config.RAW_FEATURES], test[config.TARGET]
    k = config.DAILY_CAPACITY

    data_report["day1"] = {"rows": len(train), "fraud": int(y_train.sum()), "fraud_value": float(train.loc[y_train == 1, "Amount"].sum())}
    data_report["day2"] = {"rows": len(test), "fraud": int(y_test.sum()), "fraud_value": float(test.loc[y_test == 1, "Amount"].sum())}

    _setup_mlflow()
    batch = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    common_tags = {"batch": batch, "split": "train day 1 -> test day 2"}

    # --- baselines: what happens without a model ------------------------------
    prevalence = float(y_test.mean())
    baselines = {
        "all_legit": {
            "accuracy": 1 - prevalence,
            "frauds_in_queue": 0,
            "precision": 0.0,
            "recall": 0.0,
            "pr_auc": prevalence,
        },
        "random_review": {
            "frauds_in_queue": k * prevalence,
            "precision": prevalence,
            "recall": k * prevalence / int(y_test.sum()),
            "pr_auc": prevalence,
        },
        "largest_amount_first": {
            **ranking_metrics(y_test, test["Amount"]),
            **queue_metrics(y_test, test["Amount"], test["Amount"], k),
        },
    }
    for name, m in baselines.items():
        with mlflow.start_run(run_name=name, tags={**common_tags, "role": "baseline"}):
            mlflow.log_param("imbalance_handling", "not a model")
            mlflow.log_metrics(
                {
                    "day2_pr_auc": m["pr_auc"],
                    "day2_precision_at_200": m["precision"],
                    "day2_recall_at_200": m["recall"],
                    "day2_frauds_in_top_200": m["frauds_in_queue"],
                }
            )

    # --- candidates: cross-validate on day 1, then score day 2 ----------------
    pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    models, fitted, oof_scores, test_scores = {}, {}, {}, {}
    for name, pipe in make_candidates(pos_weight).items():
        print(f"[train] {name} ...", flush=True)
        with mlflow.start_run(run_name=name, tags={**common_tags, "role": "candidate"}):
            oof, fold_pr_auc = cross_validate(pipe, X_train, y_train)
            pipe.fit(X_train, y_train)
            scores = ranking_scores(pipe, X_test)

            result = {
                "imbalance_handling": IMBALANCE_HANDLING[name],
                "cv_pr_auc_mean": float(np.mean(fold_pr_auc)),
                "cv_pr_auc_std": float(np.std(fold_pr_auc)),
                "cv_roc_auc": ranking_metrics(y_train, oof)["roc_auc"],
                "day1_oof_queue": queue_metrics(y_train, oof, train["Amount"], k),
                "day2": {**ranking_metrics(y_test, scores), **queue_metrics(y_test, scores, test["Amount"], k)},
            }
            models[name], fitted[name], oof_scores[name], test_scores[name] = result, pipe, oof, scores

            mlflow.log_params({"imbalance_handling": IMBALANCE_HANDLING[name], "cv_folds": config.CV_FOLDS, "seed": config.SEED})
            mlflow.log_params({f"clf__{p}": v for p, v in pipe[-1].get_params().items() if p in LOGGED_PARAMS and v is not None})
            mlflow.log_metrics(
                {
                    "cv_pr_auc_mean": result["cv_pr_auc_mean"],
                    "cv_pr_auc_std": result["cv_pr_auc_std"],
                    "cv_roc_auc": result["cv_roc_auc"],
                    "day2_pr_auc": result["day2"]["pr_auc"],
                    "day2_roc_auc": result["day2"]["roc_auc"],
                    "day2_precision_at_200": result["day2"]["precision"],
                    "day2_recall_at_200": result["day2"]["recall"],
                    "day2_frauds_in_top_200": result["day2"]["frauds_in_queue"],
                    "day2_value_recall_at_200": result["day2"]["value_recall"],
                }
            )

    # --- champion: chosen on day-1 cross-validation, never on day 2 -----------
    champion = max(models, key=lambda n: models[n]["cv_pr_auc_mean"])
    print(f"[train] champion = {champion}", flush=True)
    champ_scores = test_scores[champion]
    ci = bootstrap_queue_ci(y_test, champ_scores, k)
    bands = {
        "day1_out_of_fold": band_profile(y_train, oof_scores[champion], config.RISK_BANDS, k),
        "day2": band_profile(y_test, champ_scores, config.RISK_BANDS, k),
    }
    curve = capacity_curve(y_test, champ_scores, test["Amount"], np.arange(10, 1001, 10))

    with mlflow.start_run(run_name=f"champion_{champion}", tags={**common_tags, "role": "champion"}):
        mlflow.log_params({"selected_model": champion, "selection_rule": "highest day-1 CV PR-AUC", "capacity_k": k})
        mlflow.log_metrics(
            {
                "cv_pr_auc_mean": models[champion]["cv_pr_auc_mean"],
                "day2_pr_auc": models[champion]["day2"]["pr_auc"],
                "day2_precision_at_200": models[champion]["day2"]["precision"],
                "day2_recall_at_200": models[champion]["day2"]["recall"],
                "day2_frauds_in_top_200": models[champion]["day2"]["frauds_in_queue"],
                "day2_precision_ci_low": ci["precision_ci"][0],
                "day2_precision_ci_high": ci["precision_ci"][1],
            }
        )
        mlflow.sklearn.log_model(
            fitted[champion],
            name="model",
            input_example=X_train.head(3),
            registered_model_name="fraudlens-triage",
            # MLflow saves scikit-learn models with skops, which only loads types it has been told to trust.
            skops_trusted_types=TRUSTED_MODEL_TYPES,
        )

    # --- comparability: the textbook random split ----------------------------
    print("[train] stratified benchmark ...", flush=True)
    strat_train, strat_test = data.split_stratified(df)
    benchmark = {}
    for name in ("logistic_balanced", champion):
        pipe = make_candidates(pos_weight)[name]
        pipe.fit(strat_train[config.RAW_FEATURES], strat_train[config.TARGET])
        s = ranking_scores(pipe, strat_test[config.RAW_FEATURES])
        benchmark[name] = {
            **ranking_metrics(strat_test[config.TARGET], s),
            **queue_metrics(strat_test[config.TARGET], s, strat_test["Amount"], k),
            "test_rows": len(strat_test),
        }

    # --- the trap, measured ---------------------------------------------------
    print("[train] leakage exhibit ...", flush=True)
    exhibit = leakage_exhibit(df)
    with mlflow.start_run(run_name="LEAKY_smote_before_split", tags={**common_tags, "role": "exhibit_do_not_use"}):
        mlflow.log_param("imbalance_handling", "SMOTE before the split (wrong on purpose)")
        mlflow.log_metrics(
            {"benchmark_pr_auc": exhibit["smote_before_split"]["pr_auc"], "benchmark_roc_auc": exhibit["smote_before_split"]["roc_auc"]}
        )
    with mlflow.start_run(run_name="honest_smote_after_split", tags={**common_tags, "role": "exhibit"}):
        mlflow.log_param("imbalance_handling", "SMOTE on the training part only")
        mlflow.log_metrics(
            {"benchmark_pr_auc": exhibit["smote_after_split"]["pr_auc"], "benchmark_roc_auc": exhibit["smote_after_split"]["roc_auc"]}
        )

    # --- artefacts ------------------------------------------------------------
    for folder in (config.OUTPUTS, config.FIGURES, config.DEMO_DIR, config.MODEL_BUNDLE.parent):
        folder.mkdir(parents=True, exist_ok=True)

    legit_train = train[y_train == 0]
    hourly = train.assign(hour=hour_of_day(train["Time"]).astype(int)).groupby("hour")[config.TARGET].mean()
    bundle = {
        "name": champion,
        "pipeline": fitted[champion],
        "band_thresholds": config.RISK_BANDS,
        "queue_cutoff_score": models[champion]["day2"]["cutoff_score"],
        "capacity": k,
        # Reference points for plain-language reasons: where a value sits among genuine training transactions.
        "ref_quantiles": legit_train[[*config.V_COLS, "Amount"]].quantile(np.linspace(0, 1, 1001)),
        "hour_lift": (hourly / y_train.mean()).reindex(range(24), fill_value=0.0),
        "trained_on": "day 1, duplicates removed",
        "created": batch,
        "versions": {"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__},
    }
    joblib.dump(bundle, config.MODEL_BUNDLE, compress=3)

    scored = test[["txn_id", "Time", "Amount", config.TARGET]].assign(score=champ_scores)
    scored.to_parquet(config.DAY2_EVAL, index=False)

    # Live-scoring sample: the 1,000 highest scores, every day-2 fraud, and 3,000 random others.
    rng = np.random.default_rng(config.SEED)
    top_ids = scored.nlargest(1000, "score")["txn_id"]
    fraud_ids = scored.loc[scored[config.TARGET] == 1, "txn_id"]
    random_ids = rng.choice(scored["txn_id"].to_numpy(), 3000, replace=False)
    sample_ids = pd.unique(np.concatenate([top_ids, fraud_ids, random_ids]))
    test[test["txn_id"].isin(sample_ids)][["txn_id", *config.RAW_FEATURES, config.TARGET]].to_parquet(config.DAY2_SAMPLE, index=False)

    pd.concat(
        [
            _curve_points("logistic_balanced", y_test, test_scores["logistic_balanced"]),
            _curve_points(champion, y_test, champ_scores),
        ]
    ).to_csv(config.CURVES_CSV, index=False)
    curve.to_csv(config.OUTPUTS / "capacity_curve.csv", index=False)

    runs = mlflow.search_runs(experiment_names=[config.MLFLOW_EXPERIMENT], filter_string=f"tags.batch = '{batch}'")
    keep = ["tags.mlflow.runName", "tags.role", "params.imbalance_handling"] + sorted(c for c in runs.columns if c.startswith("metrics."))
    runs[keep].rename(columns=lambda c: c.split(".", 1)[1].replace("mlflow.", "")).sort_values("day2_pr_auc", ascending=False).to_csv(
        config.RUN_COMPARISON_CSV, index=False
    )

    metrics = {
        "created": batch,
        "capacity": k,
        "data": data_report,
        "baselines": baselines,
        "models": models,
        "champion": {
            "name": champion,
            "selection_rule": "highest mean PR-AUC in 5-fold cross-validation on day 1",
            "day2": models[champion]["day2"],
            "day2_ci": ci,
            "band_thresholds": config.RISK_BANDS,
            "band_profile": bands,
        },
        "stratified_benchmark": benchmark,
        "leakage_exhibit": exhibit,
    }
    config.METRICS_JSON.write_text(json.dumps(metrics, indent=2))
    return metrics
