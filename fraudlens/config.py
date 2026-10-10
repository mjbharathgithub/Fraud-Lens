"""Central configuration: paths, constants and the client's stated capacity."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# --- data -------------------------------------------------------------------
RAW_CSV = ROOT / "data" / "raw" / "creditcard.csv"
DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
DATA_SHA256 = "76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89"
EXPECTED_ROWS = 284_807

V_COLS = [f"V{i}" for i in range(1, 29)]
RAW_FEATURES = ["Time", *V_COLS, "Amount"]
TARGET = "Class"
SECONDS_PER_DAY = 86_400

# --- artefacts committed to the repo so the app runs without retraining -------
DEMO_DIR = ROOT / "data" / "demo"
DAY2_EVAL = DEMO_DIR / "day2_eval.parquet"  # txn_id, score, Class, Amount for every Day-2 transaction
DAY2_SAMPLE = DEMO_DIR / "day2_sample.parquet"  # raw features for a Day-2 sample (live scoring)
MODEL_BUNDLE = ROOT / "models" / "champion.joblib"

OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
METRICS_JSON = OUTPUTS / "metrics.json"
RUN_COMPARISON_CSV = OUTPUTS / "run_comparison.csv"
CURVES_CSV = OUTPUTS / "curves.csv"
QUEUE_CSV = OUTPUTS / "alert_queue_top200.csv"
QUEUE_EVALUATED_CSV = OUTPUTS / "alert_queue_top200_evaluated.csv"
QUEUE_STATEMENT_TXT = OUTPUTS / "queue_statement.txt"
LLM_NOTES_JSON = OUTPUTS / "llm_notes.json"
CASES_DB = OUTPUTS / "cases.db"
PROMPTS = ROOT / "prompts"

# --- experiment tracking ----------------------------------------------------
MLFLOW_TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
MLFLOW_ARTIFACTS = ROOT / "mlartifacts"
MLFLOW_EXPERIMENT = "fraudlens"

# --- modelling --------------------------------------------------------------
SEED = 42
CV_FOLDS = 5
DAILY_CAPACITY = 200  # the client's review team handles 200 alerts per day

# Risk bands decide how strong the drafted action is. They are policy settings owned by the
# fraud-operations lead, not tuned for a metric: holding a payment and calling the customer
# should need near-certainty. How often each band was right is reported in metrics.json.
RISK_BANDS = {"CRITICAL": 0.99, "HIGH": 0.90}

# --- local LLM (optional; the template note always works without it) ----------
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "phi3.5"
