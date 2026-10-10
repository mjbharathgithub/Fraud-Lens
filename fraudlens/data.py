"""Data access: download, validate, de-duplicate and split.

Order matters for leakage: duplicates are removed first, the split happens
second, and nothing downstream is fitted before the split.
"""

import hashlib
from pathlib import Path

import pandas as pd
import requests
from sklearn.model_selection import train_test_split

from . import config


def download(dest: Path = config.RAW_CSV) -> Path:
    """Fetch the public dataset (no login needed) and verify its checksum."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with requests.get(config.DATA_URL, stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in response.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                digest.update(chunk)
    if digest.hexdigest() != config.DATA_SHA256:
        dest.unlink()
        raise ValueError("Downloaded file does not match the expected checksum.")
    return dest


def load_raw(path: Path = config.RAW_CSV) -> pd.DataFrame:
    """Read the CSV, check the schema and give every row a stable transaction id."""
    df = pd.read_csv(path)
    expected = [*config.RAW_FEATURES, config.TARGET]
    if list(df.columns) != expected:
        raise ValueError(f"Unexpected columns: {list(df.columns)}")
    if df.isna().any().any():
        raise ValueError("Dataset contains missing values; the pipeline assumes none.")
    df[config.TARGET] = df[config.TARGET].astype(int)
    df.insert(0, "txn_id", range(len(df)))
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop exact duplicate transactions and tag each row with its day (1 or 2).

    An identical row sitting in both train and test would let the model be
    graded on something it has already seen, so duplicates go before any split.
    """
    content = [*config.RAW_FEATURES, config.TARGET]
    is_dup = df.duplicated(subset=content, keep="first")
    report = {
        "rows_raw": int(len(df)),
        "fraud_raw": int(df[config.TARGET].sum()),
        "duplicates_removed": int(is_dup.sum()),
        "duplicate_frauds_removed": int(df.loc[is_dup, config.TARGET].sum()),
    }
    out = df.loc[~is_dup].copy()
    out["day"] = (out["Time"] // config.SECONDS_PER_DAY).astype(int) + 1
    report["rows_clean"] = int(len(out))
    report["fraud_clean"] = int(out[config.TARGET].sum())
    return out, report


def load_clean(path: Path = config.RAW_CSV) -> tuple[pd.DataFrame, dict]:
    return clean(load_raw(path))


def split_by_day(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on day 1, test on day 2: the model never sees the future."""
    return df[df["day"] == 1], df[df["day"] == 2]


def split_stratified(df: pd.DataFrame, test_size: float = 0.25) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The textbook random split, reported only so results are comparable with other work."""
    return train_test_split(df, test_size=test_size, stratify=df[config.TARGET], random_state=config.SEED)
