"""Small synthetic dataset with the real schema, so tests run in seconds without the 150 MB file."""

import numpy as np
import pandas as pd
import pytest

from fraudlens import config


@pytest.fixture(scope="session")
def transactions() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 6000
    df = pd.DataFrame(rng.normal(size=(n, 28)), columns=config.V_COLS)
    df.insert(0, "Time", np.sort(rng.uniform(0, 2 * config.SECONDS_PER_DAY - 1, n)).round())
    df["Amount"] = rng.lognormal(3, 1.5, n).round(2)
    fraud = rng.random(n) < 0.03
    df.loc[fraud, "V14"] -= 6  # frauds sit far out on one feature, like the real data
    df[config.TARGET] = fraud.astype(int)
    df.insert(0, "txn_id", range(n))
    return df
