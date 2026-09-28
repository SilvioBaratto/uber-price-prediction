"""NCR source adapter: the per-ride distractor empirical supports (infrastructure layer).

Reads the committed Kaggle "NCR ride bookings" dump and exposes each mapped distractor column
as a null-free empirical support that :func:`uber.datagen.sampling.sample_empirical` draws
from. Isolated from the rest of the I/O so the one upstream dataset has a single access point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from uber.domain import config
from uber.infrastructure import paths


def load_ncr() -> dict[str, np.ndarray]:
    """Load the NCR per-ride distractor columns as null-free empirical supports.

    Reads ``paths.NCR_PATH`` (the Kaggle "NCR ride bookings" dump), parsing its literal
    ``null`` tokens as missing values, and returns ``{ride_column: values}`` for each per-ride
    distractor in ``config.NCR_DISTRACTOR_COLUMNS``. Each column's NaNs are dropped
    *independently* so ``sampling.sample_empirical`` can never draw a null; columns keep
    their full non-null support (they are sampled one at a time, so ragged lengths are fine).
    """
    src_columns = list(config.NCR_DISTRACTOR_COLUMNS.values())
    df = pd.read_csv(paths.NCR_PATH, na_values=["null"], usecols=src_columns)
    return {
        ride_col: df[src_col].dropna().to_numpy()
        for ride_col, src_col in config.NCR_DISTRACTOR_COLUMNS.items()
    }
