"""The single I/O point: build DataFrames and read/write CSVs via pandas.

Imported as ``uber.io`` (absolute imports) — it does not shadow the stdlib ``io``.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from uber.domain import config
from uber.infrastructure import paths

LOCATION_COLUMNS = ["location_id", "street", "neighborhood", "district", "lat", "lon"]
TIER_COLUMNS = [
    "tier",
    "display_name",
    "capacity",
    "base_fare",
    "per_km",
    "per_min",
    "booking_fee",
    "min_fare",
]
# drivers.csv schema (SPEC §2.1b, Rev 2). Matches ``entities.Driver`` field order; the table
# itself is built by the seeded ``simulation.build_drivers`` (generative, so it lives there,
# not here) — io only owns the column contract and the write.
DRIVER_COLUMNS = [
    "driver_id",
    "home_district",
    "activity_class",
    "tenure_start",
    "tenure_end",
    "active_days",
    "n_rides",
    "driver_rating",
]


def build_locations_df() -> pd.DataFrame:
    """Load the committed street-level Madrid location table (one row per street).

    The table is a frozen snapshot built offline by ``scripts/extract_locations.py`` from the
    official callejero; reading it keeps generation deterministic and network-free.
    """
    df = pd.read_csv(paths.STREETS_SOURCE, encoding="utf-8")
    missing = set(LOCATION_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{paths.STREETS_SOURCE} is missing columns: {sorted(missing)}")
    return pd.DataFrame(df, columns=LOCATION_COLUMNS)


def build_tiers_df() -> pd.DataFrame:
    """Build the ride-tier catalog (ride_tiers.csv) from ``config.TIERS``.

    One row per tier, in the increasing-price order declared in config. Dataclass field
    order matches ``TIER_COLUMNS``.
    """
    rows = [asdict(tier) for tier in config.TIERS]
    return pd.DataFrame(rows, columns=TIER_COLUMNS)


def load_ncr() -> dict[str, np.ndarray]:
    """Load the NCR per-ride distractor columns as null-free empirical supports.

    Reads ``config.NCR_PATH`` (the Kaggle "NCR ride bookings" dump), parsing its literal
    ``null`` tokens as missing values, and returns ``{ride_column: values}`` for each per-ride
    distractor in ``config.NCR_DISTRACTOR_COLUMNS``. Each column's NaNs are dropped
    *independently* (R3) so ``sampling.sample_empirical`` can never draw a null; columns keep
    their full non-null support (they are sampled one at a time, so ragged lengths are fine).
    """
    src_columns = list(config.NCR_DISTRACTOR_COLUMNS.values())
    df = pd.read_csv(paths.NCR_PATH, na_values=["null"], usecols=src_columns)
    return {
        ride_col: df[src_col].dropna().to_numpy()
        for ride_col, src_col in config.NCR_DISTRACTOR_COLUMNS.items()
    }


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    """Write ``df`` to ``path`` as UTF-8 CSV without the index. Returns ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    return path
