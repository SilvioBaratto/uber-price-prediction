"""Repository adapters: load/build the project's domain tables (infrastructure layer).

Owns the CSV column contracts and the builders that materialise the static reference tables:
the Madrid location table (read from the committed street snapshot) and the ride-tier catalog
(projected from :data:`uber.domain.config.TIERS`). The ``drivers.csv`` / ``rides.csv`` tables
are *generated* (seeded simulation), so they are built in :mod:`uber.datagen`, not here — this
module only owns their column contract (:data:`DRIVER_COLUMNS`) and the reads/writes.
"""

from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from uber.domain import config
from uber.infrastructure import paths

# --- Column contracts (CSV schemas) ----------------------------------------
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
# itself is built by the seeded ``datagen.simulation.build_drivers`` (generative, so it lives
# there, not here) — this adapter only owns the column contract and the write.
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
