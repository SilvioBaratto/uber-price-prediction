"""Boundary tests for the infrastructure I/O layer (T1.2 split of the old ``uber.io``).

Pins the public surface of the three adapter modules the monolithic ``io`` module was split
into — ``csv_io`` (generic CSV mechanics), ``repositories`` (domain-table builders + column
contracts) and ``ncr`` (the NCR empirical distractor supports) — so the clean-architecture
layering stays enforced and the moved behaviour is identical.
"""

from __future__ import annotations

import pandas as pd

from uber.domain import config
from uber.infrastructure import csv_io, ncr, repositories


# --- column contracts live in the repositories adapter ---------------------
def test_repositories_column_contracts() -> None:
    assert repositories.LOCATION_COLUMNS == [
        "location_id", "street", "neighborhood", "district", "lat", "lon"
    ]
    assert repositories.TIER_COLUMNS == [
        "tier", "display_name", "capacity", "base_fare", "per_km", "per_min",
        "booking_fee", "min_fare",
    ]
    assert repositories.DRIVER_COLUMNS == [
        "driver_id", "home_district", "activity_class", "tenure_start", "tenure_end",
        "active_days", "n_rides", "driver_rating",
    ]


def test_build_tiers_df_matches_config_catalog() -> None:
    tiers = repositories.build_tiers_df()
    assert list(tiers.columns) == repositories.TIER_COLUMNS
    assert len(tiers) == len(config.TIERS)
    assert tiers["tier"].tolist() == [t.tier for t in config.TIERS]


# --- generic CSV round-trip lives in csv_io --------------------------------
def test_csv_io_write_roundtrip(tmp_path) -> None:
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    out = tmp_path / "sub" / "t.csv"          # parent created on write
    returned = csv_io.write_csv(df, out)
    assert returned == out and out.exists()
    reloaded = pd.read_csv(out)
    assert reloaded["a"].tolist() == [1, 2]
    assert reloaded["b"].tolist() == ["x", "y"]


# --- NCR empirical supports live in ncr ------------------------------------
def test_ncr_load_returns_nonempty_distractor_supports() -> None:
    supports = ncr.load_ncr()
    assert set(supports) == set(config.NCR_DISTRACTOR_COLUMNS)
    for values in supports.values():
        assert len(values) > 0
        assert not pd.isna(values).any()      # NaNs dropped independently (R3)
