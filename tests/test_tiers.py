"""Validation for the ride-tier catalog (ride_tiers.csv)."""

from __future__ import annotations

import pandas as pd
import pytest

from uber.datagen import generate_data
from uber.infrastructure import paths, repositories

# Declared tier ids in increasing-price order.
EXPECTED_TIER_IDS = ["uberx", "green", "comfort", "xl", "black", "van"]
N_TIERS = len(EXPECTED_TIER_IDS)


@pytest.fixture(scope="module")
def tiers() -> pd.DataFrame:
    return repositories.build_tiers_df()


def test_tiers_columns_and_order(tiers: pd.DataFrame) -> None:
    assert list(tiers.columns) == repositories.TIER_COLUMNS
    assert repositories.TIER_COLUMNS == [
        "tier",
        "display_name",
        "capacity",
        "base_fare",
        "per_km",
        "per_min",
        "booking_fee",
        "min_fare",
    ]


def test_tiers_row_count(tiers: pd.DataFrame) -> None:
    assert len(tiers) == N_TIERS
    assert tiers["tier"].tolist() == EXPECTED_TIER_IDS


def test_tiers_no_nulls_and_no_empty_strings(tiers: pd.DataFrame) -> None:
    assert int(tiers.isnull().sum().sum()) == 0
    for col in ("tier", "display_name"):
        assert (tiers[col].str.len() > 0).all()


def test_tiers_dtypes(tiers: pd.DataFrame) -> None:
    assert pd.api.types.is_integer_dtype(tiers["capacity"])
    for col in ("base_fare", "per_km", "per_min", "booking_fee", "min_fare"):
        assert pd.api.types.is_float_dtype(tiers[col])
    for col in ("tier", "display_name"):
        assert pd.api.types.is_string_dtype(tiers[col])


def test_tiers_price_ordering(tiers: pd.DataFrame) -> None:
    # Fare parameters must be non-decreasing across the declared order (uberx <= ... <= van).
    for col in ("base_fare", "per_km", "per_min", "min_fare"):
        assert tiers[col].is_monotonic_increasing, f"{col} is not non-decreasing"


def test_tiers_values_match_spec(tiers: pd.DataFrame) -> None:
    uberx = tiers.loc[tiers["tier"] == "uberx"].iloc[0]
    assert uberx["display_name"] == "UberX"
    assert uberx["capacity"] == 4
    assert uberx["base_fare"] == pytest.approx(1.20)
    assert uberx["min_fare"] == pytest.approx(5.00)

    van = tiers.loc[tiers["tier"] == "van"].iloc[0]
    assert van["display_name"] == "Uber Van"
    assert van["capacity"] == 6
    assert van["base_fare"] == pytest.approx(4.50)
    assert van["min_fare"] == pytest.approx(14.00)


def test_generate_writes_tiers(tmp_path) -> None:
    written = generate_data.generate(tmp_path, n_drivers=5)  # tiny sim: this test only checks tiers
    assert written[paths.TIERS_CSV] == N_TIERS
    out = tmp_path / paths.TIERS_CSV
    assert out.exists()
    reloaded = pd.read_csv(out)
    assert list(reloaded.columns) == repositories.TIER_COLUMNS
    assert len(reloaded) == N_TIERS
    assert reloaded["tier"].tolist() == EXPECTED_TIER_IDS
