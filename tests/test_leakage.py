"""Data-leakage trap: ``commission_eur`` (SPEC §2.3 / §5.8, Task 4.2).

``commission_eur = COMMISSION_RATE × price_eur + negligible noise`` is derived straight from
the target, so it is *excluded* from the signal formula and the pinned OLS feature set.
Including it among the X features pushes in-sample R² ≈ 1 — the concrete leakage example Part 8
must recognise and drop. Excluding it leaves R² in the calibrated band. Run with
``pytest -k leakage``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from uber.datagen import generate_data, sampling
from uber.domain import config
from uber.infrastructure import repositories

N_DRIVERS_TEST = 250  # ~100k emergent rides — enough for a stable in-sample R²
PINNED_NUMERIC = ["distance_km", "duration_min", "surge_multiplier", "hour", "day_of_week", "month"]


@pytest.fixture(scope="module")
def rides() -> pd.DataFrame:
    locations = repositories.build_locations_df()
    _drivers, rides = generate_data.build_dataset(
        sampling.make_rng(config.SEED), locations, N_DRIVERS_TEST
    )
    return rides


def _ols_r2(rides: pd.DataFrame, extra_numeric: list[str] = []) -> float:
    """In-sample OLS R² on the pinned signal feature set, optionally plus extra numeric cols."""
    dummies = pd.get_dummies(rides[["tier", "pickup_district"]], drop_first=True)
    X = pd.concat([rides[PINNED_NUMERIC + extra_numeric], dummies], axis=1).to_numpy(dtype=float)
    y = rides["price_eur"].to_numpy(dtype=float)
    return LinearRegression().fit(X, y).score(X, y)


def test_leakage_column_present_and_excluded_from_signal(rides: pd.DataFrame) -> None:
    assert "commission_eur" in rides.columns
    assert rides["commission_eur"].notna().all()
    # the leakage column is NOT a signal feature and NOT a weak distractor
    assert "commission_eur" not in generate_data.SIGNAL_COLUMNS
    assert "commission_eur" not in generate_data.DISTRACTOR_COLUMNS
    assert generate_data.LEAKAGE_COLUMNS == ["commission_eur"]


def test_leakage_commission_is_quarter_of_price(rides: pd.DataFrame) -> None:
    resid = rides["commission_eur"].to_numpy(dtype=float) - config.COMMISSION_RATE * rides[
        "price_eur"
    ].to_numpy(dtype=float)
    # only negligible Gaussian noise separates commission from 0.25 x price
    assert float(np.abs(resid).mean()) < 0.1
    assert float(resid.std()) == pytest.approx(config.COMMISSION_NOISE_STD, abs=0.01)
    assert (rides["commission_eur"] > 0).all()


def test_leakage_including_commission_pushes_r2_to_one(rides: pd.DataFrame) -> None:
    r2 = _ols_r2(rides, extra_numeric=["commission_eur"])
    assert r2 > 0.999, f"leakage R²={r2:.5f} did not approach 1"


def test_leakage_excluding_commission_keeps_r2_in_band(rides: pd.DataFrame) -> None:
    lo, hi = config.R2_BAND
    r2 = _ols_r2(rides)
    assert lo <= r2 <= hi, f"signal R²={r2:.4f} outside band {config.R2_BAND}"
