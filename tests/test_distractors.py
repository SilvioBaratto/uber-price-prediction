"""Empirical distractor columns (SPEC §2.3, Task 4.1).

``payment_method`` / ``customer_rating`` / ``avg_vtat`` are sampled with replacement from
the NCR empirical distributions; ``driver_rating`` is a per-driver attribute (drivers.csv,
Phase 3R) joined onto each ride so it is constant within a driver. All four are *weak*
features — true coefficient 0 — so their |Pearson r| with ``price_eur`` is < 0.05 and no
nulls are introduced. Run with ``pytest -k distractor``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from uber import config, generate_data, io, sampling

N_DRIVERS_TEST = 250  # ~100k emergent rides — enough for a stable correlation estimate

NUMERIC_DISTRACTORS = ["driver_rating", "customer_rating", "avg_vtat"]


@pytest.fixture(scope="module")
def dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    locations = io.build_locations_df()
    return generate_data.build_dataset(sampling.make_rng(config.SEED), locations, N_DRIVERS_TEST)


@pytest.fixture(scope="module")
def rides(dataset: tuple[pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
    return dataset[1]


@pytest.fixture(scope="module")
def drivers(dataset: tuple[pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
    return dataset[0]


# --- schema ----------------------------------------------------------------
def test_distractor_columns_present_and_ordered(rides: pd.DataFrame) -> None:
    assert list(rides.columns) == generate_data.RIDE_COLUMNS
    assert generate_data.DISTRACTOR_COLUMNS == [
        "payment_method",
        "driver_rating",
        "customer_rating",
        "avg_vtat",
    ]
    # the distractor block sits immediately after the signal + target columns
    n = len(generate_data.SIGNAL_COLUMNS)
    assert generate_data.RIDE_COLUMNS[n:n + len(generate_data.DISTRACTOR_COLUMNS)] == generate_data.DISTRACTOR_COLUMNS


def test_distractors_have_no_nulls(rides: pd.DataFrame) -> None:
    cols = ["payment_method", "driver_rating", "customer_rating", "avg_vtat"]
    assert int(rides[cols].isnull().sum().sum()) == 0


# --- weak features: correlation with the target is ~0 ----------------------
def test_numeric_distractors_uncorrelated_with_price(rides: pd.DataFrame) -> None:
    price = rides["price_eur"].to_numpy(dtype=float)
    for col in NUMERIC_DISTRACTORS:
        r = np.corrcoef(rides[col].to_numpy(dtype=float), price)[0, 1]
        assert abs(r) < 0.05, f"{col} correlation with price_eur = {r:.4f}"


def test_payment_method_uncorrelated_with_price(rides: pd.DataFrame) -> None:
    # categorical: each one-hot indicator must be ~uncorrelated with the target
    price = rides["price_eur"].to_numpy(dtype=float)
    for cat, indicator in pd.get_dummies(rides["payment_method"]).items():
        r = np.corrcoef(indicator.to_numpy(dtype=float), price)[0, 1]
        assert abs(r) < 0.05, f"payment_method={cat} correlation with price_eur = {r:.4f}"


# --- value ranges match the NCR empirical support --------------------------
def test_distractor_values_match_ncr_support(rides: pd.DataFrame) -> None:
    ncr = io.load_ncr()
    assert set(rides["payment_method"]) <= set(ncr["payment_method"].tolist())
    assert rides["customer_rating"].isin(np.unique(ncr["customer_rating"])).all()
    assert rides["avg_vtat"].isin(np.unique(ncr["avg_vtat"])).all()


# --- driver_rating: per-driver, joined onto rides --------------------------
def test_driver_rating_in_calibrated_range(rides: pd.DataFrame) -> None:
    assert rides["driver_rating"].between(config.DRIVER_RATING_MIN, config.DRIVER_RATING_MAX).all()


def test_driver_rating_constant_within_driver(rides: pd.DataFrame) -> None:
    per_driver = rides.groupby("driver_id")["driver_rating"].nunique()
    assert (per_driver == 1).all()


def test_driver_rating_matches_roster(rides: pd.DataFrame, drivers: pd.DataFrame) -> None:
    roster = drivers.set_index("driver_id")["driver_rating"]
    joined = rides["driver_id"].map(roster).to_numpy()
    assert np.array_equal(rides["driver_rating"].to_numpy(), joined)


# --- io.load_ncr / sampling.sample_empirical units -------------------------
def test_load_ncr_drops_nulls_per_column() -> None:
    ncr = io.load_ncr()
    assert set(ncr) == {"payment_method", "customer_rating", "avg_vtat"}
    for values in ncr.values():
        assert len(values) > 0
        assert not pd.isna(values).any()


def test_sample_empirical_is_deterministic() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0])
    a = sampling.sample_empirical(sampling.make_rng(7), values, 100)
    b = sampling.sample_empirical(sampling.make_rng(7), values, 100)
    assert np.array_equal(a, b)


def test_sample_empirical_draws_only_from_support() -> None:
    values = np.array(["a", "b", "c"])
    draws = sampling.sample_empirical(sampling.make_rng(1), values, 500)
    assert set(np.unique(draws)) <= {"a", "b", "c"}
    assert len(draws) == 500


def test_sample_empirical_reproduces_distribution() -> None:
    values = np.array([0.0] + [1.0] * 9)  # 90% ones
    draws = sampling.sample_empirical(sampling.make_rng(3), values, 20000)
    assert float(draws.mean()) == pytest.approx(0.9, abs=0.02)
