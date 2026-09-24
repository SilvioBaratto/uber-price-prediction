"""Tasks 3R.2 / 3R.3 — the day-by-day driver simulation (SPEC §2.6, Rev 2).

Uses a modest ``N`` so the vectorized simulation stays fast; the shipped default is 15k
drivers (millions of rides). ``build_drivers`` produces the roster + tenure/churn;
``simulate_ride_events`` produces the per-driver, per-day ride events.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from uber.datagen import sampling, simulation
from uber.domain import config
from uber.infrastructure import repositories

N = 3000  # enough for stable class-mix / stationarity checks, still fast
YEAR_DAYS = 366  # 2024 is a leap year
JAN1 = np.datetime64("2024-01-01")
DEC31 = np.datetime64("2024-12-31")


@pytest.fixture(scope="module")
def locations() -> pd.DataFrame:
    return repositories.build_locations_df()


@pytest.fixture(scope="module")
def drivers(locations: pd.DataFrame) -> pd.DataFrame:
    return simulation.build_drivers(sampling.make_rng(config.SEED), N, locations)


# --- Task 3R.2: roster + tenure/churn -------------------------------------
def test_roster_shape_and_keys(drivers: pd.DataFrame) -> None:
    assert list(drivers.columns) == repositories.DRIVER_COLUMNS
    assert len(drivers) == N
    assert drivers["driver_id"].tolist() == list(range(1, N + 1))
    assert drivers["driver_id"].is_unique


def test_roster_reproducible(locations: pd.DataFrame) -> None:
    a = simulation.build_drivers(sampling.make_rng(config.SEED), N, locations)
    b = simulation.build_drivers(sampling.make_rng(config.SEED), N, locations)
    pd.testing.assert_frame_equal(a, b)


def test_activity_class_mix_matches_weights(drivers: pd.DataFrame) -> None:
    frac = drivers["activity_class"].value_counts(normalize=True)
    for cls, w in zip(config.ACTIVITY_CLASSES, config.ACTIVITY_CLASS_WEIGHTS):
        assert frac[cls] == pytest.approx(w, abs=0.04)


def test_home_district_valid(drivers: pd.DataFrame, locations: pd.DataFrame) -> None:
    assert set(drivers["home_district"]) <= set(locations["district"])


def test_driver_rating_in_range(drivers: pd.DataFrame) -> None:
    assert (
        drivers["driver_rating"].between(config.DRIVER_RATING_MIN, config.DRIVER_RATING_MAX).all()
    )


def test_tenure_within_year_and_ordered(drivers: pd.DataFrame) -> None:
    start = drivers["tenure_start"].to_numpy(dtype="datetime64[D]")
    end = drivers["tenure_end"].to_numpy(dtype="datetime64[D]")
    assert (start >= JAN1).all() and (end <= DEC31).all()
    assert (start <= end).all()


def test_churn_present(drivers: pd.DataFrame) -> None:
    end = drivers["tenure_end"].to_numpy(dtype="datetime64[D]")
    churned = (end < DEC31).mean()
    assert 0.3 < churned < 0.95  # a real churned fraction, but not everyone
    # all three tenure cohorts should be represented in the tenure-length spread
    start = drivers["tenure_start"].to_numpy(dtype="datetime64[D]")
    length = (end - start).astype("timedelta64[D]").astype(int) + 1
    assert length.min() < 130  # short (~3mo) cohort present
    assert length.max() > 300  # long (~12mo) cohort present


def test_active_fleet_is_stationary(drivers: pd.DataFrame) -> None:
    # No January cliff: the number of drivers active on each calendar day is ~flat.
    start = (drivers["tenure_start"].to_numpy(dtype="datetime64[D]") - JAN1).astype(int)
    end = (drivers["tenure_end"].to_numpy(dtype="datetime64[D]") - JAN1).astype(int)
    diff = np.zeros(YEAR_DAYS + 1, dtype=int)  # +1 so end==last-day decrements out of range
    np.add.at(diff, start, 1)
    np.add.at(diff, end + 1, -1)
    active = np.cumsum(diff)[:YEAR_DAYS]
    mean = active.mean()
    assert active.min() > 0.75 * mean
    assert active.max() < 1.25 * mean


# --- Task 3R.3: per-day ride events ---------------------------------------
@pytest.fixture(scope="module")
def sim(drivers: pd.DataFrame):
    events, active_days, n_rides = simulation.simulate_ride_events(
        sampling.make_rng(config.SEED + 1), drivers
    )
    return events, active_days, n_rides


def test_events_reproducible(drivers: pd.DataFrame) -> None:
    e1, ad1, nr1 = simulation.simulate_ride_events(sampling.make_rng(7), drivers)
    e2, ad2, nr2 = simulation.simulate_ride_events(sampling.make_rng(7), drivers)
    pd.testing.assert_frame_equal(e1, e2)
    assert np.array_equal(ad1, ad2) and np.array_equal(nr1, nr2)


def test_every_driver_has_at_least_one_ride(sim) -> None:
    _events, _active_days, n_rides = sim
    assert (n_rides >= 1).all()
    assert len(n_rides) == N


def test_counts_consistent_with_events(sim) -> None:
    events, active_days, n_rides = sim
    assert len(events) == int(n_rides.sum())
    counts = events["driver_id"].value_counts().sort_index()
    assert counts.reindex(range(1, N + 1), fill_value=0).to_numpy().tolist() == n_rides.tolist()


def test_ride_counts_are_right_skewed(sim) -> None:
    _events, _active_days, n_rides = sim
    assert (n_rides >= 2).mean() > 0.5  # most drivers have multiple rides
    assert n_rides.max() > 5 * np.median(n_rides)  # heavy tail (full-timers)


def test_rides_fall_within_tenure(sim, drivers: pd.DataFrame) -> None:
    events, _ad, _nr = sim
    ts = events["timestamp"].to_numpy(dtype="datetime64[D]")
    did = events["driver_id"].to_numpy()
    start = drivers["tenure_start"].to_numpy(dtype="datetime64[D]")[did - 1]
    end = drivers["tenure_end"].to_numpy(dtype="datetime64[D]")[did - 1]
    assert (ts >= start).all() and (ts <= end).all()


def test_hour_distribution_is_non_uniform(sim) -> None:
    events, _ad, _nr = sim
    hours = pd.DatetimeIndex(events["timestamp"]).hour
    counts = hours.value_counts()
    # busy evening hours must dominate the small-hours trough
    assert counts.get(19, 0) > 2 * counts.get(4, 1)


def test_year_and_week_coverage(sim) -> None:
    events, _ad, _nr = sim
    idx = pd.DatetimeIndex(events["timestamp"])
    assert set(idx.month) == set(range(1, 13))
    assert set(idx.dayofweek) == set(range(7))
