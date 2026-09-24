"""Consolidated SPEC §5 validation suite (Task 5.1, Checkpoint 4).

This is the single end-to-end validation of the generated dataset. It builds the driver
population + the emergent ride log **once** (a modest ``N_DRIVERS`` — R² and the structural
properties are ~invariant to scale, SPEC §2.6) and asserts every SPEC §5 property against
that one coherent dataset, plus byte-identical reproducibility (§5.1). The per-module unit
tests (``test_pricing``, ``test_sampling``, ``test_simulation`` …) remain the fine-grained
checks; this file proves they hold *together*. There is one test per SPEC §5 assertion,
numbered ``test_s5_<n>_...`` to match the spec.

The OLS feature set is *pinned* (A3) so the R² band is stable across runs: numeric
``distance_km, duration_min, surge_multiplier, hour, day_of_week, month`` plus one-hot
``tier`` and one-hot ``pickup_district``. ``driver_id`` is deliberately excluded — it is
metadata, not a price signal (SPEC §2.3).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from uber import generate_data, io, sampling, simulation
from uber.domain import config, pricing
from uber.infrastructure import paths

PINNED_NUMERIC = ["distance_km", "duration_min", "surge_multiplier", "hour", "day_of_week", "month"]
PINNED_CATEGORICAL = ["tier", "pickup_district"]

N_DRIVERS_TEST = 250   # emergent rides comfortably exceed 50k, but the suite stays fast
N_ROSTER = 3000        # the tenure/stationarity checks need a larger roster; building it is cheap
N_LOCATIONS = 8655

YEAR_DAYS = 366        # 2024 is a leap year
JAN1 = np.datetime64("2024-01-01")
DEC31 = np.datetime64("2024-12-31")


# --- fixtures --------------------------------------------------------------
@pytest.fixture(scope="module")
def locations() -> pd.DataFrame:
    return io.build_locations_df()


@pytest.fixture(scope="module")
def dataset(locations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return generate_data.build_dataset(sampling.make_rng(config.SEED), locations, N_DRIVERS_TEST)


@pytest.fixture(scope="module")
def rides(dataset: tuple[pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
    return dataset[1]


@pytest.fixture(scope="module")
def drivers(dataset: tuple[pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
    return dataset[0]


@pytest.fixture(scope="module")
def roster(locations: pd.DataFrame) -> pd.DataFrame:
    # A larger population than the ride-level dataset: the active-fleet stationarity and cohort
    # spread (§5.11) are only crisp at scale, and building the roster alone (no rides) is cheap.
    return simulation.build_drivers(sampling.make_rng(config.SEED), N_ROSTER, locations)


def _ols_r2(rides: pd.DataFrame, extra_numeric: list[str] = []) -> float:
    """In-sample OLS R² on the pinned signal feature set, optionally plus extra numeric cols."""
    dummies = pd.get_dummies(rides[PINNED_CATEGORICAL], drop_first=True)
    X = pd.concat([rides[PINNED_NUMERIC + extra_numeric], dummies], axis=1).to_numpy(dtype=float)
    y = rides["price_eur"].to_numpy(dtype=float)
    return LinearRegression().fit(X, y).score(X, y)


# --- §5.1 Reproducibility: same seed => byte-identical output ---------------
def test_s5_1_reproducible_byte_identical(tmp_path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    generate_data.generate(a, seed=config.SEED, n_drivers=40)
    generate_data.generate(b, seed=config.SEED, n_drivers=40)
    for name in (paths.LOCATIONS_CSV, paths.TIERS_CSV, paths.DRIVERS_CSV, paths.RIDES_CSV):
        assert (a / name).read_bytes() == (b / name).read_bytes(), f"{name} not byte-identical"


# --- §5.2 Integrity: columns, types, no nulls, emergent row count ----------
def test_s5_2_integrity(rides: pd.DataFrame) -> None:
    assert list(rides.columns) == generate_data.RIDE_COLUMNS
    assert int(rides.isnull().sum().sum()) == 0
    # row count is emergent from the driver simulation, well above 50k even at 250 drivers
    assert len(rides) > 50_000
    for col in ("ride_id", "driver_id", "pickup_location_id", "dropoff_location_id", "hour", "day_of_week", "month"):
        assert pd.api.types.is_integer_dtype(rides[col]), f"{col} is not integer"
    for col in ("distance_km", "duration_min", "surge_multiplier", "price_eur", "commission_eur"):
        assert pd.api.types.is_float_dtype(rides[col]), f"{col} is not float"
    for col in ("tier", "pickup_district", "payment_method"):
        assert pd.api.types.is_object_dtype(rides[col]) or pd.api.types.is_string_dtype(rides[col])


# --- §5.3 Price constraints: price > 0 and >= the tier's min_fare ----------
def test_s5_3_price_constraints(rides: pd.DataFrame) -> None:
    assert (rides["price_eur"] > 0).all()
    min_fare = rides["tier"].map({t.tier: t.min_fare for t in config.TIERS})
    assert (rides["price_eur"] >= min_fare - 1e-9).all()


# --- §5.4 Geographic consistency: distance == haversine×road-factor --------
def test_s5_4_geographic_consistency(rides: pd.DataFrame, locations: pd.DataFrame) -> None:
    lats = locations["lat"].to_numpy()
    lons = locations["lon"].to_numpy()
    assert (rides["duration_min"] > 0).all()
    for row in rides.head(500).itertuples():  # spot-check; the vectorized path is uniform
        pi, di = row.pickup_location_id - 1, row.dropoff_location_id - 1
        expected = pricing.road_distance(lats[pi], lons[pi], lats[di], lons[di])
        assert row.distance_km == pytest.approx(expected)
        assert row.duration_min == pytest.approx(pricing.duration_min(row.distance_km, row.hour))


# --- §5.5 Tier ordering: equivalent trips priced uberx <= ... <= van -------
def test_s5_5_tier_ordering(rides: pd.DataFrame) -> None:
    order = [t.tier for t in config.TIERS]              # increasing-price order (SPEC §2.2)
    band = rides["distance_km"].between(5.0, 10.0)       # "equivalent" trips: one distance band
    means = rides.loc[band].groupby("tier")["price_eur"].mean().reindex(order)
    # Non-decreasing across the declared order. uberx/green share identical tariffs (a deliberate
    # tie), so allow a small tolerance there; the genuinely distinct tiers are strictly ordered.
    assert (means.diff().dropna() >= -0.75).all(), f"tier means not ordered: {means.round(2).to_dict()}"
    assert means.idxmax() == "van" and means.idxmin() in ("uberx", "green")
    assert means["green"] < means["comfort"] < means["xl"] < means["black"] < means["van"]


# --- §5.6 Signal band: OLS R² in the calibrated band -----------------------
def test_s5_6_signal_band(rides: pd.DataFrame) -> None:
    lo, hi = config.R2_BAND
    r2 = _ols_r2(rides)
    assert lo <= r2 <= hi, f"OLS R²={r2:.4f} outside band {config.R2_BAND}"


# --- §5.7 Useless distractors: correlation ~0 with the target --------------
def test_s5_7_distractors_uncorrelated(rides: pd.DataFrame) -> None:
    price = rides["price_eur"].to_numpy(dtype=float)
    for col in ("driver_rating", "customer_rating", "avg_vtat"):
        r = np.corrcoef(rides[col].to_numpy(dtype=float), price)[0, 1]
        assert abs(r) < 0.05, f"{col} correlation with price_eur = {r:.4f}"
    for cat, indicator in pd.get_dummies(rides["payment_method"]).items():
        r = np.corrcoef(indicator.to_numpy(dtype=float), price)[0, 1]
        assert abs(r) < 0.05, f"payment_method={cat} correlation with price_eur = {r:.4f}"


# --- §5.8 Leakage trap: including commission_eur pushes R² ~ 1 -------------
def test_s5_8_leakage_trap(rides: pd.DataFrame) -> None:
    assert "commission_eur" not in generate_data.SIGNAL_COLUMNS
    assert generate_data.LEAKAGE_COLUMNS == ["commission_eur"]
    r2 = _ols_r2(rides, extra_numeric=["commission_eur"])
    assert r2 > 0.999, f"leakage R²={r2:.5f} did not approach 1"


# --- §5.9 Time-of-call pricing: weekend nights materially pricier ----------
def test_s5_9_time_of_call(rides: pd.DataFrame) -> None:
    hour, dow = rides["hour"], rides["day_of_week"]
    # Fri & Sat nights (21:00–03:59): Fri evening, Sat small hours + evening, Sun small hours.
    weekend_night = (
        ((dow == 4) & (hour >= 21))
        | ((dow == 5) & ((hour <= 3) | (hour >= 21)))
        | ((dow == 6) & (hour <= 3))
    )
    weekday_late_morning = (dow <= 4) & hour.isin([10, 11])
    band = (rides["tier"] == "uberx") & rides["distance_km"].between(5.0, 10.0)  # equivalent trips
    night_mean = rides.loc[band & weekend_night, "price_eur"].mean()
    morning_mean = rides.loc[band & weekday_late_morning, "price_eur"].mean()
    assert night_mean >= 1.5 * morning_mean, (
        f"weekend-night mean {night_mean:.2f} not >= 1.5x weekday late-morning {morning_mean:.2f}"
    )


# --- §5.10 Driver panel structure: valid FK, repeat/right-skewed, no leak --
def test_s5_10_driver_panel(rides: pd.DataFrame, drivers: pd.DataFrame) -> None:
    assert rides["driver_id"].between(1, len(drivers)).all()
    # every simulated driver gives at least one ride, so the id sets coincide exactly
    assert set(rides["driver_id"]) == set(drivers["driver_id"])
    counts = rides["driver_id"].value_counts()
    assert (counts >= 2).mean() > 0.5              # most drivers are repeat drivers
    assert counts.max() > 5 * counts.median()      # right-skewed: a heavy tail of full-timers
    # driver_id / driver_rating are metadata — never part of the pinned OLS signal set
    assert "driver_id" not in PINNED_NUMERIC and "driver_id" not in PINNED_CATEGORICAL
    r = np.corrcoef(rides["driver_rating"].to_numpy(dtype=float), rides["price_eur"].to_numpy(dtype=float))[0, 1]
    assert abs(r) < 0.05, f"driver_rating correlation with price_eur = {r:.4f}"


# --- §5.11 Tenure / churn: churned cohort, cohort spread, stationary fleet -
def test_s5_11_tenure_churn(roster: pd.DataFrame) -> None:
    start = roster["tenure_start"].to_numpy(dtype="datetime64[D]")
    end = roster["tenure_end"].to_numpy(dtype="datetime64[D]")
    churned = (end < DEC31).mean()
    assert 0.3 < churned < 0.95                    # a real churned fraction, but not everyone
    length = (end - start).astype("timedelta64[D]").astype(int) + 1
    assert length.min() < 130                      # short (~3mo) cohort present
    assert length.max() > 300                      # long (~12mo) cohort present
    # No January cliff: the number of drivers active on each calendar day is ~flat.
    s = (start - JAN1).astype(int)
    e = (end - JAN1).astype(int)
    diff = np.zeros(YEAR_DAYS + 1, dtype=int)       # +1 so end==last-day decrements out of range
    np.add.at(diff, s, 1)
    np.add.at(diff, e + 1, -1)
    active = np.cumsum(diff)[:YEAR_DAYS]
    mean = active.mean()
    assert active.min() > 0.75 * mean and active.max() < 1.25 * mean


# --- §5.12 Year coverage: all months/weekdays, non-uniform hours -----------
def test_s5_12_year_coverage(rides: pd.DataFrame) -> None:
    assert set(rides["month"].unique()) == set(range(1, 13))
    assert set(rides["day_of_week"].unique()) == set(range(7))
    counts = rides["hour"].value_counts()
    # busy evening hours must dominate the small-hours trough (hourly volume profile, not uniform)
    assert counts.get(19, 0) > 2 * counts.get(4, 1)
