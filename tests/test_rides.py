"""Integrity, geographic consistency and reproducibility of the signal-only rides table.

Rev 2 (Phase 3R): rides are the *emergent* output of the day-by-day driver simulation
(SPEC §2.6). ``build_dataset`` returns ``(drivers, rides)``; ``rides`` carries the signal
columns of SPEC §2.3 + ``price_eur`` plus the ``driver_id`` foreign key (metadata / weak
distractor). Checks use a modest driver count that still yields > 50k rides.
"""

from __future__ import annotations

import pandas as pd
import pytest

from uber.datagen import generate_data, sampling
from uber.domain import config, pricing
from uber.infrastructure import paths, repositories

N_LOCATIONS = 8655
N_DRIVERS_TEST = 250  # emergent rides comfortably exceed 50k, but the suite stays fast


@pytest.fixture(scope="module")
def locations() -> pd.DataFrame:
    return repositories.build_locations_df()


@pytest.fixture(scope="module")
def dataset(locations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return generate_data.build_dataset(sampling.make_rng(config.SEED), locations, N_DRIVERS_TEST)


@pytest.fixture(scope="module")
def rides(dataset: tuple[pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
    return dataset[1]


@pytest.fixture(scope="module")
def drivers(dataset: tuple[pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
    return dataset[0]


def test_rides_columns_and_order(rides: pd.DataFrame) -> None:
    assert list(rides.columns) == generate_data.RIDE_COLUMNS
    assert generate_data.SIGNAL_COLUMNS == [
        "ride_id",
        "driver_id",
        "timestamp",
        "pickup_location_id",
        "dropoff_location_id",
        "pickup_district",
        "distance_km",
        "duration_min",
        "tier",
        "hour",
        "day_of_week",
        "month",
        "surge_multiplier",
        "price_eur",
    ]
    # signal + target, then the weak distractors (Task 4.1), then the leakage column (Task 4.2)
    assert generate_data.RIDE_COLUMNS == generate_data.SIGNAL_COLUMNS + [
        "payment_method",
        "driver_rating",
        "customer_rating",
        "avg_vtat",
        "commission_eur",
    ]


def test_rides_integrity(rides: pd.DataFrame) -> None:
    assert len(rides) > 0
    assert int(rides.isnull().sum().sum()) == 0
    assert (rides["price_eur"] > 0).all()
    assert (rides["duration_min"] > 0).all()

    min_fare = rides["tier"].map({t.tier: t.min_fare for t in config.TIERS})
    assert (rides["price_eur"] >= min_fare - 1e-9).all()

    assert rides["pickup_location_id"].between(1, N_LOCATIONS).all()
    assert rides["dropoff_location_id"].between(1, N_LOCATIONS).all()
    assert (rides["pickup_location_id"] != rides["dropoff_location_id"]).all()
    assert rides["hour"].between(0, 23).all()
    assert rides["day_of_week"].between(0, 6).all()
    assert rides["month"].between(1, 12).all()
    assert rides["surge_multiplier"].between(1.0, 3.0).all()

    lo, hi = config.SURGE_CLAMP
    assert rides["ride_id"].tolist() == list(range(1, len(rides) + 1))
    assert set(rides["tier"]) <= {t.tier for t in config.TIERS}
    assert lo <= rides["surge_multiplier"].min()
    assert rides["surge_multiplier"].max() <= hi


def test_rides_row_count_is_emergent_and_large(rides: pd.DataFrame) -> None:
    # Row count is driven by the driver simulation, not a fixed knob; even a few hundred
    # drivers produce a big log (SPEC §2.6 sizing note: full 15k scale ⇒ millions).
    assert len(rides) > 50_000


def test_rides_sorted_chronologically(rides: pd.DataFrame) -> None:
    ts = rides["timestamp"].to_numpy()
    assert (ts[:-1] <= ts[1:]).all()


def test_driver_id_is_valid_foreign_key(rides: pd.DataFrame, drivers: pd.DataFrame) -> None:
    assert rides["driver_id"].between(1, len(drivers)).all()
    # every simulated driver gives at least one ride, so the id sets coincide exactly
    assert set(rides["driver_id"]) == set(drivers["driver_id"])


def test_driver_panel_has_repeat_rides(rides: pd.DataFrame, drivers: pd.DataFrame) -> None:
    counts = rides["driver_id"].value_counts()
    assert counts.max() > 1                       # a driver with many rides
    assert (counts >= 2).mean() > 0.5             # most drivers are repeat drivers
    # the per-driver n_rides recorded on the roster matches the actual ride log
    logged = counts.reindex(drivers["driver_id"]).fillna(0).astype(int).to_numpy()
    assert logged.tolist() == drivers["n_rides"].tolist()


def test_pickup_biased_toward_home_district(rides: pd.DataFrame, drivers: pd.DataFrame) -> None:
    home = drivers.set_index("driver_id")["home_district"]
    ride_home = rides["driver_id"].map(home)
    match = (rides["pickup_district"] == ride_home).mean()
    # baseline (no bias) would be ~1/21 ≈ 0.05; the 40% home-bias lifts this far above chance
    assert match > 0.30


def test_rides_geo_distance_and_duration(rides: pd.DataFrame, locations: pd.DataFrame) -> None:
    lats = locations["lat"].to_numpy()
    lons = locations["lon"].to_numpy()
    # distance_km must equal haversine(A,B) x ROAD_FACTOR of the two location ids, and
    # duration_min must follow from distance and the hour's speed.
    for row in rides.head(500).itertuples():
        pi, di = row.pickup_location_id - 1, row.dropoff_location_id - 1
        expected_dist = pricing.road_distance(lats[pi], lons[pi], lats[di], lons[di])
        assert row.distance_km == pytest.approx(expected_dist)
        assert row.duration_min == pytest.approx(pricing.duration_min(row.distance_km, row.hour))


def test_rides_pickup_district_matches_location(rides: pd.DataFrame, locations: pd.DataFrame) -> None:
    districts = locations["district"].to_numpy()
    for row in rides.head(500).itertuples():
        assert row.pickup_district == districts[row.pickup_location_id - 1]


def test_rides_reproducible_same_seed(locations: pd.DataFrame) -> None:
    _d1, r1 = generate_data.build_dataset(sampling.make_rng(config.SEED), locations, 40)
    _d2, r2 = generate_data.build_dataset(sampling.make_rng(config.SEED), locations, 40)
    pd.testing.assert_frame_equal(r1, r2)


def test_reproducible_byte_identical_csvs(tmp_path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    generate_data.generate(a, seed=config.SEED, n_drivers=40)
    generate_data.generate(b, seed=config.SEED, n_drivers=40)
    assert (a / paths.RIDES_CSV).read_bytes() == (b / paths.RIDES_CSV).read_bytes()
    assert (a / paths.DRIVERS_CSV).read_bytes() == (b / paths.DRIVERS_CSV).read_bytes()


def test_generate_writes_rides_and_drivers(tmp_path) -> None:
    written = generate_data.generate(tmp_path, seed=config.SEED, n_drivers=40)
    assert written[paths.DRIVERS_CSV] == 40

    rides_out = tmp_path / paths.RIDES_CSV
    assert rides_out.exists()
    reloaded = pd.read_csv(rides_out)
    assert list(reloaded.columns) == generate_data.RIDE_COLUMNS
    assert len(reloaded) == written[paths.RIDES_CSV]

    drivers_out = tmp_path / paths.DRIVERS_CSV
    assert drivers_out.exists()
    drivers = pd.read_csv(drivers_out)
    assert list(drivers.columns) == repositories.DRIVER_COLUMNS
    assert len(drivers) == 40
