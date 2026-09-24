"""Orchestration + CLI entry point: ``python -m uber.generate_data``.

Writes the dataset CSVs into ``data/raw/``: ``madrid_locations.csv``, ``ride_tiers.csv``,
``drivers.csv`` and the signal-only ``rides.csv`` (distractors + ``commission_eur`` land in
Phase 4). Rev 2: rides are the *emergent* output of a day-by-day driver simulation (SPEC
§2.6) rather than a fixed count of uniform events — the sizing knob is ``--n-drivers``. All
randomness flows through one seeded generator so runs are reproducible (SPEC §6).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from uber import sampling, simulation
from uber.domain import config, pricing
from uber.infrastructure import csv_io, ncr, paths, repositories

# Signal columns of SPEC §2.3 + the target ``price_eur`` (no distractors/commission yet).
# ``driver_id`` is metadata / a weak distractor (SPEC §2.3): it is NOT part of the pinned OLS
# feature set and does not enter the price formula.
SIGNAL_COLUMNS = [
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

# Weak features / distractors (SPEC §2.3, Task 4.1): true coefficient 0, so each is ~uncorrelated
# with ``price_eur``. ``payment_method`` / ``customer_rating`` / ``avg_vtat`` are sampled from the
# NCR empirical distributions; ``driver_rating`` is per-driver (drivers.csv) joined onto rides.
DISTRACTOR_COLUMNS = ["payment_method", "driver_rating", "customer_rating", "avg_vtat"]

# Data-leakage trap (SPEC §2.3, Task 4.2): derived from the target, so it is excluded from the
# signal set — feeding it in as an X feature pushes R² ≈ 1 (the Part 8 leakage example).
LEAKAGE_COLUMNS = ["commission_eur"]

# Full rides.csv column order: signal + target, then the weak distractors, then the leakage
# column. The pinned OLS feature set (SPEC §5.6) still draws only from ``SIGNAL_COLUMNS``.
RIDE_COLUMNS = SIGNAL_COLUMNS + DISTRACTOR_COLUMNS + LEAKAGE_COLUMNS


def _district_members(locations: pd.DataFrame) -> tuple[pd.Index, np.ndarray, np.ndarray, np.ndarray]:
    """Group location ids by district for the home-district pickup bias.

    Returns ``(district_index, member_ids, block_start, block_size)`` where ``member_ids`` is a
    flat array of 1-based location ids grouped by district, and ``block_start``/``block_size``
    slice each district's ids out of it (indexed by the district's code in ``district_index``).
    """
    codes, districts = pd.factorize(locations["district"].to_numpy())
    order = np.argsort(codes, kind="stable")
    member_ids = order.astype(np.int64) + 1  # 1-based location ids, grouped by district code
    size = np.bincount(codes, minlength=len(districts)).astype(np.int64)
    start = np.cumsum(size) - size
    return pd.Index(districts), member_ids, start, size


def build_rides_df(
    rng: np.random.Generator,
    locations: pd.DataFrame,
    drivers: pd.DataFrame,
    events: pd.DataFrame,
    ncr: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Assemble the rides frame (signal + distractors) from simulated (driver, timestamp) events.

    Fully vectorized (no per-ride Python loop): OD pairs — with a share of pickups biased into
    the driver's home district — tier, surge jitter and price noise are drawn from ``rng`` in a
    fixed order, then the per-ride physics reuse the canonical *array* functions in
    :mod:`uber.domain.pricing` (the documented ground truth) so the formula is never duplicated. The
    weak distractors (SPEC §2.3) are added last: ``payment_method`` / ``customer_rating`` /
    ``avg_vtat`` are sampled from the NCR empirical supports in ``ncr`` (drawn *after* the signal
    draws so the signal columns stay byte-identical), and per-driver ``driver_rating`` is joined
    on. Finally ``commission_eur`` (the leakage trap) is derived from ``price_eur``. Rides are
    sorted chronologically and assigned a contiguous ``ride_id``.
    """
    lats = locations["lat"].to_numpy()
    lons = locations["lon"].to_numpy()
    loc_districts = locations["district"].to_numpy()
    n_locations = len(locations)
    tier_by_id = {t.tier: t for t in config.TIERS}
    tier_ids = list(tier_by_id)

    n = len(events)
    driver_id = events["driver_id"].to_numpy()
    timestamps = events["timestamp"].to_numpy()

    # --- calendar fields (weekday(): Mon=0 .. Sun=6), derived from the simulated timestamps ---
    idx = pd.DatetimeIndex(timestamps)
    hour = idx.hour.to_numpy()
    dow = idx.dayofweek.to_numpy()
    month = idx.month.to_numpy()

    # --- per-driver attributes joined onto each ride (home district + weak-distractor rating) ---
    district_index, member_ids, block_start, block_size = _district_members(locations)
    home_per_driver = drivers["home_district"].to_numpy()
    ride_home_code = pd.Categorical(home_per_driver[driver_id - 1], categories=district_index).codes
    driver_rating = drivers["driver_rating"].to_numpy()[driver_id - 1]  # constant within a driver

    # --- draws (fixed order: bias mask, uniform pickup, home-pick, dropoff, tiers, jitter, noise) ---
    biased = rng.random(n) < config.HOME_DISTRICT_PICKUP_PROB
    pickup_uniform = rng.integers(1, n_locations + 1, size=n, dtype=np.int64)
    u_home = rng.random(n)
    local = np.floor(u_home * block_size[ride_home_code]).astype(np.int64)
    pickup_biased = member_ids[block_start[ride_home_code] + local]
    pickup = np.where(biased, pickup_biased, pickup_uniform)

    offset = rng.integers(1, n_locations, size=n, dtype=np.int64)  # 1 .. n_locations-1
    dropoff = ((pickup - 1 + offset) % n_locations) + 1  # guaranteed != pickup

    tiers = sampling.sample_tiers(rng, n, tier_ids)
    jitter = sampling.sample_jitter(rng, n)
    noise = rng.normal(0.0, config.NOISE_STD, size=n)

    # Weak distractors last (fixed order), so adding them never perturbs the signal draws above.
    payment_method = sampling.sample_empirical(rng, ncr["payment_method"], n)
    customer_rating = sampling.sample_empirical(rng, ncr["customer_rating"], n)
    avg_vtat = sampling.sample_empirical(rng, ncr["avg_vtat"], n)
    # Leakage trap: derived from the target (needs price_eur, so drawn after it below).

    # --- coordinate lookup (ids are 1-based and sequential) ---
    p_lat, p_lon = lats[pickup - 1], lons[pickup - 1]
    d_lat, d_lon = lats[dropoff - 1], lons[dropoff - 1]

    # --- per-ride physics via the canonical vectorized pricing functions ---
    distance_km = pricing.road_distance(p_lat, p_lon, d_lat, d_lon)
    duration_min = pricing.duration_min_array(distance_km, hour)
    surge_multiplier = pricing.surge_array(hour, dow, jitter)

    # tier tariffs as per-ride arrays (factorize once, then index the small per-tier lookups)
    tcodes, tuniques = pd.factorize(tiers)
    base_fare = np.array([tier_by_id[t].base_fare for t in tuniques])[tcodes]
    per_km = np.array([tier_by_id[t].per_km for t in tuniques])[tcodes]
    per_min = np.array([tier_by_id[t].per_min for t in tuniques])[tcodes]
    booking_fee = np.array([tier_by_id[t].booking_fee for t in tuniques])[tcodes]
    min_fare = np.array([tier_by_id[t].min_fare for t in tuniques])[tcodes]
    price_eur = pricing.price_array(
        base_fare, per_km, per_min, booking_fee, min_fare,
        distance_km, duration_min, surge_multiplier, noise,
    )
    # Leakage column: a near-perfect linear function of the target (Uber's cut) + tiny noise.
    commission_eur = config.COMMISSION_RATE * price_eur + rng.normal(0.0, config.COMMISSION_NOISE_STD, size=n)

    df = pd.DataFrame(
        {
            "driver_id": driver_id,
            "timestamp": timestamps,
            "pickup_location_id": pickup,
            "dropoff_location_id": dropoff,
            "pickup_district": loc_districts[pickup - 1],
            "distance_km": distance_km,
            "duration_min": duration_min,
            "tier": tiers,
            "hour": hour,
            "day_of_week": dow,
            "month": month,
            "surge_multiplier": surge_multiplier,
            "price_eur": price_eur,
            "payment_method": payment_method,
            "driver_rating": driver_rating,
            "customer_rating": customer_rating,
            "avg_vtat": avg_vtat,
            "commission_eur": commission_eur,
        }
    )
    # Chronological order, then a contiguous 1-based ride_id (stable sort keeps ties in the
    # deterministic event order, so the assignment stays reproducible).
    df = df.sort_values("timestamp", kind="stable", ignore_index=True)
    df.insert(0, "ride_id", np.arange(1, len(df) + 1, dtype=np.int64))
    return df[RIDE_COLUMNS]


def build_dataset(
    rng: np.random.Generator, locations: pd.DataFrame, n_drivers: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full driver→ride pipeline. Returns ``(drivers, rides)``.

    Builds the roster, simulates every ride each driver gives, folds the per-driver
    ``active_days`` / ``n_rides`` counts back into the driver table, then assembles the rides
    (joining the NCR empirical distractor supports, loaded once). A single ``rng`` threads
    through all three steps in a fixed order for reproducibility.
    """
    drivers = simulation.build_drivers(rng, n_drivers, locations)
    events, active_days, n_rides = simulation.simulate_ride_events(rng, drivers)
    drivers["active_days"] = active_days
    drivers["n_rides"] = n_rides
    rides = build_rides_df(rng, locations, drivers, events, ncr.load_ncr())
    return drivers, rides


def generate(out_dir: Path, seed: int = config.SEED, n_drivers: int = config.N_DRIVERS) -> dict[str, int]:
    """Generate the CSVs into ``out_dir``. Returns ``{filename: row_count}``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, int] = {}

    locations = repositories.build_locations_df()
    csv_io.write_csv(locations, out_dir / paths.LOCATIONS_CSV)
    written[paths.LOCATIONS_CSV] = len(locations)

    tiers = repositories.build_tiers_df()
    csv_io.write_csv(tiers, out_dir / paths.TIERS_CSV)
    written[paths.TIERS_CSV] = len(tiers)

    drivers, rides = build_dataset(sampling.make_rng(seed), locations, n_drivers)
    csv_io.write_csv(drivers, out_dir / paths.DRIVERS_CSV)
    written[paths.DRIVERS_CSV] = len(drivers)
    csv_io.write_csv(rides, out_dir / paths.RIDES_CSV)
    written[paths.RIDES_CSV] = len(rides)

    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic Madrid ride dataset.")
    parser.add_argument("--seed", type=int, default=config.SEED, help="RNG seed (default: %(default)s)")
    parser.add_argument(
        "--n-drivers", type=int, default=config.N_DRIVERS,
        help="number of drivers to simulate; rides emerge from the simulation (default: %(default)s)",
    )
    parser.add_argument("--out-dir", type=Path, default=paths.RAW_DIR, help="output directory (default: data/raw)")
    args = parser.parse_args(argv)

    written = generate(args.out_dir, seed=args.seed, n_drivers=args.n_drivers)
    for name, rows in written.items():
        print(f"wrote {args.out_dir / name} ({rows} rows)")


if __name__ == "__main__":
    main()
