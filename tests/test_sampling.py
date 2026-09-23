"""Unit tests for seeded sampling (no filesystem I/O).

Determinism (same seed => identical arrays) and coverage (all tiers / all 12 months /
distinct in-range OD pairs) per Task 3.2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from uber import config, sampling

TIER_IDS = [t.tier for t in config.TIERS]
N_LOCATIONS = 8655


# --- rng -------------------------------------------------------------------
def test_make_rng_is_deterministic() -> None:
    a = sampling.make_rng(123).integers(0, 1_000_000, size=50)
    b = sampling.make_rng(123).integers(0, 1_000_000, size=50)
    assert np.array_equal(a, b)


# --- timestamps ------------------------------------------------------------
def test_timestamps_deterministic() -> None:
    ts1 = sampling.sample_timestamps(sampling.make_rng(config.SEED), 1000, config.YEAR)
    ts2 = sampling.sample_timestamps(sampling.make_rng(config.SEED), 1000, config.YEAR)
    assert np.array_equal(ts1, ts2)


def test_timestamps_within_the_year() -> None:
    ts = sampling.sample_timestamps(sampling.make_rng(config.SEED), 5000, config.YEAR)
    idx = pd.DatetimeIndex(ts)
    assert (idx.year == config.YEAR).all()


def test_timestamps_cover_all_months() -> None:
    ts = sampling.sample_timestamps(sampling.make_rng(config.SEED), 10000, config.YEAR)
    months = set(pd.DatetimeIndex(ts).month.tolist())
    assert months == set(range(1, 13))


# --- OD pairs --------------------------------------------------------------
def test_od_pairs_deterministic() -> None:
    p1, d1 = sampling.sample_od_pairs(sampling.make_rng(config.SEED), 1000, N_LOCATIONS)
    p2, d2 = sampling.sample_od_pairs(sampling.make_rng(config.SEED), 1000, N_LOCATIONS)
    assert np.array_equal(p1, p2)
    assert np.array_equal(d1, d2)


def test_od_pairs_in_range_and_distinct() -> None:
    pickup, dropoff = sampling.sample_od_pairs(sampling.make_rng(config.SEED), 20000, N_LOCATIONS)
    for arr in (pickup, dropoff):
        assert arr.min() >= 1
        assert arr.max() <= N_LOCATIONS
    assert (pickup != dropoff).all()


def test_od_pairs_distinct_on_tiny_map() -> None:
    # Even with only 2 locations, pickup != dropoff must hold (R4 zero-distance guard).
    pickup, dropoff = sampling.sample_od_pairs(sampling.make_rng(config.SEED), 500, 2)
    assert set(np.unique(pickup).tolist()) <= {1, 2}
    assert (pickup != dropoff).all()


# --- tiers -----------------------------------------------------------------
def test_tiers_deterministic() -> None:
    t1 = sampling.sample_tiers(sampling.make_rng(config.SEED), 1000, TIER_IDS)
    t2 = sampling.sample_tiers(sampling.make_rng(config.SEED), 1000, TIER_IDS)
    assert np.array_equal(t1, t2)


def test_tiers_cover_all_and_stay_in_set() -> None:
    tiers = sampling.sample_tiers(sampling.make_rng(config.SEED), 5000, TIER_IDS)
    assert set(tiers.tolist()) == set(TIER_IDS)


# --- jitter ----------------------------------------------------------------
def test_jitter_deterministic() -> None:
    j1 = sampling.sample_jitter(sampling.make_rng(config.SEED), 1000)
    j2 = sampling.sample_jitter(sampling.make_rng(config.SEED), 1000)
    assert np.array_equal(j1, j2)


def test_jitter_positive_and_centered_near_one() -> None:
    j = sampling.sample_jitter(sampling.make_rng(config.SEED), 20000)
    assert (j > 0).all()
    assert float(np.median(j)) == pytest.approx(1.0, abs=0.05)
