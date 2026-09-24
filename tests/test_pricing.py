"""Unit tests for the pure pricing + geography functions (SPEC §2.4 / §2.5).

No I/O: these exercise ``uber.domain.pricing`` in isolation. dow uses datetime.weekday()
convention (0=Mon .. 6=Sun).
"""

from __future__ import annotations

import numpy as np
import pytest

from uber.domain import config, pricing

MON, TUE, WED, THU, FRI, SAT, SUN = range(7)


# --- haversine -------------------------------------------------------------
def test_haversine_zero_distance() -> None:
    assert pricing.haversine(40.4169, -3.7029, 40.4169, -3.7029) == 0.0


def test_haversine_symmetric() -> None:
    a = pricing.haversine(40.4169, -3.7029, 40.4066, -3.6907)
    b = pricing.haversine(40.4066, -3.6907, 40.4169, -3.7029)
    assert a == pytest.approx(b)


def test_haversine_one_degree_of_latitude() -> None:
    # One degree of latitude is ~111.19 km anywhere on the sphere (known reference).
    d = pricing.haversine(40.0, -3.7, 41.0, -3.7)
    assert d == pytest.approx(111.19, rel=0.02)


def test_haversine_one_degree_of_longitude_at_madrid() -> None:
    # One degree of longitude at Madrid's latitude (~40.4 N) is ~84.6 km.
    d = pricing.haversine(40.4, -3.7, 40.4, -2.7)
    assert d == pytest.approx(84.6, rel=0.02)


def test_haversine_known_madrid_pair() -> None:
    # Puerta del Sol -> Atocha station, ~1.5 km straight line (coordinate-dependent).
    d = pricing.haversine(40.4169, -3.7029, 40.4066, -3.6907)
    assert d == pytest.approx(1.5, rel=0.1)


# --- road_distance ---------------------------------------------------------
def test_road_distance_applies_factor() -> None:
    args = (40.4169, -3.7029, 40.4066, -3.6907)
    assert pricing.road_distance(*args) == pytest.approx(
        pricing.haversine(*args) * config.ROAD_FACTOR
    )


# --- speed / duration ------------------------------------------------------
def test_speed_positive_for_every_hour() -> None:
    for hour in range(24):
        assert pricing.speed_kmh(hour) > 0


def test_speed_rush_slower_than_overnight() -> None:
    assert pricing.speed_kmh(8) < pricing.speed_kmh(3)   # morning rush
    assert pricing.speed_kmh(19) < pricing.speed_kmh(3)  # evening rush


def test_duration_min_matches_speed() -> None:
    assert pricing.duration_min(30.0, 3) == pytest.approx(30.0 / pricing.speed_kmh(3) * 60.0)


def test_duration_min_scales_with_distance_and_congestion() -> None:
    assert pricing.duration_min(20.0, 3) > pricing.duration_min(10.0, 3)  # farther
    assert pricing.duration_min(10.0, 8) > pricing.duration_min(10.0, 3)  # slower hour


# --- demand profile / surge ------------------------------------------------
def test_weekend_night_is_the_expensive_case() -> None:
    # Sat 02:00 and Fri 22:00 must beat a weekday late morning (Wed 11:00).
    weekday_late_morning = pricing.demand_profile(11, WED)
    assert pricing.demand_profile(2, SAT) > weekday_late_morning
    assert pricing.demand_profile(22, FRI) > weekday_late_morning


def test_weekday_rush_above_offpeak() -> None:
    offpeak = pricing.demand_profile(11, WED)
    assert pricing.demand_profile(8, WED) > offpeak    # morning rush
    assert pricing.demand_profile(19, WED) > offpeak   # evening rush


def test_early_friday_morning_is_not_weekend_night() -> None:
    # Fri 02:00 is Thursday-night carryover (weekday late night), not a Fri/Sat peak.
    assert pricing.demand_profile(2, FRI) < pricing.demand_profile(2, SAT)


def test_surge_clamped_to_bounds() -> None:
    lo, hi = config.SURGE_CLAMP
    assert pricing.surge(2, SAT, jitter=10.0) == pytest.approx(hi)     # clamp up
    assert pricing.surge(11, WED, jitter=0.01) == pytest.approx(lo)    # clamp down


def test_surge_default_jitter_equals_profile() -> None:
    assert pricing.surge(8, WED, jitter=1.0) == pytest.approx(pricing.demand_profile(8, WED))


# --- price formula (SPEC §2.4) --------------------------------------------
def test_price_matches_formula() -> None:
    tier = config.TIERS[0]  # uberx
    distance, duration, surge_mult = 10.0, 20.0, 1.0
    ride_base = tier.base_fare + tier.per_km * distance + tier.per_min * duration
    expected = ride_base * surge_mult + tier.booking_fee
    assert pricing.price(tier, distance, duration, surge_mult, noise=0.0) == pytest.approx(expected)


def test_price_never_below_min_fare_at_zero_trip() -> None:
    tier = config.TIERS[0]
    assert pricing.price(tier, 0.0, 0.0, 1.0, noise=0.0) == pytest.approx(tier.min_fare)


def test_price_floored_by_min_fare_with_large_negative_noise() -> None:
    tier = config.TIERS[0]
    assert pricing.price(tier, 10.0, 20.0, 1.0, noise=-1000.0) == pytest.approx(tier.min_fare)


def test_price_rises_with_surge() -> None:
    tier = config.TIERS[0]
    low = pricing.price(tier, 10.0, 20.0, 1.0, noise=0.0)
    high = pricing.price(tier, 10.0, 20.0, 2.0, noise=0.0)
    assert high > low


# --- vectorized == scalar (the generation path reuses these; guard against drift) ----------
def test_haversine_array_matches_scalar() -> None:
    rng = np.random.default_rng(0)
    lat1, lon1 = rng.uniform(40.31, 40.65, 200), rng.uniform(-3.84, -3.52, 200)
    lat2, lon2 = rng.uniform(40.31, 40.65, 200), rng.uniform(-3.84, -3.52, 200)
    vec = pricing.haversine(lat1, lon1, lat2, lon2)
    scalar = [pricing.haversine(*p) for p in zip(lat1, lon1, lat2, lon2)]
    assert np.allclose(vec, scalar)


def test_duration_min_array_matches_scalar() -> None:
    rng = np.random.default_rng(1)
    dist = rng.uniform(0.5, 30.0, 200)
    hour = rng.integers(0, 24, 200)
    vec = pricing.duration_min_array(dist, hour)
    scalar = [pricing.duration_min(d, int(h)) for d, h in zip(dist, hour)]
    assert np.allclose(vec, scalar)


def test_demand_table_matches_scalar_profile() -> None:
    table = pricing.demand_table()
    assert table.shape == (7, 24)
    for d in range(7):
        for h in range(24):
            assert table[d, h] == pytest.approx(pricing.demand_profile(h, d))


def test_surge_array_matches_scalar() -> None:
    rng = np.random.default_rng(2)
    hour = rng.integers(0, 24, 300)
    dow = rng.integers(0, 7, 300)
    jitter = rng.lognormal(0.0, config.JITTER_SIGMA, 300)
    vec = pricing.surge_array(hour, dow, jitter)
    scalar = [pricing.surge(int(h), int(d), float(j)) for h, d, j in zip(hour, dow, jitter)]
    assert np.allclose(vec, scalar)


def test_price_array_matches_scalar() -> None:
    rng = np.random.default_rng(3)
    n = 300
    tiers = [config.TIERS[i] for i in rng.integers(0, len(config.TIERS), n)]
    dist = rng.uniform(0.0, 30.0, n)
    dur = rng.uniform(0.0, 60.0, n)
    surge = rng.uniform(1.0, 3.0, n)
    noise = rng.normal(0.0, 3.0, n)
    vec = pricing.price_array(
        np.array([t.base_fare for t in tiers]),
        np.array([t.per_km for t in tiers]),
        np.array([t.per_min for t in tiers]),
        np.array([t.booking_fee for t in tiers]),
        np.array([t.min_fare for t in tiers]),
        dist, dur, surge, noise,
    )
    scalar = [pricing.price(t, d, u, s, float(nz)) for t, d, u, s, nz in zip(tiers, dist, dur, surge, noise)]
    assert np.allclose(vec, scalar)
