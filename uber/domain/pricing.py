"""Pure pricing + geography functions (no I/O).

Implements the documented ground-truth price formula and the temporal demand
profile — the "answer" the ML models must reconstruct. Every function is pure;
tunable constants live in ``uber.domain.config``. ``dow`` follows ``datetime.weekday()``:
0=Mon .. 6=Sun.
"""

from __future__ import annotations

import numpy as np

from uber.domain import config
from uber.domain.entities import RideTier

# Mean Earth radius (km) — the haversine reference sphere (WGS84 mean radius).
EARTH_RADIUS_KM = 6371.0088


# --- Geography -------------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two WGS84 points.

    Written with numpy so it accepts scalars *or* aligned arrays (the vectorized generation
    path passes whole coordinate columns); scalar callers get a float-like ``np.float64``.
    """
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def road_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Approximate road distance (km): straight-line haversine × ``ROAD_FACTOR``."""
    return haversine(lat1, lon1, lat2, lon2) * config.ROAD_FACTOR


def speed_kmh(hour: int) -> float:
    """Average road speed (km/h) for the given hour of day (0..23)."""
    return config.SPEED_KMH_BY_HOUR[hour]


def duration_min(distance_km: float, hour: int) -> float:
    """Trip duration in minutes from distance and the hour's average speed."""
    return distance_km / speed_kmh(hour) * 60.0


# Speed-by-hour as an array for vectorized lookup (same values as ``speed_kmh``).
_SPEED_ARR = np.asarray(config.SPEED_KMH_BY_HOUR, dtype=float)


def duration_min_array(distance_km: np.ndarray, hour: np.ndarray) -> np.ndarray:
    """Vectorized :func:`duration_min` over aligned distance/hour arrays."""
    return distance_km / _SPEED_ARR[hour] * 60.0


# --- Temporal demand / surge -----------------------------------
def _is_weekend_night(hour: int, dow: int) -> bool:
    """True inside the Fri & Sat night window (21:00–03:59).

    A "Friday night" runs Fri 21:00 → Sat 03:59; a "Saturday night" runs Sat 21:00 →
    Sun 03:59. So the small hours of Sat and Sun still count as the previous evening's peak,
    while the small hours of Fri (Thursday-night carryover) do not.
    """
    late = hour >= 21  # 21, 22, 23
    early = hour <= 3  # 00, 01, 02, 03
    if dow == 4 and late:  # Fri evening
        return True
    if dow == 5 and (early or late):  # Sat small hours + Sat evening
        return True
    if dow == 6 and early:  # Sun small hours
        return True
    return False


def demand_profile(hour: int, dow: int) -> float:
    """Deterministic base surge for a (hour, dow) bucket, before jitter."""
    if _is_weekend_night(hour, dow):
        return config.SURGE_WEEKEND_NIGHT
    if dow in (5, 6):  # remaining Sat/Sun hours = weekend daytime
        return config.SURGE_WEEKEND_DAY
    # Weekdays (Mon–Fri)
    if hour <= 5:
        return config.SURGE_LATE_NIGHT
    if 7 <= hour <= 9:
        return config.SURGE_MORNING_RUSH
    if 18 <= hour <= 20:
        return config.SURGE_EVENING_RUSH
    return config.SURGE_OFFPEAK


def surge(hour: int, dow: int, jitter: float = 1.0) -> float:
    """Surge multiplier: ``demand_profile(hour, dow) × jitter`` clamped to ``SURGE_CLAMP``."""
    lo, hi = config.SURGE_CLAMP
    return min(hi, max(lo, demand_profile(hour, dow) * jitter))


_DEMAND_TABLE: np.ndarray | None = None


def demand_table() -> np.ndarray:
    """7×24 base-surge lookup ``[dow, hour]`` built once from the scalar :func:`demand_profile`.

    Vectorized surge indexes into this table instead of re-deriving the bucket logic, so the
    array path and the scalar ground truth cannot drift apart.
    """
    global _DEMAND_TABLE
    if _DEMAND_TABLE is None:
        table = np.empty((7, 24), dtype=float)
        for d in range(7):
            for h in range(24):
                table[d, h] = demand_profile(h, d)
        _DEMAND_TABLE = table
    return _DEMAND_TABLE


def surge_array(hour: np.ndarray, dow: np.ndarray, jitter: np.ndarray) -> np.ndarray:
    """Vectorized :func:`surge` over aligned hour/dow/jitter arrays."""
    lo, hi = config.SURGE_CLAMP
    base = demand_table()[dow, hour]
    return np.clip(base * jitter, lo, hi)


# --- Price formula ---------------------------------------------
def price(
    tier: RideTier,
    distance_km: float,
    duration_min: float,
    surge_multiplier: float,
    noise: float = 0.0,
) -> float:
    """Ground-truth ride price in EUR::

        ride_base = base_fare + per_km * distance_km + per_min * duration_min
        price_raw = ride_base * surge_multiplier + booking_fee
        price_eur = max(min_fare, price_raw + noise)

    The weak distractors and ``commission_eur`` do not enter this formula.
    """
    ride_base = tier.base_fare + tier.per_km * distance_km + tier.per_min * duration_min
    price_raw = ride_base * surge_multiplier + tier.booking_fee
    return max(tier.min_fare, price_raw + noise)


def price_array(
    base_fare: np.ndarray,
    per_km: np.ndarray,
    per_min: np.ndarray,
    booking_fee: np.ndarray,
    min_fare: np.ndarray,
    distance_km: np.ndarray,
    duration_min: np.ndarray,
    surge_multiplier: np.ndarray,
    noise: np.ndarray,
) -> np.ndarray:
    """Vectorized :func:`price` — identical formula over per-ride tariff/feature arrays.

    The tariff arrays (``base_fare`` … ``min_fare``) are the per-ride tier parameters; keeping
    the arithmetic byte-for-byte the same as :func:`price` is what makes the two paths agree.
    """
    ride_base = base_fare + per_km * distance_km + per_min * duration_min
    price_raw = ride_base * surge_multiplier + booking_fee
    return np.maximum(min_fare, price_raw + noise)
