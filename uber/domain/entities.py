"""Pure domain dataclasses (no I/O).

``Location``, ``RideTier`` and ``Driver`` are defined; ``Ride`` is represented directly as
DataFrame columns in generation (see tasks/plan.md).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Location:
    """A Madrid street (vial) with its representative point and administrative tags.

    ``lat``/``lon`` are the median coordinate of the street's official addresses;
    ``neighborhood`` (barrio) and ``district`` (distrito) are its modal administrative area.
    """

    location_id: int
    street: str
    neighborhood: str
    district: str
    lat: float
    lon: float


@dataclass(frozen=True, slots=True)
class RideTier:
    """A service tier and its fare parameters (SPEC §2.2).

    Tariffs are plausible but invented (EUR, distances in km, time in minutes). The
    catalog in ``config.TIERS`` lists the tiers in increasing-price order. Field order
    matches the ``ride_tiers.csv`` column order.
    """

    tier: str          # id: uberx, green, comfort, xl, black, van
    display_name: str  # on-screen name
    capacity: int      # passenger seats
    base_fare: float   # € base fare
    per_km: float      # €/km distance rate
    per_min: float     # €/min time rate
    booking_fee: float # € fixed booking fee
    min_fare: float    # € guaranteed minimum price


@dataclass(frozen=True, slots=True)
class Driver:
    """A simulated driver in the year's roster (SPEC §2.1b / §2.6, Rev 2).

    All attributes are **per-driver** (constant across that driver's rides) and none enter
    the price formula — ``driver_id`` is metadata / a weak distractor. ``tenure_start`` and
    ``tenure_end`` are ISO ``YYYY-MM-DD`` date strings; a ``tenure_end`` before Dec 31 means
    the driver churned mid-year. Field order matches the ``drivers.csv`` column order.
    """

    driver_id: int          # key (1-based); FK target of rides.driver_id
    home_district: str      # base district (biases where the driver's rides start)
    activity_class: str     # casual / part_time / full_time
    tenure_start: str       # first active day (ISO date)
    tenure_end: str         # last active day (ISO date); < Dec 31 => churned
    active_days: int        # days actually worked in 2024
    n_rides: int            # total rides produced in 2024
    driver_rating: float    # 1–5 rating (per-driver weak distractor)
