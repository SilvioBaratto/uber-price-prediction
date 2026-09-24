"""Pure domain dataclasses (no I/O).

``Location``, ``RideTier`` and ``Driver`` model the generated dataset; ``Ride`` is represented
directly as DataFrame columns in generation (see tasks/plan.md). ``RideOption`` and
``TripRequest`` are the quote value objects the simulator's application layer produces and
consumes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


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

    tier: str  # id: uberx, green, comfort, xl, black, van
    display_name: str  # on-screen name
    capacity: int  # passenger seats
    base_fare: float  # € base fare
    per_km: float  # €/km distance rate
    per_min: float  # €/min time rate
    booking_fee: float  # € fixed booking fee
    min_fare: float  # € guaranteed minimum price


@dataclass(frozen=True, slots=True)
class Driver:
    """A simulated driver in the year's roster (SPEC §2.1b / §2.6, Rev 2).

    All attributes are **per-driver** (constant across that driver's rides) and none enter
    the price formula — ``driver_id`` is metadata / a weak distractor. ``tenure_start`` and
    ``tenure_end`` are ISO ``YYYY-MM-DD`` date strings; a ``tenure_end`` before Dec 31 means
    the driver churned mid-year. Field order matches the ``drivers.csv`` column order.
    """

    driver_id: int  # key (1-based); FK target of rides.driver_id
    home_district: str  # base district (biases where the driver's rides start)
    activity_class: str  # casual / part_time / full_time
    tenure_start: str  # first active day (ISO date)
    tenure_end: str  # last active day (ISO date); < Dec 31 => churned
    active_days: int  # days actually worked in 2024
    n_rides: int  # total rides produced in 2024
    driver_rating: float  # 1–5 rating (per-driver weak distractor)


@dataclass(frozen=True, slots=True)
class RideOption:
    """One priced service tier offered for a trip (a row in the simulator's quote table).

    A frozen value object produced by the application layer: ``price_eur`` is the model's
    prediction for this tier on the requested trip and ``eta_min`` its estimated duration.
    Ordering is by price so a quote can be presented cheapest-first (``sorted(options)``);
    equality still compares every field.
    """

    tier: str  # tier id (uberx, green, ...); matches RideTier.tier
    display_name: str  # on-screen name
    price_eur: float  # model-predicted fare for this tier, in EUR (> 0)
    eta_min: float  # estimated trip duration, in minutes (>= 0)
    capacity: int  # passenger seats (>= 1)

    def __post_init__(self) -> None:
        if self.price_eur <= 0:
            raise ValueError(f"price_eur must be positive, got {self.price_eur}")
        if self.eta_min < 0:
            raise ValueError(f"eta_min must be non-negative, got {self.eta_min}")
        if self.capacity < 1:
            raise ValueError(f"capacity must be at least 1, got {self.capacity}")

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, RideOption):
            return NotImplemented
        return self.price_eur < other.price_eur


@dataclass(frozen=True, slots=True)
class TripRequest:
    """A request to price a trip: pick-up ``origin`` -> drop-off ``destination`` at ``when``.

    A frozen value object handed to the ``QuoteService``; the two endpoints must differ.
    """

    origin: Location
    destination: Location
    when: datetime

    def __post_init__(self) -> None:
        if self.origin.location_id == self.destination.location_id:
            raise ValueError("origin and destination must be different locations")
