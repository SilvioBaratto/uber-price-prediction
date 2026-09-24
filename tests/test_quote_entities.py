"""T2.1 — the pure quote entities ``RideOption`` and ``TripRequest`` (domain layer).

These are I/O-free value objects the application layer produces (``RideOption``) and consumes
(``TripRequest``). They must be frozen (hashable value objects), validate their invariants at
construction, and ``RideOption`` must sort by price so a quote can be rendered cheapest-first.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime

import pytest

from uber.domain.entities import Location, RideOption, TripRequest

MADRID = Location(1, "Plaza de la Puerta del Sol", "Sol", "Centro", 40.4169, -3.7029)
CHAMARTIN = Location(2, "Calle de Alberto Alcocer", "Nueva España", "Chamartín", 40.4620, -3.6790)
WHEN = datetime(2024, 6, 15, 19, 30)


# --- RideOption ------------------------------------------------------------
def test_ride_option_constructs_and_exposes_fields() -> None:
    opt = RideOption(tier="uberx", display_name="UberX", price_eur=9.40, eta_min=12.5, capacity=4)
    assert opt.tier == "uberx"
    assert opt.display_name == "UberX"
    assert opt.price_eur == pytest.approx(9.40)
    assert opt.eta_min == pytest.approx(12.5)
    assert opt.capacity == 4


def test_ride_option_is_frozen() -> None:
    opt = RideOption("uberx", "UberX", 9.40, 12.5, 4)
    with pytest.raises(FrozenInstanceError):
        opt.price_eur = 1.0  # type: ignore[misc]


def test_ride_option_is_hashable() -> None:
    opt = RideOption("uberx", "UberX", 9.40, 12.5, 4)
    assert opt in {opt}


def test_ride_options_sort_by_price_ascending() -> None:
    black = RideOption("black", "Uber Black", 22.10, 11.0, 4)
    uberx = RideOption("uberx", "UberX", 9.40, 12.5, 4)
    comfort = RideOption("comfort", "Comfort", 14.30, 12.0, 4)
    assert [o.tier for o in sorted([black, uberx, comfort])] == ["uberx", "comfort", "black"]


@pytest.mark.parametrize(
    "price,eta,cap",
    [(-1.0, 12.0, 4), (0.0, 12.0, 4), (9.4, -1.0, 4), (9.4, 12.0, 0)],
)
def test_ride_option_rejects_invalid_values(price: float, eta: float, cap: int) -> None:
    with pytest.raises(ValueError):
        RideOption("uberx", "UberX", price, eta, cap)


# --- TripRequest -----------------------------------------------------------
def test_trip_request_constructs_and_exposes_fields() -> None:
    req = TripRequest(origin=MADRID, destination=CHAMARTIN, when=WHEN)
    assert req.origin is MADRID
    assert req.destination is CHAMARTIN
    assert req.when == WHEN


def test_trip_request_is_frozen() -> None:
    req = TripRequest(MADRID, CHAMARTIN, WHEN)
    with pytest.raises(FrozenInstanceError):
        req.when = WHEN  # type: ignore[misc]


def test_trip_request_rejects_same_origin_and_destination() -> None:
    with pytest.raises(ValueError):
        TripRequest(MADRID, MADRID, WHEN)
