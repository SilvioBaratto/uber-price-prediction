"""T2.3 — the QuoteService use case (application layer).

Given an origin, a destination and a time, QuoteService builds one feature row per tier, calls
the injected PricePredictor exactly once, and returns a RideOption per tier sorted cheapest
first. Distance/eta come from the pure domain pricing, so they are asserted against it directly.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from uber.application.quoting import QuoteService
from uber.domain import pricing
from uber.domain.entities import Location, RideOption, RideTier

ORIGIN = Location(1, "Plaza de la Puerta del Sol", "Sol", "Centro", 40.4169, -3.7029)
DEST = Location(2, "Calle de Alberto Alcocer", "Nueva España", "Chamartín", 40.4620, -3.6790)
WHEN = datetime(2024, 6, 15, 19, 30)

TIERS = [
    RideTier("uberx", "UberX", 4, 1.20, 0.90, 0.20, 1.0, 5.0),
    RideTier("comfort", "Comfort", 4, 2.00, 1.20, 0.30, 1.5, 8.0),
    RideTier("black", "Uber Black", 4, 3.50, 1.80, 0.45, 2.5, 12.0),
]


class FakeTierRepo:
    def all(self) -> list[RideTier]:
        return list(TIERS)


class FakeLocationRepo:
    def all(self) -> list[Location]:
        return [ORIGIN, DEST]

    def get(self, location_id: int) -> Location:
        return {1: ORIGIN, 2: DEST}[location_id]

    def search(self, query: str) -> list[Location]:
        return [ORIGIN, DEST]


class RecordingPredictor:
    """Returns a fixed price per tier (descending) and records how many times it was called."""

    def __init__(self) -> None:
        self.calls = 0
        self.last_frame: pd.DataFrame | None = None

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        self.calls += 1
        self.last_frame = frame.copy()
        # uberx=30, comfort=20, black=10 -> deliberately NOT sorted ascending by tier order
        prices = {"uberx": 30.0, "comfort": 20.0, "black": 10.0}
        return np.array([prices[t] for t in frame["tier"]], dtype=float)


def _service(predictor: RecordingPredictor) -> QuoteService:
    return QuoteService(FakeLocationRepo(), FakeTierRepo(), predictor)


def test_quote_returns_one_option_per_tier() -> None:
    options = _service(RecordingPredictor()).quote(ORIGIN, DEST, WHEN)
    assert len(options) == len(TIERS)
    assert all(isinstance(o, RideOption) for o in options)
    assert {o.tier for o in options} == {"uberx", "comfort", "black"}


def test_quote_calls_predictor_exactly_once() -> None:
    pred = RecordingPredictor()
    _service(pred).quote(ORIGIN, DEST, WHEN)
    assert pred.calls == 1


def test_quote_sorted_ascending_by_price() -> None:
    options = _service(RecordingPredictor()).quote(ORIGIN, DEST, WHEN)
    assert [o.tier for o in options] == ["black", "comfort", "uberx"]
    prices = [o.price_eur for o in options]
    assert prices == sorted(prices)


def test_quote_distance_and_eta_match_domain_pricing() -> None:
    options = _service(RecordingPredictor()).quote(ORIGIN, DEST, WHEN)
    expected_dist = pricing.road_distance(ORIGIN.lat, ORIGIN.lon, DEST.lat, DEST.lon)
    expected_eta = pricing.duration_min(expected_dist, WHEN.hour)
    for o in options:
        assert o.eta_min == pytest.approx(expected_eta)


def test_quote_feature_frame_has_pinned_columns_and_time_fields() -> None:
    pred = RecordingPredictor()
    _service(pred).quote(ORIGIN, DEST, WHEN)
    frame = pred.last_frame
    assert frame is not None
    for col in (
        "distance_km",
        "duration_min",
        "surge_multiplier",
        "hour",
        "day_of_week",
        "month",
        "tier",
        "pickup_district",
    ):
        assert col in frame.columns
    assert (frame["hour"] == WHEN.hour).all()
    assert (frame["day_of_week"] == WHEN.weekday()).all()
    assert (frame["month"] == WHEN.month).all()
    assert (frame["pickup_district"] == ORIGIN.district).all()
    expected_surge = pricing.surge(WHEN.hour, WHEN.weekday())
    assert frame["surge_multiplier"].to_numpy() == pytest.approx(expected_surge)


def test_quote_capacity_and_display_name_come_from_tier() -> None:
    options = {o.tier: o for o in _service(RecordingPredictor()).quote(ORIGIN, DEST, WHEN)}
    assert options["black"].display_name == "Uber Black"
    assert options["black"].capacity == 4


def test_quote_rejects_identical_endpoints() -> None:
    with pytest.raises(ValueError):
        _service(RecordingPredictor()).quote(ORIGIN, ORIGIN, WHEN)
