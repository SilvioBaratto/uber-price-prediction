"""The application ports (interfaces the use cases depend on).

The ports are ``@runtime_checkable`` Protocols so a concrete adapter (or a test fake) is
recognised structurally: any object with the right methods satisfies ``isinstance``. This keeps
the application layer depending on abstractions, with adapters injected at the composition edge.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from uber.application.ports import Clock, LocationRepository, PricePredictor, TierRepository
from uber.domain.entities import Location, RideTier


class FakeLocationRepo:
    def all(self) -> list[Location]:
        return []

    def get(self, location_id: int) -> Location:
        return Location(location_id, "s", "n", "d", 0.0, 0.0)

    def search(self, query: str) -> list[Location]:
        return []


class FakeTierRepo:
    def all(self) -> list[RideTier]:
        return []


class FakePredictor:
    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(frame))


class FakeClock:
    def now(self) -> datetime:
        return datetime(2024, 1, 1)


def test_fakes_satisfy_ports() -> None:
    assert isinstance(FakeLocationRepo(), LocationRepository)
    assert isinstance(FakeTierRepo(), TierRepository)
    assert isinstance(FakePredictor(), PricePredictor)
    assert isinstance(FakeClock(), Clock)


def test_missing_method_fails_port_check() -> None:
    class NotARepo:
        def all(self) -> list[Location]:
            return []

    # missing get/search -> not a LocationRepository
    assert not isinstance(NotARepo(), LocationRepository)


def test_ports_are_runtime_checkable_protocols() -> None:
    # a bare object satisfies none of them
    for port in (LocationRepository, TierRepository, PricePredictor, Clock):
        assert not isinstance(object(), port)
