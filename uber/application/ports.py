"""Application ports — the interfaces the use cases depend on (dependency inversion).

Each port is a ``@runtime_checkable`` :class:`typing.Protocol`, so concrete adapters in
:mod:`uber.infrastructure` (and test fakes) satisfy them structurally without importing this
module. The application layer programs against these abstractions; the composition root
(:mod:`uber.cli`) wires in the real implementations.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from uber.domain.entities import Location, RideTier


@runtime_checkable
class LocationRepository(Protocol):
    """Read access to the Madrid locations catalog."""

    def all(self) -> list[Location]:
        """Every known location."""
        ...

    def get(self, location_id: int) -> Location:
        """The location with this 1-based id (raises ``KeyError`` if absent)."""
        ...

    def search(self, query: str) -> list[Location]:
        """Locations whose street name matches ``query`` (case-insensitive substring)."""
        ...


@runtime_checkable
class TierRepository(Protocol):
    """Read access to the ride-tier catalog."""

    def all(self) -> list[RideTier]:
        """Every service tier, in catalog (increasing-price) order."""
        ...


@runtime_checkable
class PricePredictor(Protocol):
    """A fitted price model: map a feature frame to predicted fares (EUR)."""

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict ``price_eur`` for each row of ``frame``."""
        ...


@runtime_checkable
class Clock(Protocol):
    """A source of the current time (injected so quoting is testable)."""

    def now(self) -> datetime:
        """The current local time."""
        ...
