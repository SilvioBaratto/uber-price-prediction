"""QuoteService — the price-a-trip use case (application layer).

Given a trip (origin -> destination at a time), build one feature row per service tier, ask the
injected :class:`~uber.application.ports.PricePredictor` for all tier prices in a single call,
and return a :class:`~uber.domain.entities.RideOption` per tier sorted cheapest-first. Distance,
duration and surge come from the pure domain pricing; the fare itself is the model's prediction
(never the ground-truth formula). Depends only on the domain and the ports — no I/O here.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from uber.application.ports import Clock, LocationRepository, PricePredictor, TierRepository
from uber.domain import pricing
from uber.domain.entities import Location, RideOption, TripRequest

# The pinned feature columns the predictor consumes (the model's data contract). Kept here because
# producing this row is the use case's responsibility; the concrete predictor selects by name.
FEATURE_COLUMNS = [
    "tier",
    "pickup_district",
    "distance_km",
    "duration_min",
    "surge_multiplier",
    "hour",
    "day_of_week",
    "month",
]


class QuoteService:
    """Turn a trip request into a sorted list of priced ride options."""

    def __init__(
        self,
        locations: LocationRepository,
        tiers: TierRepository,
        predictor: PricePredictor,
        clock: Clock | None = None,
    ) -> None:
        self._locations = locations
        self._tiers = tiers
        self._predictor = predictor
        self._clock = clock

    def quote(
        self, origin: Location, destination: Location, when: datetime | None = None
    ) -> list[RideOption]:
        """Price every tier for ``origin -> destination`` at ``when`` (defaults to ``clock.now()``).

        Returns one :class:`RideOption` per tier, sorted by ascending price.
        """
        if when is None:
            if self._clock is None:
                raise ValueError("no time supplied and no clock injected")
            when = self._clock.now()

        request = TripRequest(origin, destination, when)  # validates distinct endpoints
        hour, dow, month = request.when.hour, request.when.weekday(), request.when.month

        distance_km = pricing.road_distance(
            request.origin.lat,
            request.origin.lon,
            request.destination.lat,
            request.destination.lon,
        )
        duration = pricing.duration_min(distance_km, hour)
        surge_multiplier = pricing.surge(hour, dow)

        tiers = self._tiers.all()
        frame = pd.DataFrame(
            {
                "tier": [t.tier for t in tiers],
                "pickup_district": [request.origin.district] * len(tiers),
                "distance_km": distance_km,
                "duration_min": duration,
                "surge_multiplier": surge_multiplier,
                "hour": hour,
                "day_of_week": dow,
                "month": month,
            },
            columns=FEATURE_COLUMNS,
        )

        prices = self._predictor.predict(frame)
        options = [
            RideOption(t.tier, t.display_name, float(p), float(duration), t.capacity)
            for t, p in zip(tiers, prices)
        ]
        return sorted(options)
