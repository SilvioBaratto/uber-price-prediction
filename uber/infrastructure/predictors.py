"""Price predictors — adapters implementing the ``PricePredictor`` port (infrastructure layer).

``ModelPredictor`` is the production adapter: it *retrains on launch* (M3) — fitting a pinned
poly-OLS model (the Part-5 model, ~R^2 0.88) from the rides data every time — so no model is
persisted. ``FormulaPredictor`` is a ground-truth test double / no-data fallback that prices each
row directly with :func:`uber.domain.pricing.price`. Both consume the quote feature frame the
:class:`~uber.application.quoting.QuoteService` produces.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from uber.domain import config, pricing
from uber.domain.entities import RideTier
from uber.infrastructure import paths
from uber.modeling import pipeline

# Positive floor for model output (EUR): a fare cannot be zero/negative, but a raw regression
# can extrapolate there on degenerate inputs. Deliberately tiny so it never shapes real quotes.
_MIN_FARE_FLOOR_EUR = 0.01


class ModelPredictor:
    """A ``PricePredictor`` that fits a pinned poly-OLS model from rides data at construction.

    Pass a ``rides_path`` (defaults to ``data/raw/rides.csv``) and, optionally, ``max_rows`` to
    cap the training read for responsiveness, or use :meth:`from_frame` when the rides are
    already in memory (tests). ``poly_degree=2`` reproduces the Part-5 surge*distance model.
    """

    def __init__(
        self,
        rides_path: Path | str = paths.RAW_DIR / paths.RIDES_CSV,
        *,
        max_rows: int | None = None,
        poly_degree: int = 2,
        rides: pd.DataFrame | None = None,
    ) -> None:
        frame = rides if rides is not None else pipeline.load_rides(rides_path, max_rows=max_rows)
        X, y = pipeline.make_xy(frame)
        self._model = pipeline.ols_pipeline(poly_degree=poly_degree).fit(X, y)
        self.n_train = int(len(frame))

    @classmethod
    def from_frame(cls, rides: pd.DataFrame, *, poly_degree: int = 2) -> "ModelPredictor":
        """Fit directly from an in-memory rides frame (no file read)."""
        return cls(rides=rides, poly_degree=poly_degree)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        preds = np.asarray(self._model.predict(frame), dtype=float)
        # A fare is positive by definition; OLS can extrapolate below zero on degenerate
        # (near-zero-distance) inputs, so floor the output to keep the port's contract and
        # satisfy RideOption's positive-price invariant.
        return np.maximum(preds, _MIN_FARE_FLOOR_EUR)


class FormulaPredictor:
    """A ``PricePredictor`` test double / no-data fallback using the ground-truth price formula.

    Each row is priced with :func:`uber.domain.pricing.price` from the tier's fare parameters;
    this is the "answer" the ML model reconstructs, so it is handy for tests and as a fallback
    when no rides data is available to train :class:`ModelPredictor`.
    """

    def __init__(self, tiers: list[RideTier] | None = None) -> None:
        catalog = tiers if tiers is not None else list(config.TIERS)
        self._by_id = {tier.tier: tier for tier in catalog}

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        prices = [
            pricing.price(self._by_id[str(tier)], float(distance), float(duration), float(surge))
            for tier, distance, duration, surge in zip(
                frame["tier"], frame["distance_km"],
                frame["duration_min"], frame["surge_multiplier"],
            )
        ]
        return np.asarray(prices, dtype=float)
