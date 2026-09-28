"""Price predictors — adapters implementing the ``PricePredictor`` port (infrastructure layer).

``ModelPredictor`` is the production adapter: it fits a **tier-interaction poly-OLS** model
(:func:`~uber.modeling.pipeline.interaction_ols_pipeline`, ~R^2 0.98) so every tier gets its own
distance/duration/surge slopes, then **clamps each quote up to that tier's ``min_fare``** (mirrors
the ground-truth ``max(min_fare, …)``). It can either *retrain on launch* (construct from a rides
path) or be **persisted once and reloaded instantly** via :meth:`save` / :meth:`load` — the latter
is what ``uber train`` + ``uber simulate --model`` use to avoid refitting on every start.
``FormulaPredictor`` is a ground-truth test double / no-data fallback that prices each row directly
with :func:`uber.domain.pricing.price`. Both consume the quote feature frame the
:class:`~uber.application.quoting.QuoteService` produces.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from uber.domain import config, pricing
from uber.domain.entities import RideTier
from uber.infrastructure import paths
from uber.modeling import pipeline

# Positive floor for model output (EUR): used only as a fallback for tiers absent from the tariff
# catalog. A fare cannot be zero/negative, but a raw regression can extrapolate there; deliberately
# tiny so it never shapes real quotes for known tiers (those clamp to their own min_fare below).
_MIN_FARE_FLOOR_EUR = 0.01

# Per-tier price floors (EUR) from the tariff catalog. Applied to every quote so a prediction never
# falls below the tier's guaranteed minimum — the serve-time analogue of the ground-truth
# ``max(min_fare, …)`` in :func:`uber.domain.pricing.price`.
_MIN_FARE_BY_TIER: dict[str, float] = {tier.tier: tier.min_fare for tier in config.TIERS}


class ModelPredictor:
    """A ``PricePredictor`` that fits a tier-interaction poly-OLS model from rides at construction.

    Pass a ``rides_path`` (defaults to ``data/raw/rides.csv``) and, optionally, ``max_rows`` to
    cap the training read (responsiveness *and* memory — the interaction design is wide), or use
    :meth:`from_frame` when the rides are already in memory (tests). ``poly_degree=2`` is the
    tier×feature interaction model (:func:`~uber.modeling.pipeline.interaction_ols_pipeline`).
    """

    def __init__(
        self,
        rides_path: Path | str = paths.RAW_DIR / paths.RIDES_CSV,
        *,
        max_rows: int | None = None,
        poly_degree: int = 2,
        seed: int = config.SEED,
        rides: pd.DataFrame | None = None,
    ) -> None:
        frame = rides if rides is not None else pipeline.load_rides(rides_path)
        if max_rows is not None and len(frame) > max_rows:
            # A seeded RANDOM sample, never the first N rows: rides.csv is sorted chronologically,
            # so a head-slice would train on early-year months only (``month`` near-constant) and
            # skew every quote. Sampling keeps the training slice representative.
            frame = frame.sample(max_rows, random_state=seed).reset_index(drop=True)
        X, y = pipeline.make_xy(frame)
        self._model = pipeline.interaction_ols_pipeline(poly_degree=poly_degree).fit(X, y)
        self.n_train = int(len(frame))
        self._poly_degree = int(poly_degree)

    @classmethod
    def from_frame(cls, rides: pd.DataFrame, *, poly_degree: int = 2) -> ModelPredictor:
        """Fit directly from an in-memory rides frame (no file read)."""
        return cls(rides=rides, poly_degree=poly_degree)

    def save(self, path: Path | str = paths.MODELS_DIR / paths.MODEL_JOBLIB) -> Path:
        """Persist the fitted model (+ metadata) to ``path`` with joblib; return the path.

        Creates the parent directory if needed. The payload bundles the fitted sklearn pipeline
        with ``n_train`` and ``poly_degree`` so :meth:`load` restores a fully-formed predictor.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self._model,
            "n_train": self.n_train,
            "poly_degree": self._poly_degree,
        }
        joblib.dump(payload, path)
        return path

    @classmethod
    def load(cls, path: Path | str = paths.MODELS_DIR / paths.MODEL_JOBLIB) -> ModelPredictor:
        """Reconstruct a predictor from weights saved by :meth:`save` (no refitting).

        Bypasses ``__init__`` (which would refit): the fitted pipeline and its metadata are read
        straight from the joblib payload, so loading is near-instant regardless of dataset size.
        """
        payload = joblib.load(Path(path))
        predictor = cls.__new__(cls)
        predictor._model = payload["model"]
        predictor.n_train = int(payload["n_train"])
        predictor._poly_degree = int(payload.get("poly_degree", 2))
        return predictor

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        preds = np.asarray(self._model.predict(frame), dtype=float)
        # Clamp each quote UP to its tier's min_fare (the serve-time analogue of the ground-truth
        # ``max(min_fare, …)``): OLS can still dip just under the floor near-zero distance, and the
        # floor also guards the RideOption positive-price invariant. Unknown tiers (not in the
        # catalog) fall back to the tiny positivity floor so a fare is never zero/negative.
        floors = (
            frame["tier"]
            .map(_MIN_FARE_BY_TIER.get)
            .fillna(_MIN_FARE_FLOOR_EUR)
            .to_numpy(dtype=float)
        )
        return np.maximum(preds, floors)


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
                frame["tier"],
                frame["distance_km"],
                frame["duration_min"],
                frame["surge_multiplier"],
            )
        ]
        return np.asarray(prices, dtype=float)
