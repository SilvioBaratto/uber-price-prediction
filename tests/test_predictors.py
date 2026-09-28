"""The price predictors (infrastructure layer).

``ModelPredictor`` retrains a pinned poly-OLS model on launch from rides data and predicts fares
for a quote's feature frame; ``FormulaPredictor`` is a ground-truth test double / no-data
fallback built on ``domain.pricing.price``. Both satisfy the ``PricePredictor`` port.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import r2_score

from uber.application.ports import PricePredictor
from uber.datagen import generate_data, sampling
from uber.domain import config, pricing
from uber.infrastructure import repositories
from uber.infrastructure.predictors import (
    _MIN_FARE_FLOOR_EUR,
    FormulaPredictor,
    ModelPredictor,
)

PINNED = [
    "tier",
    "pickup_district",
    "distance_km",
    "duration_min",
    "surge_multiplier",
    "hour",
    "day_of_week",
    "month",
]


@pytest.fixture(scope="module")
def full_rides() -> pd.DataFrame:
    """The full-year, chronologically-sorted rides frame (spans all 12 months)."""
    locations = repositories.build_locations_df()
    _drivers, rides = generate_data.build_dataset(sampling.make_rng(config.SEED), locations, 250)
    return rides


@pytest.fixture(scope="module")
def rides(full_rides: pd.DataFrame) -> pd.DataFrame:
    return full_rides.sample(8000, random_state=config.SEED).reset_index(drop=True)


# --- ModelPredictor --------------------------------------------------------
def test_model_predictor_satisfies_port(rides: pd.DataFrame) -> None:
    predictor = ModelPredictor.from_frame(rides.iloc[:2000])
    assert isinstance(predictor, PricePredictor)


def test_model_predictor_predicts_positive_prices(rides: pd.DataFrame) -> None:
    predictor = ModelPredictor.from_frame(rides.iloc[:6000])
    preds = predictor.predict(rides.iloc[6000:][PINNED])
    assert len(preds) == len(rides) - 6000
    assert (preds > 0).all()


def test_model_predictor_r2_in_loose_band(rides: pd.DataFrame) -> None:
    train, test = rides.iloc[:6400], rides.iloc[6400:]
    predictor = ModelPredictor.from_frame(train)
    preds = predictor.predict(test[PINNED])
    r2 = r2_score(test["price_eur"].to_numpy(dtype=float), preds)
    # pinned poly-deg2 tracks the ground truth closely; keep a generous band for the small sample
    assert 0.70 < r2 < 0.99


def test_model_predictor_max_rows_caps_training(rides: pd.DataFrame, tmp_path) -> None:
    path = tmp_path / "rides.csv"
    rides.to_csv(path, index=False)
    predictor = ModelPredictor(path, max_rows=1000)
    assert predictor.n_train == 1000


def test_model_predictor_max_rows_samples_representatively(full_rides: pd.DataFrame) -> None:
    # `full_rides` is chronological, so a head-slice cap would train on early months only.
    # The true fare is month-independent, so a representative sample must price a given trip
    # ~identically across months; a chronological head-slice would extrapolate wildly.
    cap = min(6000, len(full_rides) // 2)
    predictor = ModelPredictor(rides=full_rides, max_rows=cap, seed=config.SEED)
    assert predictor.n_train == cap

    def trip(month: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "tier": ["uberx"],
                "pickup_district": ["Centro"],
                "distance_km": [8.0],
                "duration_min": [20.0],
                "surge_multiplier": [1.2],
                "hour": [19],
                "day_of_week": [2],
                "month": [month],
            }
        )

    jan = float(predictor.predict(trip(1))[0])
    jul = float(predictor.predict(trip(7))[0])
    dec = float(predictor.predict(trip(12))[0])
    # months barely move the true price; a representative fit keeps them within a couple of euros
    assert abs(jul - jan) < 2.5
    assert abs(dec - jan) < 2.5


def test_model_predictor_save_load_round_trips(rides: pd.DataFrame, tmp_path) -> None:
    # train once, persist, reload: the reloaded predictor must be identical (no refitting)
    trained = ModelPredictor.from_frame(rides.iloc[:6000])
    dest = tmp_path / "models" / "price_model.joblib"
    saved = trained.save(dest)
    assert saved == dest and dest.exists()  # save() creates the dir and returns the path

    reloaded = ModelPredictor.load(dest)
    assert isinstance(reloaded, PricePredictor)
    assert reloaded.n_train == trained.n_train

    holdout = rides.iloc[6000:][PINNED]
    # identical inputs -> byte-identical predictions from the deserialized model
    assert (reloaded.predict(holdout) == trained.predict(holdout)).all()


def test_model_predictor_load_does_not_refit(rides: pd.DataFrame, tmp_path, monkeypatch) -> None:
    # loading weights must never call the fitting pipeline (that's the whole point of persistence)
    dest = tmp_path / "price_model.joblib"
    ModelPredictor.from_frame(rides.iloc[:4000]).save(dest)

    from uber.infrastructure import predictors as predictors_mod

    def _boom(*_args, **_kwargs):  # pragma: no cover - only runs on regression
        raise AssertionError("load() must not fit a new pipeline")

    # patch the function the fit path actually calls (interaction_ols_pipeline), not ols_pipeline
    monkeypatch.setattr(predictors_mod.pipeline, "interaction_ols_pipeline", _boom)
    reloaded = ModelPredictor.load(dest)  # would raise if it tried to refit
    assert reloaded.n_train == 4000


def test_model_predictor_clamps_each_quote_to_its_tier_min_fare(rides: pd.DataFrame) -> None:
    predictor = ModelPredictor.from_frame(rides.iloc[:6000])
    min_fare = {t.tier: t.min_fare for t in config.TIERS}
    # a near-zero-distance trip: the raw model can dip below the floor, but every QUOTE must land at
    # (or above) that tier's guaranteed minimum — the serve-time analogue of max(min_fare, …). The
    # OLD behaviour floored to a generic €0.01 (the visible bug); now it floors PER TIER.
    short = pd.DataFrame(
        {
            "tier": ["uberx", "black"],
            "pickup_district": ["Centro", "Centro"],
            "distance_km": [0.1, 0.1],
            "duration_min": [0.5, 0.5],
            "surge_multiplier": [1.0, 1.0],
            "hour": [3, 3],
            "day_of_week": [2, 2],
            "month": [1, 1],
        }
    )
    raw = predictor._model.predict(short)
    floors = np.array([min_fare["uberx"], min_fare["black"]])
    preds = predictor.predict(short)
    # precondition: raw dips below the floor here, so the clamp does real work (not tautological)
    assert (raw < floors).any()
    # the clamp only ever raises to the floor, never lowers a valid fare
    assert np.allclose(preds, np.maximum(raw, floors))
    assert preds[0] >= min_fare["uberx"]  # >= €5.00, not the old generic €0.01
    assert preds[1] >= min_fare["black"]  # >= €12.00


def test_model_predictor_never_quotes_below_min_fare_for_any_tier(rides: pd.DataFrame) -> None:
    predictor = ModelPredictor.from_frame(rides.iloc[:6000])
    min_fare = {t.tier: t.min_fare for t in config.TIERS}
    tiers = list(min_fare)
    # a very short trip across ALL tiers at once: the floor-violation rate must be exactly zero
    frame = pd.DataFrame(
        {
            "tier": tiers,
            "pickup_district": ["Centro"] * len(tiers),
            "distance_km": [0.2] * len(tiers),
            "duration_min": [0.8] * len(tiers),
            "surge_multiplier": [1.0] * len(tiers),
            "hour": [4] * len(tiers),
            "day_of_week": [1] * len(tiers),
            "month": [6] * len(tiers),
        }
    )
    raw = predictor._model.predict(frame)
    floors = np.array([min_fare[t] for t in tiers])
    preds = predictor.predict(frame)
    assert (raw < floors).any()  # precondition: at least one tier's raw dips below its floor
    assert (preds >= floors).all()  # every quote lands at or above its tier's min_fare
    assert np.allclose(preds, np.maximum(raw, floors))


def test_model_predictor_unknown_tier_falls_back_to_positivity_floor(rides: pd.DataFrame) -> None:
    predictor = ModelPredictor.from_frame(rides.iloc[:6000])
    # a tier absent from the catalog must not crash or return NaN; it floors to _MIN_FARE_FLOOR_EUR
    frame = pd.DataFrame(
        {
            "tier": ["mystery"],
            "pickup_district": ["Centro"],
            "distance_km": [5.0],
            "duration_min": [15.0],
            "surge_multiplier": [1.2],
            "hour": [19],
            "day_of_week": [5],
            "month": [6],
        }
    )
    preds = predictor.predict(frame)
    assert np.isfinite(preds).all()
    assert (preds >= _MIN_FARE_FLOOR_EUR).all()


# --- FormulaPredictor ------------------------------------------------------
def test_formula_predictor_satisfies_port() -> None:
    assert isinstance(FormulaPredictor(), PricePredictor)


def test_formula_predictor_matches_ground_truth_formula() -> None:
    frame = pd.DataFrame(
        {
            "tier": ["uberx", "black"],
            "pickup_district": ["Centro", "Centro"],
            "distance_km": [5.0, 5.0],
            "duration_min": [15.0, 15.0],
            "surge_multiplier": [1.5, 1.5],
            "hour": [19, 19],
            "day_of_week": [5, 5],
            "month": [6, 6],
        }
    )
    preds = FormulaPredictor().predict(frame)
    by_id = {t.tier: t for t in config.TIERS}
    assert preds[0] == pytest.approx(pricing.price(by_id["uberx"], 5.0, 15.0, 1.5))
    assert preds[1] == pytest.approx(pricing.price(by_id["black"], 5.0, 15.0, 1.5))
    assert preds[1] > preds[0]  # black is pricier than uberx
