"""T2.5 — the price predictors (infrastructure layer).

``ModelPredictor`` retrains a pinned poly-OLS model on launch from rides data and predicts fares
for a quote's feature frame; ``FormulaPredictor`` is a ground-truth test double / no-data
fallback built on ``domain.pricing.price``. Both satisfy the ``PricePredictor`` port.
"""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.metrics import r2_score

from uber.application.ports import PricePredictor
from uber.datagen import generate_data, sampling
from uber.domain import config, pricing
from uber.infrastructure import repositories
from uber.infrastructure.predictors import FormulaPredictor, ModelPredictor

PINNED = ["tier", "pickup_district", "distance_km", "duration_min",
          "surge_multiplier", "hour", "day_of_week", "month"]


@pytest.fixture(scope="module")
def rides() -> pd.DataFrame:
    locations = repositories.build_locations_df()
    _drivers, rides = generate_data.build_dataset(
        sampling.make_rng(config.SEED), locations, 250
    )
    return rides.sample(8000, random_state=config.SEED).reset_index(drop=True)


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
