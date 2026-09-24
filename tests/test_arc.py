"""Per-part headline claims for the eight-part arc + a run.py composition smoke test.

Each test pins the one *aha* number the corresponding short is built around, produced by calling
the real ``run.run_partN`` on the established ~107k-row fixture
(``build_dataset(make_rng(SEED), build_locations_df(), 250)``) — never the 1.2 GB
``data/raw/rides.csv``. kNN-bearing parts (2, 3, 6) subsample a few thousand rows because
brute-force neighbour prediction is superlinear; the linear/CV parts use the full fixture. R²/RMSE
are ~N-invariant, so the fixture reproduces every calibrated number the full run prints.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import KFold, cross_validate

from uber.datagen import generate_data, sampling
from uber.modeling import arc as run
from uber.modeling import pipeline as modeling
from uber.domain import config
from uber.infrastructure import repositories

N_DRIVERS_TEST = 250  # ~107k emergent rides
SEED = config.SEED


@pytest.fixture(scope="module")
def rides() -> pd.DataFrame:
    locations = repositories.build_locations_df()
    _drivers, rides = generate_data.build_dataset(
        sampling.make_rng(config.SEED), locations, N_DRIVERS_TEST
    )
    return rides


def _subsample(rides: pd.DataFrame, n: int) -> pd.DataFrame:
    """Deterministic row subsample for the kNN parts (brute-force neighbours are superlinear)."""
    return rides.sample(n, random_state=SEED).reset_index(drop=True)


# --- Part 1 — constant predictor & RMSE ------------------------------------
def test_part1_baseline_is_std(rides: pd.DataFrame) -> None:
    split = modeling.make_split(rides)
    res = run.run_part1(split)

    # the mean is the optimal constant: constant_ == mean(y_train), train RMSE == std(y_train)
    model = DummyRegressor(strategy="mean").fit(split.X_train, split.y_train)
    assert float(model.constant_.ravel()[0]) == pytest.approx(float(split.y_train.mean()))
    assert res.rmse_train == pytest.approx(float(np.std(split.y_train, ddof=0)))
    r2_train = modeling.r2(split.y_train, np.asarray(model.predict(split.X_train)))
    assert r2_train == pytest.approx(0.0, abs=1e-9)  # R2 == 0 by construction
    assert 24.0 < res.rmse_test < 29.0


# --- Part 2 — kNN + standardization ----------------------------------------
def test_part2_standardization_halves_error(rides: pd.DataFrame) -> None:
    split = modeling.make_split(_subsample(rides, 8000))
    res, k_star = run.run_part2(split)

    assert k_star in modeling.K_GRID_TUNE  # k* earned from the validation sweep, in-grid
    baseline = run.run_part1(split).rmse_test
    raw = modeling.knn_pipeline(k_star, scale=False).fit(split.X_train, split.y_train)
    raw_rmse = modeling.rmse(split.y_test, np.asarray(raw.predict(split.X_test)))
    assert res.rmse_test < 0.75 * raw_rmse   # scaling roughly halves the error
    assert res.rmse_test < baseline          # a real model crushes the mean baseline

    k1 = modeling.knn_pipeline(1).fit(split.X_train, split.y_train)
    assert modeling.rmse(split.y_train, np.asarray(k1.predict(split.X_train))) < 1e-6


# --- Part 3 — k=1 overfitting & curse of dimensionality --------------------
def test_part3_overfit_and_curse(rides: pd.DataFrame) -> None:
    split = modeling.make_split(_subsample(rides, 6000))
    k_star = modeling.tune_k(split)
    res = run.run_part3(split, k_star)

    assert res.rmse_train is not None and res.rmse_train < 1e-9  # k=1 memorizes: 0 train error
    kstar = modeling.knn_pipeline(k_star).fit(split.X_train, split.y_train)
    kstar_test = modeling.rmse(split.y_test, np.asarray(kstar.predict(split.X_test)))
    assert res.rmse_test > kstar_test                            # yet worse on held-out data

    def k1_test(numeric: list[str], categorical: list[str]) -> float:
        model = modeling.knn_pipeline(1, numeric, categorical).fit(split.X_train, split.y_train)
        return modeling.rmse(split.y_test, np.asarray(model.predict(split.X_test)))

    r11 = k1_test(modeling.PINNED_NUMERIC, ["tier"])                                    # 11 cols
    r31 = k1_test(modeling.PINNED_NUMERIC, modeling.PINNED_CATEGORICAL)                 # 31 cols
    r38 = k1_test(modeling.AUGMENTED_NUMERIC, modeling.AUGMENTED_CATEGORICAL)           # 38 cols
    assert r11 < r31 < r38  # test RMSE rises monotonically as columns are layered on


# --- Part 4 — OLS via normal equations -------------------------------------
def test_part4_ols_in_band_low_variance(rides: pd.DataFrame) -> None:
    split = modeling.make_split(rides)
    res = run.run_part4(split)

    lo, hi = modeling.R2_BAND
    assert res.r2_test is not None and lo <= res.r2_test <= hi   # R2 ~ 0.85, inside the band
    assert res.rmse_train is not None
    assert abs(res.rmse_train - res.rmse_test) / res.rmse_test < 0.05  # low variance (train ~ test)
    assert res.rmse_test < 0.5 * float(np.std(split.y_test, ddof=0))   # ~2.6x below baseline


# --- Part 5 — coefficient standard errors & polynomial surge ---------------
def test_part5_poly_breaks_the_wall(rides: pd.DataFrame) -> None:
    split = modeling.make_split(rides)
    base = run.run_part4(split)
    res = run.run_part5(split)

    assert base.r2_test is not None and res.r2_test is not None
    assert res.r2_test > 0.87                       # degree-2 clears the 0.87 wall
    assert res.r2_test > base.r2_test + 0.02        # a real jump over plain OLS
    assert res.rmse_test < base.rmse_test - 0.5     # >= 0.5 EUR RMSE improvement, no new data

    # significance is read off the interpretable pinned OLS (see run_part5 for why, not the poly design)
    pinned = modeling.ols_pipeline().fit(split.X_train, split.y_train)
    table = modeling.ols_coef_table(pinned, split.X_train, split.y_train)
    feats = [str(f) for f in table["feature"].to_numpy()]
    ts = table["t"].to_numpy(dtype=float)
    tmap = dict(zip(feats, ts))
    assert abs(tmap["surge_multiplier"]) > 50       # the pricing physics is overwhelmingly real
    assert abs(tmap["distance_km"]) > 50
    districts = [abs(t) for f, t in zip(feats, ts) if f.startswith("pickup_district")]
    assert min(districts) < 1.96                    # >= 1 district dummy is insignificant


# --- Part 6 — bias-variance & the U-curve ----------------------------------
def test_part6_ucurve(rides: pd.DataFrame, tmp_path) -> None:
    split = modeling.make_split(_subsample(rides, 4000))
    args = run.parse_cli(["--output-dir", str(tmp_path)])
    res = run.run_part6(split, args)

    csv_path = tmp_path / "part6_ucurve.csv"
    assert csv_path.exists()                        # the sweep is persisted
    curve = pd.read_csv(csv_path)
    ks = curve["k"].to_numpy()
    rt = curve["rmse_test"].to_numpy()

    def rmse_at(k: int) -> float:
        return float(rt[ks == k][0])

    assert res.rmse_train is not None and res.rmse_train < 1e-6   # k=1 train error ~ 0
    assert res.rmse_test < rmse_at(1)               # interior k* beats the overfit end
    assert res.rmse_test < rmse_at(1000)            # ... and the underfit end
    assert rmse_at(1000) < float(np.std(split.y_test, ddof=0))    # large-k still beats the mean


# --- Part 7 — Ridge (L2) & Lasso (L1) --------------------------------------
def test_part7_lasso_selects_ridge_shrinks(rides: pd.DataFrame) -> None:
    split = modeling.make_split(rides)
    res = run.run_part7(split)

    ols = modeling.ols_pipeline(modeling.AUGMENTED_NUMERIC, modeling.AUGMENTED_CATEGORICAL)
    ols.fit(split.X_train, split.y_train)
    ols_rmse = modeling.rmse(split.y_test, np.asarray(ols.predict(split.X_test)))
    assert res.rmse_test <= ols_rmse + 0.25         # same accuracy, simpler model

    lasso = modeling.make_pipeline(
        Lasso(alpha=0.1, max_iter=50000),
        modeling.AUGMENTED_NUMERIC, modeling.AUGMENTED_CATEGORICAL,
    ).fit(split.X_train, split.y_train)
    ridge = modeling.make_pipeline(
        Ridge(alpha=1.0), modeling.AUGMENTED_NUMERIC, modeling.AUGMENTED_CATEGORICAL
    ).fit(split.X_train, split.y_train)

    lasso_distractors = run._distractor_coefs(lasso)
    ridge_distractors = run._distractor_coefs(ridge)
    # all 4 distractor groups present (3 numeric + payment_method dummies)
    assert {"driver_rating", "customer_rating", "avg_vtat"} <= set(lasso_distractors)
    assert any(name.startswith("payment_method") for name in lasso_distractors)
    assert max(lasso_distractors.values()) < 1e-8   # L1 zeros every distractor exactly
    assert min(ridge_distractors.values()) > 0.0    # L2 shrinks but never selects


# --- Part 8 — k-fold CV, splits & data leakage -----------------------------
def test_part8_cv_and_leakage(rides: pd.DataFrame) -> None:
    args = run.parse_cli([])
    res = run.run_part8(rides, args)

    lo, hi = modeling.R2_BAND
    assert res.r2_test is not None and lo <= res.r2_test <= hi   # CV R2 mean inside the band

    X, y = modeling.make_xy(rides)
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    cv = cross_validate(modeling.ols_pipeline(), X, y, cv=kf, scoring=["r2"])
    assert float(cv["test_r2"].std()) < 0.02        # tiny fold spread = trustworthy estimate

    leak = modeling.ols_pipeline(
        modeling.PINNED_NUMERIC + [modeling.LEAKAGE_FEATURE], modeling.PINNED_CATEGORICAL
    )
    leak_cv = cross_validate(leak, X, y, cv=kf, scoring=["r2"])
    assert float(leak_cv["test_r2"].mean()) > 0.999  # commission_eur leaks the target


# --- run.py composition smoke test -----------------------------------------
def test_run_composition_smoke(rides: pd.DataFrame, tmp_path) -> None:
    """End-to-end: every part runs on the fixture, returns a PartResult, and the report renders."""
    small = _subsample(rides, 3000)
    split = modeling.make_split(small)
    args = run.parse_cli(["--output-dir", str(tmp_path)])

    r1 = run.run_part1(split)
    r2, k_star = run.run_part2(split)
    r3 = run.run_part3(split, k_star)
    r4 = run.run_part4(split)
    r5 = run.run_part5(split)
    r6 = run.run_part6(split, args)
    r7 = run.run_part7(split)
    r8 = run.run_part8(small, args)

    results = [r1, r2, r3, r4, r5, r6, r7, r8]
    assert all(isinstance(r, modeling.PartResult) for r in results)
    assert [r.part for r in results] == [1, 2, 3, 4, 5, 6, 7, 8]
    report = modeling.format_report(results)
    assert "Part" in report and "test RMSE" in report
