"""Unit tests for the shared arc plumbing (``uber.modeling``).

These exercise the helpers the eight parts rely on — data loading/splitting, the preprocessing
widths, the metrics, OLS coefficient inference, the k tuner and the report formatter. They use
the established ~107k-row fixture (``build_dataset(make_rng(SEED), build_locations_df(), 250)``)
and never read the 1.2 GB ``data/raw/rides.csv``. Run with ``pytest tests/test_modeling.py``.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from uber import config, generate_data, io, modeling, sampling

N_DRIVERS_TEST = 250  # ~107k emergent rides


@pytest.fixture(scope="module")
def rides() -> pd.DataFrame:
    locations = io.build_locations_df()
    _drivers, rides = generate_data.build_dataset(
        sampling.make_rng(config.SEED), locations, N_DRIVERS_TEST
    )
    return rides


@pytest.fixture(scope="module")
def rides_csv(tmp_path_factory: pytest.TempPathFactory, rides: pd.DataFrame) -> str:
    """The fixture rides written to a CSV (all 19 columns incl. the 5 IDs) for ``load_rides``."""
    path = tmp_path_factory.mktemp("modeling") / "rides.csv"
    io.write_csv(rides, path)
    return str(path)


# --- constants --------------------------------------------------------------
def test_feature_set_shapes() -> None:
    assert modeling.TARGET == "price_eur"
    assert modeling.CANDIDATE_FEATURES == (
        modeling.AUGMENTED_NUMERIC + modeling.AUGMENTED_CATEGORICAL + [modeling.LEAKAGE_FEATURE]
    )
    # the leakage column and the 5 IDs are never in the pinned/augmented feature sets
    assert modeling.LEAKAGE_FEATURE not in modeling.AUGMENTED_NUMERIC
    assert not set(modeling.EXCLUDED_IDS) & set(modeling.CANDIDATE_FEATURES)
    # re-exports match config
    assert (modeling.SEED, modeling.R2_BAND, modeling.NOISE_STD) == (
        config.SEED, config.R2_BAND, config.NOISE_STD
    )
    assert 10 in modeling.K_GRID_TUNE  # the SPEC-anchored good k is a candidate


# --- load_rides -------------------------------------------------------------
def test_load_rides_keeps_only_candidate_features_and_target(rides_csv: str, rides: pd.DataFrame) -> None:
    loaded = modeling.load_rides(rides_csv)
    assert list(loaded.columns) == modeling.CANDIDATE_FEATURES + [modeling.TARGET]
    assert len(loaded) == len(rides)
    for col in modeling.EXCLUDED_IDS:
        assert col not in loaded.columns


def test_load_rides_dtypes(rides_csv: str) -> None:
    loaded = modeling.load_rides(rides_csv)
    for col in ("tier", "pickup_district", "payment_method"):
        assert isinstance(loaded[col].dtype, pd.CategoricalDtype)
    for col in ("hour", "day_of_week", "month"):
        assert loaded[col].dtype == np.int16
    for col in ("distance_km", "surge_multiplier", "commission_eur", modeling.TARGET):
        assert loaded[col].dtype == np.float64


# --- make_xy / make_split ---------------------------------------------------
def test_make_xy(rides: pd.DataFrame) -> None:
    X, y = modeling.make_xy(rides)
    assert list(X.columns) == modeling.CANDIDATE_FEATURES
    assert y.dtype == np.float64 and y.shape == (len(rides),)


def test_split_supports_unpack_and_attribute_access(rides: pd.DataFrame) -> None:
    split = modeling.make_split(rides)
    X_train, X_test, y_train, y_test = split
    assert X_train.equals(split.X_train) and X_test.equals(split.X_test)
    assert np.array_equal(y_train, split.y_train) and np.array_equal(y_test, split.y_test)
    assert len(X_test) == pytest.approx(0.2 * len(rides), rel=0.01)


def test_split_is_deterministic(rides: pd.DataFrame) -> None:
    a, b = modeling.make_split(rides), modeling.make_split(rides)
    assert a.X_train.index.equals(b.X_train.index)
    assert np.array_equal(a.y_train, b.y_train)


def test_split_row_partition_is_column_invariant(rides: pd.DataFrame) -> None:
    """The 'one fixed split': the row partition is identical regardless of which columns X carries."""
    _, y = modeling.make_xy(rides)
    pinned = cast(pd.DataFrame, rides[modeling.PINNED_NUMERIC + modeling.PINNED_CATEGORICAL])
    candidate = cast(pd.DataFrame, rides[modeling.CANDIDATE_FEATURES])
    idx_pinned = cast(
        pd.DataFrame, train_test_split(pinned, y, test_size=0.2, random_state=modeling.SEED)[0]
    ).index
    idx_candidate = cast(
        pd.DataFrame, train_test_split(candidate, y, test_size=0.2, random_state=modeling.SEED)[0]
    ).index
    assert idx_pinned.equals(idx_candidate)


# --- build_preprocessor widths ---------------------------------------------
@pytest.mark.parametrize(
    "numeric, categorical, poly, expected",
    [
        (modeling.PINNED_NUMERIC, modeling.PINNED_CATEGORICAL, None, 31),
        (modeling.AUGMENTED_NUMERIC, modeling.AUGMENTED_CATEGORICAL, None, 38),
        (modeling.PINNED_NUMERIC, modeling.PINNED_CATEGORICAL, 2, 52),
    ],
)
def test_preprocessor_widths(
    rides: pd.DataFrame, numeric: list[str], categorical: list[str], poly: int | None, expected: int
) -> None:
    pre = modeling.build_preprocessor(numeric, categorical, poly_degree=poly)
    assert np.asarray(pre.fit_transform(rides)).shape[1] == expected


def test_preprocessor_passthrough_unscaled(rides: pd.DataFrame) -> None:
    pre = modeling.build_preprocessor(modeling.PINNED_NUMERIC, modeling.PINNED_CATEGORICAL, scale=False)
    out = pre.fit_transform(rides)
    # unscaled numeric branch preserves the raw distance_km column values (first numeric col)
    assert np.allclose(np.sort(out[:, 0]), np.sort(rides["distance_km"].to_numpy()))


# --- metrics ----------------------------------------------------------------
def test_rmse_of_mean_equals_population_std() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
    assert modeling.rmse(y, np.full_like(y, y.mean())) == pytest.approx(np.std(y, ddof=0))


def test_score_matches_component_metrics(rides: pd.DataFrame) -> None:
    split = modeling.make_split(rides)
    model = modeling.ols_pipeline().fit(split.X_train, split.y_train)
    rmse_v, r2_v = modeling.score(model, split.X_test, split.y_test)
    assert rmse_v == pytest.approx(modeling.rmse(split.y_test, model.predict(split.X_test)))
    assert r2_v == pytest.approx(modeling.r2(split.y_test, model.predict(split.X_test)))


# --- ols_coef_table ---------------------------------------------------------
def test_ols_coef_table_shape_and_significance(rides: pd.DataFrame) -> None:
    split = modeling.make_split(rides)
    model = modeling.ols_pipeline().fit(split.X_train, split.y_train)
    table = modeling.ols_coef_table(model, split.X_train, split.y_train)
    assert list(table.columns) == ["feature", "coef", "std_err", "t", "p_value"]
    assert len(table) == 32  # 31 design columns + intercept
    assert (table["p_value"].between(0.0, 1.0)).all()
    tmap = table.set_index("feature")["t"].abs()
    assert tmap["surge_multiplier"] > 50 and tmap["distance_km"] > 50


# --- tune_k -----------------------------------------------------------------
def test_tune_k_returns_in_grid_and_is_deterministic(rides: pd.DataFrame) -> None:
    small = rides.sample(n=5000, random_state=modeling.SEED)
    split = modeling.make_split(small)
    k_a = modeling.tune_k(split)
    k_b = modeling.tune_k(split)
    assert k_a in modeling.K_GRID_TUNE
    assert k_a == k_b


# --- format_report ----------------------------------------------------------
def test_format_report_renders_header_and_rows() -> None:
    results = [
        modeling.PartResult(1, "mean baseline", None, 26.6, 26.6, -3e-5, note="R2=0 reference"),
        modeling.PartResult(4, "OLS (pinned)", 31, 10.3, 10.3, 0.85, note=""),
    ]
    report = modeling.format_report(results)
    assert "Part-1 baseline" in report
    assert "mean baseline" in report and "OLS (pinned)" in report
    assert "R2=0 reference" in report  # the note column renders
    assert report.count("\n") >= 4  # header line + separator + 2 rows + baseline line
