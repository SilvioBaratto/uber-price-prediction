"""Shared helpers for the eight-part regression arc (README "The model, part by part").

Single source of truth for the arc's feature sets, the one fixed train/test split, the
scikit-learn preprocessing/pipeline wiring, the metric helpers, OLS coefficient inference and
the final summary-table formatter. Every ``run_partN`` in ``run.py`` imports from here so the
preprocessing is byte-identical across parts — ``run.py`` never re-declares a
``ColumnTransformer``. scikit-learn throughout; ``root_mean_squared_error`` is used because
sklearn 1.9 removed ``mean_squared_error(squared=False)``.

Design-column widths (fit on the real data): pinned = 31, augmented = 38, poly-degree-2 = 52.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, cast

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn.base import RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

from uber.domain import config
from uber.infrastructure import paths

# --- Feature sets (single source of truth, SPEC Data Contract) -------------
TARGET = "price_eur"
EXCLUDED_IDS = ["ride_id", "driver_id", "timestamp", "pickup_location_id", "dropoff_location_id"]
PINNED_NUMERIC = ["distance_km", "duration_min", "surge_multiplier", "hour", "day_of_week", "month"]
PINNED_CATEGORICAL = ["tier", "pickup_district"]
DISTRACTOR_NUMERIC = ["driver_rating", "customer_rating", "avg_vtat"]
DISTRACTOR_CATEGORICAL = ["payment_method"]
AUGMENTED_NUMERIC = PINNED_NUMERIC + DISTRACTOR_NUMERIC
AUGMENTED_CATEGORICAL = PINNED_CATEGORICAL + DISTRACTOR_CATEGORICAL
LEAKAGE_FEATURE = "commission_eur"
# One wide candidate-feature frame; each part slices it via its own ColumnTransformer.
CANDIDATE_FEATURES = AUGMENTED_NUMERIC + AUGMENTED_CATEGORICAL + [LEAKAGE_FEATURE]

# k-grids. Part 2 tunes k over K_GRID_TUNE (its argmin is the "good k" Part 3 inherits); Part 6
# sweeps the wider PART6_K_GRID to trace the bias-variance U (reading its OWN k* from the sweep).
K_GRID_TUNE = (1, 3, 5, 8, 10, 15, 20, 30, 50)
PART6_K_GRID = (1, 2, 3, 5, 8, 12, 20, 40, 80, 160, 320, 640, 1000)

# Re-exported from config so every part imports one module.
SEED = config.SEED
R2_BAND = config.R2_BAND
NOISE_STD = config.NOISE_STD

# Explicit read dtypes: float64 preserves the calibrated R²; the calendar fields stay numeric.
_DTYPES: dict[str, str] = {
    **{c: "category" for c in ("tier", "pickup_district", "payment_method")},
    **{c: "int16" for c in ("hour", "day_of_week", "month")},
    **{c: "float64" for c in ("distance_km", "duration_min", "surge_multiplier", "driver_rating",
                              "customer_rating", "avg_vtat", "commission_eur", TARGET)},
}


class Split(NamedTuple):
    """One train/test partition; supports both ``a, b, c, d = split`` and ``split.X_train``."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray


@dataclass
class PartResult:
    """One row of the final summary table (``None`` where a metric is not applicable)."""

    part: int
    name: str
    n_features: int | None
    rmse_train: float | None
    rmse_test: float
    r2_test: float | None
    note: str = ""


# --- Data loading & splitting ----------------------------------------------
def load_rides(path: Path | str = paths.RAW_DIR / paths.RIDES_CSV) -> pd.DataFrame:
    """Read ``rides.csv`` once, keeping only the candidate features + target (the 5 IDs dropped).

    ``usecols`` drops ``ride_id/driver_id/timestamp/pickup_location_id/dropoff_location_id``;
    the result is re-ordered to ``CANDIDATE_FEATURES + [TARGET]`` (``read_csv`` returns file
    order) with the declared dtypes.
    """
    usecols = CANDIDATE_FEATURES + [TARGET]
    frame = pd.read_csv(path, usecols=usecols, dtype=cast(Any, _DTYPES))
    return cast(pd.DataFrame, frame[usecols])


def make_xy(rides: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Return the wide candidate-feature ``X`` and the float target ``y``."""
    return cast(pd.DataFrame, rides[CANDIDATE_FEATURES]), rides[TARGET].to_numpy(dtype=float)


def make_split(rides: pd.DataFrame, *, test_size: float = 0.2, seed: int = SEED) -> Split:
    """One fixed train/test partition (Parts 1-7 share it).

    ``random_state`` fixes the shuffle independently of which columns ``X`` carries, so every
    part slices the *same* rows — the "one fixed split" guarantee. Part 8 calls ``make_xy``
    directly for its own KFold + 60/20/20.
    """
    X, y = make_xy(rides)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    return Split(
        cast(pd.DataFrame, X_train), cast(pd.DataFrame, X_test),
        np.asarray(y_train), np.asarray(y_test),
    )


# --- Preprocessing & pipelines ---------------------------------------------
def build_preprocessor(
    numeric: list[str],
    categorical: list[str],
    *,
    scale: bool = True,
    poly_degree: int | None = None,
    dense: bool = True,
) -> ColumnTransformer:
    """Numeric + one-hot categorical preprocessing (fit inside the pipeline, no pre-split leakage).

    Numeric branch = ``PolynomialFeatures(poly_degree) -> StandardScaler`` if ``poly_degree`` else
    (``StandardScaler`` if ``scale`` else passthrough). Categorical branch =
    ``OneHotEncoder(drop="first", handle_unknown="ignore")``. ``dense=True`` (linear/inference
    parts 4,5,7,8) yields a dense design for exact coef inference; kNN parts (2,3,6) pass
    ``dense=False``.
    """
    numeric_tf: Pipeline | StandardScaler | str
    if poly_degree is not None:
        numeric_tf = Pipeline(
            [("poly", PolynomialFeatures(degree=poly_degree, include_bias=False)),
             ("scale", StandardScaler())]
        )
    elif scale:
        numeric_tf = StandardScaler()
    else:
        numeric_tf = "passthrough"
    categorical_tf = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=not dense)
    return ColumnTransformer(
        [("num", numeric_tf, list(numeric)), ("cat", categorical_tf, list(categorical))]
    )


def make_pipeline(
    estimator: RegressorMixin,
    numeric: list[str] = PINNED_NUMERIC,
    categorical: list[str] = PINNED_CATEGORICAL,
    *,
    scale: bool = True,
    poly_degree: int | None = None,
    dense: bool = True,
) -> Pipeline:
    """``Pipeline([("pre", preprocessor), ("est", estimator)])`` — step names are load-bearing
    (coef extraction relies on ``"pre"``/``"est"``)."""
    pre = build_preprocessor(numeric, categorical, scale=scale, poly_degree=poly_degree, dense=dense)
    return Pipeline([("pre", pre), ("est", estimator)])


def knn_pipeline(
    k: int,
    numeric: list[str] = PINNED_NUMERIC,
    categorical: list[str] = PINNED_CATEGORICAL,
    *,
    scale: bool = True,
    dense: bool = False,
) -> Pipeline:
    """Standardized (by default) kNN pipeline; ``dense=False`` keeps the one-hot block sparse."""
    return make_pipeline(
        KNeighborsRegressor(n_neighbors=k), numeric, categorical, scale=scale, dense=dense
    )


def ols_pipeline(
    numeric: list[str] = PINNED_NUMERIC,
    categorical: list[str] = PINNED_CATEGORICAL,
    *,
    poly_degree: int | None = None,
    dense: bool = True,
) -> Pipeline:
    """OLS (``LinearRegression``) pipeline; dense design for exact coefficient inference."""
    return make_pipeline(LinearRegression(), numeric, categorical, poly_degree=poly_degree, dense=dense)


# --- Metrics ----------------------------------------------------------------
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error (EUR)."""
    return float(root_mean_squared_error(y_true, y_pred))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination R²."""
    return float(r2_score(y_true, y_pred))


def score(model, X: pd.DataFrame, y: np.ndarray) -> tuple[float, float]:
    """Return ``(rmse, r2)`` of ``model`` on ``(X, y)``."""
    y_pred = model.predict(X)
    return rmse(y, y_pred), r2(y, y_pred)


# --- OLS coefficient inference ---------------------------------------------
def _strip_prefix(name: str) -> str:
    """Drop the ColumnTransformer branch prefix (``num__``/``cat__``) for a readable table."""
    return name.split("__", 1)[-1]


def ols_coef_table(
    fitted_pipeline: Pipeline, X_train: pd.DataFrame, y_train: np.ndarray
) -> pd.DataFrame:
    """Coefficient table ``[feature, coef, std_err, t, p_value]`` for a fitted OLS pipeline.

    Standard-error estimates use ``sigma² · (DᵀD)⁻¹`` on the dense design ``D`` (an intercept
    column prepended to match ``LinearRegression(fit_intercept=True)``); two-sided p-values use
    Student's t with ``df = n − (p + 1)`` (``scipy.stats.t``). Feature names come from the fitted
    ``"pre"`` step with the branch prefix stripped.
    """
    pre = fitted_pipeline.named_steps["pre"]
    est = fitted_pipeline.named_steps["est"]
    design = np.asarray(pre.transform(X_train), dtype=float)
    n = design.shape[0]
    Xd = np.column_stack([np.ones(n), design])              # prepend intercept column
    beta = np.concatenate([[float(est.intercept_)], np.asarray(est.coef_, dtype=float).ravel()])

    resid = np.asarray(y_train, dtype=float) - fitted_pipeline.predict(X_train)
    df = n - Xd.shape[1]
    sigma2 = float(resid @ resid) / df
    xtx_inv = np.linalg.inv(Xd.T @ Xd)
    std_err = np.sqrt(sigma2 * np.diag(xtx_inv))
    t_stat = beta / std_err
    p_value = 2.0 * student_t.sf(np.abs(t_stat), df)

    features = ["intercept"] + [_strip_prefix(f) for f in pre.get_feature_names_out()]
    return pd.DataFrame(
        {"feature": features, "coef": beta, "std_err": std_err, "t": t_stat, "p_value": p_value}
    )


# --- k tuning (Part 2 earns K_STAR; Part 3 inherits it) --------------------
def tune_k(split: Split, k_grid: tuple[int, ...] = K_GRID_TUNE, *, seed: int = SEED) -> int:
    """Pick k by minimizing validation RMSE over ``k_grid``.

    Carves an internal train/validation *partition* out of ``split.X_train`` (a partition, not a
    performance subsample): each candidate fits a standardized kNN on the inner-train rows and is
    scored on the held-out inner-validation rows. Deterministic at a fixed ``seed``.
    """
    X_inner, X_val, y_inner, y_val = train_test_split(
        split.X_train, split.y_train, test_size=0.25, random_state=seed
    )
    y_inner, y_val = np.asarray(y_inner), np.asarray(y_val)
    best_k, best_rmse = int(k_grid[0]), np.inf
    for k in k_grid:
        model = knn_pipeline(int(k)).fit(X_inner, y_inner)
        val_rmse = rmse(y_val, model.predict(X_val))
        if val_rmse < best_rmse:
            best_k, best_rmse = int(k), val_rmse
    return best_k


# --- Reporting --------------------------------------------------------------
def format_report(results: list[PartResult]) -> str:
    """Fixed-width summary table ``Part | model | #feat | train RMSE | test RMSE | test R² | note``.

    The header states the Part-1 baseline (~26.6 EUR) and units. The ``note`` column carries the
    honest framing (a well-tuned kNN can beat plain OLS, so RMSE is not strictly monotone).
    """
    baseline = next((r.rmse_test for r in results if r.part == 1), None)
    lines: list[str] = []
    if baseline is not None:
        lines.append(
            f"Part-1 baseline (mean predictor) test RMSE = {baseline:.2f} EUR; "
            "RMSE in EUR, R^2 on the held-out test set."
        )
    else:
        lines.append("RMSE in EUR, R^2 on the held-out test set.")
    head = (
        f"{'Part':>4} | {'model':<24} | {'#feat':>5} | {'train RMSE':>10} | "
        f"{'test RMSE':>9} | {'test R2':>8} | note"
    )
    lines.append(head)
    lines.append("-" * len(head))
    for r in sorted(results, key=lambda x: x.part):
        nf = "-" if r.n_features is None else str(r.n_features)
        tr = "-" if r.rmse_train is None else f"{r.rmse_train:.3f}"
        r2v = "-" if r.r2_test is None else f"{r.r2_test:.4f}"
        lines.append(
            f"{r.part:>4} | {r.name:<24} | {nf:>5} | {tr:>10} | "
            f"{r.rmse_test:>9.3f} | {r2v:>8} | {r.note}"
        )
    return "\n".join(lines)
