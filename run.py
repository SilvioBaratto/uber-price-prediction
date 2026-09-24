"""Orchestrator for the eight-part regression arc (README "The model, part by part").

Loads the full ``rides.csv`` **once**, carves the one fixed 80/20 split (Parts 1-7 share it),
then runs Parts 1->8 sequentially as ``run_partN`` section functions and prints the summary
table. Every number is produced here by running scikit-learn at ``SEED=42`` — never hand-written.
All preprocessing goes through ``uber.modeling`` (this file never re-declares a
``ColumnTransformer``). Part 8 does its own KFold + 60/20/20 on the full frame.

Usage::

    python run.py                                        # full data/raw/rides.csv
    python run.py --rides /tmp/uber_small/rides.csv      # fast dev build (still full data OF THAT FILE)
    python run.py --seed 42 --test-size 0.2 --output-dir output --charts
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from uber import modeling
from uber.domain import config
from uber.infrastructure import paths


# --- CLI --------------------------------------------------------------------
def parse_cli(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the five run.py flags (no ``--parts``/``--sample``/``--frac`` — fixed decisions)."""
    parser = argparse.ArgumentParser(description="Run the eight-part regression arc on rides.csv.")
    parser.add_argument(
        "--rides", type=Path, default=paths.RAW_DIR / paths.RIDES_CSV,
        help="path to rides.csv (default: data/raw/rides.csv)",
    )
    parser.add_argument("--seed", type=int, default=config.SEED, help="random_state (default: 42)")
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="held-out test fraction (default: 0.2)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output"),
        help="where part6_ucurve.csv (+ --charts PNG) is written (default: output/)",
    )
    parser.add_argument(
        "--charts", action="store_true", help="also emit the Part-6 U-curve PNG (CSV is always written)"
    )
    return parser.parse_args(argv)


def _strip_prefix(name: str) -> str:
    """Drop the ColumnTransformer branch prefix (``num__``/``cat__``)."""
    return name.split("__", 1)[-1]


# --- Part 1 — constant predictor & RMSE ------------------------------------
def run_part1(split: modeling.Split) -> modeling.PartResult:
    """Fit the mean-only predictor on the shared split and print the baseline RMSE."""
    model = DummyRegressor(strategy="mean").fit(split.X_train, split.y_train)
    mean_price = float(model.constant_.ravel()[0])            # == mean(y_train), TRAIN only

    pred_train = np.asarray(model.predict(split.X_train))
    pred_test = np.asarray(model.predict(split.X_test))
    rmse_train = modeling.rmse(split.y_train, pred_train)     # == np.std(y_train)
    rmse_test = modeling.rmse(split.y_test, pred_test)
    r2_test = modeling.r2(split.y_test, pred_test)            # ~ -3e-5

    print(f"[Part 1] constant predictor  mean(price_eur)={mean_price:.2f} EUR")
    print(f"[Part 1]   train RMSE={rmse_train:.3f} (= std of y_train={np.std(split.y_train):.3f})")
    print(f"[Part 1]   test  RMSE={rmse_test:.3f}   R2_test={r2_test:.6f}")
    return modeling.PartResult(1, "mean baseline", None, rmse_train, rmse_test, r2_test,
                               note="R2=0 reference")


# --- Part 2 — kNN + standardization ----------------------------------------
def run_part2(split: modeling.Split) -> tuple[modeling.PartResult, int]:
    """Standardized vs. raw kNN on the pinned set; tune k over ``K_GRID_TUNE`` and return k*.

    The tuned k* is *earned* from a validation-partition sweep (not hard-coded) and threaded
    into Part 3. Standardization roughly halves the error; k=1 memorizes (train RMSE ~ 0).
    """
    k_star = modeling.tune_k(split)

    std_model = modeling.knn_pipeline(k_star, scale=True).fit(split.X_train, split.y_train)
    raw_model = modeling.knn_pipeline(k_star, scale=False).fit(split.X_train, split.y_train)
    std_rmse, std_r2 = modeling.score(std_model, split.X_test, split.y_test)
    raw_rmse, _ = modeling.score(raw_model, split.X_test, split.y_test)
    std_rmse_train = modeling.rmse(split.y_train, std_model.predict(split.X_train))

    k1_model = modeling.knn_pipeline(1, scale=True).fit(split.X_train, split.y_train)
    k1_rmse_train = modeling.rmse(split.y_train, k1_model.predict(split.X_train))

    print(f"[Part 2] tuned k*={k_star} (argmin val RMSE over {list(modeling.K_GRID_TUNE)})")
    print(f"[Part 2]   standardized kNN  test RMSE={std_rmse:.3f}  R2={std_r2:.4f}")
    print(f"[Part 2]   raw (unscaled) kNN test RMSE={raw_rmse:.3f}  (std/raw={std_rmse / raw_rmse:.2f})")
    print(f"[Part 2]   k=1 standardized train RMSE={k1_rmse_train:.6f} (memorization)")
    return (
        modeling.PartResult(2, f"kNN std (k={k_star})", 31, std_rmse_train, std_rmse, std_r2,
                            note="scaling ~halves error"),
        k_star,
    )


# --- Part 3 — k=1 overfitting & curse of dimensionality --------------------
def run_part3(split: modeling.Split, k_star: int) -> modeling.PartResult:
    """k=1 (memorizing) vs. the tuned k*, then a fixed-k=1 dimensionality sweep (11 < 31 < 38)."""
    k1 = modeling.knn_pipeline(1).fit(split.X_train, split.y_train)
    kstar = modeling.knn_pipeline(k_star).fit(split.X_train, split.y_train)
    k1_rmse_train = modeling.rmse(split.y_train, k1.predict(split.X_train))
    k1_rmse_test = modeling.rmse(split.y_test, k1.predict(split.X_test))
    kstar_rmse_test = modeling.rmse(split.y_test, kstar.predict(split.X_test))
    k1_r2_test = modeling.r2(split.y_test, k1.predict(split.X_test))

    # Fixed-k=1 curse-of-dimensionality sweep: test RMSE rises as columns are layered on.
    sweeps = [
        ("num+tier (11)", modeling.PINNED_NUMERIC, ["tier"]),
        ("pinned (31)", modeling.PINNED_NUMERIC, modeling.PINNED_CATEGORICAL),
        ("pinned+distractors (38)", modeling.AUGMENTED_NUMERIC, modeling.AUGMENTED_CATEGORICAL),
    ]
    sweep_rmse: list[float] = []
    for label, num, cat in sweeps:
        model = modeling.knn_pipeline(1, num, cat).fit(split.X_train, split.y_train)
        r = modeling.rmse(split.y_test, model.predict(split.X_test))
        sweep_rmse.append(r)
        print(f"[Part 3]   k=1 curse {label:<24} test RMSE={r:.3f}")

    print(f"[Part 3] k=1 train RMSE={k1_rmse_train:.9f} (each point its own neighbor)")
    print(f"[Part 3]   k=1 test RMSE={k1_rmse_test:.3f} > k*={k_star} test RMSE={kstar_rmse_test:.3f}")
    return modeling.PartResult(3, "kNN k=1 (memorize)", 31, k1_rmse_train, k1_rmse_test, k1_r2_test,
                               note="0 train err = overfit; +cols hurt")


# --- Part 4 — OLS via normal equations -------------------------------------
def run_part4(split: modeling.Split) -> modeling.PartResult:
    """OLS on the pinned set: low-variance (train ~ test), R^2 in band; leakage sanity refit."""
    model = modeling.ols_pipeline().fit(split.X_train, split.y_train)
    rmse_train = modeling.rmse(split.y_train, model.predict(split.X_train))
    rmse_test, r2_test = modeling.score(model, split.X_test, split.y_test)

    # Normal-equations aside: the closed form matches sklearn's SVD/lstsq to ~1e-8.
    pre = model.named_steps["pre"]
    est = model.named_steps["est"]
    design = np.asarray(pre.transform(split.X_train), dtype=float)
    Xd = np.column_stack([np.ones(design.shape[0]), design])
    beta_closed = np.linalg.solve(Xd.T @ Xd, Xd.T @ split.y_train)
    beta_sklearn = np.concatenate([[float(est.intercept_)], np.asarray(est.coef_).ravel()])
    max_diff = float(np.max(np.abs(beta_closed - beta_sklearn)))

    # Sanity refit WITH commission_eur — a preview of the Part-8 leakage trap.
    leak = modeling.ols_pipeline(
        modeling.PINNED_NUMERIC + [modeling.LEAKAGE_FEATURE], modeling.PINNED_CATEGORICAL
    ).fit(split.X_train, split.y_train)
    _, leak_r2 = modeling.score(leak, split.X_test, split.y_test)

    print(f"[Part 4] OLS pinned  train RMSE={rmse_train:.3f}  test RMSE={rmse_test:.3f}  R2={r2_test:.4f}")
    print(f"[Part 4]   normal-equations closed form matches sklearn to {max_diff:.2e}")
    print(f"[Part 4]   sanity refit WITH commission_eur -> R2={leak_r2:.6f} (leakage preview)")
    return modeling.PartResult(4, "OLS (pinned)", 31, rmse_train, rmse_test, r2_test,
                               note="~2.6x below baseline; low variance")


# --- Part 5 — coefficient standard errors & polynomial surge ---------------
def run_part5(split: modeling.Split) -> modeling.PartResult:
    """Degree-2 polynomial (52 cols) breaks the R^2~0.85 wall; print the OLS coefficient table."""
    base = modeling.ols_pipeline().fit(split.X_train, split.y_train)
    base_rmse, base_r2 = modeling.score(base, split.X_test, split.y_test)

    poly = modeling.ols_pipeline(poly_degree=2).fit(split.X_train, split.y_train)
    poly_rmse_train = modeling.rmse(split.y_train, poly.predict(split.X_train))
    poly_rmse, poly_r2 = modeling.score(poly, split.X_test, split.y_test)

    # Standard-error lesson: read significance off the INTERPRETABLE pinned OLS (the poly design's
    # linear terms are collinear with their own squares/interactions, which inflates their SEs).
    table = modeling.ols_coef_table(base, split.X_train, split.y_train)
    feats = [str(f) for f in table["feature"].to_numpy()]
    t_vals = table["t"].to_numpy(dtype=float)
    abs_t = np.abs(t_vals)
    top = table.iloc[np.argsort(abs_t)[::-1][:8]]
    district_mask = np.array([f.startswith("pickup_district") for f in feats])
    n_districts = int(district_mask.sum())
    n_insignificant = int((abs_t[district_mask] < 1.96).sum())

    print(f"[Part 5] base OLS test R2={base_r2:.4f}  RMSE={base_rmse:.3f}")
    print(f"[Part 5] poly-deg2 test R2={poly_r2:.4f}  RMSE={poly_rmse:.3f} (surge*distance interactions)")
    print(f"[Part 5]   pinned-OLS top |t| coefficients (standard-error lesson):")
    for _, row in top.iterrows():
        print(f"[Part 5]     {row['feature']:<28} coef={row['coef']:+9.3f}  t={row['t']:+11.2f}")
    print(f"[Part 5]   {n_insignificant}/{n_districts} pickup_district dummies insignificant (|t|<1.96)")
    return modeling.PartResult(5, "OLS poly-deg2 (pinned)", 52, poly_rmse_train, poly_rmse, poly_r2,
                               note="surge*distance breaks the 0.85 wall")


# --- Part 6 — bias-variance & the U-curve ----------------------------------
def run_part6(split: modeling.Split, args: argparse.Namespace) -> modeling.PartResult:
    """Sweep kNN over ``PART6_K_GRID`` to trace the bias-variance U; write part6_ucurve.csv."""
    rows: list[dict[str, float]] = []
    for k in modeling.PART6_K_GRID:
        model = modeling.knn_pipeline(int(k)).fit(split.X_train, split.y_train)
        r_train = modeling.rmse(split.y_train, np.asarray(model.predict(split.X_train)))
        r_test = modeling.rmse(split.y_test, np.asarray(model.predict(split.X_test)))
        rows.append({"k": float(k), "rmse_train": r_train, "rmse_test": r_test,
                     "gap": r_test - r_train})
    curve = pd.DataFrame(rows)

    # k* is READ from the sweep (interior argmin), never hard-coded.
    ks = np.asarray([r["k"] for r in rows])
    rmse_test_arr = np.asarray([r["rmse_test"] for r in rows])
    rmse_train_arr = np.asarray([r["rmse_train"] for r in rows])
    k_max = float(ks.max())
    k_star = int(ks[int(np.argmin(rmse_test_arr))])
    rmse_k1 = float(rmse_test_arr[ks == 1][0])
    rmse_kstar = float(rmse_test_arr.min())
    rmse_klarge = float(rmse_test_arr[ks == k_max][0])
    train_k1 = float(rmse_train_arr[ks == 1][0])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "part6_ucurve.csv"
    curve.to_csv(csv_path, index=False)
    if args.charts:
        _plot_ucurve(curve, args.output_dir / "part6_ucurve.png")

    print(f"[Part 6] U-curve k*={k_star}: test RMSE(k*)={rmse_kstar:.3f} < "
          f"RMSE(k=1)={rmse_k1:.3f} and < RMSE(k={int(k_max)})={rmse_klarge:.3f}")
    print(f"[Part 6]   k=1 train RMSE={train_k1:.6f}; small k = low bias/high variance, "
          f"large k -> high bias/low variance (collapses to the mean)")
    print(f"[Part 6]   wrote {csv_path}" + (" (+ PNG)" if args.charts else ""))
    return modeling.PartResult(6, f"kNN U-curve (k*={k_star})", 31, train_k1, rmse_kstar,
                               None, note="one dial travels overfit->underfit")


def _plot_ucurve(curve: pd.DataFrame, path: Path) -> None:
    """Emit the Part-6 U-curve PNG (only called behind ``--charts``)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(curve["k"], curve["rmse_test"], marker="o", label="test RMSE")
    ax.plot(curve["k"], curve["rmse_train"], marker="s", label="train RMSE")
    ax.set_xscale("log")
    ax.set_xlabel("k (neighbors)")
    ax.set_ylabel("RMSE (EUR)")
    ax.set_title("Part 6 — bias-variance U-curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --- Part 7 — Ridge (L2) & Lasso (L1) --------------------------------------
def run_part7(split: modeling.Split) -> modeling.PartResult:
    """Augmented set (38): Lasso zeros all 4 distractor groups; Ridge keeps them nonzero."""
    ols = modeling.ols_pipeline(modeling.AUGMENTED_NUMERIC, modeling.AUGMENTED_CATEGORICAL)
    ols.fit(split.X_train, split.y_train)
    ols_rmse, _ = modeling.score(ols, split.X_test, split.y_test)

    lasso = modeling.make_pipeline(
        Lasso(alpha=0.1, max_iter=50000), modeling.AUGMENTED_NUMERIC, modeling.AUGMENTED_CATEGORICAL
    ).fit(split.X_train, split.y_train)
    ridge = modeling.make_pipeline(
        Ridge(alpha=1.0), modeling.AUGMENTED_NUMERIC, modeling.AUGMENTED_CATEGORICAL
    ).fit(split.X_train, split.y_train)

    lasso_rmse_train = modeling.rmse(split.y_train, lasso.predict(split.X_train))
    lasso_rmse, lasso_r2 = modeling.score(lasso, split.X_test, split.y_test)

    lasso_zeros = _distractor_coefs(lasso)
    ridge_nonzeros = _distractor_coefs(ridge)
    n_kept = int(np.sum(np.abs(np.asarray(lasso.named_steps["est"].coef_)) > 1e-8))

    print(f"[Part 7] OLS test RMSE={ols_rmse:.3f}  Lasso test RMSE={lasso_rmse:.3f} "
          f"(delta={lasso_rmse - ols_rmse:+.3f} EUR, same accuracy)")
    print(f"[Part 7]   Lasso keeps {n_kept}/38 columns; distractor |coef| max={max(lasso_zeros.values()):.2e}")
    print(f"[Part 7]   Ridge distractor |coef| min={min(ridge_nonzeros.values()):.4f} (L2 never selects)")
    return modeling.PartResult(7, "Lasso L1 (augmented)", 38, lasso_rmse_train, lasso_rmse, lasso_r2,
                               note="drops 4 distractor groups; same accuracy")


def _distractor_coefs(fitted: Pipeline) -> dict[str, float]:
    """Map each distractor feature (3 numeric + payment_method dummies) to its |coefficient|."""
    pre = fitted.named_steps["pre"]
    coefs = np.asarray(fitted.named_steps["est"].coef_).ravel()
    out: dict[str, float] = {}
    for name, c in zip(pre.get_feature_names_out(), coefs):
        stripped = _strip_prefix(name)
        if stripped in modeling.DISTRACTOR_NUMERIC or stripped.startswith("payment_method"):
            out[stripped] = float(abs(c))
    return out


# --- Part 8 — k-fold CV, splits & data leakage -----------------------------
def run_part8(rides: pd.DataFrame, args: argparse.Namespace) -> modeling.PartResult:
    """5-fold CV on pinned OLS (trustworthy estimate) then the commission_eur leakage trap."""
    X, y = modeling.make_xy(rides)
    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)

    scoring = ["r2", "neg_root_mean_squared_error"]
    cv = cross_validate(modeling.ols_pipeline(), X, y, cv=kf, scoring=scoring)
    r2_mean = float(cv["test_r2"].mean())
    r2_std = float(cv["test_r2"].std())
    rmse_mean = float(-cv["test_neg_root_mean_squared_error"].mean())

    # 60/20/20 corroboration of the CV estimate.
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, random_state=args.seed)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=args.seed)
    holdout = modeling.ols_pipeline().fit(cast(pd.DataFrame, X_tr), np.asarray(y_tr))
    _, val_r2 = modeling.score(holdout, cast(pd.DataFrame, X_val), np.asarray(y_val))
    _, te_r2 = modeling.score(holdout, cast(pd.DataFrame, X_te), np.asarray(y_te))

    # Adding commission_eur leaks the target -> CV R2 collapses to ~1.
    leak = modeling.ols_pipeline(modeling.PINNED_NUMERIC + [modeling.LEAKAGE_FEATURE],
                                 modeling.PINNED_CATEGORICAL)
    leak_cv = cross_validate(leak, X, y, cv=kf, scoring=["r2"])
    leak_r2 = float(leak_cv["test_r2"].mean())

    print(f"[Part 8] 5-fold CV pinned OLS  R2 mean={r2_mean:.4f} std={r2_std:.4f}  RMSE={rmse_mean:.3f}")
    print(f"[Part 8]   60/20/20 corroboration  val R2={val_r2:.4f}  test R2={te_r2:.4f}")
    print(f"[Part 8]   +commission_eur CV R2={leak_r2:.6f} -> LEAKAGE, flagged and dropped")
    return modeling.PartResult(8, "OLS 5-fold CV (pinned)", 31, None, rmse_mean, r2_mean,
                               note=f"trustworthy (fold std={r2_std:.3f}); commission=leak")


# --- Orchestrator -----------------------------------------------------------
def main(argv: list[str] | None = None) -> list[modeling.PartResult]:
    """Load once, one fixed split, run Parts 1->8 (threading k* from Part 2 into Part 3)."""
    args = parse_cli(argv)
    print(f"Loading {args.rides} ...")
    rides = modeling.load_rides(args.rides)
    print(f"Loaded {len(rides):,} rides; carving the one fixed {1 - args.test_size:.0%}/"
          f"{args.test_size:.0%} split (seed={args.seed}).")
    split = modeling.make_split(rides, test_size=args.test_size, seed=args.seed)

    results: list[modeling.PartResult] = []
    results.append(run_part1(split))
    part2, k_star = run_part2(split)
    results.append(part2)
    results.append(run_part3(split, k_star))
    results.append(run_part4(split))
    results.append(run_part5(split))
    results.append(run_part6(split, args))
    results.append(run_part7(split))
    results.append(run_part8(rides, args))

    print()
    print(modeling.format_report(results))
    return results


if __name__ == "__main__":
    main()
