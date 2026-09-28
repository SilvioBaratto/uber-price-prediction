# uber-price-prediction

[![CI](https://github.com/neuroespresso/uber-price-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/neuroespresso/uber-price-prediction/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

Companion project to the series **"Let's Build Uber's Algorithm"** (neuroespresso).
Build a ride-hailing price engine for **Madrid** from scratch — a synthetic dataset,
an eight-part regression arc, and a terminal simulator that prices every ride tier
with a model retrained on launch.

> Simplified for teaching: this is not Uber's real price list, it's the core idea —
> estimate a price from a few variables and present the options — rebuilt as a project
> you can run from the terminal.

```text
Pickup (street name or id, 'q' to quit): Puerta del Sol
  -> Plaza de la Puerta del Sol (Centro)
Drop-off (street name or id, 'q' to quit): Gran Vía
  -> Calle Gran Vía (Centro)

Plaza de la Puerta del Sol -> Calle Gran Vía
Tier           Price    ETA  Seats
------------  ------  -----  -----
UberX          €6.90  4 min      4
Uber Green     €7.10  4 min      4
Uber Comfort   €8.40  4 min      4
UberXL         €9.20  4 min      6
Uber Black    €12.75  4 min      4
Uber Van      €14.10  4 min      6
```

_(Illustrative figures — real prices come from the model retrained at launch.)_

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

uber generate-data --n-drivers 200   # build a fast, dev-sized dataset into data/raw/
uber run-arc                         # train the models, print RMSE/R² part by part
uber train                           # fit the price model once, save weights to models/
uber simulate --model models/price_model.joblib   # interactive A -> B simulator (loads weights)
```

Every command is also reachable as `python -m uber <subcommand>`. Run
`uber --help` (or `uber <subcommand> --help`) for the full options.

### Train once, then serve fast

`uber train` fits the tier-interaction poly-OLS price model and **persists its weights** to
`models/price_model.joblib` (~13 KB — a committed, ready-to-run model, so `uber simulate --model`
works out of the box; regenerate it any time with `uber train`, noting joblib binaries are
scikit-learn-version-sensitive). `uber simulate --model <path>` **loads those weights instantly**
instead of refitting, so startup is immediate regardless of dataset size:

```bash
uber simulate --model models/price_model.joblib   # uses the committed model, no retraining
uber train                                        # refit on a seeded 500k-row sample -> models/
```

Training defaults to a seeded 500,000-row sample (`--max-train-rows`): the interaction design is
wide (~500 columns), so an uncapped fit on the multi-million-row `rides.csv` would need tens of GB —
and R² is ~invariant to N here, so the sample is quality-equivalent. Pass a larger value to use
more rows.

If you skip `--model` (or the file is missing), the simulator falls back to
**retraining on launch** from `rides.csv`; cap that read for a snappy start:

```bash
uber simulate --max-train-rows 200000         # no saved model → retrain on a capped sample
```

## How it works

1. pick **A** and **B** among Madrid's streets (by name or id);
2. compute the trip **distance** (km) and **estimated duration** (min);
3. read the **current conditions**: hour of day, day of week, surge/demand;
4. for **each tier**, the model **predicts the price**;
5. the terminal prints the **full list**: tier, price (€), ETA, seats.

Each price is produced by a fitted regression model — never hand-written.

## Architecture

The code follows **clean architecture**: dependencies point inward and inner layers
never import outer ones. The pure business rules sit at the center; I/O and the CLI
live at the edges and are wired together only in the composition root (`uber/cli/app.py`).

```
        cli  ─────────▶  application  ─────────▶  domain
   (entry points,     (use cases + ports,      (entities, pricing,
    composition)         Protocols)             business config)
        │                    ▲
        │                    │ implements ports
        └────────▶  infrastructure  (CSV repositories, predictors, clock, paths)
                             ▲
        datagen ─────────────┘   modeling ─────▶  domain + infrastructure
   (synthetic-data pipeline)   (the ML arc — supporting feature packages)
```

```
uber/
  __main__.py               python -m uber -> cli.app.main()
  domain/
    entities.py             Location, RideTier, Driver, RideOption, TripRequest
    pricing.py              haversine, road_distance, duration, surge, price (ground truth)
    config.py               business/sim constants + the tier catalog
  application/
    ports.py                Protocols: LocationRepository, TierRepository, PricePredictor, Clock
    quoting.py              QuoteService.quote(origin, dest, when) -> list[RideOption]
  infrastructure/
    paths.py                project paths and filenames
    csv_io.py  ncr.py       CSV write mechanics; NCR empirical supports
    repositories.py         Csv{Location,Tier}Repository + the reference-table builders
    predictors.py           ModelPredictor (retrain-on-launch), FormulaPredictor (test double)
    clock.py                SystemClock
  datagen/                  sampling, simulation, generate_data  (the synthetic dataset)
  modeling/
    pipeline.py             feature sets, the fixed split, preprocessing, metrics, report
    arc.py                  the eight parts + orchestration
  cli/
    app.py                  the `uber` CLI (generate-data | run-arc | simulate)
    simulator.py            the interactive A->B loop
    render.py               the aligned quote table
run.py                      back-compat shim -> uber.modeling.arc.main
```

## The data

Generated into `data/raw/` by `uber generate-data`. The three small reference tables are
committed; the large `rides.csv` is git-ignored and rebuilt from the seed.

### `madrid_locations.csv` — the map (streets in Madrid)

| Column | Type | Note |
|---|---|---|
| `location_id` | int | key |
| `street` | str | street name (vial) |
| `neighborhood` | str | Madrid neighborhood (barrio) |
| `district` | str | district (distrito) it belongs to |
| `lat`, `lon` | float | coordinates, for distance (haversine × road factor) |

### `ride_tiers.csv` — the ride catalog

| Column | Type | Note |
|---|---|---|
| `tier` | str | id: `uberx`, `green`, `comfort`, `xl`, `black`, `van` |
| `display_name` | str | name shown on screen |
| `capacity` | int | passenger seats |
| `base_fare` | € | base fare |
| `per_km` | €/km | distance rate |
| `per_min` | €/min | time rate |
| `booking_fee` | € | fixed booking fee |
| `min_fare` | € | guaranteed minimum price |

### `rides.csv` — the ride history (what the model trains on)

Emergent output of a seeded, day-by-day driver-population simulation. Here the price
**genuinely depends** on the features.

| Column | Type | Role |
|---|---|---|
| `distance_km` | float | primary price driver |
| `duration_min` | float | second driver (traffic/time) |
| `tier` | cat | service tier (one-hot) |
| `surge_multiplier` | float | demand; **non-linear** effect (part 5) |
| `hour` | int | hour of day (0–23) |
| `day_of_week` | cat | day |
| `pickup_district` | cat | pickup zone |
| `price_eur` | € | **target** |

> ⚠️ The Kaggle *NCR Uber 2024* (India) dataset is **not** suitable as the target: its
> price is uncorrelated with everything (R² ≈ 0) and the city is wrong. `rides.csv` is a
> Madrid-specific dataset with real price signal. We only reuse NCR's empirical column
> distributions to build realistic distractor features.

## The model, part by part

The arc teaches a **progression**: each part lowers the price-estimation error (RMSE).
The simulator builds on Part 5's polynomial idea but goes one step further: it runs the
degree-2 polynomial over the **full one-hot design**, so it forms `tier × distance`/`duration`/
`surge` interactions and each tier gets its **own slopes** (the arc's Part 5 keeps the
numeric-only polynomial for the teaching narrative). It then clamps every quote up to that tier's
`min_fare`. This lifts held-out R² from ≈0.88 to ≈0.98 and guarantees a quote never falls below a
tier's minimum — see `uber/modeling/pipeline.py` `interaction_ols_pipeline` and
`uber/infrastructure/predictors.py` `ModelPredictor`.

| Part | Model / concept |
|---|---|
| 1 | constant predictor = mean of prices; squared loss (RMSE) |
| 2 | **k-NN** regression + standardization; train vs test error |
| 3 | k=1 → zero train error, high test error; curse of dimensionality |
| 4 | **linear regression (OLS)** + normal equations |
| 5 | standard error of coefficients; polynomial expansion (non-linear surge) |
| 6 | **bias–variance** decomposition; the under/overfitting U-curve |
| 7 | **Ridge (L2)** and **Lasso (L1)** (feature selection) |
| 8 | **k-fold cross-validation** + train/val/test split; data leakage |

Every RMSE/R² number is produced by `uber run-arc` on the real data — never by hand.

## ML project checklist → where it lives in the code

This project is organized to follow the canonical eight-step **Machine Learning Project
Checklist** (Aurélien Géron, *Hands-On Machine Learning*, Appendix B). Each step below points
to the code that implements it.

**1. Frame the problem & look at the big picture** — predict a ride's price (€) from trip +
demand features, then present per-tier options. The ground truth the model learns and the
business constants live in the domain layer.
→ `uber/domain/pricing.py` (`price()` — the ground-truth fare), `uber/domain/config.py`
(business/sim constants + tier catalog), `uber/domain/entities.py` (`TripRequest`,
`RideOption`, `RideTier`). Narrative: the *How it works* section above.

**2. Get the data** *(automated & reproducible from a seed)* — a seeded, day-by-day
driver-population simulation emits `rides.csv`; NCR empirical distributions feed realistic
distractor columns.
→ `uber generate-data` = `uber/datagen/generate_data.py` (`main`, `build_dataset`,
`build_rides_df`); `uber/datagen/simulation.py` (`build_drivers`, `simulate_ride_events`);
`uber/datagen/sampling.py` (RNG, timestamps, O-D pairs, empirical sampling);
`uber/infrastructure/ncr.py` (`load_ncr`). One-time map authoring: `scripts/extract_locations.py`.

**3. Explore the data** — attribute types/roles are documented in *The data* tables above;
spatial balance is visualized as standalone maps.
→ `scripts/plot_balance_map.py` → `output/madrid_balanced.png` / `madrid_unbalanced.png`.
Attribute schema: the tables in *The data*. *(EDA is intentionally light — the dataset is
synthetic and fully specified.)*

**4. Prepare the data** *(every transform is a reusable function)* — load → build X/y → one
fixed train/test split (test set set aside) → preprocessing (scaling, one-hot, polynomial
expansion).
→ `uber/modeling/pipeline.py`: `load_rides`, `make_xy`, `make_split` (holds out the test
set), `build_preprocessor` (`StandardScaler` + `OneHotEncoder`), `knn_pipeline`, `ols_pipeline`
(`PolynomialFeatures`).

**5. Shortlist promising models** — several model families are trained and compared (kNN,
OLS, polynomial OLS, Ridge, Lasso), with error/coefficient analysis.
→ `uber/modeling/arc.py`: `run_part2` (kNN + scaling), `run_part3` (curse of dimensionality),
`run_part4` (OLS), `run_part5` (polynomial OLS + coefficient t-stats via
`pipeline.ols_coef_table`), `run_part7` (Ridge/Lasso feature selection).

**6. Fine-tune the system** *(cross-validation; test set touched once)* — hyperparameters
chosen on validation folds; bias–variance swept; generalization confirmed by k-fold CV.
→ `uber/modeling/pipeline.py` `tune_k` (validation-partition k sweep); `uber/modeling/arc.py`
`run_part6` (U-curve over `PART6_K_GRID`), `run_part8` (`KFold` + `cross_validate`, 60/20/20
corroboration, leakage check).

**7. Present your solution** — a part-by-part metrics report, the U-curve chart, and the
interactive per-tier quote table.
→ `uber/modeling/pipeline.py` `format_report`; `uber/modeling/arc.py` `_plot_ucurve`;
`uber/cli/render.py` `render_options`; plus this README.

**8. Launch, monitor & maintain** — the model is **trained once and persisted** (`uber train`),
then **loaded to serve** (`uber simulate --model`), or retrained on launch as a fallback;
everything is wired in the composition root and guarded by unit tests + CI.
→ `uber/infrastructure/predictors.py` `ModelPredictor` (`save`/`load` + retrain-on-launch);
`uber/cli/app.py` `run_train` / `run_simulate` (composition root); `uber/cli/simulator.py`
`run_simulator`; `tests/` (hermetic suite); `.github/workflows/ci.yml`. *(Live drift monitoring
is out of scope for a teaching project; re-running `uber train` on fresh data stands in for
scheduled retraining.)*

> Checklist source: Aurélien Géron, *Hands-On Machine Learning with Scikit-Learn, Keras &
> TensorFlow*, Appendix B — "Machine Learning Project Checklist."

## Development

```bash
pip install -e ".[dev]"
pytest -q                            # the full test suite
ruff check . && ruff format --check .
pyright                              # informational; a known pandas/numpy stub baseline is tolerated
```

Tests never read the 1.2 GB `rides.csv`; they build small in-memory datasets, so the
suite is fast and hermetic. See [CONTRIBUTING.md](CONTRIBUTING.md) for the test-first
workflow and the dependency rule.

## License

Released under the [MIT License](LICENSE).
