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
uber simulate                        # interactive A -> B ride-price simulator
```

Every command is also reachable as `python -m uber <subcommand>`. Run
`uber --help` (or `uber <subcommand> --help`) for the full options.

The simulator **retrains the price model on every launch** from `rides.csv` (no model
file is shipped). On the full dataset that read can be large — cap it for a snappy start:

```bash
uber simulate --max-train-rows 200000
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
The final model (Part 5's polynomial OLS) is the one the simulator uses.

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
