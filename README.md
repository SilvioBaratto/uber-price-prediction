# uber-price-prediction

Companion project to the series **"Let's Build Uber's Algorithm"** (neuroespresso channel).

## Project goal

Simulate, **in the terminal**, a ride-hailing platform in **Madrid**.

The user imagines living in Madrid, picks a start point **A** and a destination **B**
in the city, and the program shows the **full list of available rides** — UberX,
Comfort, XL, Black, Van… — each with its **estimated price** and ETA. Exactly what you
see when you open the app and all the options appear.

Each ride's price is **not hand-written**: it is produced by the regression model built
throughout the series. Every episode improves the estimate; in the end the model powers
the simulator.

> Simplified for teaching: this is not Uber's real price list, it's the core idea
> (estimate a price from a few variables and present the options) rebuilt as a project
> you can run from the terminal.

## How it works (flow)

1. pick **A** and **B** among Madrid's points (or type them);
2. the program computes the trip **distance** (km) and **estimated duration** (min);
3. it reads the **current conditions**: hour of day, day of week, surge/demand;
4. for **each available tier**, the model **predicts the price**;
5. the terminal prints the **full list**: tier, price (€), ETA.

## The data

Three tables, each with a precise role in the flow above.

### `madrid_locations.csv` — the map (points in Madrid)
Used in step 2 to compute the A→B distance.

| Column | Type | Note |
|---|---|---|
| `location_id` | int | key |
| `neighborhood` | str | Madrid neighborhood (barrio) |
| `district` | str | district (distrito) it belongs to |
| `lat`, `lon` | float | coordinates, for distance (haversine × road factor) |

### `ride_tiers.csv` — the ride catalog
Used in step 4: which options exist and with which price parameters.

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
Used to **train** and validate the price model of steps 4–5. Here the price
**genuinely depends** on the features (distance, duration, tier, surge, hour…).

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

> ⚠️ The Kaggle dataset downloaded at the start (*NCR Uber 2024*, India) is **not
> suitable** as the target data: it's synthetic dashboard data, the price is
> uncorrelated with everything (R² ≈ 0), and the city is wrong. `rides.csv` is a
> **Madrid-specific** dataset with real price signal. (We do reuse NCR's empirical
> column distributions to make realistic distractor features — see `SPEC.md`.)

## The model, part by part

The arc teaches a **progression**: each part lowers the price-estimation error (RMSE).
The final model is the one the simulator uses in step 4.

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

The RMSE numbers shown in the shorts are produced by `run.py` on the real data — never
by hand.

## Structure

```
data/madrid_locations.csv   the map of Madrid points
data/ride_tiers.csv         the tier catalog and price parameters
data/raw/                   raw / generated data, e.g. rides.csv
data/processed/             cleaned dataset and features
uber/                       the code: data generation, models, evaluation, simulator
output/                     numbers and charts for the shorts (out of git)
run.py                      orchestrator: from raw data to each part's numbers
```

## Usage

```bash
python -m venv .venv && .venv/Scripts/activate   # Windows
pip install -e .
python -m uber.generate_data   # generate the three CSVs
python run.py                  # train the models and print RMSE part by part
python -m uber.simulator       # start the terminal simulator (A → B in Madrid)
```
