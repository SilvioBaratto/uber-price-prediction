"""Signal-band calibration and time-of-call pricing (SPEC §5.6 / §5.9, Task 3.4).

The OLS feature set is *pinned* here (A3) so the R² band is stable across runs: numeric
``distance_km, duration_min, surge_multiplier, hour, day_of_week, month`` plus one-hot
``tier`` and one-hot ``pickup_district``. ``driver_id`` is deliberately excluded — it is
metadata, not a price signal (SPEC §2.3). The R² band is ~invariant to dataset size, so the
checks run on a reduced driver count rather than the shipped 15k (SPEC §2.6).
"""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from uber import config, generate_data, io, sampling

PINNED_NUMERIC = ["distance_km", "duration_min", "surge_multiplier", "hour", "day_of_week", "month"]

N_DRIVERS_TEST = 250  # yields ~100k rides — plenty for a stable in-sample R²


@pytest.fixture(scope="module")
def rides() -> pd.DataFrame:
    locations = io.build_locations_df()
    _drivers, rides = generate_data.build_dataset(sampling.make_rng(config.SEED), locations, N_DRIVERS_TEST)
    return rides


def _ols_r2(rides: pd.DataFrame) -> float:
    """In-sample OLS R² on the pinned signal feature set."""
    dummies = pd.get_dummies(rides[["tier", "pickup_district"]], drop_first=True)
    X = pd.concat([rides[PINNED_NUMERIC], dummies], axis=1).to_numpy(dtype=float)
    y = rides["price_eur"].to_numpy(dtype=float)
    return LinearRegression().fit(X, y).score(X, y)


def test_signal_band(rides: pd.DataFrame) -> None:
    lo, hi = config.R2_BAND
    r2 = _ols_r2(rides)
    assert lo <= r2 <= hi, f"OLS R²={r2:.4f} outside band {config.R2_BAND}"


def test_time_of_call(rides: pd.DataFrame) -> None:
    hour, dow = rides["hour"], rides["day_of_week"]
    # Fri & Sat nights (21:00–03:59): Fri evening, Sat small hours + evening, Sun small hours.
    weekend_night = (
        ((dow == 4) & (hour >= 21))
        | ((dow == 5) & ((hour <= 3) | (hour >= 21)))
        | ((dow == 6) & (hour <= 3))
    )
    weekday_late_morning = (dow <= 4) & hour.isin([10, 11])

    # Equivalent trips: same tier + distance band.
    band = (rides["tier"] == "uberx") & rides["distance_km"].between(5.0, 10.0)
    night_mean = rides.loc[band & weekend_night, "price_eur"].mean()
    morning_mean = rides.loc[band & weekday_late_morning, "price_eur"].mean()

    assert night_mean >= 1.5 * morning_mean, (
        f"weekend-night mean {night_mean:.2f} not >= 1.5x weekday late-morning {morning_mean:.2f}"
    )
