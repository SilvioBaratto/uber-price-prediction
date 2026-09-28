"""Day-by-day driver-population simulation.

Two pure, fully-vectorized steps (no per-ride Python loops — the shipped default is 15k
drivers ⇒ millions of rides, so every array op stays in numpy):

- ``build_drivers`` — the year's roster: right-skewed activity classes, staggered tenure
  windows with churn (a stationary active fleet, no January cliff), a home district and a
  per-driver rating.
- ``simulate_ride_events`` — for each driver × day in tenure, Bernoulli(work) × Poisson(rides
  per worked day, ≥1), with the ride hour drawn from the hourly volume profile (not uniform).

All randomness flows through the caller's seeded ``numpy.random.Generator`` in a FIXED draw
order, so the same seed reproduces byte-identical output despite the data-dependent sizes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from uber.domain import config
from uber.infrastructure import repositories


def _year_bounds() -> tuple[np.datetime64, int]:
    """Return (first day of the configured year, number of days in it)."""
    start = np.datetime64(f"{config.YEAR}-01-01", "D")
    days = int((np.datetime64(f"{config.YEAR + 1}-01-01", "D") - start).astype(int))
    return start, days


def _normalized(weights: tuple[float, ...]) -> np.ndarray:
    """Weights as a probability vector (numpy ``choice`` needs an exact unit sum)."""
    w = np.asarray(weights, dtype=float)
    return w / w.sum()


def build_drivers(
    rng: np.random.Generator, n_drivers: int, locations: pd.DataFrame
) -> pd.DataFrame:
    """Build the year's driver roster, one row per driver.

    ``active_days`` / ``n_rides`` are placeholders (0) here — they are filled from the ride
    simulation by :func:`simulate_ride_events`. Columns match :data:`repositories.DRIVER_COLUMNS`.
    """
    n = int(n_drivers)
    year_start, year_days = _year_bounds()

    # 1. Activity class, right-skewed toward casual.
    class_idx = rng.choice(
        len(config.ACTIVITY_CLASSES), size=n, p=_normalized(config.ACTIVITY_CLASS_WEIGHTS)
    )

    # 2. Tenure cohort + per-driver jitter, clipped to at most the whole year.
    coh_idx = rng.choice(
        len(config.TENURE_COHORTS), size=n, p=_normalized(config.TENURE_COHORT_WEIGHTS)
    )
    cohort_days = np.array([config.TENURE_COHORT_DAYS[c] for c in config.TENURE_COHORTS])
    jitter = rng.integers(-config.TENURE_JITTER_DAYS, config.TENURE_JITTER_DAYS + 1, size=n)
    tenure_len = np.clip(cohort_days[coh_idx] + jitter, 1, year_days)

    # 3. Staggered start so the active fleet is stationary (no January cliff): draw a 0-based
    #    start uniformly over [-(tenure_len-1), year_days-1] — a negative start means the driver
    #    was already active before Jan 1 (an incumbent) — then clip the window into the year. For
    #    a start uniform over this full range every calendar day gets identical expected coverage.
    span = year_days + tenure_len - 1
    start_day = np.floor(rng.random(n) * span).astype(np.int64) - (tenure_len - 1)
    active_start = np.maximum(start_day, 0)
    active_end = np.minimum(start_day + tenure_len - 1, year_days - 1)

    # 4. Per-driver rating (a weak distractor; not a price signal).
    rating = np.round(rng.uniform(config.DRIVER_RATING_MIN, config.DRIVER_RATING_MAX, size=n), 1)

    # 5. Home district, weighted by each district's share of streets (denser districts host more).
    counts = locations["district"].value_counts()
    districts = counts.index.to_numpy()
    home_idx = rng.choice(len(districts), size=n, p=(counts.to_numpy(dtype=float) / counts.sum()))

    tenure_start = (year_start + active_start.astype("timedelta64[D]")).astype(str)
    tenure_end = (year_start + active_end.astype("timedelta64[D]")).astype(str)

    return pd.DataFrame(
        {
            "driver_id": np.arange(1, n + 1, dtype=np.int64),
            "home_district": districts[home_idx],
            "activity_class": np.asarray(config.ACTIVITY_CLASSES)[class_idx],
            "tenure_start": tenure_start,
            "tenure_end": tenure_end,
            "active_days": np.zeros(n, dtype=np.int64),
            "n_rides": np.zeros(n, dtype=np.int64),
            "driver_rating": rating,
        },
        columns=repositories.DRIVER_COLUMNS,
    )


def _hour_cdf_table() -> np.ndarray:
    """Per-day-of-week (7×24) CDF of ride volume over the hour of day.

    Weekdays use the weekday profile, Sat/Sun the weekend one; Fri & Sat nights (21–03) get an
    extra volume boost. Each row is normalized to end at 1.0 so it can be sampled by inversion.
    """
    weekday = np.asarray(config.HOUR_VOLUME_WEEKDAY, dtype=float)
    weekend = np.asarray(config.HOUR_VOLUME_WEEKEND, dtype=float)
    vol = np.stack([weekday if d < 5 else weekend for d in range(7)])
    night = np.zeros(24, dtype=bool)
    night[[21, 22, 23, 0, 1, 2, 3]] = True
    for d in (4, 5):  # Fri, Sat nights
        vol[d, night] *= config.WEEKEND_NIGHT_VOLUME_BOOST
    cdf = np.cumsum(vol, axis=1)
    return cdf / cdf[:, -1:]


def simulate_ride_events(
    rng: np.random.Generator, drivers: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Simulate every ride each driver gives across their tenure.

    Returns ``(events, active_days, n_rides)`` where ``events`` has columns ``driver_id`` and
    ``timestamp`` (second resolution, in driver/day order — the caller sorts chronologically),
    and ``active_days`` / ``n_rides`` are per-driver counts aligned to ``drivers`` row order.
    Every driver produces at least one ride.
    """
    n = len(drivers)
    year_start, _year_days = _year_bounds()

    start_day = (drivers["tenure_start"].to_numpy(dtype="datetime64[D]") - year_start).astype(
        np.int64
    )
    end_day = (drivers["tenure_end"].to_numpy(dtype="datetime64[D]") - year_start).astype(np.int64)
    tenure_len = end_day - start_day + 1  # >= 1 by construction

    # Per-driver activity parameters, looked up by class.
    cls_to_idx = {c: i for i, c in enumerate(config.ACTIVITY_CLASSES)}
    class_idx = drivers["activity_class"].map(cls_to_idx).to_numpy()
    work_per_week = np.array([config.WORKDAYS_PER_WEEK[c] for c in config.ACTIVITY_CLASSES])
    rides_mean = np.array([config.RIDES_PER_WORKDAY_MEAN[c] for c in config.ACTIVITY_CLASSES])
    work_prob = work_per_week[class_idx] / 7.0
    ride_mean = rides_mean[class_idx]

    # Expand the roster into one row per (driver, calendar day in tenure).
    total_days = int(tenure_len.sum())
    driver_of_day = np.repeat(np.arange(n), tenure_len)
    block_start = np.repeat(np.cumsum(tenure_len) - tenure_len, tenure_len)
    day_of_year = start_day[driver_of_day] + (np.arange(total_days) - block_start)
    dow = day_of_year % 7  # config.YEAR (2024) starts on a Monday, so day 0 == Mon(0)

    # Bernoulli(work) per driver-day, with weekends busier (probability capped at 1).
    p = work_prob[driver_of_day].copy()
    weekend = dow >= 5
    p[weekend] = np.minimum(p[weekend] * config.WEEKEND_WORK_BOOST, 1.0)
    worked = rng.random(total_days) < p

    # Poisson(rides) on each worked day, at least one ride.
    w_driver = driver_of_day[worked]
    w_day = day_of_year[worked]
    w_dow = dow[worked]
    rides_per_day = np.maximum(1, rng.poisson(ride_mean[w_driver]))

    # Guarantee every driver appears: a driver who happened to work zero days gets a single ride
    # on their first tenure day (rare, and needs no extra draws so the RNG order is unchanged).
    idle = np.flatnonzero(np.bincount(w_driver, minlength=n) == 0)
    if idle.size:
        w_driver = np.concatenate([w_driver, idle])
        w_day = np.concatenate([w_day, start_day[idle]])
        w_dow = np.concatenate([w_dow, start_day[idle] % 7])
        rides_per_day = np.concatenate(
            [rides_per_day, np.ones(idle.size, dtype=rides_per_day.dtype)]
        )

    active_days = np.bincount(w_driver, minlength=n).astype(np.int64)
    n_rides = np.bincount(w_driver, weights=rides_per_day, minlength=n).astype(np.int64)

    # Expand worked days into individual rides.
    ride_driver = np.repeat(w_driver, rides_per_day)
    ride_day = np.repeat(w_day, rides_per_day)
    ride_dow = np.repeat(w_dow, rides_per_day)
    total = ride_driver.size

    # Ride hour from the volume profile (inversion per day-of-week); minute/second uniform.
    cdf = _hour_cdf_table()
    u = rng.random(total)
    hour = np.empty(total, dtype=np.int64)
    for d in range(7):
        mask = ride_dow == d
        if mask.any():
            hour[mask] = np.searchsorted(cdf[d], u[mask], side="right")
    np.clip(hour, 0, 23, out=hour)
    minute = rng.integers(0, 60, size=total)
    second = rng.integers(0, 60, size=total)

    seconds = ride_day * 86400 + hour * 3600 + minute * 60 + second
    epoch = np.datetime64(f"{config.YEAR}-01-01T00:00:00", "s")
    timestamp = epoch + seconds.astype("timedelta64[s]")

    events = pd.DataFrame({"driver_id": (ride_driver + 1).astype(np.int64), "timestamp": timestamp})
    return events, active_days, n_rides
