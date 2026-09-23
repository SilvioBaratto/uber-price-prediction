"""Configuration constants for dataset generation.

Central place for paths, the RNG seed, dataset sizing, the tier tariffs, the pricing
dynamics (speed-by-hour + surge demand profile) and the Madrid bounding box. Noise
calibration is added here when Phase 3.4 lands (see SPEC.md and tasks/plan.md).
"""

from __future__ import annotations

from pathlib import Path

from uber.entities import RideTier

# --- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"          # generated CSVs live here (SPEC §2)
SOURCES_DIR = DATA_DIR / "sources"  # committed upstream source files
NCR_PATH = RAW_DIR / "ncr_ride_bookings.csv"

# Per-ride distractor columns (SPEC §2.3): each is sampled with replacement from the empirical
# distribution of the mapped NCR source column (``io.load_ncr`` / ``sampling.sample_empirical``).
# ``driver_rating`` is NOT here — it is a per-driver attribute (drivers.csv) joined onto rides.
NCR_DISTRACTOR_COLUMNS = {
    "payment_method": "Payment Method",
    "customer_rating": "Customer Rating",
    "avg_vtat": "Avg VTAT",
}

# Committed, frozen street-level location snapshot (built by scripts/extract_locations.py
# from the official Madrid callejero; see data/sources/SOURCE.md). Generation reads this so
# it stays offline and reproducible even though the upstream callejero updates daily.
STREETS_SOURCE = SOURCES_DIR / "madrid_streets.csv"

# Output CSV filenames (all written to RAW_DIR).
LOCATIONS_CSV = "madrid_locations.csv"
TIERS_CSV = "ride_tiers.csv"
DRIVERS_CSV = "drivers.csv"
RIDES_CSV = "rides.csv"

# --- Reproducibility / sizing ---------------------------------------------
SEED = 42
YEAR = 2024

# Rev 2: the ride count is EMERGENT from the driver simulation (§2.6). ``N_DRIVERS`` is the
# sizing knob (CLI ``--n-drivers``); the default is Madrid's full operating scale, which
# yields millions of rides (so ``rides.csv`` is gitignored / regenerated on demand).
N_DRIVERS = 15_000

# Legacy: the old uniform-timestamp model used a fixed ride count. Kept only so the retired
# ``sample_timestamps`` path and any pre-Rev-2 callers still import; unused by the driver sim.
N_RIDES = 50_000

# --- Ride tiers (SPEC §2.2) ------------------------------------------------
# Plausible but invented EUR tariffs, listed in increasing-price order. Adjustable here.
TIERS: tuple[RideTier, ...] = (
    RideTier("uberx",   "UberX",        4, 1.20, 0.90, 0.18, 0.90,  5.00),
    RideTier("green",   "Uber Green",   4, 1.20, 0.90, 0.18, 0.90,  5.00),
    RideTier("comfort", "Uber Comfort", 4, 1.80, 1.10, 0.22, 1.00,  7.00),
    RideTier("xl",      "UberXL",       6, 2.50, 1.50, 0.28, 1.20,  9.00),
    RideTier("black",   "Uber Black",   4, 4.00, 2.20, 0.45, 1.50, 12.00),
    RideTier("van",     "Uber Van",     6, 4.50, 2.40, 0.50, 1.50, 14.00),
)

# --- Pricing dynamics (SPEC §2.4 / §2.5) -----------------------------------
# Average road speed (km/h) by hour of day (index 0..23): slow in the rush peaks, fast
# overnight. Turns distance into duration. Flexible (SPEC §6); every entry is > 0 so any
# positive-distance ride has a positive duration.
SPEED_KMH_BY_HOUR: tuple[float, ...] = (
    35.0, 35.0, 35.0, 35.0, 35.0, 33.0,   # 00–05 free-flowing overnight
    28.0, 18.0, 16.0, 18.0,               # 06–09 morning ramp into rush
    24.0, 24.0, 24.0, 24.0,               # 10–13 midday
    25.0, 25.0, 23.0, 20.0,               # 14–17 afternoon
    16.0, 16.0, 18.0,                     # 18–20 evening rush
    26.0, 30.0, 32.0,                     # 21–23 late evening easing
)

# Surge demand profile (SPEC §2.5): deterministic base multiplier per time bucket, applied
# before the per-ride jitter. dow uses datetime.weekday(): 0=Mon .. 6=Sun.
SURGE_CLAMP = (1.0, 3.0)      # (min, max) surge after jitter
SURGE_OFFPEAK = 1.00          # weekday off-peak (10–12, 14–17 and other daytime)
SURGE_MORNING_RUSH = 1.35     # weekday 07–09
SURGE_EVENING_RUSH = 1.45     # weekday 18–20
SURGE_LATE_NIGHT = 1.10       # weekday 00–05
SURGE_WEEKEND_DAY = 1.05      # Sat/Sun daytime (incl. Sunday daytime)
SURGE_WEEKEND_NIGHT = 2.20    # Fri & Sat nights 21–03 (the expensive case)

# Per-ride surge jitter (SPEC §2.5): multiplicative lognormal(0, sigma) noise, median 1.0,
# so two identical trips at the same hour still differ before the [1.0, 3.0] clamp.
JITTER_SIGMA = 0.15

# Additive Gaussian price noise (EUR std, SPEC §2.4). Calibrated (Task 3.4) so in-sample OLS
# on the pinned signal feature set (numeric + one-hot tier + one-hot pickup_district;
# driver_id excluded) lands at R²≈0.85. Re-verified for Rev 2 (Task 3R.5): under the day-by-day
# driver simulation's *clustered* timestamps the value holds unchanged — SEED=42 gives R²≈0.851
# (robust ≈0.848–0.852 across seeds at n_drivers=250 ≈107k rides; R² is ~invariant to N, so it
# also characterizes the shipped 15k-driver run ≈6.4M rides). Even at NOISE_STD=0 OLS caps near
# R²≈0.869 — the base×surge product and the min_fare clamp are non-linear and cannot be fit
# exactly, which is the point of Parts 5+.
NOISE_STD = 3.75

# Target OLS R² band on the pinned signal feature set (SPEC §5.6, Task 3.4).
R2_BAND = (0.83, 0.87)

# --- Driver population & day-by-day simulation (SPEC §2.6, Rev 2) -----------
# Activity classes, right-skewed toward part-time (research: >50% of drivers work 1–5 h/week,
# ~80% <20 h/week). Per class: expected working days per week, and mean rides on a worked day
# (Poisson mean; ~2 trips/hour → part-time ≈ 6–12/day, full-time ≈ 20–25/day). Flexible (§6).
ACTIVITY_CLASSES = ("casual", "part_time", "full_time")
ACTIVITY_CLASS_WEIGHTS = (0.55, 0.30, 0.15)
WORKDAYS_PER_WEEK = {"casual": 1.5, "part_time": 3.0, "full_time": 5.5}
RIDES_PER_WORKDAY_MEAN = {"casual": 5.0, "part_time": 9.0, "full_time": 20.0}
WEEKEND_WORK_BOOST = 1.20  # weekends are busier: scale the daily work probability (capped at 1)

# Tenure / churn cohorts (§2.6): drivers split into ~3-month, ~6-month and full-year cohorts
# (Stanford/Uber: ~68% quit within 6 months, ~50% within a year). Per-driver uniform jitter of
# ±TENURE_JITTER_DAYS is added to the cohort length. Start dates are staggered for a stationary
# active fleet (see simulation.py).
TENURE_COHORTS = ("short", "medium", "long")
TENURE_COHORT_WEIGHTS = (0.55, 0.25, 0.20)
TENURE_COHORT_DAYS = {"short": 90, "medium": 180, "long": 366}
TENURE_JITTER_DAYS = 20

# Hourly ride-VOLUME profiles (relative weights over hours 0..23; distinct from price surge,
# though correlated). Rides concentrate at commute peaks + evenings on weekdays, and shift to
# late nights on weekends. Fri & Sat nights get an extra boost in simulation.py.
HOUR_VOLUME_WEEKDAY = (
    0.25, 0.15, 0.10, 0.08, 0.10, 0.25,   # 00–05 sparse overnight
    0.60, 1.10, 1.30, 0.95,               # 06–09 morning commute peak
    0.70, 0.80, 0.85, 0.80,               # 10–13 midday
    0.75, 0.80, 0.95, 1.15,               # 14–17 afternoon build
    1.35, 1.40, 1.15, 0.90,               # 18–21 evening peak
    0.60, 0.40,                           # 22–23 wind-down
)
HOUR_VOLUME_WEEKEND = (
    0.90, 0.75, 0.60, 0.45, 0.30, 0.25,   # 00–05 nightlife carry-over then quiet
    0.30, 0.45, 0.60, 0.75,               # 06–09 slow start
    0.90, 1.05, 1.10, 1.05,               # 10–13 brunch/midday
    1.00, 1.00, 1.05, 1.10,               # 14–17 afternoon
    1.15, 1.20, 1.25, 1.30,               # 18–21 evening
    1.35, 1.20,                           # 22–23 into the night
)
WEEKEND_NIGHT_VOLUME_BOOST = 1.5  # extra ride volume in 21–03 on Fri(4) & Sat(5) nights

# Fraction of a driver's rides whose pickup is biased into their home district (rest uniform).
HOME_DISTRICT_PICKUP_PROB = 0.40

# Per-driver rating (a weak distractor, joined onto rides in Phase 4). Range calibrated to the
# NCR ``Driver Ratings`` support; refined against NCR empirically in Task 4.1.
DRIVER_RATING_MIN, DRIVER_RATING_MAX = 3.5, 5.0

# --- Geography -------------------------------------------------------------
ROAD_FACTOR = 1.3  # haversine (straight line) -> road distance multiplier

# Madrid municipality bounding box (WGS84), used to validate the location table.
# Bounds taken with a small margin around the street-level coordinate extent
# (observed lat 40.3204..40.6432, lon -3.8367..-3.5296).
LAT_MIN, LAT_MAX = 40.31, 40.65
LON_MIN, LON_MAX = -3.84, -3.52
