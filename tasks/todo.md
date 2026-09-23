# TODO — Madrid synthetic data generation

Task list for `tasks/plan.md`. Check off as completed. ⛳ = checkpoint (pause for review).

## Phase 0 — Skeleton & contracts
- [~] 0.1 PARTIAL (done ahead of order to support the locations slice): `uber/__init__.py`,
      `config.py` (paths, seed, n_rides, year, road-factor, bbox), `entities.py` (`Location`
      only), `uber/io.py`, `uber/generate_data.py` CLI, `tests/`; `[dev]=["pytest"]` added;
      `pip install -e ".[dev]"` ok. Tier tariffs + `RideTier` added in Task 1.1. TODO later:
      surge table, speed-by-hour, noise std, R² band in config; `Ride` in entities.

## Phase 1 — ride_tiers.csv
- [x] 1.1 DONE — `RideTier` dataclass (entities), `config.TIERS` tariff table (SPEC §2.2,
      increasing-price order), `io.TIER_COLUMNS` + `io.build_tiers_df` (via `asdict`),
      `generate_data.generate` writes `data/raw/ride_tiers.csv`. CLI already had
      `--seed/--n-rides/--out-dir`.
      - AC met: 6 rows, SPEC §2.2 columns/types, price order uberx≤…≤van, no nulls.
      - Verified: `tests/test_tiers.py` 7/7 green; `pytest` 17/17; generator writes the CSV.
- [x] ⛳ **CHECKPOINT 1** — CLI + config + entities + io wiring proven on `ride_tiers.csv`.

## Phase 2 — madrid_locations.csv (STREET-LEVEL, per user directive)
- [x] 2.0 SOURCES FOUND & VALIDATED (CC BY 4.0): (a) callejero "Direcciones vigentes con
      coordenadas" (~33 MB, latin-1, DMS coords, 8 655 streets) for coordinates; (b) Barrios
      TopoJSON (UTF-8, committed) for district/barrio names. See data/sources/SOURCE.md.
- [x] 2.1 DONE — STREET-LEVEL map, 8 655 streets. `scripts/extract_locations.py` aggregates
      one median point per `COD_VIA`, modal district/barrio, barrio-name keyed by **COD_DISB**
      (fixes the Sol/NUM_BAR trap → 0 unmapped). Frozen snapshot committed at
      `data/sources/madrid_streets.csv`; raw 33 MB stays gitignored in `.scratch/`.
      `io.build_locations_df` reads the snapshot; `generate_data` writes
      `data/raw/madrid_locations.csv`. Schema: `location_id, street, neighborhood, district,
      lat, lon`. `uber/locations_data.py` (old 131-barrio literal) removed. bbox widened to
      lat∈[40.31,40.65] lon∈[-3.84,-3.52].
      - AC met: 8 655 ids (1-based, unique), 131 barrios, 21 districts, no nulls/empties,
        UTF-8 accents verified by code point, coords in bbox.
      - Verified: `tests/test_locations.py` 10/10 green; generator deterministic (identical
        md5 on re-run); Sol/Gran Vía spot-checks correct.

## Phase 3 — rides.csv (signal + target)
- [x] 3.1 DONE — `uber/pricing.py`: `haversine`, `road_distance` (×ROAD_FACTOR),
      `speed_kmh`/`duration_min` (config `SPEED_KMH_BY_HOUR`), `demand_profile` +
      `surge` (clamp 1.0–3.0), `price` (SPEC §2.4, `max(min_fare, …)`). Added
      `SPEED_KMH_BY_HOUR` + surge constants (`SURGE_*`, `SURGE_CLAMP`) to config.
      - AC met: haversine ±2% on known refs; surge clamps; price ≥ min_fare;
        Fri/Sat-night profile (2.2) > weekday late-morning (1.0).
      - Verified: `tests/test_pricing.py` 19/19 green; `pytest` 36/36; imports compile.
- [x] 3.2 DONE — `uber/sampling.py`: `make_rng` (seeded `default_rng`), `sample_timestamps`
      (uniform over 2024, leap-year-safe), `sample_od_pairs` (1-based ids, offset-mod trick →
      always distinct, no rejection loop), `sample_tiers` (uniform), `sample_jitter`
      (lognormal(0, `JITTER_SIGMA`=0.15), median 1.0). All take an rng + return numpy arrays.
      - AC met: same seed ⇒ identical arrays; OD ids ∈ [1, 8655] & pickup≠dropoff; tiers &
        all 12 months covered.
      - Verified: `tests/test_sampling.py` 11/11 green; `pytest` 47/47; imports compile.
- [x] 3.3 DONE — `generate_data.build_rides_df` (+ `SIGNAL_COLUMNS`) wires locations+tiers+
      pricing+sampling into the 13 signal cols of SPEC §2.3 + `price_eur`. One seeded rng,
      fixed draw order (timestamps→OD→tiers→jitter→noise); per-ride physics reuse the
      canonical `pricing` scalar funcs (no formula duplication). `generate(out_dir, seed,
      n_rides)` now writes `rides.csv`; CLI `--seed/--n-rides` wired through. Added
      `NOISE_STD` (placeholder 3.0) + `R2_BAND` to config.
      - AC met: 50 000 rows, 0 nulls, price>0 & ≥min_fare, duration>0, distance=haversine×
        ROAD_FACTOR of the two ids (verified per-row). Full 50k generation ~1.8s.
      - Verified: `pytest -k "rides_integrity or geo or reproducible"` 4/4; `test_rides.py`
        8/8; same seed ⇒ equal df + byte-identical CSV; `pytest` 55/55.
- [x] 3.4 DONE — Calibrated `NOISE_STD=3.75` → in-sample OLS R²=0.8504 (SEED=42, 50k) on the
      pinned feature set (numeric + one-hot tier + one-hot pickup_district), robust ≈0.850
      across seeds. Added `R2_BAND=(0.83,0.87)`; documented the value + the ~0.872 zero-noise
      cap (multiplicative surge + min_fare clamp) in config. Committed dataset regenerated.
      - AC met: R²∈band; uberx 5–10km weekend-night mean ≈1.86× weekday 10–12 (≥1.5×).
      - Verified: `pytest -k "signal_band or time_of_call"` 2/2; `pytest` 57/57; committed
        rides.csv R²=0.8504.
- [x] ⛳ **CHECKPOINT 2** — rides.csv reproducible, signal-calibrated (R²≈0.85), time-of-call
      pricing bites. Critical pedagogical gate reached.

## Phase 3R — driver-centric ride simulation (SPEC Rev 2; supersedes uniform-timestamp model)
SPEC/plan approved; **O1 = Option C** (`N_DRIVERS=15000`, `rides.csv` gitignored).
Tasks 3.1–3.3 are reused; `sample_timestamps` is retired for the simulation's hour-of-day draw.
- [x] 3R.1 DONE — `Driver` frozen/slotted dataclass (entities); driver-population config block
      (`N_DRIVERS=15000`, `ACTIVITY_CLASSES`/weights, `WORKDAYS_PER_WEEK`, `RIDES_PER_WORKDAY_MEAN`,
      `TENURE_COHORTS`/weights/days + jitter, `HOUR_VOLUME_WEEKDAY/WEEKEND` + Fri/Sat-night boost,
      `HOME_DISTRICT_PICKUP_PROB`, `DRIVER_RATING_MIN/MAX`); `N_RIDES` demoted to legacy comment.
      - AC met: imports; param shapes/weights coherent (right-skewed, means sorted, profiles 24×>0).
      - Verified: `tests/test_driver_config.py` 6/6; `pytest` 63/63.
- [x] 3R.2 DONE — `uber/simulation.py` `build_drivers` (roster + tenure/churn): right-skewed
      activity class, cohort tenure + jitter clipped to 2024, **staggered starts** for a stationary
      active fleet (negative start = incumbent, window clipped in-year), home_district weighted by
      street share, per-driver rating; `io.DRIVER_COLUMNS` added. Fully vectorized (numpy).
      - AC met: reproducible; class mix ≈ target (±0.04); churned fraction ~0.3–0.95; cohorts span
        (min<130d, max>300d); daily active count flat (min/max within ±25% of mean → no Jan cliff).
      - Verified: roster/tenure/churn tests green (part of `test_simulation.py`).
- [x] 3R.3 DONE — `uber/simulation.py` `simulate_ride_events` (per-day events): Bernoulli(work |
      class, weekend boost) × Poisson(rides | class) ≥1; ride hour by inversion of a 7×24 volume
      CDF (weekday/weekend + Fri/Sat-night boost); minute/second uniform. Returns `(events,
      active_days, n_rides)`; every driver ≥1 ride; fixed RNG draw order despite data-dependent
      sizes. Fully vectorized (no per-ride loop).
      - AC met: >50% drivers ≥2 rides, heavy tail (max>5×median); rides only within tenure; hour
        non-uniform (19h > 2×4h); all 12 months + 7 weekdays covered; reproducible.
      - Verified: `tests/test_simulation.py` 15/15; `pytest` 78/78 (no regressions).
- [x] 3R.4 DONE — vectorized rewrite of `generate_data`: `build_dataset(rng, locations,
      n_drivers)→(drivers, rides)`; `build_rides_df(rng, locations, drivers, events)` replaces the
      per-row loop with numpy array ops (OD with home-district pickup bias, tier/jitter/noise in a
      fixed draw order); rides sorted chronologically → contiguous `ride_id`; `driver_id` added to
      `SIGNAL_COLUMNS` (excluded from OLS). Emits `drivers.csv` (`io.DRIVER_COLUMNS`, `active_days`/
      `n_rides` folded back). Added vectorized `pricing.haversine`/`duration_min_array`/
      `demand_table`/`surge_array`/`price_array` (reuse scalar formula — guarded by array==scalar
      tests) + `config.DRIVERS_CSV`. `generate(..., n_drivers)`; CLI `--n-drivers`; fixed the two
      generate_writes_* tests + rewrote test_rides/test_signal to the new API.
      - AC met: 4 CSVs; driver_id FK valid (id sets coincide); emergent rows > 50k (250 drivers →
        ~107k; ~427/driver ⇒ ~6.4M at 15k); §2.4 holds per-row; byte-identical re-run (rides+drivers).
      - Verified: `test_rides.py` 12/12; `test_pricing.py` 24/24; `pytest` 82→87.
- [x] 3R.5 DONE — NOISE_STD=3.75 **re-verified unchanged** under clustered timestamps (pinned
      feature set, driver_id excluded): SEED=42 R²=0.8510, robust 0.848–0.852 across seeds (n_drivers
      =250 ≈107k rows; R² ~invariant to N). Zero-noise cap R²≈0.869. Recorded in `config.NOISE_STD`
      comment. driver_rating corr≈0 is validated in Phase 4.1 (the column is joined onto rides there).
      - AC met: R²∈[0.83,0.87] robust; time-of-call weekend-night/weekday-morning = 1.81× (≥1.5×).
      - Verified: `pytest -k "signal_band or time_of_call"` 2/2; `pytest` 87/87.
- [x] ⛳ **CHECKPOINT 2R** — driver-centric rides.csv + drivers.csv reproducible (byte-identical),
      R² re-verified in band (≈0.85), panel/tenure/churn structure valid, generation fully
      vectorized. `pytest` 87/87 green.

## Phase 4 — distractors + leakage
- [x] 4.1 DONE — `config.NCR_DISTRACTOR_COLUMNS` maps each per-ride distractor→NCR source col;
      `io.load_ncr()` returns null-free empirical supports per column (R3, `na_values=["null"]`);
      `sampling.sample_empirical(rng, values, n)` draws with replacement (rng-first, module
      convention). `generate_data`: added `DISTRACTOR_COLUMNS`/`RIDE_COLUMNS`; `build_rides_df`
      now samples `payment_method`/`customer_rating`/`avg_vtat` **after** the signal draws (so
      signal cols stay byte-identical) and joins per-driver `driver_rating` (constant within a
      driver); `build_dataset` loads NCR once. Updated the two schema tests to `RIDE_COLUMNS`.
      - AC met: |corr with price| ≤ 0.0064 each (driver_rating −0.0005, customer_rating −0.0004,
        avg_vtat −0.0029, payment_method max dummy 0.0064); payment ∈ 5 NCR cats, customer_rating
        ∈[3,5], avg_vtat∈[2,20], driver_rating∈[3.5,5.0]; 0 nulls; pinned OLS R²=0.8510 (unchanged,
        in band). driver_rating verified constant per driver + matches roster.
      - Verified: `tests/test_distractors.py` 12/12; `pytest` 99/99 (was 87); compileall+imports OK.
- [x] 4.2 DONE — `config.COMMISSION_RATE=0.25` + `COMMISSION_NOISE_STD=0.02`; `generate_data`
      adds `LEAKAGE_COLUMNS=["commission_eur"]`, extends `RIDE_COLUMNS` (signal+distractors+
      leakage); `build_rides_df` derives `commission_eur = 0.25×price_eur + N(0,0.02)` right
      after `price_eur` (its rng draw comes last, so signal + distractor cols stay byte-identical).
      Updated the two schema pins (test_rides `RIDE_COLUMNS`, test_distractors → position-based).
      - AC met: leakage R²=0.999991 (>0.999); signal R²=0.8510 unchanged (in band); commission =
        0.25×price + noise (residual std 0.0200, min commission 1.16 > 0, 0 nulls); excluded from
        SIGNAL_COLUMNS + DISTRACTOR_COLUMNS.
      - Verified: `tests/test_leakage.py` 4/4; `pytest` 103/103 (was 99); compileall+imports OK.
- [x] ⛳ **CHECKPOINT 3** — distractors inert, leakage trap fires. Reached: `pytest` 103/103 green.

## Phase 5 — full validation & docs
- [ ] 5.1 Consolidate all 9 SPEC §5 assertions + byte-identical reproducibility test;
      reconcile README to `data/raw/` layout.
      - AC: clean `data/raw/` → `python -m uber.generate_data && pytest` all green.
      - Verify: delete generated CSVs, regenerate, `pytest` → 0 failures.
- [ ] ⛳ **CHECKPOINT 4 (final)** — full pipeline reproducible end-to-end; ready to commit.

## Open risks (see plan §7)
- R1 coordinate sourcing (online fetch, license, normalization) · R2 R² band sensitivity ·
  R3 NCR nulls/quoting · R4 zero-distance OD.
