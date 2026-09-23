# Implementation Plan — Madrid synthetic data generation

Derived from `SPEC.md` (status: awaiting approval, 2026-09-23). Scope is the
**data-generation step only**: build the three CSVs (`ride_tiers.csv`,
`madrid_locations.csv`, `rides.csv`) plus the generator package and the pytest
validation suite. **Non-goals:** no ML model, no simulator, no external API calls.

Companion task list: `tasks/todo.md`.

---

## 1. Starting state

- `uber/` package is **empty** — greenfield build.
- `data/raw/ncr_ride_bookings.csv` exists (150 000 rows) — source of the empirical
  distributions for the distractor features (`Payment Method`, `Driver Ratings`,
  `Customer Rating`, `Avg VTAT`).
- `pyproject.toml` declares `pandas`, `numpy`, `scikit-learn`, `matplotlib`; **no**
  `[project.optional-dependencies].dev` yet (SPEC §5 requires `pytest` there).
- `data/raw/.gitkeep`, `data/processed/.gitkeep`, `output/.gitkeep` present.

## 2. Authoritative decisions & assumptions

**From SPEC (locked):**
- 3 CSVs, all written to `data/raw/` (SPEC §2, "Locked decisions").
- 6 tiers with the exact starting tariffs in SPEC §2.2.
- ~50 000 rides over the 2024 horizon; **8 655 streets** across 131 barrios / 21 districts
  (street-level location table, per user directive — supersedes SPEC's ~130-barrio table).
- Ground-truth price formula (SPEC §2.4) and single surge mechanism (SPEC §2.5).
- OLS on signal features must land **R² ∈ [0.83, 0.87]** (target ≈ 0.85).
- Distractors sampled from NCR empirical distributions; `commission_eur` leakage trap.
- Fixed configurable seed → reproducible; full pytest suite (9 assertions, SPEC §5).

**Assumptions taken as sensible defaults (SPEC §6 marks these "flexible"):**
- **A1 — Location table sourcing (CONFIRMED & BUILT — street-level, per user directive):**
  the table is **street-level** (8 655 official *viales*), not the ~130 barrios of the SPEC.
  Two official Ayuntamiento de Madrid datasets, both **CC BY 4.0** (see data/sources/SOURCE.md):
  1. **Callejero oficial — "Direcciones vigentes con coordenadas"** (dataset page
     `https://datos.madrid.es/dataset/213605-0-callejero-oficial-madrid`): per-address CSV
     (~33 MB, latin-1) with `LATITUD`/`LONGITUD` in **DMS** and `DISTRITO`/`BARRIO` codes.
     The coordinate source. Too large + daily-updated → **not committed**; auto-downloaded to
     `.scratch/` when the extractor runs.
  2. **Barrios TopoJSON** (`https://datos.madrid.es/dataset/300496-0-barrios-madrid`, UTF-8,
     committed) → maps codes to official names: `CODDIS`→district, and **`COD_DISB`** (e.g.
     `"1-6"`) →barrio. Keying the barrio on `COD_DISB` (not `NUM_BAR`) is essential — Sol has
     `NUM_BAR=8` but `COD_DISB="1-6"`; using `NUM_BAR` silently drops all of Sol's streets.

  Method (no geo libs): parse DMS→decimal, group addresses by `COD_VIA`, take the **median**
  point + **modal** district/barrio per street → `(location_id, street, neighborhood,
  district, lat, lon)`. Extraction is a **one-time authoring step**
  (`scripts/extract_locations.py`); its **frozen output is committed** at
  `data/sources/madrid_streets.csv`, and the generator reads that snapshot, so generation is
  **100% offline, deterministic, and byte-reproducible** even though the callejero changes
  daily (consistent with SPEC §6 — the offline boundary concerns generation + ride-hailing
  /fare data, not public geographic reference data). Built & validated: 8 655 streets, 131
  barrios, 21 districts, 0 unmapped; Sol → *Plaza de la Puerta del Sol* 40.4169, −3.7029;
  *Calle Gran Vía* 40.4204, −3.7048. See risk R1.

  Note on names: both sources decode as documented (callejero **latin-1**, TopoJSON
  **UTF-8**); the derived snapshot is written UTF-8, accents verified by **code point**
  (a Windows cp1252 console *displays* correct accents as `�` — verify by code point, not
  console output).
- **A2 — Single I/O point:** `uber/io.py` is the only module that touches the filesystem.
  It loads NCR and writes the 3 CSVs; `sampling.py` receives already-loaded arrays.
- **A3 — OLS in tests** uses numpy `lstsq` (or the already-present scikit-learn) on a
  **pinned** signal-feature set (see Task 3.4) so the R² band is stable across runs.
- **A4 — Doc reconciliation (confirmed by user):** all three CSVs are written to
  `data/raw/` per SPEC's locked decision; README's structure block (which shows two at the
  `data/` top level) is fixed in Phase 5.

## 3. Component dependency graph

```
config.py ─┐  (constants: seed, n_rides, paths, tariffs, road-factor,
           │   surge table, speed-by-hour, noise std, R² band, NCR + streets path)
entities.py┤  (pure dataclasses: Location, RideTier, Ride)
madrid_    ┤  (A1: 8 655 streets + coords, committed CSV snapshot read by io.py)
 streets.csv│
           ├─> pricing.py   (pure: haversine, road-factor, speed-by-hour,
           │                 demand_profile→surge, price formula)
           ├─> sampling.py  (seeded RNG: timestamps, OD pairs, tiers, jitter,
           │                 empirical sampling of NCR distractors)
           └─> io.py        (pandas read NCR + streets snapshot / write 3 CSVs — single I/O point)
                    │
                    └─> generate_data.py  (orchestration + CLI: python -m uber.generate_data)
                              │
                              └─> tests/test_generation.py  (9 assertions)
```

Topological layers: **(1)** `config`, `entities`, `madrid_streets.csv` → **(2)** `pricing`,
`sampling`, `io` → **(3)** `generate_data` → **(4)** tests. Work is sliced *vertically*
across these layers (below), not layer-by-layer.

## 4. Vertical slicing strategy

Each phase delivers a **runnable, verifiable increment** — a complete path from config to
a written artifact (or a validated behaviour), with its own tests — rather than a
horizontal layer. Ordering follows the data dependencies: tiers (self-contained) →
locations (needed by rides) → rides signal (needs both) → distractors/leakage → full
validation.

---

## 5. Phases & tasks

### Phase 0 — Skeleton & contracts
**Goal:** installable package, shared constants and types, dev tooling.

- **Task 0.1 — Package skeleton + config + entities + dev deps**
  - Create `uber/__init__.py`, `uber/config.py`, `uber/entities.py`, `tests/`.
  - `config.py`: SEED, N_RIDES, OUT_DIR (`data/raw`), NCR_PATH, ROAD_FACTOR, tier tariff
    table (SPEC §2.2), surge `demand_profile` table (SPEC §2.5), speed-by-hour table,
    NOISE_STD placeholder, R²_BAND = (0.83, 0.87), YEAR = 2024.
  - `entities.py`: frozen dataclasses `Location`, `RideTier`, `Ride` (pure, no I/O).
  - Add `[project.optional-dependencies].dev = ["pytest"]` to `pyproject.toml`.
  - **Acceptance:** `pip install -e ".[dev]"` succeeds; `python -c "import uber, uber.config, uber.entities"` exits 0; `pytest` collects (0 tests OK).
  - **Verify:** run the three commands above; all exit 0.

### Phase 1 — `ride_tiers.csv` (simplest end-to-end path)
**Goal:** prove the full config→entities→io→CLI pipeline on the smallest artifact.

- **Task 1.1 — Tiers slice + io write path + CLI scaffold**
  - `io.py`: `write_csv(df, name)` into `OUT_DIR`; `build_tiers_df()` from config tariffs.
  - `generate_data.py`: CLI (`argparse`: `--seed`, `--n-rides`, `--out-dir`) that writes
    `ride_tiers.csv`. Other CSVs stubbed as TODO for later phases.
  - **Acceptance:** `python -m uber.generate_data` writes `data/raw/ride_tiers.csv` with
    columns/types of SPEC §2.2, 6 rows, prices ordered uberx ≤ green ≤ comfort ≤ xl ≤
    black ≤ van.
  - **Verify:** run generator; `pytest tests/test_generation.py -k tiers` passes
    (columns, 6 rows, min_fare/base ordering, no nulls).
  - **⛳ CHECKPOINT 1:** CLI + config + entities + io wiring proven on a real file.

### Phase 2 — `madrid_locations.csv` (STREET-LEVEL — ✅ DONE)
**Goal:** extract, freeze, and emit the street-level map used for A→B distance. Extraction is
a one-time authoring step; the committed generator reads a frozen snapshot and stays offline (A1).

- **Task 2.0 — Extract streets from the callejero (authoring-time) — ✅ DONE**
  - Sources **located and validated** (A1): callejero addresses CSV (coordinates, DMS) +
    Barrios TopoJSON (names). `scripts/extract_locations.py` auto-downloads the raw callejero
    to `.scratch/` (gitignored), parses DMS→decimal, groups by `COD_VIA`, takes median point +
    modal district/barrio (barrio keyed by **`COD_DISB`**), and writes the frozen snapshot
    `data/sources/madrid_streets.csv`. Source URLs/date/license recorded in the script header
    and `data/sources/SOURCE.md`.
  - **Acceptance met:** 8 655 streets / 131 barrios / 21 districts, **0 unmapped**; all coords
    inside the Madrid bbox; committed snapshot makes generation reproducible offline.
  - **Verified:** spot-checks — *Plaza de la Puerta del Sol* → Sol 40.4169, −3.7029;
    *Calle Gran Vía* → 40.4204, −3.7048; `COD_DISB` fix resolves all Centro/Sol streets.

- **Task 2.1 — Emit locations CSV — ✅ DONE**
  - `io.build_locations_df()` reads `data/sources/madrid_streets.csv` (validates columns);
    `generate_data` writes `data/raw/madrid_locations.csv`. (`uber/locations_data.py` removed.)
  - **Acceptance met:** columns `location_id, street, neighborhood, district, lat, lon`;
    **8 655** unique 1-based `location_id`; 131 barrios / 21 districts; every
    `lat ∈ [40.31, 40.65]`, `lon ∈ [-3.84, -3.52]`; no nulls/empty strings; UTF-8 accents
    verified by code point.
  - **Verified:** `pytest -k locations` → **10/10**; generator deterministic (identical md5
    on re-run); runs with no network access.

### Phase 3 — `rides.csv` signal + target (the core)
**Goal:** produce reproducible rides whose price genuinely depends on features, calibrated
to the R² band.

- **Task 3.1 — `pricing.py` pure geo + price functions**
  - `haversine(lat1,lon1,lat2,lon2)`, `road_distance = haversine × ROAD_FACTOR`,
    `speed_kmh(hour)`, `duration_min(distance, hour)`, `demand_profile(hour, dow)`,
    `surge(hour, dow, jitter)` clamped [1.0, 3.0], `price(tier, distance, duration, surge,
    noise)` implementing SPEC §2.4 with `max(min_fare, …)`.
  - **Acceptance:** unit tests — haversine matches known Madrid pairs (±2%); surge clamps;
    price ≥ min_fare; Fri/Sat-night profile > weekday-late-morning profile.
  - **Verify:** `pytest tests/test_pricing.py` passes (pure-function unit tests, no I/O).

- **Task 3.2 — `sampling.py` seeded signal sampling**
  - Seeded `numpy.random.Generator`; sample: uniform 2024 timestamps → hour/dow/month;
    **street-level** OD pairs (pickup/dropoff `location_id` ∈ [1, 8655], distinct); tier
    choice; lognormal jitter.
  - **Acceptance:** given a fixed seed, two calls produce identical arrays; OD ids in
    [1, 8655] and pickup ≠ dropoff; distributions cover all tiers and all 12 months.
  - **Verify:** `pytest -k sampling` (determinism + coverage) passes.

- **Task 3.3 — Assemble signal-only `rides.csv`**
  - `generate_data` wires locations + tiers + pricing + sampling → DataFrame with the
    **signal** columns of SPEC §2.3 + `price_eur` (no distractors/commission yet).
  - **Acceptance:** ~50 000 rows; no nulls; `price_eur > 0` and `≥ min_fare`;
    `duration_min > 0`; `distance_km` ≈ haversine×ROAD_FACTOR of the two location ids.
  - **Verify:** `pytest -k "rides_integrity or geo or reproducible"` passes; re-running
    the generator with the same seed yields an equal DataFrame (byte-identical CSV).

- **Task 3.4 — Calibrate NOISE_STD to the R² band + time-of-call test**
  - Pin the OLS feature set (A3): `distance_km, duration_min, surge_multiplier, hour,
    day_of_week, month`, one-hot `tier`, one-hot `pickup_district`. Tune `NOISE_STD` in
    `config.py` so OLS R² ∈ [0.83, 0.87]; document the chosen value.
  - **Acceptance:** OLS R² in band; equivalent trips (same distance band + tier) on Fri/Sat
    night (21–03) average **≥ 50%** higher than weekday 10–12.
  - **Verify:** `pytest -k "signal_band or time_of_call"` passes.
  - **⛳ CHECKPOINT 2:** rides.csv is reproducible, signal-calibrated, and time-of-call
    pricing bites. This is the critical pedagogical gate — review before adding noise cols.

### Phase 3R — Driver-centric ride simulation (SPEC Rev 2 — supersedes the uniform-timestamp model)
**Goal:** replace the "independent, uniformly-timestamped ride" model with a **day-by-day,
12-month simulation of a driver population** (tenure/churn + realistic per-day activity), add
`drivers.csv`, and put a `driver_id` on every ride — **without** changing the price formula
(§2.4) or breaking the R² band / reproducibility. Tasks 3.1–3.3 (pricing, sampling primitives,
assembly) are **reused**; only the *temporal/driver structure* and the calibration change.
Task 3.2's `sample_timestamps` is retired (replaced by the simulation's hour-of-day draw).

- **Task 3R.1 — `Driver` entity + driver/config params**
  - `entities.py`: add frozen `Driver(driver_id, home_district, activity_class, tenure_start,
    tenure_end, active_days, n_rides, driver_rating)`.
  - `config.py`: `N_DRIVERS` (default, **the sizing knob** — see open item O1); activity-class
    mix (`casual`/`part_time`/`full_time` weights) + per-class {days-per-week prob, Poisson mean
    rides/working-day} tuned to the 6–12 / 20–25 research ranges; tenure cohort mixture
    (~3/6/12-month weights + jitter); `HOUR_VOLUME_WEIGHTS` (24) + weekend-night boost; home-
    district pickup-bias weight; driver_rating range (calibrated to NCR). Retire `N_RIDES`.
  - **Acceptance:** package imports; param tables have valid shapes/weights; constants only
    (no draws). **Verify:** `pytest -k "entities or driver_config"`.

- **Task 3R.2 — `simulation.py`: roster + tenure/churn**
  - `build_drivers(rng, n_drivers, locations)` → drivers with skewed `activity_class`,
    **staggered** tenure windows (incumbents + joiners, cohort tenure, clipped to 2024),
    home_district (weighted by each district's street share), per-driver driver_rating.
  - **Acceptance:** reproducible; class mix ≈ target; a non-trivial churned fraction
    (`tenure_end` < Dec 31); tenures span the 3/6/12-month cohorts; **active-driver count
    roughly stationary month-to-month** (no January cliff). **Verify:** `pytest -k "roster or
    tenure or churn"`.

- **Task 3R.3 — `simulation.py`: per-day ride events**
  - For each driver × each day in its tenure window: `work_today ~ Bernoulli(p | class, weekend)`;
    if working, `n ~ Poisson(mean | class)`, `n ≥ 1`; each ride's **hour** drawn from
    `HOUR_VOLUME_WEIGHTS` for that day-of-week (minute/second uniform) → `timestamp`. Returns
    ordered `(driver_id, timestamp)` events.
  - **Acceptance:** many drivers have ≥2 rides; the per-driver ride-count distribution is
    right-skewed; rides fall **only** inside tenure windows; hour distribution matches the
    profile (busy buckets heavier, not uniform); reproducible. **Verify:** `pytest -k
    "day_events or rides_per_driver or hour_profile"`.

- **Task 3R.4 — Rewire `generate_data` onto the simulation (+ emit `drivers.csv`)**
  - `build_rides_df` consumes the events: per ride draw OD (**pickup biased to the driver's
    home_district**), tier, jitter, noise; compute distance/duration/surge/price via the
    unchanged `pricing` funcs; sort by `timestamp`; assign sequential `ride_id`; attach
    `driver_id`. Add `io.build_drivers_df` + write `drivers.csv`. Change `generate(out_dir,
    seed, n_drivers)` (was `n_rides`); swap CLI `--n-rides`→`--n-drivers`; update the two
    locations/tiers `generate_writes_*` tests to pass a small `n_drivers`.
  - **Acceptance:** `drivers.csv` + `rides.csv` emitted; every `rides.driver_id` ∈ `drivers`;
    row count emergent and > 50k at the default; all §2.4 constraints hold; **same seed ⇒
    byte-identical both CSVs**. **Verify:** `pytest -k "rides_integrity or driver_panel or
    reproducible"`.

- **Task 3R.5 — Re-calibrate `NOISE_STD` to the R² band (new temporal distribution)**
  - Re-fit the **unchanged** pinned OLS feature set (numeric + one-hot tier + one-hot
    pickup_district; **`driver_id` excluded**); sweep `NOISE_STD` → R² ∈ [0.83, 0.87]. Record
    the new value, the chosen `N_DRIVERS` default, and the realized row count in `config.py`.
    Re-verify time-of-call ≥ 1.5×; confirm `driver_rating` corr ≈ 0.
  - **Acceptance:** R² in band (robust across seeds); time-of-call holds; row count documented.
    **Verify:** `pytest -k "signal_band or time_of_call"`.
  - **⛳ CHECKPOINT 2R:** driver-centric `rides.csv` + `drivers.csv` reproducible, signal
    re-calibrated to R² ≈ 0.85, panel/tenure structure valid. Re-gates the pedagogical core.

### Phase 4 — Distractors + leakage trap
**Goal:** add the columns that parts 7–8 must handle, without disturbing the signal.

- **Task 4.1 — Empirical distractor sampling from NCR (+ per-driver rating join)**
  - `io.load_ncr()` returns the source columns; `sampling.sample_empirical(values, n)` draws
    with replacement (seeded). Add **per-ride** `payment_method`, `customer_rating`, `avg_vtat`
    to `rides.csv`; `driver_rating` is now a **per-driver** attribute (set in Phase 3R,
    `drivers.csv`) **joined** onto each ride so it is constant within a driver.
  - **Acceptance:** each distractor’s Pearson correlation with `price_eur` is ≈ 0
    (|r| < 0.05, including the joined `driver_rating`); value ranges match NCR empirical
    support; no nulls introduced.
  - **Verify:** `pytest -k distractor` passes; signal-band test (3R.5) still green.

- **Task 4.2 — `commission_eur` leakage column**
  - `commission_eur = 0.25 × price_eur + small noise`; excluded from the signal formula.
  - **Acceptance:** including `commission_eur` in the OLS features pushes R² ≈ 1 (> 0.999);
    excluding it leaves R² in band.
  - **Verify:** `pytest -k leakage` passes.
  - **⛳ CHECKPOINT 3:** distractors are inert, leakage trap fires as designed.

### Phase 5 — Full validation & doc reconciliation
**Goal:** the complete SPEC §5 suite green; docs consistent.

- **Task 5.1 — Consolidate suite + reproducibility + docs**
  - Ensure all 9 SPEC §5 assertions live in `tests/test_generation.py` and pass together;
    add the byte-identical reproducibility test (same seed ⇒ identical CSVs).
  - Reconcile README structure block with `data/raw/` layout (A4); note in the change log
    that all three CSVs live in `data/raw/`.
  - **Acceptance:** `python -m uber.generate_data && pytest` → all green from a clean
    `data/raw/` (only NCR + `.gitkeep` present beforehand).
  - **Verify:** delete generated CSVs, run generator, run `pytest` → 0 failures.
  - **⛳ CHECKPOINT 4 (final):** full pipeline reproducible end-to-end; ready to commit.

---

## 6. Checkpoints summary

| # | After | Gate |
|---|---|---|
| 1 | Phase 1 | CLI + config/entities/io wiring proven on `ride_tiers.csv` |
| 2 | Phase 3 | `rides.csv` reproducible, R² ∈ band, time-of-call bites (uniform-timestamp model) |
| 2R | Phase 3R | driver-centric `rides.csv` + `drivers.csv` reproducible, R² re-calibrated, panel/tenure valid |
| 3 | Phase 4 | distractors inert, leakage trap fires |
| 4 | Phase 5 | full SPEC §5 suite green from clean state; docs reconciled |

## 7. Risks & open items

- **R1 — Coordinate sourcing (A1) — RETIRED:** the street-level table is built and validated
  (8 655 streets, 0 unmapped). Residuals: (a) CC BY 4.0 attribution — recorded in
  `data/sources/SOURCE.md`, still to add to the README in Phase 5; (b) the raw callejero
  changes daily and is not committed — mitigated by committing the **frozen**
  `data/sources/madrid_streets.csv` snapshot, so generation never needs the network and stays
  byte-reproducible (re-running the extractor later may drift as the callejero evolves); (c)
  long radial streets (e.g. *Calle de Alcalá*) get a whole-street median point that can land
  outside their nominal central barrio — acceptable, since the *ML signal depends only on
  plausible pairwise distances*.
- **R2 — R² band sensitivity (3.4):** R² depends on both `NOISE_STD` **and** the pinned
  OLS feature encoding. Mitigation: pin the feature set in the test; calibrate `NOISE_STD`
  against that exact set; keep the band test deterministic via the seed.
- **R3 — NCR data hygiene:** distractor columns contain `null`s and quoting artifacts.
  Mitigation: `io.load_ncr()` drops nulls per column before empirical sampling.
- **R4 — Distance/duration edge cases:** identical pickup=dropoff → zero distance; clamp
  to `min_fare`. Mitigation: enforce distinct OD pairs in sampling (3.2).
- **R5 — Dataset size vs. realism (Rev 2):** realistic activity ⇒ hundreds of rides/driver/yr,
  so the full ~15k Madrid roster is *millions* of rows (hundreds of MB / GB) — too big to
  commit and slow for the k-NN/CV exercises. Mitigation: `N_DRIVERS` is a knob; ship a
  representative default (open item **O1**); optionally commit `rides.csv.gz` if the default
  CSV exceeds a comfortable git size.
- **R6 — R² re-calibration under clustered timestamps (3R.5):** the hour distribution is no
  longer uniform, so `NOISE_STD` shifts. Mitigation: same pinned-feature calibration method as
  Task 3.4, re-swept; `driver_id` stays out of the feature set so it cannot leak.
- **R7 — Determinism with data-dependent sizes:** ride count now depends on Poisson draws, so
  arrays can't be pre-sized. Mitigation: fix the RNG draw order (all driver attributes → per-
  driver/per-day activity in a fixed iteration order → per-ride draws) and build/concatenate
  lists; assert byte-identical re-runs.
- **R8 — Performance at full scale (O1=C, 15k drivers ⇒ millions of rides):** nested Python
  loops over ~millions of driver-days/rides would be far too slow. Mitigation: **vectorize** in
  numpy — expand driver-days as arrays, draw work-day mask + Poisson counts in bulk, `np.repeat`
  to per-ride arrays, sample hours via inverse-CDF indexed by day-of-week, assemble timestamps
  arithmetically. Keep the per-ride price physics vectorized too (the current per-row loop in
  `build_rides_df` must be replaced with array ops). Signal/calibration tests use a small
  `n_drivers` so they stay fast.

## 9. Open items for approval (Rev 2)

- **O1 — RESOLVED (2026-09-23): Option C.** User chose **full Madrid scale `N_DRIVERS = 15000`**
  (⇒ millions of rides) and to **gitignore the ride log** (`data/raw/rides.csv` + `.gz` added to
  `.gitignore`; regenerated from the seed, not committed). The 3 small tables stay tracked.
  Implications baked into Phase 3R: (a) the simulation must be **vectorized** (numpy, no per-ride
  Python loops) to build millions of rows in reasonable time — see risk R8; (b) the R²/signal
  tests run on a **reduced `n_drivers`** for speed (R² ~invariant to N), with the full-scale R²
  confirmed once at calibration. Original options kept below for the record:
  | Option | `N_DRIVERS` | ≈ rides | ≈ CSV size | Notes |
  |---|---|---|---|---|
  | A (recommended) | ~1,500 | ~250–400k | ~40–60 MB | "pretty big" (5–8× the old 50k), still git-able & fast |
  | B (leaner) | ~800 | ~120–200k | ~20–30 MB | comfortable git size, quick CV/k-NN |
  | C (full scale) | 15,000 | several **million** | hundreds of MB–GB | realistic Madrid; commit as `.gz` or don't commit |
  Any option can be reproduced later via `--n-drivers`; this only fixes the **committed
  default**. Recommendation: **A**, and gzip the committed `rides.csv` if it clears ~50 MB.

## 8. Commands (reference)

```bash
pip install -e ".[dev]"
python -m uber.generate_data
python -m uber.generate_data --seed 42 --n-drivers 500   # smaller run (rides emerge from the sim)
pytest
```
