# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-09-24

### Added

- **Synthetic data pipeline** — a seeded, day-by-day driver-population simulation
  that emits `madrid_locations.csv`, `ride_tiers.csv`, `drivers.csv` and the
  signal-only `rides.csv` (`uber generate-data`).
- **The eight-part regression arc** — Parts 1–8 (constant baseline, kNN &
  standardization, overfitting & the curse of dimensionality, OLS, polynomial
  surge interactions, the bias–variance U-curve, Ridge/Lasso, k-fold CV & the
  `commission_eur` leakage trap) with a summary report (`uber run-arc`).
- **Interactive price simulator** — pick a pickup and drop-off in Madrid (by
  street name or id) and see every ride tier priced by a model retrained on
  launch, with an estimated ETA (`uber simulate`).
- **Unified `uber` CLI** with `generate-data | run-arc | simulate` subcommands
  and a `python -m uber` entry point.
- Clean-architecture layout (`domain` / `application` / `infrastructure` / `cli`
  plus `datagen` and `modeling` feature packages) with ports & adapters.
- Project tooling: packaging metadata, `ruff` lint + format, a `pyright` config,
  a GitHub Actions CI workflow, and issue/PR templates.

[Unreleased]: https://github.com/neuroespresso/uber-price-prediction/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/neuroespresso/uber-price-prediction/releases/tag/v0.1.0
