"""Filesystem paths for the project's data files (infrastructure layer).

The single place that knows *where* committed source files and generated CSVs live. Kept out
of :mod:`uber.domain.config` so the pure domain never needs to know about the filesystem; the
business/simulation constants (seed, tariffs, dynamics) stay in the domain.
"""

from __future__ import annotations

from pathlib import Path

# --- Directories -----------------------------------------------------------
# paths.py lives at uber/infrastructure/paths.py, so the project root is two levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"  # generated CSVs live here (SPEC §2)
SOURCES_DIR = DATA_DIR / "sources"  # committed upstream source files

# --- Source files (committed) ----------------------------------------------
NCR_PATH = RAW_DIR / "ncr_ride_bookings.csv"
# Frozen street-level location snapshot built by scripts/extract_locations.py from the official
# Madrid callejero (see data/sources/SOURCE.md); read so generation stays offline and reproducible.
STREETS_SOURCE = SOURCES_DIR / "madrid_streets.csv"

# --- Output CSV filenames (all written to RAW_DIR) -------------------------
LOCATIONS_CSV = "madrid_locations.csv"
TIERS_CSV = "ride_tiers.csv"
DRIVERS_CSV = "drivers.csv"
RIDES_CSV = "rides.csv"
