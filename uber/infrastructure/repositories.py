"""Repository adapters: load/build the project's domain tables (infrastructure layer).

Owns the CSV column contracts and the builders that materialise the static reference tables:
the Madrid location table (read from the committed street snapshot) and the ride-tier catalog
(projected from :data:`uber.domain.config.TIERS`). The ``drivers.csv`` / ``rides.csv`` tables
are *generated* (seeded simulation), so they are built in :mod:`uber.datagen`, not here — this
module only owns their column contract (:data:`DRIVER_COLUMNS`) and the reads/writes.

For the simulator, :class:`CsvLocationRepository` and :class:`CsvTierRepository` implement the
application ports over the committed ``data/raw`` CSVs, returning domain entities.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from uber.domain import config
from uber.domain.entities import Location, RideTier
from uber.infrastructure import paths

# --- Column contracts (CSV schemas) ----------------------------------------
LOCATION_COLUMNS = ["location_id", "street", "neighborhood", "district", "lat", "lon"]
TIER_COLUMNS = [
    "tier",
    "display_name",
    "capacity",
    "base_fare",
    "per_km",
    "per_min",
    "booking_fee",
    "min_fare",
]
# drivers.csv schema (SPEC §2.1b, Rev 2). Matches ``entities.Driver`` field order; the table
# itself is built by the seeded ``datagen.simulation.build_drivers`` (generative, so it lives
# there, not here) — this adapter only owns the column contract and the write.
DRIVER_COLUMNS = [
    "driver_id",
    "home_district",
    "activity_class",
    "tenure_start",
    "tenure_end",
    "active_days",
    "n_rides",
    "driver_rating",
]


def build_locations_df() -> pd.DataFrame:
    """Load the committed street-level Madrid location table (one row per street).

    The table is a frozen snapshot built offline by ``scripts/extract_locations.py`` from the
    official callejero; reading it keeps generation deterministic and network-free.
    """
    df = pd.read_csv(paths.STREETS_SOURCE, encoding="utf-8")
    missing = set(LOCATION_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{paths.STREETS_SOURCE} is missing columns: {sorted(missing)}")
    return pd.DataFrame(df, columns=LOCATION_COLUMNS)


def build_tiers_df() -> pd.DataFrame:
    """Build the ride-tier catalog (ride_tiers.csv) from ``config.TIERS``.

    One row per tier, in the increasing-price order declared in config. Dataclass field
    order matches ``TIER_COLUMNS``.
    """
    rows = [asdict(tier) for tier in config.TIERS]
    return pd.DataFrame(rows, columns=TIER_COLUMNS)


# --- Port adapters over the committed CSVs ---------------------------------
class CsvLocationRepository:
    """``LocationRepository`` backed by the committed ``data/raw/madrid_locations.csv``.

    Rows are read once, lazily, and cached as :class:`~uber.domain.entities.Location` objects.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path if path is not None else paths.RAW_DIR / paths.LOCATIONS_CSV
        self._locations: list[Location] | None = None
        self._by_id: dict[int, Location] = {}

    def _ensure_loaded(self) -> list[Location]:
        if self._locations is None:
            df = pd.read_csv(self._path, encoding="utf-8")
            missing = set(LOCATION_COLUMNS) - set(df.columns)
            if missing:
                raise ValueError(f"{self._path} is missing columns: {sorted(missing)}")
            locations = [
                Location(int(lid), str(street), str(neigh), str(district), float(lat), float(lon))
                for lid, street, neigh, district, lat, lon in zip(
                    df["location_id"], df["street"], df["neighborhood"],
                    df["district"], df["lat"], df["lon"],
                )
            ]
            self._locations = locations
            self._by_id = {loc.location_id: loc for loc in locations}
        return self._locations

    def all(self) -> list[Location]:
        return list(self._ensure_loaded())

    def get(self, location_id: int) -> Location:
        self._ensure_loaded()
        try:
            return self._by_id[location_id]
        except KeyError:
            raise KeyError(f"no location with id {location_id}") from None

    def search(self, query: str) -> list[Location]:
        locations = self._ensure_loaded()
        needle = query.strip().lower()
        if not needle:
            return []
        return [loc for loc in locations if needle in loc.street.lower()]


class CsvTierRepository:
    """``TierRepository`` backed by the committed ``data/raw/ride_tiers.csv``.

    Rows are read once, lazily, and cached as :class:`~uber.domain.entities.RideTier` objects in
    the file's (increasing-price) order.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path if path is not None else paths.RAW_DIR / paths.TIERS_CSV
        self._tiers: list[RideTier] | None = None

    def _ensure_loaded(self) -> list[RideTier]:
        if self._tiers is None:
            df = pd.read_csv(self._path, encoding="utf-8")
            missing = set(TIER_COLUMNS) - set(df.columns)
            if missing:
                raise ValueError(f"{self._path} is missing columns: {sorted(missing)}")
            self._tiers = [
                RideTier(
                    str(tier), str(display_name), int(capacity), float(base_fare),
                    float(per_km), float(per_min), float(booking_fee), float(min_fare),
                )
                for tier, display_name, capacity, base_fare, per_km, per_min, booking_fee, min_fare
                in zip(
                    df["tier"], df["display_name"], df["capacity"], df["base_fare"],
                    df["per_km"], df["per_min"], df["booking_fee"], df["min_fare"],
                )
            ]
        return self._tiers

    def all(self) -> list[RideTier]:
        return list(self._ensure_loaded())
