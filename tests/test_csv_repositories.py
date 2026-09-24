"""T2.4 — CSV-backed repository adapters (infrastructure layer).

CsvLocationRepository and CsvTierRepository implement the application ports over the committed
``data/raw/madrid_locations.csv`` and ``data/raw/ride_tiers.csv``, returning domain entities.
They satisfy the ports structurally, expose the real catalog counts, look up by id and search
street names case-insensitively.
"""

from __future__ import annotations

import pytest

from uber.application.ports import LocationRepository, TierRepository
from uber.domain import config
from uber.domain.entities import Location, RideTier
from uber.infrastructure.repositories import CsvLocationRepository, CsvTierRepository

N_STREETS = 8655
N_TIERS = 6


@pytest.fixture(scope="module")
def loc_repo() -> CsvLocationRepository:
    return CsvLocationRepository()


@pytest.fixture(scope="module")
def tier_repo() -> CsvTierRepository:
    return CsvTierRepository()


# --- CsvLocationRepository -------------------------------------------------
def test_location_repo_satisfies_port(loc_repo: CsvLocationRepository) -> None:
    assert isinstance(loc_repo, LocationRepository)


def test_location_repo_all_count_and_type(loc_repo: CsvLocationRepository) -> None:
    locations = loc_repo.all()
    assert len(locations) == N_STREETS
    assert all(isinstance(loc, Location) for loc in locations)


def test_location_repo_get_by_id(loc_repo: CsvLocationRepository) -> None:
    first = loc_repo.get(1)
    assert isinstance(first, Location)
    assert first.location_id == 1


def test_location_repo_get_missing_raises(loc_repo: CsvLocationRepository) -> None:
    with pytest.raises(KeyError):
        loc_repo.get(10_000_000)


def test_location_repo_search_is_case_insensitive(loc_repo: CsvLocationRepository) -> None:
    hits_lower = loc_repo.search("gran vía")
    hits_upper = loc_repo.search("GRAN VÍA")
    assert len(hits_lower) == len(hits_upper) >= 1
    assert any(loc.street == "Calle Gran Vía" for loc in hits_lower)


def test_location_repo_search_no_match_returns_empty(loc_repo: CsvLocationRepository) -> None:
    assert loc_repo.search("zzzzz-no-such-street") == []


# --- CsvTierRepository -----------------------------------------------------
def test_tier_repo_satisfies_port(tier_repo: CsvTierRepository) -> None:
    assert isinstance(tier_repo, TierRepository)


def test_tier_repo_all_count_order_and_type(tier_repo: CsvTierRepository) -> None:
    tiers = tier_repo.all()
    assert len(tiers) == N_TIERS
    assert all(isinstance(t, RideTier) for t in tiers)
    # same increasing-price order as the config catalog (single source of truth)
    assert [t.tier for t in tiers] == [t.tier for t in config.TIERS]


def test_tier_repo_values_match_config(tier_repo: CsvTierRepository) -> None:
    by_id = {t.tier: t for t in tier_repo.all()}
    for cfg_tier in config.TIERS:
        loaded = by_id[cfg_tier.tier]
        assert loaded.display_name == cfg_tier.display_name
        assert loaded.capacity == cfg_tier.capacity
        assert loaded.base_fare == pytest.approx(cfg_tier.base_fare)
        assert loaded.min_fare == pytest.approx(cfg_tier.min_fare)
