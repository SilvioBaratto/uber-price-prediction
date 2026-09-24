"""Validation for the street-level Madrid locations table (madrid_locations.csv)."""

from __future__ import annotations

import pandas as pd
import pytest

from uber import generate_data
from uber.domain import config
from uber.infrastructure import paths, repositories

N_STREETS = 8655   # official viales in the Madrid callejero snapshot
N_BARRIOS = 131    # municipal barrios
N_DISTRICTS = 21   # distritos


@pytest.fixture(scope="module")
def locations() -> pd.DataFrame:
    return repositories.build_locations_df()


def test_columns_and_order(locations: pd.DataFrame) -> None:
    assert list(locations.columns) == repositories.LOCATION_COLUMNS
    assert repositories.LOCATION_COLUMNS == ["location_id", "street", "neighborhood", "district", "lat", "lon"]


def test_row_and_area_counts(locations: pd.DataFrame) -> None:
    assert len(locations) == N_STREETS
    assert locations["neighborhood"].nunique() == N_BARRIOS
    assert locations["district"].nunique() == N_DISTRICTS


def test_location_id_is_sequential_and_unique(locations: pd.DataFrame) -> None:
    ids = locations["location_id"]
    assert ids.is_unique
    assert ids.tolist() == list(range(1, len(locations) + 1))


def test_no_nulls_and_no_empty_strings(locations: pd.DataFrame) -> None:
    assert int(locations.isnull().sum().sum()) == 0
    for col in ("street", "neighborhood", "district"):
        assert (locations[col].str.len() > 0).all()


def test_dtypes(locations: pd.DataFrame) -> None:
    assert pd.api.types.is_integer_dtype(locations["location_id"])
    assert pd.api.types.is_float_dtype(locations["lat"])
    assert pd.api.types.is_float_dtype(locations["lon"])
    # pandas may infer the native "str" dtype or fall back to object; both are strings.
    for col in ("street", "neighborhood", "district"):
        assert pd.api.types.is_string_dtype(locations[col])


def test_coordinates_within_madrid_bbox(locations: pd.DataFrame) -> None:
    assert locations["lat"].between(config.LAT_MIN, config.LAT_MAX).all()
    assert locations["lon"].between(config.LON_MIN, config.LON_MAX).all()


def test_every_barrio_has_streets(locations: pd.DataFrame) -> None:
    # The COD_DISB join must resolve every street to a barrio (the Sol / NUM_BAR trap).
    per_barrio = locations.groupby("neighborhood").size()
    assert per_barrio.min() >= 1
    assert "Sol" in per_barrio.index  # regression guard for the Centro/Sol mapping


def test_known_street_coordinates(locations: pd.DataFrame) -> None:
    sol = locations.loc[locations["street"] == "Plaza de la Puerta del Sol"].iloc[0]
    assert sol["neighborhood"] == "Sol"
    assert sol["district"] == "Centro"
    assert sol["lat"] == pytest.approx(40.4169, abs=0.01)
    assert sol["lon"] == pytest.approx(-3.7029, abs=0.01)

    gran_via = locations.loc[locations["street"] == "Calle Gran Vía"].iloc[0]
    assert gran_via["district"] == "Centro"
    assert gran_via["lat"] == pytest.approx(40.4204, abs=0.01)
    assert gran_via["lon"] == pytest.approx(-3.7048, abs=0.01)


def test_accents_preserved(locations: pd.DataFrame) -> None:
    # Verified by code point, not console rendering (the source is UTF-8).
    assert "Calle Gran Vía" in set(locations["street"])  # contains U+00ED
    barrios = set(locations["neighborhood"])
    assert {"Ríos Rosas", "Almagro"} <= barrios
    assert "Chamberí" not in barrios  # Chamberí is a district, not a barrio
    assert "Chamberí" in set(locations["district"])


def test_generate_writes_locations(tmp_path) -> None:
    written = generate_data.generate(tmp_path, n_drivers=5)  # tiny sim: this test only checks locations
    assert written[paths.LOCATIONS_CSV] == N_STREETS
    out = tmp_path / paths.LOCATIONS_CSV
    assert out.exists()
    reloaded = pd.read_csv(out)
    assert list(reloaded.columns) == repositories.LOCATION_COLUMNS
    assert len(reloaded) == N_STREETS
