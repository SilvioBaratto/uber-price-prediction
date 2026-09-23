# Location data sources

The location table is **street-level**: one representative point per official Madrid street
(*vial*), tagged with its barrio (neighbourhood) and distrito (district). Two official
Ayuntamiento de Madrid datasets are used, both licensed **CC BY 4.0**.

## 1. Callejero oficial — street coordinates (primary)

- **Publisher:** Ayuntamiento de Madrid — Portal de datos abiertos.
- **Dataset:** "Callejero oficial" → *Direcciones vigentes con coordenadas*.
- **Dataset page:** https://datos.madrid.es/dataset/213605-0-callejero-oficial-madrid
- **File fetched:** https://datos.madrid.es/dataset/213605-0-callejero-oficial-madrid/resource/213605-4-callejero-oficial-madrid-csv/download/213605-4-callejero-oficial-madrid-csv.csv
- **Retrieved:** 2026-09-23.
- **Format:** CSV, `;`-separated, **latin-1**; ~33 MB, ~214k address rows, 8 655 unique
  streets (`COD_VIA`). `LATITUD`/`LONGITUD` are in DMS (e.g. `40°25'12.34'' N`);
  `DISTRITO`/`BARRIO` carry the administrative codes.

This raw file is **not committed** (too large, and it is regenerated upstream daily). It is
auto-downloaded to `.scratch/` (gitignored) by the extractor when absent.

## 2. Barrios municipales — administrative names (secondary)

- **Dataset:** "Barrios municipales de Madrid".
- **Dataset page:** https://datos.madrid.es/dataset/300496-0-barrios-madrid
- **File committed:** `barrios_madrid.topojson`
  (from https://geoportal.madrid.es/fsdescargas/IDEAM_WBGEOPORTAL/LIMITES_ADMINISTRATIVOS/Barrios/TopoJSON/Barrios.json)
- **Retrieved:** 2026-09-23. **Encoding:** UTF-8 (accents like *Ríos Rosas* decode correctly).

Used only to map codes → official names: `CODDIS` → district name (`NOMDIS`) and `COD_DISB`
(e.g. `"1-6"`) → barrio name (`NOMBRE`).

> **Mapping note:** the barrio number must be read from `COD_DISB`, **not** `NUM_BAR`. Sol has
> `NUM_BAR=8` but `COD_DISB="1-6"`, and the callejero uses `DISTRITO=1, BARRIO=6`; keying on
> `NUM_BAR` silently drops all of Sol's streets.

## Derived, committed snapshot

`madrid_streets.csv` — the frozen output of `scripts/extract_locations.py`
(columns `location_id, street, neighborhood, district, lat, lon`; 8 655 rows; UTF-8).
Per street it holds the **median** of that street's address coordinates and the **modal**
district/barrio. Committing this snapshot makes data generation fully offline and
byte-reproducible, independent of the daily-changing upstream callejero. Re-running the
extractor against a fresh download may differ slightly as the callejero evolves.

- Bounding box: lat 40.3204…40.6432, lon −3.8367…−3.5296.

## Attribution

> Contains data from the Ayuntamiento de Madrid — "Callejero oficial" and "Barrios
> municipales de Madrid", licensed under CC BY 4.0.
