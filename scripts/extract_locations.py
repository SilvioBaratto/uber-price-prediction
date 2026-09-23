"""Build the street-level Madrid location map from the official callejero.

One-time **authoring** step (not part of data generation). It produces a committed
snapshot ``data/sources/madrid_streets.csv`` with one representative point per official
street (vial), tagged with its district and barrio. The generator later reads that frozen
snapshot, so *generation* stays offline and reproducible even though the upstream callejero
updates daily.

Inputs (both from the Ayuntamiento de Madrid, CC BY 4.0 — see data/sources/SOURCE.md):
  1. Callejero oficial — "Direcciones vigentes con coordenadas" (per-address CSV, ~33 MB,
     latin-1, ``;``-separated, with LATITUD/LONGITUD in DMS and DISTRITO/BARRIO codes).
     Too large to commit; auto-downloaded to .scratch/ (gitignored) if not already present.
  2. ``data/sources/barrios_madrid.topojson`` (committed) — for district and barrio names.

Method: group addresses by ``COD_VIA`` (a street), take the median lat/lon as the
representative point, and the modal (district, barrio) codes -> official names.

Run from the project root:

    python scripts/extract_locations.py
"""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BARRIOS_TOPOJSON = PROJECT_ROOT / "data" / "sources" / "barrios_madrid.topojson"
TARGET = PROJECT_ROOT / "data" / "sources" / "madrid_streets.csv"
RAW_CACHE = PROJECT_ROOT / ".scratch" / "direcciones_callejero.csv"

ADDRESSES_URL = (
    "https://datos.madrid.es/dataset/213605-0-callejero-oficial-madrid/resource/"
    "213605-4-callejero-oficial-madrid-csv/download/213605-4-callejero-oficial-madrid-csv.csv"
)
RETRIEVED = "2026-09-23"

# Spanish connector words rendered lower-case in a display name (unless first).
_CONNECTORS = {"de", "del", "la", "las", "los", "el", "y", "e", "al", "a", "en", "con"}
_DMS = re.compile(r"(\d+)\D+(\d+)'([\d.]+)''\s*([NSEWnsew])")


def _spanish_titlecase(text: str) -> str:
    words = text.split()
    out = []
    for i, w in enumerate(words):
        low = w.lower()
        out.append(low if (i > 0 and low in _CONNECTORS) else low.capitalize())
    return " ".join(out)


def _dms_to_decimal(value: str) -> float | None:
    m = _DMS.search(value)
    if not m:
        return None
    deg, minutes, seconds, hemi = m.groups()
    dec = int(deg) + int(minutes) / 60 + float(seconds) / 3600
    return -dec if hemi.upper() in ("S", "W") else dec


def _name_maps() -> tuple[dict[int, str], dict[str, str]]:
    """(district code -> name, "d-b" code -> barrio name) from the barrio TopoJSON."""
    topo = json.loads(BARRIOS_TOPOJSON.read_text(encoding="utf-8"))
    district_name: dict[int, str] = {}
    barrio_name: dict[str, str] = {}
    for geom in topo["objects"]["Barrios"]["geometries"]:
        p = geom["properties"]
        district_name[int(p["CODDIS"])] = p["NOMDIS"].strip()
        barrio_name[p["COD_DISB"]] = p["NOMBRE"].strip()  # e.g. "1-6" -> "Sol"
    return district_name, barrio_name


def _ensure_raw() -> Path:
    if not RAW_CACHE.exists():
        RAW_CACHE.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading callejero addresses -> {RAW_CACHE} ...")
        req = urllib.request.Request(ADDRESSES_URL, headers={"User-Agent": "uber-price-prediction/0.1"})
        with urllib.request.urlopen(req, timeout=300) as resp, RAW_CACHE.open("wb") as fh:
            fh.write(resp.read())
    return RAW_CACHE


def _mode_first(column):
    """Most common value in a column (first by index on ties)."""
    series = pd.Series(column)
    modes = series.mode()
    return modes.iloc[0] if len(modes) else series.iloc[0]


def build() -> pd.DataFrame:
    district_name, barrio_name = _name_maps()
    df = pd.read_csv(_ensure_raw(), sep=";", encoding="latin-1", dtype=str, low_memory=False)

    df["lat"] = df["LATITUD"].map(_dms_to_decimal)
    df["lon"] = df["LONGITUD"].map(_dms_to_decimal)
    df["dcode"] = df["DISTRITO"].astype(int)
    df["bcode"] = df["BARRIO"].astype(int)

    rows = []
    for _cod, sub in df.groupby("COD_VIA"):
        dcode = int(_mode_first(sub["dcode"]))
        bcode = int(_mode_first(sub["bcode"]))
        raw_name = " ".join(
            part for part in (
                _mode_first(sub["VIA_CLASE"]).strip(),
                str(_mode_first(sub["VIA_PAR"])).strip(),
                _mode_first(sub["VIA_NOMBRE_ACENTOS"]).strip(),
            ) if part and part != "nan"
        )
        rows.append(
            {
                "street": _spanish_titlecase(raw_name),
                "neighborhood": barrio_name.get(f"{dcode}-{bcode}", ""),
                "district": district_name[dcode],
                "lat": round(sub["lat"].median(), 6),
                "lon": round(sub["lon"].median(), 6),
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(["district", "neighborhood", "street"], kind="stable").reset_index(drop=True)
    out.insert(0, "location_id", range(1, len(out) + 1))
    return out


def main() -> None:
    out = build()
    unmapped = int((out["neighborhood"] == "").sum())
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(TARGET, index=False, encoding="utf-8")
    print(
        f"wrote {TARGET.relative_to(PROJECT_ROOT)}: {len(out)} streets, "
        f"{out['district'].nunique()} districts, {unmapped} without a barrio "
        f"(retrieved {RETRIEVED})"
    )


if __name__ == "__main__":
    main()
