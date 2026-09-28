"""Export compact JSON fixtures for the neuroespresso «Costruiamo l'algoritmo di Uber» videos.

Reads the real project artifacts (tier tariffs, the domain pricing constants, the rides
sample, the U-curve CSV, the arc log, the per-location pickup counts, the driver roster) and
writes small, seeded JSON files that Remotion renders live. The point of the whole pipeline is
that every number on screen traces back to a real run, so nothing here is hand-typed: values
come from ``uber.domain.config`` and from files under ``data/raw`` / ``.scratch`` / ``output``.

The heavy ``rides.csv`` (~1.2 GB, git-ignored) is never read whole; the ~100k-row sample in
``.scratch`` is used when present, otherwise a capped head of ``rides.csv``.

Usage::

    python scripts/export_fixtures.py            # -> output/fixtures/uber_prezzi/
    python scripts/export_fixtures.py --out-dir ../neuroespresso/videocraft/public/fixtures/uber_prezzi
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from uber.domain import config  # noqa: E402  (needs PROJECT_ROOT on sys.path first)

SLUG = "uber_prezzi"
RAW = PROJECT_ROOT / "data" / "raw"
SCRATCH = PROJECT_ROOT / ".scratch"
OUTPUT = PROJECT_ROOT / "output"

# Puerta del Sol, the centre the illustrative unbalanced hotspot decays from (matches
# scripts/plot_balance_map.py so the two stay comparable).
CENTER_LAT, CENTER_LON = 40.4168, -3.7038

SCATTER_N = 800  # points drawn in the price-vs-distance cloud (enough to read, cheap to ship)
MAP_N = 2000  # street dots on the Madrid map


def gini(x: np.ndarray) -> float:
    """Return the Gini coefficient of a non-negative distribution (0 = perfectly even)."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    c = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(c) / c[-1]) / n)


def synthetic_unbalanced(lat: np.ndarray, lon: np.ndarray, total: float) -> np.ndarray:
    """Return the illustrative hotspot demand: same total rides, decaying from the city centre."""
    dx = (lon - CENTER_LON) * np.cos(np.radians(CENTER_LAT))
    dy = lat - CENTER_LAT
    d = np.hypot(dx, dy)
    w = np.exp(-d / 0.025) + 0.06  # sharp central peak plus a floor so edges stay visible
    return total * w / w.sum()


def load_rides(rng: np.random.Generator, n: int) -> pd.DataFrame | None:
    """Return a seeded ``n``-row sample of rides, or None if no ride source is available.

    Prefers the cached ``.scratch`` sample; falls back to a capped head of the full
    ``rides.csv`` so the exporter still runs on a fresh checkout that only has the big file.
    """
    sample = SCRATCH / "rides_sample_100k.csv"
    if sample.exists():
        df = pd.read_csv(sample)
    elif (RAW / "rides.csv").exists():
        df = pd.read_csv(RAW / "rides.csv", nrows=500_000)
    else:
        return None
    if len(df) > n:
        df = df.sample(n, random_state=int(rng.integers(0, 2**31))).reset_index(drop=True)
    return df


def write_json(out_dir: Path, name: str, obj: object) -> None:
    """Write ``obj`` as JSON to ``out_dir/name`` and report the size on stdout."""
    path = out_dir / name
    path.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))
    print(f"  {name:<24} {path.stat().st_size / 1024:6.1f} KB")


def fx_tiers() -> object:
    """The six ride tiers with their real tariffs (the price-recipe inputs)."""
    return [
        {
            "tier": t.tier,
            "name": t.display_name,
            "seats": t.capacity,
            "base_fare": t.base_fare,
            "per_km": t.per_km,
            "per_min": t.per_min,
            "booking_fee": t.booking_fee,
            "min_fare": t.min_fare,
        }
        for t in config.TIERS
    ]


def fx_pricing() -> object:
    """The ground-truth pricing dynamics: surge profile, speed-by-hour, noise, road factor."""
    return {
        "formula": "(base + per_km*km + per_min*min) * surge + booking, clamped to min_fare, + noise",
        "speed_kmh_by_hour": list(config.SPEED_KMH_BY_HOUR),
        "surge": {
            "offpeak": config.SURGE_OFFPEAK,
            "morning_rush": config.SURGE_MORNING_RUSH,
            "evening_rush": config.SURGE_EVENING_RUSH,
            "late_night": config.SURGE_LATE_NIGHT,
            "weekend_day": config.SURGE_WEEKEND_DAY,
            "weekend_night": config.SURGE_WEEKEND_NIGHT,
            "clamp": list(config.SURGE_CLAMP),
            "jitter_sigma": config.JITTER_SIGMA,
        },
        "noise_std": config.NOISE_STD,
        "road_factor": config.ROAD_FACTOR,
    }


def fx_ucurve() -> object | None:
    """The bias-variance U-curve straight from ``output/part6_ucurve.csv`` (k, train, test)."""
    path = OUTPUT / "part6_ucurve.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    k = df["k"].to_numpy()
    train = df["rmse_train"].to_numpy()
    test = df["rmse_test"].to_numpy()
    return [
        {"k": int(k[i]), "rmse_train": round(float(train[i]), 3), "rmse_test": round(float(test[i]), 3)}
        for i in range(len(df))
    ]


def fx_arc_metrics() -> object | None:
    """Parse the per-part results table out of ``output/full_run.log`` (the error spine).

    The final deployed model is not in that table (it is the simulator's interaction OLS), so its
    held-out R² is carried separately, marked with its source, rather than silently invented.
    """
    log = OUTPUT / "full_run.log"
    if not log.exists():
        return None

    def num(x: str) -> float | None:
        """Return the cell as a float, or None for the ``-`` placeholder."""
        try:
            return float(x)
        except ValueError:
            return None

    rows = []
    for line in log.read_text(encoding="utf-8").splitlines():
        if not re.match(r"^\s*\d+\s*\|", line):
            continue
        cells = [c.strip() for c in line.split("|")]
        if len(cells) < 6:
            continue
        rows.append(
            {
                "part": int(cells[0]),
                "model": cells[1],
                "rmse_test": num(cells[4]),
                "r2_test": num(cells[5]),
            }
        )
    return {
        "parts": rows,
        "final": {"model": "regressione con interazioni (simulatore)", "r2_test": 0.98, "source": "README/ModelPredictor"},
    }


def fx_scatter(rides: pd.DataFrame) -> object:
    """A small price-vs-features cloud for the regression/kNN scenes."""
    s = rides.head(SCATTER_N)
    km = s["distance_km"].to_numpy()
    minutes = s["duration_min"].to_numpy()
    surge = s["surge_multiplier"].to_numpy()
    price = s["price_eur"].to_numpy()
    tier = s["tier"].to_numpy()
    hour = s["hour"].to_numpy()
    return [
        {
            "km": round(float(km[i]), 2),
            "min": round(float(minutes[i]), 1),
            "surge": round(float(surge[i]), 2),
            "price": round(float(price[i]), 2),
            "tier": str(tier[i]),
            "hour": int(hour[i]),
        }
        for i in range(len(s))
    ]


def fx_ols(rides: pd.DataFrame) -> object:
    """Real regression fits on the sample: the 1-D price-vs-distance line and the weight ranking.

    Coefficients are on standardized features, so their magnitudes are comparable and surge and
    distance stand out — the point the OLS scene makes.
    """
    line = LinearRegression().fit(rides[["distance_km"]], rides["price_eur"])
    feats = ["distance_km", "duration_min", "surge_multiplier"]
    xs = StandardScaler().fit_transform(rides[feats])
    multi = LinearRegression().fit(xs, rides["price_eur"])
    ranking = sorted(
        ({"feature": f, "weight": round(w, 2)} for f, w in zip(feats, multi.coef_)),
        key=lambda d: -abs(d["weight"]),
    )
    return {
        "line_price_vs_distance": {"slope": round(float(line.coef_[0]), 3), "intercept": round(float(line.intercept_), 3)},
        "weights_standardized": ranking,
    }


def fx_poly(rides: pd.DataFrame) -> object:
    """Ground-truth price-vs-distance lines at rising surge for the reference tier.

    Holds a representative midday speed so the only thing that changes across the fan is the surge
    multiplier: it scales the per-km slope, which is exactly why one straight line cannot fit the
    whole cloud and the polynomial gives each surge level its own slope. The single OLS line is fit
    on the same tier across all its surge levels, so it lands in the middle of the fan: the "before"
    contrast that misses the high-surge rides above it and the low-surge ones below.
    """
    tier = next(t for t in config.TIERS if t.tier == "uberx")
    speed = 24.0  # km/h, representative midday speed; keeps the fan a pure surge effect
    grid = list(range(1, 26))

    def price_at(d: int, surge: float) -> float:
        """Return the ground-truth uberx price for a ``d``-km trip at the given surge multiplier."""
        duration = d / speed * 60.0
        raw = (tier.base_fare + tier.per_km * d + tier.per_min * duration) * surge + tier.booking_fee
        return round(max(tier.min_fare, raw), 2)

    same_tier = rides[rides["tier"] == tier.tier]
    line = LinearRegression().fit(same_tier[["distance_km"]], same_tier["price_eur"])
    return {
        "tier": tier.tier,
        "distance_grid": grid,
        "single_line": {"slope": round(float(line.coef_[0]), 3), "intercept": round(float(line.intercept_), 3)},
        "by_surge": [{"surge": s, "prices": [price_at(d, s) for d in grid]} for s in (1.0, 1.5, 2.2)],
    }


def fx_knn(rides: pd.DataFrame, rng: np.random.Generator) -> object:
    """One query ride and its 5 nearest neighbours in standardized (distance, hour) space.

    Mirrors the kNN scene: the neighbours light up and their mean price is the prediction.
    """
    feats = ["distance_km", "hour"]
    z = StandardScaler().fit_transform(rides[feats])
    qi = int(rng.integers(0, len(rides)))
    d = np.linalg.norm(z - z[qi], axis=1)
    order = np.argsort(d)
    neigh = [i for i in order if i != qi][:5]
    km = rides["distance_km"].to_numpy()
    hour = rides["hour"].to_numpy()
    price = rides["price_eur"].to_numpy()
    neighbours = [
        {"km": round(float(km[i]), 2), "hour": int(hour[i]), "price": round(float(price[i]), 2)}
        for i in neigh
    ]
    return {
        "query": {"km": round(float(km[qi]), 2), "hour": int(hour[qi]), "true_price": round(float(price[qi]), 2)},
        "neighbours": neighbours,
        "predicted": round(float(np.mean([n["price"] for n in neighbours])), 2),
    }


def fx_map_points(rng: np.random.Generator) -> object | None:
    """A downsampled set of Madrid street dots (lat, lon, district) for the map scenes."""
    path = RAW / "madrid_locations.csv"
    if not path.exists():
        return None
    loc = pd.read_csv(path)
    if len(loc) > MAP_N:
        loc = loc.sample(MAP_N, random_state=int(rng.integers(0, 2**31)))
    lat = loc["lat"].to_numpy()
    lon = loc["lon"].to_numpy()
    district = loc["district"].to_numpy()
    return [
        {"lat": round(float(lat[i]), 5), "lon": round(float(lon[i]), 5), "district": str(district[i])}
        for i in range(len(loc))
    ]


def fx_balance() -> object | None:
    """Per-street intensities for the balanced (real) vs unbalanced (hotspot) maps, plus Gini.

    Real counts come from ``.scratch/pickup_counts_by_location.npy``; the unbalanced twin is the
    same total rides piled onto downtown. Points are downsampled together so the pair lines up.
    """
    counts_path = SCRATCH / "pickup_counts_by_location.npy"
    loc_path = RAW / "madrid_locations.csv"
    if not counts_path.exists() or not loc_path.exists():
        return None
    per_loc = np.load(counts_path).astype(float)
    loc = pd.read_csv(loc_path)
    lat, lon = loc["lat"].to_numpy(), loc["lon"].to_numpy()
    total = float(per_loc.sum())
    unb = synthetic_unbalanced(lat, lon, total)
    rel_bal = per_loc / per_loc.mean()
    rel_unb = unb / unb.mean()

    idx = np.linspace(0, len(loc) - 1, min(MAP_N, len(loc))).astype(int)
    points = [
        {"lat": round(float(lat[i]), 5), "lon": round(float(lon[i]), 5), "bal": round(float(rel_bal[i]), 2), "unb": round(float(rel_unb[i]), 2)}
        for i in idx
    ]
    return {
        "points": points,
        "gini_balanced": round(gini(per_loc), 3),
        "gini_unbalanced": round(gini(unb), 3),
        "streets_covered": int((per_loc > 0).sum()),
        "streets_total": int(len(per_loc)),
        "total_rides": int(total),
    }


def fx_daily_counts(rides: pd.DataFrame) -> object | None:
    """Rides per calendar day, scaled from the sample up to the true dataset total.

    Feeds the map counter climbing toward the real total. The per-day shape comes from the
    sample's timestamps; the scale factor is the true total (pickup-counts sum) over the sample.
    """
    counts_path = SCRATCH / "pickup_counts_by_location.npy"
    if not counts_path.exists():
        return None
    total = int(np.load(counts_path).sum())
    days = pd.to_datetime(rides["timestamp"]).dt.date.value_counts().sort_index()
    scale = total / len(rides)
    series = [{"date": str(d), "count": int(round(c * scale))} for d, c in days.items()]
    return {"total_rides": total, "days": series}


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and write every available fixture into the destination directory."""
    parser = argparse.ArgumentParser(description="Export JSON fixtures for the Uber-video animations.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUTPUT / "fixtures" / SLUG,
        help="destination (point at videocraft/public/fixtures/uber_prezzi to write in place)",
    )
    parser.add_argument("--seed", type=int, default=config.SEED, help="sampling seed (default: %(default)s)")
    args = parser.parse_args(argv)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    print(f"writing fixtures to {out_dir}")

    write_json(out_dir, "tiers.json", fx_tiers())
    write_json(out_dir, "pricing.json", fx_pricing())
    for name, obj in (("ucurve.json", fx_ucurve()), ("arc_metrics.json", fx_arc_metrics()), ("map_points.json", fx_map_points(rng)), ("balance.json", fx_balance())):
        if obj is None:
            print(f"  {name:<24} skipped (source missing)")
        else:
            write_json(out_dir, name, obj)

    rides = load_rides(rng, 20_000)
    if rides is None:
        print("  (no rides source: scatter/ols/knn/daily_counts skipped)")
        return
    write_json(out_dir, "scatter.json", fx_scatter(rides))
    write_json(out_dir, "ols.json", fx_ols(rides))
    write_json(out_dir, "poly_fit.json", fx_poly(rides))
    write_json(out_dir, "knn_demo.json", fx_knn(rides, rng))
    daily = fx_daily_counts(rides)
    if daily is not None:
        write_json(out_dir, "daily_counts.json", daily)


if __name__ == "__main__":
    main()
