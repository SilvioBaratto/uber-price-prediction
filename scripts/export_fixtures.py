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
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from uber.domain import config  # noqa: E402  (needs PROJECT_ROOT on sys.path first)
from uber.modeling import pipeline as modeling  # noqa: E402

SLUG = "uber_prezzi"
RAW = PROJECT_ROOT / "data" / "raw"
SCRATCH = PROJECT_ROOT / ".scratch"
OUTPUT = PROJECT_ROOT / "output"

# Puerta del Sol, the centre the illustrative unbalanced hotspot decays from (matches
# scripts/plot_balance_map.py so the two stay comparable).
CENTER_LAT, CENTER_LON = 40.4168, -3.7038

SCATTER_N = 800  # points drawn in the price-vs-distance cloud (enough to read, cheap to ship)
MAP_N = 2000  # street dots on the Madrid map

# The arc's published numbers come from this exact sample of rides.csv: it is what reproduces the
# Part-1 baseline of 26.458, so every fixture that quotes the arc must read the same rows.
ARC_SAMPLE = SCRATCH / "rides_sample_100k.csv"
ARC_SAMPLE_N = 100_000

# Video 03's scenes. One tier only: across tiers, two rides with the same km and hour differ by up
# to 4x in price, which on a 2-D plane reads as noise instead of "close rides cost alike".
KNN_TIER = "uberx"
# Sparse on purpose: on all 80k training rides the nearest neighbours sit on the query's own pixel.
KNN_CLOUD_N = 300
# The unscaled neighbours must look identical on the two columns the voice names (km, minutes),
# so the price gap on screen can only come from what scaling would have weighed: tier and surge.
TWIN_KM = 0.5
TWIN_MIN = 1.5
# "Typical" ride: both its errors within this factor of the test-set median, so the example is
# neither a lucky hit for the scaled model nor a freak miss for the unscaled one.
TYPICAL_FACTOR = 1.5


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


def ensure_arc_sample() -> bool:
    """Write the arc's 100k-row sample from ``rides.csv`` if it is not cached yet.

    Returns:
        True when the sample exists afterwards, False when neither it nor ``rides.csv`` is there.
    """
    if ARC_SAMPLE.exists():
        return True
    if not (RAW / "rides.csv").exists():
        return False
    full = pd.read_csv(RAW / "rides.csv")
    ARC_SAMPLE.parent.mkdir(parents=True, exist_ok=True)
    full.sample(ARC_SAMPLE_N, random_state=config.SEED).to_csv(ARC_SAMPLE, index=False)
    return True


def load_rides(rng: np.random.Generator, n: int) -> pd.DataFrame | None:
    """Return a seeded ``n``-row sample of rides, or None if no ride source is available.

    Prefers the cached ``.scratch`` sample; falls back to a capped head of the full
    ``rides.csv`` so the exporter still runs on a fresh checkout that only has the big file.
    """
    if ARC_SAMPLE.exists():
        df = pd.read_csv(ARC_SAMPLE)
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


def fx_tree() -> object:
    """The repository's own layout, as git tracks it.

    Video 01 animates a ``tree`` of this project, so the folders on screen must be the
    folders that exist — not a drawing of them. The listing comes from ``git ls-files``
    rather than from walking the disk: the working copy also holds ``.venv``, caches and
    the 1,2 GB of generated rides, none of which is the project. The commit travels with
    it so a frame can be traced back to the snapshot it was rendered from.
    """
    def git(*argv: str) -> str:
        return subprocess.run(
            ["git", *argv], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    tracciati = [Path(p) for p in git("ls-files").splitlines()]
    primo_livello = sorted(
        {(p.parts[0], "dir" if len(p.parts) > 1 else "file") for p in tracciati},
        key=lambda x: (x[1] == "file", x[0]),
    )
    pacchetto = PROJECT_ROOT.name.replace("-", "_").split("_")[0]
    strati: list[dict[str, object]] = []
    for nome in sorted({p.parts[1] for p in tracciati if p.parts[0] == "uber" and len(p.parts) > 2}):
        file = sorted(p.name for p in tracciati if p.parts[:2] == ("uber", nome))
        strati.append({"nome": nome, "file": file, "nFile": len(file)})
    nel_pacchetto = [p for p in tracciati if p.parts[0] == "uber" and len(p.parts) > 1]
    return {
        "repo": PROJECT_ROOT.name,
        "commit": git("rev-parse", "--short", "HEAD"),
        "pacchetto": pacchetto,
        "nFileTracciati": len(tracciati),
        # What `tree uber/` prints on its last line, counted the way tree counts it.
        "nCartellePacchetto": len(strati),
        "nFilePacchetto": len(nel_pacchetto),
        "primoLivello": [{"nome": n, "tipo": t} for n, t in primo_livello],
        "strati": strati,
    }


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


def _corsa(row: pd.Series, price: float) -> dict[str, object]:
    """Return one ride as the kNN scenes label it (tier, km, minutes, hour, surge, ..., price)."""
    return {
        "tier": str(row["tier"]),
        "km": round(float(row["distance_km"]), 2),
        "min": round(float(row["duration_min"]), 1),
        "ora": int(row["hour"]),
        "surge": round(float(row["surge_multiplier"]), 2),
        "giorno": int(row["day_of_week"]),
        "mese": int(row["month"]),
        "quartiere": str(row["pickup_district"]),
        "prezzo": round(float(price), 2),
    }


def fx_knn_vicini() -> object | None:
    """Video 03's kNN scenes, on the arc's own split and the arc's own Part-2 pipeline.

    ``knn_demo.json`` (video 00) is a free-standing 2-D toy. Video 03 narrates the real model
    instead — its tuned k, a StandardScaler fit on the training rows only, and why the unscaled
    twin doubles the error — so everything here comes from ``knn_pipeline`` fit on ``make_split``
    of the same 100k sample ``uber run-arc`` reads, and the scaler's numbers are the fitted ones.

    One test ride is the protagonist of every scene, chosen by rule rather than by hand: a typical
    ride (both errors near the test median) whose scaled neighbours share its tier, whose unscaled
    neighbours are km/minute twins that include another tier, and whose scaled neighbours are also
    its three nearest on the drawn km-by-hour plane. Among those, the widest unscaled price spread
    wins: the failure the voice describes has to be visible, not just true.

    Returns:
        The fixture, or None when the arc sample or ``output/full_run.log`` is missing.

    Raises:
        RuntimeError: No test ride satisfies the scene constraints.
    """
    log = OUTPUT / "full_run.log"
    if not ARC_SAMPLE.exists() or not log.exists():
        return None
    text = log.read_text(encoding="utf-8")
    k_found = re.search(r"\[Part 2\] tuned k\*=(\d+)", text)
    std_found = re.search(r"standardized kNN\s+test RMSE=([\d.]+)\s+R2=([\d.]+)", text)
    raw_found = re.search(r"raw \(unscaled\) kNN test RMSE=([\d.]+)", text)
    if k_found is None or std_found is None or raw_found is None:
        return None
    k = int(k_found.group(1))

    split = modeling.make_split(modeling.load_rides(ARC_SAMPLE))
    train = split.X_train.reset_index(drop=True)
    test = split.X_test.reset_index(drop=True)
    scaled = modeling.knn_pipeline(k, scale=True).fit(split.X_train, split.y_train)
    unscaled = modeling.knn_pipeline(k, scale=False).fit(split.X_train, split.y_train)

    def neighbours(model: object) -> np.ndarray:
        """Return the k training-row positions each test ride's prediction averages."""
        pre, est = model.named_steps["pre"], model.named_steps["est"]  # type: ignore[attr-defined]
        return est.kneighbors(pre.transform(test), return_distance=False)

    near_std, near_raw = neighbours(scaled), neighbours(unscaled)
    err_std = np.abs(split.y_train[near_std].mean(axis=1) - split.y_test)
    err_raw = np.abs(split.y_train[near_raw].mean(axis=1) - split.y_test)
    med_std, med_raw = float(np.median(err_std)), float(np.median(err_raw))

    scaler = scaled.named_steps["pre"].named_transformers_["num"]
    mean = dict(zip(modeling.PINNED_NUMERIC, scaler.mean_))
    dev = dict(zip(modeling.PINNED_NUMERIC, scaler.scale_))

    def z(frame: pd.DataFrame, col: str) -> np.ndarray:
        """Return ``col`` as the fitted scaler sees it: (value - mean) / deviation."""
        return (frame[col].to_numpy(dtype=float) - mean[col]) / dev[col]

    def plane(frame: pd.DataFrame) -> np.ndarray:
        """Return the scaled (km, hour) coordinates the plane scenes draw."""
        return np.column_stack([z(frame, "distance_km"), z(frame, "hour")])

    cloud = train[train["tier"] == KNN_TIER].sample(KNN_CLOUD_N, random_state=config.SEED)
    tier_train = train["tier"].to_numpy()
    km_train, min_train = train["distance_km"].to_numpy(), train["duration_min"].to_numpy()

    def typical(q: int) -> bool:
        """Report whether both of ride ``q``'s errors sit near their test-set medians."""
        return all(
            med / TYPICAL_FACTOR <= err[q] <= med * TYPICAL_FACTOR
            for err, med in ((err_std, med_std), (err_raw, med_raw))
        )

    best, best_spread = None, -1.0
    for q in np.flatnonzero(test["tier"].to_numpy() == KNN_TIER):
        ns, nr = near_std[q], near_raw[q]
        if not (tier_train[ns] == KNN_TIER).all() or (tier_train[nr] == KNN_TIER).all():
            continue
        if (np.abs(km_train[nr] - test.at[q, "distance_km"]) > TWIN_KM).any():
            continue
        if (np.abs(min_train[nr] - test.at[q, "duration_min"]) > TWIN_MIN).any():
            continue
        if err_raw[q] <= err_std[q] or not typical(q):
            continue
        zq = plane(test.iloc[[q]])[0]
        reach = np.linalg.norm(plane(train.iloc[ns]) - zq, axis=1).max()
        others = cloud.drop(index=ns, errors="ignore")
        if (np.linalg.norm(plane(others) - zq, axis=1) < reach).any():
            continue
        spread = float(np.ptp(split.y_train[nr]))
        if spread > best_spread:
            best, best_spread = q, spread
    if best is None:
        raise RuntimeError("no test ride satisfies the video-03 scene constraints")

    query = test.iloc[[best]]
    shown = pd.concat([cloud, train.iloc[[i for i in near_std[best] if i not in cloud.index]]])
    pos = {int(i): p for p, i in enumerate(shown.index)}
    zz = np.column_stack([z(shown, c) for c in ("distance_km", "duration_min", "hour")])
    pts = plane(shown)
    gaps = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    np.fill_diagonal(gaps, np.inf)
    twins = np.unravel_index(np.argmin(gaps), gaps.shape)
    np.fill_diagonal(gaps, -np.inf)
    apart = np.unravel_index(np.argmax(gaps), gaps.shape)
    y_shown = split.y_train[shown.index.to_numpy()]

    def mismatch(near: np.ndarray) -> dict[str, dict[str, float]]:
        """Summarise, over the whole test set, how far the chosen neighbours sit per column.

        ``delta`` is the mean absolute gap on each numeric column; ``diverso`` the share of
        neighbours whose category differs from the test ride's.
        """
        delta = {}
        for c in modeling.PINNED_NUMERIC:
            gap = train[c].to_numpy(dtype=float)[near] - test[c].to_numpy(dtype=float)[:, None]
            delta[c] = round(float(np.abs(gap).mean()), 2)
        diverso = {
            c: round(float((train[c].to_numpy()[near] != test[c].to_numpy()[:, None]).mean()), 3)
            for c in ("tier", "pickup_district")
        }
        return {"delta": delta, "diverso": diverso}

    return {
        "fonte": {
            "campione": f"rides.csv .sample({ARC_SAMPLE_N}, random_state={config.SEED})",
            "nTrain": len(train),
            "nTest": len(test),
            "k": k,
        },
        "metriche": {
            "rmseScalato": float(std_found.group(1)),
            "r2Scalato": float(std_found.group(2)),
            "rmseGrezzo": float(raw_found.group(1)),
            "medianaErroreScalato": round(med_std, 2),
            "medianaErroreGrezzo": round(med_raw, 2),
        },
        "logPart2": [line for line in text.splitlines() if line.startswith("[Part 2]")],
        "scala": {
            c: {
                "media": round(float(mean[c]), 2),
                "dev": round(float(dev[c]), 2),
                "p05": round(float(train[c].quantile(0.05)), 1),
                "p95": round(float(train[c].quantile(0.95)), 1),
            }
            for c in modeling.PINNED_NUMERIC
        },
        "corsaNuova": {
            **_corsa(query.iloc[0], split.y_test[best]),
            "z": {c: round(float(z(query, c)[0]), 2) for c in modeling.PINNED_NUMERIC},
        },
        "scalati": {
            "vicini": [_corsa(train.iloc[i], split.y_train[i]) for i in near_std[best]],
            "stima": round(float(scaled.predict(query)[0]), 2),
        },
        "grezzi": {
            "vicini": [_corsa(train.iloc[i], split.y_train[i]) for i in near_raw[best]],
            "stima": round(float(unscaled.predict(query)[0]), 2),
        },
        "piano": {
            "tier": KNN_TIER,
            "punti": [
                {
                    "km": round(float(r["distance_km"]), 2),
                    "min": round(float(r["duration_min"]), 1),
                    "ora": int(r["hour"]),
                    "surge": round(float(r["surge_multiplier"]), 2),
                    "prezzo": round(float(y_shown[p]), 2),
                    "zKm": round(float(zz[p, 0]), 3),
                    "zMin": round(float(zz[p, 1]), 3),
                    "zOra": round(float(zz[p, 2]), 3),
                }
                for p, (_, r) in enumerate(shown.iterrows())
            ],
            "vicini": [pos[int(i)] for i in near_std[best]],
            "gemelle": [int(twins[0]), int(twins[1])],
            "opposte": [int(apart[0]), int(apart[1])],
        },
        "perche": {"grezzi": mismatch(near_raw), "scalati": mismatch(near_std)},
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

    write_json(out_dir, "tree.json", fx_tree())
    write_json(out_dir, "tiers.json", fx_tiers())
    write_json(out_dir, "pricing.json", fx_pricing())
    for name, obj in (("ucurve.json", fx_ucurve()), ("arc_metrics.json", fx_arc_metrics()), ("map_points.json", fx_map_points(rng)), ("balance.json", fx_balance())):
        if obj is None:
            print(f"  {name:<24} skipped (source missing)")
        else:
            write_json(out_dir, name, obj)

    if not ensure_arc_sample():
        print("  (no rides.csv: the arc sample cannot be built)")
    vicini = fx_knn_vicini()
    if vicini is None:
        print(f"  {'knn_vicini.json':<24} skipped (arc sample or full_run.log missing)")
    else:
        write_json(out_dir, "knn_vicini.json", vicini)

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
