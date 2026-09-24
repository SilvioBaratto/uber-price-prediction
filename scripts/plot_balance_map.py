"""Render TWO standalone 1920x1920 Madrid maps: one BALANCED, one UNBALANCED.

Just the map of Madrid, full-frame (no captions/legend — text is added over the reel).
Both are the same city and the same total number of rides; only the *spatial
distribution* differs, on one shared colour scale so the pair stays comparable.

  output/madrid_balanced.png    our real rides.csv  -> pickups spread evenly (flat tone)
  output/madrid_unbalanced.png  illustrative        -> demand piled onto downtown (hotspot)

Colour = pickups per street relative to the city average (1.0 = perfectly even), on
one sequential-blue ramp (light -> dark).  Palette follows the `dataviz` skill.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

from uber import config

# --- design tokens (dataviz skill, light mode) -----------------------------
SURFACE = "#fcfcfb"   # chart surface (the whole frame)
# Sequential blue ramp, steps 100->700 (near-zero -> high). Lightest is light-blue, not
# white, so even the sparsest streets stay visible and the city outline is preserved.
BLUE_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
CMAP = LinearSegmentedColormap.from_list("uber_blue_seq", BLUE_RAMP)

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})

CENTER_LAT, CENTER_LON = 40.4168, -3.7038  # Puerta del Sol
VMAX = 3.0  # colour scale cap, in multiples of the city average (>=3x clips to darkest)


def gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    c = np.cumsum(x)
    return (n + 1 - 2 * np.sum(c) / c[-1]) / n


def actual_counts(n_loc: int) -> np.ndarray:
    """Per-location pickup counts from rides.csv (cached under .scratch for fast re-runs)."""
    cache = config.PROJECT_ROOT / ".scratch" / "pickup_counts_by_location.npy"
    if cache.exists():
        return np.load(cache)
    counts = np.zeros(n_loc + 1, dtype=np.int64)  # 1-based ids
    for ch in pd.read_csv(config.RAW_DIR / config.RIDES_CSV,
                          usecols=["pickup_location_id"], chunksize=2_000_000):
        np.add.at(counts, ch["pickup_location_id"].to_numpy(), 1)
    per_loc = counts[1:]
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, per_loc)
    return per_loc


def synthetic_unbalanced(lat: np.ndarray, lon: np.ndarray, total: int) -> np.ndarray:
    """Same total rides, but demand decays from the city centre (a classic hotspot map)."""
    dx = (lon - CENTER_LON) * np.cos(np.radians(CENTER_LAT))
    dy = lat - CENTER_LAT
    d = np.hypot(dx, dy)                       # degrees
    w = np.exp(-d / 0.025) + 0.06              # sharp central peak + small floor (keep edges visible)
    return total * w / w.sum()


def square_window(lat, lon):
    """A square lon/lat window cropped to Madrid's dense core (drops sparse outer
    streets so the city fills the frame) while keeping true lon/lat proportions.

    In a square pixel frame that fills the axes, aspect is carried by the range
    ratio: 1° of latitude covers ASPECT× the ground of 1° of longitude here, so a
    true-shape square needs ``lon_range = ASPECT × lat_range``.
    """
    aspect = 1.0 / np.cos(np.radians(CENTER_LAT))   # ~1.314 at Madrid's latitude
    cx, cy = np.median(lon), np.median(lat)
    half = max(np.percentile(np.abs(lat - cy), 98),
               np.percentile(np.abs(lon - cx), 98) / aspect) * 1.04
    return (cx - half * aspect, cx + half * aspect), (cy - half, cy + half)


def render_map(lat, lon, rel, norm, mark_center, xlim, ylim, out_path):
    """One standalone full-bleed 1920x1920 frame: just the map of Madrid."""
    fig = plt.figure(figsize=(19.2, 19.2), dpi=100)   # square 1920x1920
    ax = fig.add_axes((0, 0, 1, 1))            # fill the whole frame
    order = np.argsort(rel)                     # densest points drawn last (on top)
    ax.scatter(lon[order], lat[order], c=rel[order], cmap=CMAP, norm=norm,
               s=22, linewidths=0, marker="o")
    if mark_center:
        ax.scatter([CENTER_LON], [CENTER_LAT], s=320, marker="*", c="#0d366b",
                   edgecolors=SURFACE, linewidths=1.8, zorder=5)
    # explicit square window (no auto-margin); range ratio keeps Madrid undistorted
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def main() -> None:
    loc = pd.read_csv(config.RAW_DIR / config.LOCATIONS_CSV)
    lat = loc["lat"].to_numpy(); lon = loc["lon"].to_numpy()
    n_loc = len(loc)

    per_loc = actual_counts(n_loc).astype(float)
    total = int(per_loc.sum())
    unbal = synthetic_unbalanced(lat, lon, total)

    rel_bal = per_loc / per_loc.mean()         # 1.0 = exactly the city average
    rel_unb = unbal / unbal.mean()
    g_bal, g_unb = gini(per_loc), gini(unbal)

    norm = Normalize(vmin=0.0, vmax=VMAX)
    out_dir = config.PROJECT_ROOT / "output"
    xlim, ylim = square_window(lat, lon)       # same crop for both, so they line up

    render_map(lat, lon, rel_bal, norm, False, xlim, ylim, out_dir / "madrid_balanced.png")
    render_map(lat, lon, rel_unb, norm, True, xlim, ylim, out_dir / "madrid_unbalanced.png")

    print(f"wrote {out_dir / 'madrid_balanced.png'}")
    print(f"wrote {out_dir / 'madrid_unbalanced.png'}")
    print(f"balanced Gini={g_bal:.3f}  unbalanced Gini={g_unb:.3f}  total rides={total:,}")


if __name__ == "__main__":
    main()
