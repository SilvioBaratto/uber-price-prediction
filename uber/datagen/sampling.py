"""Seeded random sampling for ride generation (no filesystem I/O).

All randomness flows through a single seeded ``numpy.random.Generator`` so generation is
reproducible: the same seed yields identical arrays. Functions receive the
generator and already-loaded sizes/arrays and return numpy arrays; calendar fields
(hour / day-of-week / month) are derived downstream from the sampled timestamps.
"""

from __future__ import annotations

import numpy as np

from uber.domain import config


def make_rng(seed: int) -> np.random.Generator:
    """Return a fresh seeded generator; the same seed reproduces identical draws."""
    return np.random.default_rng(seed)


def sample_timestamps(rng: np.random.Generator, n: int, year: int = config.YEAR) -> np.ndarray:
    """Sample ``n`` timestamps uniformly across ``year`` at second resolution.

    Draws uniform second offsets in ``[0, seconds_in_year)``; the span is computed from the
    year bounds, so leap years (2024) are handled automatically.
    """
    start = np.datetime64(f"{year}-01-01T00:00:00", "s")
    end = np.datetime64(f"{year + 1}-01-01T00:00:00", "s")
    span = (end - start).astype("int64")
    offsets = rng.integers(0, span, size=n, dtype=np.int64)
    return start + offsets.astype("timedelta64[s]")


def sample_od_pairs(
    rng: np.random.Generator, n: int, n_locations: int
) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``n`` (pickup, dropoff) 1-based location-id pairs that are always distinct.

    ``pickup`` is uniform over ``[1, n_locations]``; ``dropoff`` is pickup shifted by a
    non-zero offset (mod ``n_locations``), which guarantees ``pickup != dropoff`` without a
    rejection loop and keeps dropoff marginally uniform (zero-distance guard).
    """
    pickup = rng.integers(1, n_locations + 1, size=n, dtype=np.int64)
    offset = rng.integers(1, n_locations, size=n, dtype=np.int64)  # 1 .. n_locations-1
    dropoff = ((pickup - 1 + offset) % n_locations) + 1
    return pickup, dropoff


def sample_tiers(rng: np.random.Generator, n: int, tier_ids: list[str]) -> np.ndarray:
    """Sample ``n`` tier ids uniformly (with replacement) from ``tier_ids``."""
    return rng.choice(np.asarray(tier_ids), size=n)


def sample_jitter(rng: np.random.Generator, n: int) -> np.ndarray:
    """Sample ``n`` multiplicative surge jitters ~ lognormal(0, JITTER_SIGMA), median 1.0."""
    return rng.lognormal(mean=0.0, sigma=config.JITTER_SIGMA, size=n)


def sample_empirical(rng: np.random.Generator, values: np.ndarray, n: int) -> np.ndarray:
    """Draw ``n`` values with replacement from the empirical support ``values`` (seeded).

    ``values`` is an already null-free 1-D array (see
    :func:`uber.infrastructure.ncr.load_ncr`); indices are
    drawn uniformly, so the output reproduces the input's empirical distribution. Used for the
    NCR-derived per-ride distractors. Dtype (str or float) is preserved.
    """
    arr = np.asarray(values)
    if arr.size == 0:
        raise ValueError("cannot sample from an empty empirical distribution")
    idx = rng.integers(0, arr.size, size=n, dtype=np.int64)
    return arr[idx]
