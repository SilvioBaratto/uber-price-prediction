"""Task 3R.1 — the ``Driver`` entity and driver-population config params (SPEC §2.6, Rev 2).

Pure constants/shape checks (no random draws). The values themselves are tuned later in
Task 3R.5; here we only guarantee the knobs exist and have coherent shapes.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass

import pytest

from uber import config
from uber.entities import Driver

EXPECTED_DRIVER_FIELDS = [
    "driver_id",
    "home_district",
    "activity_class",
    "tenure_start",
    "tenure_end",
    "active_days",
    "n_rides",
    "driver_rating",
]


def test_driver_is_frozen_slotted_dataclass() -> None:
    assert is_dataclass(Driver)
    assert [f.name for f in fields(Driver)] == EXPECTED_DRIVER_FIELDS
    d = Driver(1, "Centro", "casual", "2024-01-01", "2024-03-31", 20, 95, 4.8)
    with pytest.raises(Exception):
        d.driver_id = 2  # type: ignore[misc]  # frozen


def test_n_drivers_default_is_full_madrid_scale() -> None:
    assert isinstance(config.N_DRIVERS, int)
    assert config.N_DRIVERS == 15_000  # O1 = Option C (full operating scale)


def test_activity_class_params_are_coherent() -> None:
    assert len(config.ACTIVITY_CLASSES) == len(config.ACTIVITY_CLASS_WEIGHTS)
    assert config.ACTIVITY_CLASS_WEIGHTS == pytest.approx(
        config.ACTIVITY_CLASS_WEIGHTS
    )  # numeric
    assert sum(config.ACTIVITY_CLASS_WEIGHTS) == pytest.approx(1.0)
    # right-skewed toward casual (research: >50% work 1-5 h/week)
    assert config.ACTIVITY_CLASS_WEIGHTS[0] > config.ACTIVITY_CLASS_WEIGHTS[-1]
    for cls in config.ACTIVITY_CLASSES:
        assert config.WORKDAYS_PER_WEEK[cls] > 0
        assert 0 < config.WORKDAYS_PER_WEEK[cls] <= 7
        assert config.RIDES_PER_WORKDAY_MEAN[cls] > 0
    # intensity increases casual -> full_time
    means = [config.RIDES_PER_WORKDAY_MEAN[c] for c in config.ACTIVITY_CLASSES]
    assert means == sorted(means)


def test_tenure_cohort_params_are_coherent() -> None:
    assert len(config.TENURE_COHORTS) == len(config.TENURE_COHORT_WEIGHTS)
    assert sum(config.TENURE_COHORT_WEIGHTS) == pytest.approx(1.0)
    for c in config.TENURE_COHORTS:
        assert config.TENURE_COHORT_DAYS[c] > 0
    assert config.TENURE_JITTER_DAYS >= 0


def test_hour_volume_profiles_are_24_positive() -> None:
    for profile in (config.HOUR_VOLUME_WEEKDAY, config.HOUR_VOLUME_WEEKEND):
        assert len(profile) == 24
        assert all(w > 0 for w in profile)


def test_bias_and_rating_ranges() -> None:
    assert 0.0 <= config.HOME_DISTRICT_PICKUP_PROB <= 1.0
    assert 1.0 <= config.DRIVER_RATING_MIN < config.DRIVER_RATING_MAX <= 5.0
