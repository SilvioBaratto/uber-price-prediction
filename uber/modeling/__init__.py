"""Modeling feature package — the eight-part regression arc.

A supporting feature package (not a clean-architecture layer): :mod:`~uber.modeling.pipeline`
holds the shared scikit-learn plumbing (feature sets, the one fixed split, preprocessing,
metrics, OLS inference, the report formatter) and :mod:`~uber.modeling.arc` orchestrates
Parts 1->8. It depends inward on :mod:`uber.domain` (config) and :mod:`uber.infrastructure`
(paths); nothing in those layers imports back into it.
"""

from __future__ import annotations
