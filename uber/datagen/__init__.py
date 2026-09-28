"""Datagen feature package — synthetic-data generation for the Madrid ride log.

A supporting feature package (not a clean-architecture layer): it orchestrates the
day-by-day driver simulation and writes the dataset CSVs. It depends
inward on :mod:`uber.domain` (pricing, config) and on :mod:`uber.infrastructure`
(CSV plumbing, empirical distractor supports); nothing imports back into it.
"""

from __future__ import annotations
