"""Domain layer — pure business rules (entities, pricing, business config).

The innermost clean-architecture layer: depends only on the standard library and numpy, and
imports nothing from :mod:`uber.application`, :mod:`uber.infrastructure` or :mod:`uber.cli`.
No filesystem or framework access lives here.
"""

from __future__ import annotations
