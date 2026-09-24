"""Infrastructure layer — adapters that implement the application ports and do the I/O.

Filesystem paths, CSV repositories, the price predictors and the system clock live here. This
layer depends on :mod:`uber.application` and :mod:`uber.domain`; neither imports back into it.
"""

from __future__ import annotations
