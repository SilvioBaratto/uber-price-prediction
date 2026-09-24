"""Application layer — use cases and the ports (interfaces) they depend on.

Depends only on :mod:`uber.domain`. Concrete adapters live in :mod:`uber.infrastructure` and
are injected at the composition boundary (see :mod:`uber.cli`).
"""

from __future__ import annotations
