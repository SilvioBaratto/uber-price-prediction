"""Clock adapter — the system wall clock (infrastructure layer).

Implements the :class:`~uber.application.ports.Clock` port so the application layer can read the
current time without importing ``datetime.now`` directly (which keeps quoting testable).
"""

from __future__ import annotations

from datetime import datetime


class SystemClock:
    """``Clock`` backed by the local system time."""

    def now(self) -> datetime:
        return datetime.now()
