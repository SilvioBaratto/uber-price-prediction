"""Render a quote as an aligned terminal table (presentation layer).

Pure string formatting — no I/O. :func:`render_options` turns the sorted
:class:`~uber.domain.entities.RideOption` list from a quote into a header line, a rule, and one
aligned row per tier. The caller (:mod:`uber.cli.simulator`) decides where to print it.
"""

from __future__ import annotations

from uber.domain.entities import RideOption

_HEADERS = ("Tier", "Price", "ETA", "Seats")
_ALIGN = ("<", ">", ">", ">")  # tier left-aligned, the numeric columns right-aligned
_GUTTER = "  "


def _row_cells(option: RideOption) -> tuple[str, str, str, str]:
    return (
        option.display_name,
        f"€{option.price_eur:.2f}",
        f"{option.eta_min:.0f} min",
        str(option.capacity),
    )


def render_options(options: list[RideOption]) -> str:
    """Format ``options`` (already sorted) as an aligned table with a header and rule.

    Returns a multi-line string; every line is padded to the same width. An empty list still
    renders the header so the output is never blank.
    """
    rows = [_row_cells(o) for o in options]
    widths = [
        max(len(_HEADERS[i]), *(len(r[i]) for r in rows)) if rows else len(_HEADERS[i])
        for i in range(len(_HEADERS))
    ]

    def line(cells: tuple[str, ...]) -> str:
        return _GUTTER.join(format(cell, f"{_ALIGN[i]}{widths[i]}") for i, cell in enumerate(cells))

    rule = tuple("-" * w for w in widths)
    return "\n".join([line(_HEADERS), line(rule), *(line(r) for r in rows)])
