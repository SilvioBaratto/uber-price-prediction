"""T2.6 — render_options: format a quote as an aligned terminal table (presentation layer).

Pure string formatting: given the sorted RideOptions from a quote, produce a header line plus
one aligned row per tier (tier name, price in EUR, ETA in minutes, seat capacity).
"""

from __future__ import annotations

from uber.cli.render import render_options
from uber.domain.entities import RideOption

OPTIONS = [
    RideOption("uberx", "UberX", 9.40, 13.0, 4),
    RideOption("comfort", "Comfort", 14.30, 12.0, 4),
    RideOption("black", "Uber Black", 22.10, 11.0, 4),
    RideOption("van", "Uber Van", 28.75, 11.0, 6),
]


def test_render_has_column_headers() -> None:
    text = render_options(OPTIONS)
    header = text.splitlines()[0]
    for col in ("Tier", "Price", "ETA", "Seats"):
        assert col in header


def test_render_shows_each_tier_row() -> None:
    text = render_options(OPTIONS)
    for opt in OPTIONS:
        assert opt.display_name in text


def test_render_formats_price_and_eta_and_seats() -> None:
    text = render_options(OPTIONS)
    assert "9.40" in text          # two-decimal euros
    assert "€" in text             # euro sign present
    assert "13 min" in text        # eta rounded to whole minutes
    # the 6-seat van's capacity shows up
    lines = [ln for ln in text.splitlines() if "Uber Van" in ln]
    assert lines and "6" in lines[0]


def test_render_preserves_input_order() -> None:
    text = render_options(OPTIONS)
    positions = [text.index(o.display_name) for o in OPTIONS]
    assert positions == sorted(positions)


def test_render_rows_are_aligned() -> None:
    text = render_options(OPTIONS)
    lines = text.splitlines()
    # every rendered line has the same visual width (padded to a common column layout)
    body = [ln for ln in lines if ln.strip()]
    assert len({len(ln) for ln in body}) == 1


def test_render_empty_is_graceful() -> None:
    text = render_options([])
    assert isinstance(text, str)
    assert "Tier" in text  # header still present
