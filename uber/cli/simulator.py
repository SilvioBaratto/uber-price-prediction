"""Interactive terminal simulator — pick A -> B in Madrid, see priced ride tiers.

The loop prompts for a pickup then a drop-off (by street name or numeric id), quotes the trip
through the injected :class:`~uber.application.quoting.QuoteService`, and prints the rendered
table, repeating until the user quits with ``q``. I/O is injected (``input_fn`` / ``out``) so the
loop is fully driven and captured in tests; the composition root wires the real ``input``/``print``.
"""

from __future__ import annotations

from typing import Callable

from uber.application.ports import LocationRepository
from uber.application.quoting import QuoteService
from uber.cli.render import render_options
from uber.domain.entities import Location

_QUIT = {"q", "quit", "exit"}
_INTRO = "Uber Madrid — trip price simulator. Enter a street name or id; 'q' to quit."


def _resolve(locations: LocationRepository, raw: str) -> Location | None:
    """Resolve raw user text to a location: numeric -> id lookup, else first name match."""
    text = raw.strip()
    if not text:
        return None
    if text.isdigit():
        try:
            return locations.get(int(text))
        except KeyError:
            return None
    hits = locations.search(text)
    return hits[0] if hits else None


def _prompt_location(
    locations: LocationRepository,
    label: str,
    input_fn: Callable[[str], str],
    out: Callable[[str], None],
) -> Location | None:
    """Prompt until a location resolves; return ``None`` if the user quits."""
    while True:
        raw = input_fn(f"{label} (street name or id, 'q' to quit): ").strip()
        if raw.lower() in _QUIT:
            return None
        location = _resolve(locations, raw)
        if location is None:
            out(f"  No match for {raw!r}. Try again.")
            continue
        out(f"  -> {location.street} ({location.district})")
        return location


def run_simulator(
    quote_service: QuoteService,
    locations: LocationRepository,
    *,
    input_fn: Callable[[str], str] = input,
    out: Callable[[str], None] = print,
) -> None:
    """Run the interactive quote loop until the user quits.

    ``quote_service`` prices each trip (its injected clock supplies "now"); ``locations`` resolves
    the typed pickup/drop-off. ``input_fn``/``out`` are injected for testability.
    """
    out(_INTRO)
    while True:
        origin = _prompt_location(locations, "Pickup", input_fn, out)
        if origin is None:
            break
        destination = _prompt_location(locations, "Drop-off", input_fn, out)
        if destination is None:
            break
        try:
            options = quote_service.quote(origin, destination)
        except ValueError as exc:
            out(f"  Cannot quote: {exc}")
            continue
        out(f"\n{origin.street} -> {destination.street}")
        out(render_options(options))
        out("")
    out("Bye!")
