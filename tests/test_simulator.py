"""The interactive terminal simulator loop (presentation layer).

``run_simulator`` prompts for a pickup then a drop-off (by street name or id), quotes the trip
through the injected QuoteService and prints the rendered table, looping until the user quits.
I/O is injected (``input_fn`` / ``out``) so the loop is driven by scripted input in tests.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from uber.application.quoting import QuoteService
from uber.cli.simulator import run_simulator
from uber.domain import config
from uber.domain.entities import Location
from uber.infrastructure.predictors import FormulaPredictor

SOL = Location(1, "Plaza de la Puerta del Sol", "Sol", "Centro", 40.4169, -3.7029)
ALCOCER = Location(2, "Calle de Alberto Alcocer", "Nueva España", "Chamartín", 40.4620, -3.6790)


class FakeLocationRepo:
    def __init__(self) -> None:
        self._locs = [SOL, ALCOCER]
        self._by_id = {loc.location_id: loc for loc in self._locs}

    def all(self) -> list[Location]:
        return list(self._locs)

    def get(self, location_id: int) -> Location:
        return self._by_id[location_id]

    def search(self, query: str) -> list[Location]:
        needle = query.strip().lower()
        return [loc for loc in self._locs if needle in loc.street.lower()]


class FakeTierRepo:
    def all(self):
        return list(config.TIERS)


class FakeClock:
    def now(self) -> datetime:
        return datetime(2024, 6, 15, 19, 30)


def _make(inputs: list[str]) -> tuple[list[str], Callable[[], None]]:
    it = iter(inputs)
    out_lines: list[str] = []
    locations = FakeLocationRepo()
    service = QuoteService(locations, FakeTierRepo(), FormulaPredictor(), FakeClock())

    def run() -> None:
        run_simulator(
            service,
            locations,
            input_fn=lambda _prompt="": next(it),
            out=out_lines.append,
        )

    return out_lines, run


def test_simulator_quotes_a_trip_then_quits() -> None:
    out_lines, run = _make(["Sol", "Alcocer", "q"])
    run()
    text = "\n".join(out_lines)
    assert "Tier" in text and "Price" in text  # the quote table rendered
    assert "UberX" in text  # at least one tier row
    assert "Puerta del Sol" in text and "Alberto Alcocer" in text  # the resolved trip echoed


def test_simulator_resolves_endpoints_by_id() -> None:
    out_lines, run = _make(["1", "2", "q"])
    run()
    text = "\n".join(out_lines)
    assert "Tier" in text and "UberX" in text


def test_simulator_quit_immediately_prints_no_table() -> None:
    out_lines, run = _make(["q"])
    run()
    text = "\n".join(out_lines)
    assert "€" not in text  # no quote table was produced


def test_simulator_reports_unknown_location_and_recovers() -> None:
    out_lines, run = _make(["zzzzz-nope", "Sol", "Alcocer", "q"])
    run()
    text = "\n".join(out_lines).lower()
    assert "no match" in text or "not found" in text
    assert "tier" in text  # it still produced a quote after the bad entry


def test_simulator_quit_at_dropoff_prints_no_table() -> None:
    # resolve the pickup, then quit at the drop-off prompt -> no quote is produced
    out_lines, run = _make(["Sol", "q"])
    run()
    text = "\n".join(out_lines)
    assert "Puerta del Sol" in text  # the pickup did resolve
    assert "€" not in text  # ...but no table, because the drop-off was abandoned


def test_simulator_reports_unknown_id_and_recovers() -> None:
    # an absent numeric id exercises the id-lookup KeyError path, then a valid id-pair recovers
    out_lines, run = _make(["999999", "1", "2", "q"])
    run()
    text = "\n".join(out_lines)
    assert "No match" in text
    assert "Tier" in text  # recovered and quoted after the bad id
