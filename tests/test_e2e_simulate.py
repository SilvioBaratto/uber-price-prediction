"""T3.2 — end-to-end simulate smoke.

Generate a tiny dataset, then run the real simulate composition (CSV repositories + a
retrain-on-launch ModelPredictor fitted on that small rides file + QuoteService + the interactive
loop) with scripted input, and assert a ride-price table is printed. Only the terminal I/O is
faked; every layer in between is the production wiring.
"""

from __future__ import annotations

from uber.cli import app
from uber.datagen import generate_data
from uber.domain import config
from uber.infrastructure import paths


def test_e2e_simulate_prints_a_quote_table(tmp_path) -> None:
    # a small driver count still writes all four CSVs and emits enough rides to fit the model
    generate_data.generate(tmp_path, seed=config.SEED, n_drivers=25)
    rides_path = tmp_path / paths.RIDES_CSV
    assert rides_path.exists()

    inputs = iter(["Sol", "Gran Vía", "q"])
    out_lines: list[str] = []
    rc = app.run_simulate(
        rides_path,
        None,
        locations_path=tmp_path / paths.LOCATIONS_CSV,
        tiers_path=tmp_path / paths.TIERS_CSV,
        input_fn=lambda _prompt="": next(inputs),
        out=out_lines.append,
    )

    text = "\n".join(out_lines)
    assert rc == 0
    assert "Tier" in text and "Price" in text  # the quote table header
    assert "€" in text  # priced rows
    assert "->" in text  # a resolved pickup -> drop-off trip was echoed
    assert "Model trained on" in text  # retrain-on-launch actually ran
