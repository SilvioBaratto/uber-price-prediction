"""End-to-end simulate smoke.

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


def test_e2e_train_then_simulate_loads_saved_model(tmp_path) -> None:
    # `uber train` persists weights; `uber simulate --model` then loads them WITHOUT refitting.
    generate_data.generate(tmp_path, seed=config.SEED, n_drivers=25)
    rides_path = tmp_path / paths.RIDES_CSV
    model_path = tmp_path / "models" / paths.MODEL_JOBLIB

    train_lines: list[str] = []
    rc_train = app.run_train(rides_path, None, model_path, out=train_lines.append)
    assert rc_train == 0
    assert model_path.exists()  # weights were written to disk
    assert "saved to" in "\n".join(train_lines)

    inputs = iter(["Sol", "Gran Vía", "q"])
    sim_lines: list[str] = []
    rc_sim = app.run_simulate(
        rides_path,
        None,
        model_path=model_path,
        locations_path=tmp_path / paths.LOCATIONS_CSV,
        tiers_path=tmp_path / paths.TIERS_CSV,
        input_fn=lambda _prompt="": next(inputs),
        out=sim_lines.append,
    )

    text = "\n".join(sim_lines)
    assert rc_sim == 0
    assert "Model loaded" in text  # it loaded the saved weights...
    assert "retrain-on-launch" not in text  # ...and did NOT retrain
    assert "Tier" in text and "€" in text  # a real quote table was still produced


def test_e2e_simulate_missing_model_falls_back_to_retrain(tmp_path) -> None:
    # a --model path that doesn't exist must not crash: it falls back to retrain-on-launch.
    generate_data.generate(tmp_path, seed=config.SEED, n_drivers=25)
    rides_path = tmp_path / paths.RIDES_CSV

    inputs = iter(["Sol", "Gran Vía", "q"])
    out_lines: list[str] = []
    rc = app.run_simulate(
        rides_path,
        None,
        model_path=tmp_path / "does_not_exist.joblib",
        locations_path=tmp_path / paths.LOCATIONS_CSV,
        tiers_path=tmp_path / paths.TIERS_CSV,
        input_fn=lambda _prompt="": next(inputs),
        out=out_lines.append,
    )

    text = "\n".join(out_lines)
    assert rc == 0
    assert "No saved model" in text and "retrain-on-launch" in text
    assert "Tier" in text and "€" in text
