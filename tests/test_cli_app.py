"""The unified ``uber`` CLI (generate-data | run-arc | simulate) + ``python -m uber``.

The top-level parser exposes three subcommands; generate-data and run-arc forward their options
to the existing module CLIs, while simulate wires the composition root. Routing is asserted by
patching the handlers, and ``python -m uber --help`` is exercised as a real subprocess.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from uber.cli import app
from uber.infrastructure import paths


def test_parser_builds_each_subcommand() -> None:
    parser = app.build_parser()
    assert parser.parse_args(["simulate"]).command == "simulate"
    assert parser.parse_args(["generate-data"]).command == "generate-data"
    assert parser.parse_args(["run-arc"]).command == "run-arc"
    assert parser.parse_args(["train"]).command == "train"


def test_top_level_help_lists_the_four_subcommands(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        app.build_parser().parse_args(["--help"])
    out = capsys.readouterr().out
    assert "generate-data" in out and "run-arc" in out and "simulate" in out and "train" in out


def test_simulate_routes_to_its_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_simulate(args: object) -> int:
        seen["cmd"] = "simulate"
        return 0

    monkeypatch.setattr(app, "_handle_simulate", fake_simulate)
    rc = app.main(["simulate"])
    assert seen["cmd"] == "simulate"
    assert rc == 0


def test_generate_data_forwards_options_to_module_main(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str] | None] = []
    monkeypatch.setattr("uber.datagen.generate_data.main", lambda argv=None: calls.append(argv))
    rc = app.main(["generate-data", "--seed", "7", "--n-drivers", "3"])
    assert calls == [["--seed", "7", "--n-drivers", "3"]]
    assert rc == 0


def test_run_arc_forwards_options_to_module_main(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str] | None] = []
    monkeypatch.setattr("uber.modeling.arc.main", lambda argv=None: calls.append(argv))
    rc = app.main(["run-arc", "--seed", "1", "--charts"])
    assert calls == [["--seed", "1", "--charts"]]
    assert rc == 0


def test_train_routes_to_its_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_train(args: object) -> int:
        seen["cmd"] = "train"
        return 0

    monkeypatch.setattr(app, "_handle_train", fake_train)
    rc = app.main(["train"])
    assert seen["cmd"] == "train"
    assert rc == 0


def test_train_out_flag_parses() -> None:
    args = app.build_parser().parse_args(["train", "--out", "models/custom.joblib"])
    assert str(args.out) == str(Path("models/custom.joblib"))


def test_simulate_accepts_model_flag() -> None:
    args = app.build_parser().parse_args(["simulate", "--model", "models/price_model.joblib"])
    assert str(args.model) == str(Path("models/price_model.joblib"))


def test_simulate_model_defaults_to_none() -> None:
    assert app.build_parser().parse_args(["simulate"]).model is None


def test_python_m_uber_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "uber", "--help"],
        capture_output=True,
        text=True,
        cwd=str(paths.PROJECT_ROOT),
    )
    assert result.returncode == 0
    assert "generate-data" in result.stdout
    assert "simulate" in result.stdout


@pytest.mark.parametrize("bad", ["0", "-5"])
def test_simulate_rejects_non_positive_max_train_rows(bad: str) -> None:
    # a clean argparse usage error (exit code 2), not an opaque crash inside model training
    with pytest.raises(SystemExit) as exc:
        app.build_parser().parse_args(["simulate", "--max-train-rows", bad])
    assert exc.value.code == 2


def test_simulate_accepts_positive_max_train_rows() -> None:
    args = app.build_parser().parse_args(["simulate", "--max-train-rows", "1000"])
    assert args.max_train_rows == 1000
