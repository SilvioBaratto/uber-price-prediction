"""T3.1 — the unified ``uber`` CLI (generate-data | run-arc | simulate) + ``python -m uber``.

The top-level parser exposes three subcommands; generate-data and run-arc forward their options
to the existing module CLIs, while simulate wires the composition root. Routing is asserted by
patching the handlers, and ``python -m uber --help`` is exercised as a real subprocess.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from uber.cli import app
from uber.infrastructure import paths


def test_parser_builds_each_subcommand() -> None:
    parser = app.build_parser()
    assert parser.parse_args(["simulate"]).command == "simulate"
    assert parser.parse_args(["generate-data"]).command == "generate-data"
    assert parser.parse_args(["run-arc"]).command == "run-arc"


def test_top_level_help_lists_the_three_subcommands(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        app.build_parser().parse_args(["--help"])
    out = capsys.readouterr().out
    assert "generate-data" in out and "run-arc" in out and "simulate" in out


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
