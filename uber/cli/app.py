"""The unified ``uber`` command-line interface (composition root).

Three subcommands:

* ``generate-data`` — build the synthetic Madrid dataset (forwards to
  :func:`uber.datagen.generate_data.main`).
* ``run-arc`` — run the eight-part regression arc (forwards to :func:`uber.modeling.arc.main`).
* ``simulate`` — interactive A->B ride-price simulator; wires the CSV repositories, the
  retrain-on-launch :class:`~uber.infrastructure.predictors.ModelPredictor`, a system clock and
  the :class:`~uber.application.quoting.QuoteService`, then runs the terminal loop.

This module is the only place the layers are wired together; everything else depends inward.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from uber.infrastructure import paths


def _handle_generate_data(args: argparse.Namespace) -> int:
    from uber.datagen import generate_data

    generate_data.main(args.forward)
    return 0


def _handle_run_arc(args: argparse.Namespace) -> int:
    from uber.modeling import arc

    arc.main(args.forward)
    return 0


def run_simulate(
    rides_path: Path | str,
    max_train_rows: int | None,
    *,
    locations_path: Path | None = None,
    tiers_path: Path | None = None,
    input_fn: Callable[[str], str] = input,
    out: Callable[[str], None] = print,
) -> int:
    """Wire the simulate composition and run the loop (the testable seam behind ``simulate``).

    Repositories default to the committed ``data/raw`` catalogs; the model is retrained on
    launch from ``rides_path`` (optionally capped by ``max_train_rows``). ``input_fn``/``out`` are
    injected so an end-to-end run can be driven and captured by tests.
    """
    from uber.application.quoting import QuoteService
    from uber.cli.simulator import run_simulator
    from uber.infrastructure.clock import SystemClock
    from uber.infrastructure.predictors import ModelPredictor
    from uber.infrastructure.repositories import CsvLocationRepository, CsvTierRepository

    locations = CsvLocationRepository(locations_path)
    tiers = CsvTierRepository(tiers_path)
    out(f"Training the price model on {rides_path} (retrain-on-launch)...")
    predictor = ModelPredictor(rides_path, max_rows=max_train_rows)
    out(f"Model trained on {predictor.n_train:,} rides.\n")
    service = QuoteService(locations, tiers, predictor, SystemClock())
    run_simulator(service, locations, input_fn=input_fn, out=out)
    return 0


def _handle_simulate(args: argparse.Namespace) -> int:
    return run_simulate(args.rides, args.max_train_rows)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level ``uber`` argument parser with its three subcommands."""
    parser = argparse.ArgumentParser(
        prog="uber",
        description="Uber price-prediction toolkit: generate data, run the modeling arc, "
        "or simulate ride quotes.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser(
        "generate-data",
        add_help=False,
        help="generate the synthetic Madrid dataset (see: uber generate-data --help)",
    )
    gen.set_defaults(func=_handle_generate_data)

    arc_cmd = sub.add_parser(
        "run-arc",
        add_help=False,
        help="run the eight-part regression arc (see: uber run-arc --help)",
    )
    arc_cmd.set_defaults(func=_handle_run_arc)

    sim = sub.add_parser("simulate", help="interactive A->B ride-price simulator")
    sim.add_argument(
        "--rides",
        type=Path,
        default=paths.RAW_DIR / paths.RIDES_CSV,
        help="rides.csv used to retrain the model on launch (default: data/raw/rides.csv)",
    )
    sim.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="cap training rows for a faster launch (default: all rows)",
    )
    sim.set_defaults(func=_handle_simulate)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv`` and dispatch to the selected subcommand's handler.

    ``generate-data`` and ``run-arc`` forward any extra options to their module CLIs, so unknown
    args are collected (``args.forward``) rather than rejected; ``simulate`` stays strict.
    """
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)
    if args.command == "simulate" and extra:
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    args.forward = extra
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
