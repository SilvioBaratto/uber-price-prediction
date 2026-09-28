"""The unified ``uber`` command-line interface (composition root).

Four subcommands:

* ``generate-data`` — build the synthetic Madrid dataset (forwards to
  :func:`uber.datagen.generate_data.main`).
* ``run-arc`` — run the eight-part regression arc (forwards to :func:`uber.modeling.arc.main`).
* ``train`` — fit the price model once and persist its weights to disk
  (:meth:`uber.infrastructure.predictors.ModelPredictor.save`).
* ``simulate`` — interactive A->B ride-price simulator; wires the CSV repositories, a
  :class:`~uber.infrastructure.predictors.ModelPredictor` (loaded from ``--model`` weights when
  given, else retrained on launch), a system clock and the
  :class:`~uber.application.quoting.QuoteService`, then runs the terminal loop.

This module is the only place the layers are wired together; everything else depends inward.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from uber.infrastructure import paths

# Default cap on training rows for the production model. The tier-interaction pipeline expands a
# degree-2 polynomial over the ~31-column one-hot design (~500 dense columns), so an uncapped fit on
# the multi-million-row rides.csv would need tens of GB. R² is ~invariant to N for this model, so a
# seeded sample of this size is memory-safe and quality-equivalent to the full set. Override with
# ``--max-train-rows`` (a larger value uses more rows; pass a huge number to effectively uncap).
DEFAULT_MAX_TRAIN_ROWS = 500_000


def _positive_int(value: str) -> int:
    """argparse type: a strictly positive integer (rejects 0 and negatives with a usage error)."""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value}")
    return number


def _handle_generate_data(args: argparse.Namespace) -> int:
    from uber.datagen import generate_data

    generate_data.main(args.forward)
    return 0


def _handle_run_arc(args: argparse.Namespace) -> int:
    from uber.modeling import arc

    arc.main(args.forward)
    return 0


def run_train(
    rides_path: Path | str,
    max_train_rows: int | None,
    out_path: Path | str,
    *,
    out: Callable[[str], None] = print,
) -> int:
    """Fit the price model and save its weights to ``out_path`` (the seam behind ``train``).

    Fits the pinned poly-OLS model from ``rides_path`` (optionally capped by ``max_train_rows``)
    and persists it with :meth:`ModelPredictor.save`, so ``simulate --model`` can reload it
    instantly instead of retraining. ``out`` is injected so tests can capture the messages.
    """
    from uber.infrastructure.predictors import ModelPredictor

    out(f"Training the price model on {rides_path} ...")
    predictor = ModelPredictor(rides_path, max_rows=max_train_rows)
    saved = predictor.save(out_path)
    out(f"Model trained on {predictor.n_train:,} rides -> saved to {saved}")
    return 0


def _handle_train(args: argparse.Namespace) -> int:
    return run_train(args.rides, args.max_train_rows, args.out)


def run_simulate(
    rides_path: Path | str,
    max_train_rows: int | None,
    *,
    model_path: Path | str | None = None,
    locations_path: Path | None = None,
    tiers_path: Path | None = None,
    input_fn: Callable[[str], str] = input,
    out: Callable[[str], None] = print,
) -> int:
    """Wire the simulate composition and run the loop (the testable seam behind ``simulate``).

    Repositories default to the committed ``data/raw`` catalogs. If ``model_path`` is given and
    exists, the model is **loaded** from those saved weights (no refitting); otherwise it is
    retrained on launch from ``rides_path`` (optionally capped by ``max_train_rows``).
    ``input_fn``/``out`` are injected so an end-to-end run can be driven and captured by tests.
    """
    from uber.application.quoting import QuoteService
    from uber.cli.simulator import run_simulator
    from uber.infrastructure.clock import SystemClock
    from uber.infrastructure.predictors import ModelPredictor
    from uber.infrastructure.repositories import CsvLocationRepository, CsvTierRepository

    locations = CsvLocationRepository(locations_path)
    tiers = CsvTierRepository(tiers_path)
    if model_path is not None and Path(model_path).exists():
        out(f"Loading saved price model from {model_path} ...")
        predictor = ModelPredictor.load(model_path)
        out(f"Model loaded (trained on {predictor.n_train:,} rides).\n")
    else:
        if model_path is not None:
            out(f"No saved model at {model_path}; retraining on launch instead.")
        out(f"Training the price model on {rides_path} (retrain-on-launch)...")
        predictor = ModelPredictor(rides_path, max_rows=max_train_rows)
        out(f"Model trained on {predictor.n_train:,} rides.\n")
    service = QuoteService(locations, tiers, predictor, SystemClock())
    run_simulator(service, locations, input_fn=input_fn, out=out)
    return 0


def _handle_simulate(args: argparse.Namespace) -> int:
    return run_simulate(args.rides, args.max_train_rows, model_path=args.model)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level ``uber`` argument parser with its four subcommands."""
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

    default_model = paths.MODELS_DIR / paths.MODEL_JOBLIB
    train_cmd = sub.add_parser("train", help="fit the price model and save its weights to disk")
    train_cmd.add_argument(
        "--rides",
        type=Path,
        default=paths.RAW_DIR / paths.RIDES_CSV,
        help="rides.csv to fit the model on (default: data/raw/rides.csv)",
    )
    train_cmd.add_argument(
        "--max-train-rows",
        type=_positive_int,
        default=DEFAULT_MAX_TRAIN_ROWS,
        help="fit on a seeded random sample of at most this many rides "
        f"(default: {DEFAULT_MAX_TRAIN_ROWS:,}; memory-safe and R²-equivalent to the full set)",
    )
    train_cmd.add_argument(
        "--out",
        type=Path,
        default=default_model,
        help=f"where to save the model weights (default: {default_model})",
    )
    train_cmd.set_defaults(func=_handle_train)

    sim = sub.add_parser("simulate", help="interactive A->B ride-price simulator")
    sim.add_argument(
        "--rides",
        type=Path,
        default=paths.RAW_DIR / paths.RIDES_CSV,
        help="rides.csv used to retrain the model on launch (default: data/raw/rides.csv)",
    )
    sim.add_argument(
        "--max-train-rows",
        type=_positive_int,
        default=DEFAULT_MAX_TRAIN_ROWS,
        help="when retraining on launch (no --model), fit on a seeded random sample of at most "
        f"this many rides (default: {DEFAULT_MAX_TRAIN_ROWS:,}); ignored when --model is loaded",
    )
    sim.add_argument(
        "--model",
        type=Path,
        default=None,
        help="load saved model weights from this path instead of retraining on launch "
        f"(e.g. {default_model}); falls back to retraining if the file is missing",
    )
    sim.set_defaults(func=_handle_simulate)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv`` and dispatch to the selected subcommand's handler.

    ``generate-data`` and ``run-arc`` forward any extra options to their module CLIs, so unknown
    args are collected (``args.forward``) rather than rejected; ``train`` and ``simulate`` stay
    strict.
    """
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)
    if args.command in {"simulate", "train"} and extra:
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    args.forward = extra
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
