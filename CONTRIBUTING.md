# Contributing

Thanks for your interest in improving **uber-price-prediction** — a teaching
companion to the *Let's Build Uber's Algorithm* series. Contributions that keep
the project a clear, honest learning resource are very welcome.

## Getting set up

```bash
git clone https://github.com/neuroespresso/uber-price-prediction
cd uber-price-prediction
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Regenerate the dataset locally (the large `rides.csv` is git-ignored and built
on demand):

```bash
uber generate-data --n-drivers 200   # a fast dev-sized dataset
```

## The development loop (test-first)

Every change earns a test. The loop we follow for each task:

1. **RED** — write a failing test that pins the behavior you want.
2. **GREEN** — write the minimum code to make it pass.
3. **Regression** — run the whole suite; keep it green.
4. **Check** — `ruff check .`, `ruff format .`, and `pyright`.
5. **Commit** — one focused commit per logical change.

```bash
pytest -q              # full suite
ruff check . && ruff format --check .
pyright                # informational; a known pandas/numpy stub baseline is tolerated
```

## Architecture: the dependency rule

The package is organized in clean-architecture layers; **dependencies always
point inward** and inner layers never import outer ones:

```
cli  ->  application  ->  domain
 \                         ^
  \--> infrastructure -----/   (implements the application ports)
datagen, modeling  ->  domain + infrastructure   (supporting feature packages)
```

- `domain/` — pure business rules (entities, pricing, config). No I/O, no framework.
- `application/` — use cases + ports (Protocols). Depends only on `domain`.
- `infrastructure/` — adapters (CSV repositories, predictors, clock, paths) that
  implement the ports and do the I/O.
- `cli/` — the terminal entry points and the composition root that wires it all up.

When adding a capability, put pure logic in `domain`, an interface in
`application/ports.py`, and the concrete adapter in `infrastructure`.

## Behavior-preserving refactors

Calibrated constants (`SEED`, `R2_BAND`, `NOISE_STD`, `COMMISSION_RATE`, the
tariffs and demand dynamics) and the CSV schema are load-bearing for the lessons.
Refactors must not change their values — the existing tests are the guard.

## Commit & PR style

- Small, focused commits with an imperative subject (e.g. `Add CsvTierRepository`).
- Reference the issue you're closing in the PR description.
- Fill in the PR checklist; make sure `ruff` and `pytest` pass.

By contributing you agree that your work is licensed under the project's
[MIT License](LICENSE).
