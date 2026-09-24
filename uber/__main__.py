"""``python -m uber`` entry point — dispatches to the unified CLI."""

from __future__ import annotations

from uber.cli.app import main

if __name__ == "__main__":
    raise SystemExit(main())
