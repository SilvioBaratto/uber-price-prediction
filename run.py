"""Back-compat shim: ``python run.py`` runs the eight-part regression arc.

The orchestration now lives in :mod:`uber.modeling.arc` (the shared plumbing in
:mod:`uber.modeling.pipeline`). This forwarder keeps the historical entry point —
``python run.py --rides ... --seed ... --output-dir ... --charts`` — working with the
same flags. Prefer ``python -m uber.modeling.arc`` (or the ``uber run-arc`` subcommand).
"""

from __future__ import annotations

from uber.modeling.arc import main

if __name__ == "__main__":
    main()
