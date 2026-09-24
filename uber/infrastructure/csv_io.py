"""Generic CSV write mechanics (infrastructure layer).

The single low-level file-writing helper, kept free of any domain knowledge so the
repository adapters and the data-generation pipeline all persist frames the same way.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    """Write ``df`` to ``path`` as UTF-8 CSV without the index. Returns ``path``.

    Creates the parent directory if it does not yet exist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    return path
