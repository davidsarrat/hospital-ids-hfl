from __future__ import annotations

import sys
from pathlib import Path


def bootstrap() -> None:
    """Make repository-root imports work when scripts are run as files."""

    root = Path(__file__).resolve().parents[1]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
