#!/usr/bin/env python3
"""Thin wrapper so the trainer runs without installing the package."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aml_gnn.cli.train import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
