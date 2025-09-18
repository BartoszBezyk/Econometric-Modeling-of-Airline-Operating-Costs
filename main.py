"""Convenience wrapper that executes the linear workflow script."""

from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    workflow_path = Path(__file__).with_name("workflow.py")
    runpy.run_path(str(workflow_path), run_name="__main__")
