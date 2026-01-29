"""Pytest configuration for backend tests.

This module ensures the project root (the `web-gui` directory) is on sys.path
so that `backend` can be imported directly in tests without modifying sys.path
in each test file.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
ROOT_STR = str(ROOT)
if ROOT_STR not in sys.path:
    # Prepend to sys.path so local `backend` package is preferred during tests
    sys.path.insert(0, ROOT_STR)
