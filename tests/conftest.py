"""Make the flat top-level modules importable from any test file, and keep tests off the real home directory.

Several scanner modules create ``~/Downloads/market_cache/...`` at import time, so ``HOME`` is pointed at a
throw-away directory before anything is imported.
"""
import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

_FAKE_HOME = tempfile.mkdtemp(prefix="bazaartalks-test-home-")
os.environ["HOME"] = _FAKE_HOME
atexit.register(shutil.rmtree, _FAKE_HOME, ignore_errors=True)

ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
