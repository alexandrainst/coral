"""
.. include:: ../../README.md
"""

import importlib.metadata
import logging
import sys

from .asr import build_and_store_ftspeech, finetune_wav2vec2  # noqa: F401
from .utils import block_terminal_output

# Fetches the version of the package as defined in pyproject.toml
__version__ = importlib.metadata.version(__package__)


# Block terminal output, if we are not running tests
if not hasattr(sys, "_called_from_test"):
    block_terminal_output()


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
