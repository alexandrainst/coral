"""
.. include:: ../../README.md
"""

import importlib.metadata
import logging
import sys

from .asr import finetune_wav2vec2, finetune_whisper
from .utils import block_terminal_output

# Fetches the version of the package as defined in pyproject.toml
__version__ = importlib.metadata.version(__package__)

# Block terminal output, if we are not running tests
if not hasattr(sys, "_called_from_test"):
    block_terminal_output()

# Set up logging
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] <%(name)s> %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
