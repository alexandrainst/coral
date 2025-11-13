"""The CoRal project.

.. include:: ../../README.md
"""

import importlib.metadata
import logging
import sys

from .utils import block_terminal_output

# Fetches the version of the package as defined in pyproject.toml
__version__ = importlib.metadata.version(__package__ or "")


# Block terminal output, if we are not running tests
if not hasattr(sys, "_called_from_test"):
    block_terminal_output()


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s ⋅ %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
